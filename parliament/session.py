"""
Science Parliament — core session logic.

This module contains only pure functions with no module-level side effects.
It can be safely imported by any script without triggering directory creation,
environment variable changes, or OASIS/patches initialization.

Initialization (patches, OASIS imports, env vars) is the caller's responsibility.
See run_parliament.py for demo usage, or judgement/run_benchmark.py for batch usage.
"""

import json
import os
import shutil
import sqlite3
import time

# ---------------------------------------------------------------------------
# Project root — all output paths are relative to this, regardless of cwd.
# parliament/session.py → dirname twice → zhizheng2/
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output")
LOG_BASE = os.path.join(PROJECT_ROOT, "log")

# Actions that change the Judge's context (new content or score changes).
CONTEXT_ACTIONS = frozenset({
    "create_post", "create_comment",
    "like_post", "dislike_post",
    "like_comment", "dislike_comment",
})

EARLY_STOP_ROUNDS = 2


# ---------------------------------------------------------------------------
# Initialization — call once per process before any run_session() call.
# ---------------------------------------------------------------------------

_initialized = False


def init(log_dir: str | None = None):
    """One-time setup: set PARLIAMENT_LOG_DIR, import patches, import OASIS,
    redirect loggers.  Safe to call multiple times (only runs once).

    Args:
        log_dir: Where OASIS logs go. Defaults to LOG_BASE/<timestamp>.
    """
    global _initialized
    if _initialized:
        return

    from datetime import datetime
    if log_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(LOG_BASE, ts)
    os.makedirs(log_dir, exist_ok=True)
    os.environ["PARLIAMENT_LOG_DIR"] = os.path.abspath(log_dir)

    # patches must be imported before oasis agents are created.
    # It applies monkey-patches and triggers OASIS module-level code
    # (which creates ./log/ — harmless, we redirect below).
    import patches  # noqa: F401
    import oasis  # noqa: F401

    patches._redirect_loggers_to_run_dir()
    _initialized = True


def create_model(base_url: str | None = None):
    """Create the CAMEL model.

    Args:
        base_url: Override the API base URL (e.g. for multi-GPU setups
                  where each GPU runs vLLM on a different port).
                  If None, uses the value from config.py.
    """
    from config import MODEL_NAME, MODEL_BASE_URL, API_KEY

    if API_KEY:
        os.environ["OPENAI_API_KEY"] = API_KEY

    url = base_url or MODEL_BASE_URL
    if url:
        os.environ["OPENAI_API_BASE_URL"] = url

    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=MODEL_NAME,
    )


# ---------------------------------------------------------------------------
# Session — run one parliament discussion
# ---------------------------------------------------------------------------

def _build_agents(model, agent_graph, tools, question, num_agents):
    """Create scientist agents with the question baked into their system prompt."""
    from camel.prompts import TextPrompt
    from config import AVAILABLE_ACTIONS_LIST, get_agent_names
    from prompts import SCIENTIST_SYSTEM
    from oasis import SocialAgent, UserInfo
    from oasis.social_platform.typing import ActionType

    template = TextPrompt(SCIENTIST_SYSTEM)
    action_types = [ActionType[a] for a in AVAILABLE_ACTIONS_LIST]
    names = get_agent_names(num_agents)

    agents = []
    for i in range(num_agents):
        name = names[i]
        agent = SocialAgent(
            agent_id=i,
            user_info=UserInfo(
                user_name=name,
                name=name,
                description=f"Scientist {name}, member of the Science Parliament",
                profile={"name": name, "question": question},
                recsys_type="reddit",
            ),
            user_info_template=template,
            agent_graph=agent_graph,
            model=model,
            available_actions=action_types,
            tools=tools,
            single_iteration=False,
        )
        agent_graph.add_agent(agent)
        agents.append(agent)
    return agents


def _print_round_stats(db_path: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM post")
    n_posts = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM comment")
    n_comments = c.fetchone()[0]
    c.execute(
        "SELECT COUNT(*) FROM trace WHERE action IN ('like_post','dislike_post')"
    )
    n_votes = c.fetchone()[0]
    conn.close()
    print(f"  → {n_posts} posts, {n_comments} comments, {n_votes} votes")


def _count_context_actions(db_path: str, prev_rowid: int) -> tuple[int, int]:
    """Count context-affecting actions since prev_rowid. Returns (count, new_max_rowid)."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    actions = list(CONTEXT_ACTIONS)
    placeholders = ",".join("?" for _ in actions)
    c.execute(f"""
        SELECT COUNT(*) FROM trace
        WHERE action IN ({placeholders}) AND rowid > ?
    """, (*actions, prev_rowid))
    n = c.fetchone()[0]
    c.execute("SELECT COALESCE(MAX(rowid), 0) FROM trace")
    new_max = c.fetchone()[0]
    conn.close()
    return n, new_max


def dump_discussion(db_path: str) -> list[dict]:
    """Export all posts and comments from the database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT p.post_id, u.name, p.content, p.num_likes, p.num_dislikes
        FROM post p JOIN user u ON p.user_id = u.user_id
        ORDER BY p.post_id
    """)
    posts = []
    for post_id, author, content, likes, dislikes in c.fetchall():
        c.execute("""
            SELECT u.name, cm.content, cm.num_likes, cm.num_dislikes
            FROM comment cm JOIN user u ON cm.user_id = u.user_id
            WHERE cm.post_id = ? ORDER BY cm.comment_id
        """, (post_id,))
        comments = [
            {"author": r[0], "content": r[1], "likes": r[2], "dislikes": r[3]}
            for r in c.fetchall()
        ]
        posts.append({
            "post_id": post_id,
            "author": author,
            "content": content,
            "score": (likes or 0) - (dislikes or 0),
            "comments": comments,
        })
    conn.close()
    return posts


def _snapshot(db_path: str) -> dict:
    """Capture current max IDs for all mutable tables, used for rollback."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    snap = {}
    for key, sql in (
        ("max_post_id",             "SELECT COALESCE(MAX(post_id), 0) FROM post"),
        ("max_comment_id",          "SELECT COALESCE(MAX(comment_id), 0) FROM comment"),
        ("max_trace_rowid",         "SELECT COALESCE(MAX(rowid), 0) FROM trace"),
        ("max_follow_rowid",        "SELECT COALESCE(MAX(rowid), 0) FROM follow"),
        ("max_like_rowid",          "SELECT COALESCE(MAX(rowid), 0) FROM 'like'"),
        ("max_dislike_rowid",       "SELECT COALESCE(MAX(rowid), 0) FROM 'dislike'"),
        ("max_comment_like_rowid",  "SELECT COALESCE(MAX(rowid), 0) FROM comment_like"),
        ("max_comment_dislike_rowid", "SELECT COALESCE(MAX(rowid), 0) FROM comment_dislike"),
    ):
        c.execute(sql)
        snap[key] = c.fetchone()[0]
    conn.close()
    return snap


async def run_session(
    question: str,
    model,
    output_dir: str,
    num_agents: int | None = None,
    num_rounds: int | None = None,
    concurrency: int | None = None,
) -> dict:
    """Run a full Science Parliament session on one question.

    The caller must have called init() first.
    """
    from config import (
        DEFAULT_NUM_AGENTS, NUM_ROUNDS, LLM_CONCURRENCY,
        MAX_ITERATION, REFRESH_REC_POST_COUNT, MAX_REC_POST_LEN,
        ALLOW_SELF_RATING, AGENT_FAIL_THRESHOLD, VLLM_MAX_MODEL_LEN,
    )
    import oasis
    from oasis import LLMAction, ManualAction, AgentGraph
    from oasis.social_platform.channel import Channel
    from oasis.social_platform.platform import Platform
    from oasis.social_platform.typing import ActionType
    from tools import load_tools
    from context import (
        reset as ctx_reset, compress_posts, rollback_to,
        build_context, context_overflows,
    )
    from patches import ContextOverflowError, reset_round_stats, round_fail_count, round_agent_count
    import patches as _patches

    if num_agents is None:
        num_agents = DEFAULT_NUM_AGENTS
    if num_rounds is None:
        num_rounds = NUM_ROUNDS
    if concurrency is None:
        concurrency = LLM_CONCURRENCY

    os.makedirs(output_dir, exist_ok=True)
    os.environ["PARLIAMENT_RUN_DIR"] = os.path.abspath(output_dir)

    db_path = os.path.join(output_dir, "parliament.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)

    ctx_reset()
    session_t0 = time.time()

    _patches.log_event({
        "event": "session_start",
        "question": question[:500],
        "num_agents": num_agents,
        "num_rounds": num_rounds,
        "concurrency": concurrency,
        "max_iteration": MAX_ITERATION,
        "vllm_max_model_len": VLLM_MAX_MODEL_LEN,
    })

    tools = load_tools()
    agent_graph = AgentGraph()
    agents = _build_agents(model, agent_graph, tools, question, num_agents)

    channel = Channel()
    platform = Platform(
        db_path=db_path,
        channel=channel,
        recsys_type="reddit",
        allow_self_rating=ALLOW_SELF_RATING,
        show_score=True,
        max_rec_post_len=MAX_REC_POST_LEN,
        refresh_rec_post_count=REFRESH_REC_POST_COUNT,
    )

    env = oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=db_path,
        semaphore=concurrency,
    )
    await env.reset()

    print(f"\n{'='*70}")
    print("SCIENCE PARLIAMENT")
    print(f"{'='*70}")
    print(f"Question: {question[:300]}{'...' if len(question) > 300 else ''}")
    print(f"Agents: {num_agents}  |  Rounds: {num_rounds}  |  "
          f"Concurrency: {concurrency}  |  Max iteration: {MAX_ITERATION}")
    print(f"{'='*70}\n")

    opening = (
        "The Science Parliament is now in session. "
        "The question is in your briefing. Let's begin."
    )
    await env.step({
        agents[0]: ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": opening},
        )
    })
    print("[Round 0] Session opened.\n")

    idle_streak = 0
    prev_rowid = 0
    early_stopped = False
    stop_reason = "max_rounds"
    actual_rounds = 0
    round_boundaries = []

    for round_num in range(1, num_rounds + 1):
        print(f"[Round {round_num}/{num_rounds}]")
        _patches.reset_round_stats()
        round_t0 = time.time()

        # Snapshot before the round (for rollback)
        snap = _snapshot(db_path)

        actions = {agent: LLMAction() for agent in agents}
        try:
            await env.step(actions)
        except ContextOverflowError:
            # An agent's context exceeded the safety limit.
            # Rollback this round, compress, then check if we can continue.
            print("  Context overflow detected — rolling back and compressing...")
            _patches.log_event({"event": "overflow", "round": round_num, "action": "rollback_and_compress"})
            rollback_to(db_path, snap)
            await compress_posts(output_dir)

            # Check if compression was enough
            test_msgs = build_context(
                agent_id=0, agent_name="test",
                system_content=agents[0].system_message.content,
            )
            if context_overflows(test_msgs):
                print("  Still too large after compression — stopping.")
                early_stopped = True
                stop_reason = "context_overflow"
                break

            # Compression helped — retry this round
            print("  Compression done — retrying round...")
            _patches.reset_round_stats()
            try:
                await env.step(actions)
            except ContextOverflowError:
                rollback_to(db_path, snap)
                print("  Still overflowing after compression — stopping.")
                early_stopped = True
                stop_reason = "context_overflow"
                break

        # Check if too many agents failed (timeout / error) this round
        fail_n = _patches.round_fail_count
        total_n = _patches.round_agent_count
        if total_n > 0 and fail_n / total_n >= AGENT_FAIL_THRESHOLD:
            rollback_to(db_path, snap)
            print(f"  {fail_n}/{total_n} agents failed — rolled back, stopping.")
            early_stopped = True
            stop_reason = "agent_failures"
            break

        _print_round_stats(db_path)

        n_ctx, prev_rowid = _count_context_actions(db_path, prev_rowid)
        if n_ctx == 0:
            idle_streak += 1
            print(f"  idle streak: {idle_streak}/{EARLY_STOP_ROUNDS}")
        else:
            idle_streak = 0

        actual_rounds = round_num
        round_duration = round(time.time() - round_t0, 2)

        end_snap = _snapshot(db_path)

        _patches.log_event({
            "event": "round_end",
            "round": round_num,
            "duration_s": round_duration,
            "agents_succeeded": _patches.round_agent_count - _patches.round_fail_count,
            "agents_failed": _patches.round_fail_count,
            "posts": end_snap["max_post_id"],
            "comments": end_snap["max_comment_id"],
            "context_actions": n_ctx,
            "idle_streak": idle_streak,
        })
        round_boundaries.append({
            "round": round_num,
            "max_post_id": end_snap["max_post_id"],
            "max_comment_id": end_snap["max_comment_id"],
        })

        try:
            from visualize import generate_html
            generate_html(db_path, output_dir,
                          question=question,
                          current_round=round_num,
                          num_rounds=num_rounds)
        except Exception:
            pass

        if idle_streak >= EARLY_STOP_ROUNDS:
            print(f"\n  Early stop: {EARLY_STOP_ROUNDS} consecutive idle rounds.")
            early_stopped = True
            stop_reason = "idle"
            break

    rb_path = os.path.join(output_dir, "round_map.json")
    with open(rb_path, "w", encoding="utf-8") as f:
        json.dump(round_boundaries, f)

    print(f"\n{'='*70}")
    label = f"STOPPED ({stop_reason})" if early_stopped else "SESSION COMPLETE"
    print(f"{label} after {actual_rounds} rounds")
    print(f"{'='*70}\n")

    posts = dump_discussion(db_path)
    session = {
        "question": question,
        "num_rounds": num_rounds,
        "num_rounds_completed": actual_rounds,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "num_agents": num_agents,
        "discussion": posts,
        "db_path": os.path.abspath(db_path),
    }

    session_path = os.path.join(output_dir, "session.json")
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    print(f"Discussion saved to {session_path}")
    print(f"Total: {len(posts)} posts")

    sorted_posts = sorted(posts, key=lambda p: p["score"], reverse=True)
    print("\n--- Top Voted Posts ---")
    for p in sorted_posts[:5]:
        score_str = f"+{p['score']}" if p['score'] >= 0 else str(p['score'])
        preview = (p["content"] or "")[:120].replace("\n", " ")
        print(f"  [{score_str}] {p['author']}: {preview}...")

    _patches.log_event({
        "event": "session_end",
        "total_rounds": actual_rounds,
        "stop_reason": stop_reason,
        "total_posts": len(posts),
        "duration_s": round(time.time() - session_t0, 2),
    })

    await env.close()
    return session
