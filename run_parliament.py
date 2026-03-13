"""
Science Parliament: A multi-agent scientific reasoning system built on OASIS.

Scientists collaborate on a Reddit-like forum to solve a scientific question.
The question is embedded in every agent's system prompt so it's always visible.
Each round, agents observe the forum, think (optionally use tools), then post
their analysis or build on others' work.
"""

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Create timestamped run directory BEFORE importing OASIS (which sets up loggers)
_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_run_dir = os.path.join(
    os.environ.get("PARLIAMENT_OUTPUT_DIR", "output"), _timestamp
)
_log_dir = os.path.join(_run_dir, "log")
os.makedirs(_log_dir, exist_ok=True)
os.environ["PARLIAMENT_LOG_DIR"] = _log_dir

import patches  # noqa: F401 — applies all monkey-patches at import time

from camel.models import ModelFactory
from camel.prompts import TextPrompt
from camel.toolkits import SymPyToolkit
from camel.types import ModelPlatformType

import oasis
from oasis import LLMAction, ManualAction, SocialAgent, AgentGraph, UserInfo
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

from config import (
    AVAILABLE_ACTIONS_LIST,
    DEFAULT_NUM_AGENTS,
    NUM_ROUNDS,
    LLM_CONCURRENCY,
    SCIENTIST_PROMPT_TEMPLATE,
    MODEL_NAME,
    MODEL_BASE_URL,
    API_KEY,
    OUTPUT_DIR,
    REFRESH_REC_POST_COUNT,
    MAX_REC_POST_LEN,
    ALLOW_SELF_RATING,
    get_agent_names,
)


def build_agents(
    model, agent_graph: AgentGraph, tools, question: str,
    num_agents: int = DEFAULT_NUM_AGENTS,
) -> list[SocialAgent]:
    """Create scientist agents with the question baked into their prompt."""
    template = TextPrompt(SCIENTIST_PROMPT_TEMPLATE)
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


def print_round_stats(db_path: str):
    """Print a quick summary of the current state."""
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


def dump_discussion(db_path: str) -> list[dict]:
    """Dump all posts and comments from the database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        SELECT p.post_id, u.name, p.content, p.num_likes, p.num_dislikes,
               p.created_at
        FROM post p JOIN user u ON p.user_id = u.user_id
        ORDER BY p.post_id
    """)
    posts = []
    for post_id, author, content, likes, dislikes, created in c.fetchall():
        c.execute("""
            SELECT u.name, cm.content, cm.num_likes, cm.num_dislikes
            FROM comment cm JOIN user u ON cm.user_id = u.user_id
            WHERE cm.post_id = ?
            ORDER BY cm.comment_id
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


async def run_parliament(
    question: str,
    model,
    num_agents: int = DEFAULT_NUM_AGENTS,
    num_rounds: int = NUM_ROUNDS,
    concurrency: int = LLM_CONCURRENCY,
    output_dir: str = OUTPUT_DIR,
):
    """Run a full Science Parliament session on one question."""
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "parliament.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)

    tools = SymPyToolkit().get_tools()

    agent_graph = AgentGraph()
    agents = build_agents(model, agent_graph, tools, question, num_agents)

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
          f"Refresh posts: {REFRESH_REC_POST_COUNT}  |  "
          f"Max iteration: {from_config('MAX_ITERATION')}")
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

    for round_num in range(1, num_rounds + 1):
        print(f"[Round {round_num}/{num_rounds}]")
        actions = {agent: LLMAction() for agent in agents}
        await env.step(actions)
        print_round_stats(db_path)

    print(f"\n{'='*70}")
    print("SESSION COMPLETE")
    print(f"{'='*70}\n")

    posts = dump_discussion(db_path)
    session = {
        "question": question,
        "num_rounds": num_rounds,
        "num_agents": num_agents,
        "discussion": posts,
    }

    log_path = os.path.join(output_dir, "session.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    print(f"Discussion saved to {log_path}")
    print(f"Total: {len(posts)} posts")

    sorted_posts = sorted(posts, key=lambda p: p["score"], reverse=True)
    print(f"\n--- Top Voted Posts ---")
    for p in sorted_posts[:5]:
        score_str = f"+{p['score']}" if p['score'] >= 0 else str(p['score'])
        preview = (p["content"] or "")[:120].replace("\n", " ")
        print(f"  [{score_str}] {p['author']}: {preview}...")

    await env.close()
    return session


def from_config(name: str):
    """Read a value from config module by name."""
    import config
    return getattr(config, name)


def parse_args():
    parser = argparse.ArgumentParser(description="Science Parliament")
    parser.add_argument(
        "--question", type=str, default=None,
        help="The scientific question to discuss",
    )
    parser.add_argument(
        "--question_file", type=str, default=None,
        help="Path to a text file containing the question",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    api_key = API_KEY
    base_url = MODEL_BASE_URL

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_API_BASE_URL"] = base_url

    question = args.question
    if question is None and args.question_file:
        with open(args.question_file, "r") as f:
            question = f.read().strip()

    if question is None:
        question = input("Enter the scientific question:\n> ")

    shutil.copy2("config.py", os.path.join(_run_dir, "config.py"))

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=MODEL_NAME,
    )

    await run_parliament(
        question=question,
        model=model,
        output_dir=_run_dir,
    )

    print(f"\nAll outputs saved to: {_run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
