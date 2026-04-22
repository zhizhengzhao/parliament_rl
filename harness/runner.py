"""Global experiment scheduler — event-driven polling mode.

Runner wakes on actor submit_event (or 60s timeout), fetches new content,
and distributes only when posts or comments exist. Votes are accumulated
and delivered alongside the next post/comment batch.

Judge votes never wake the runner. Two processing sets track actors and
judges independently. Session ends when actors idle (coupled mode) or
when every actor has called `leave` (independent mode); runner waits
for judges to finish before finalizing.

Two `config.json` flags drive the 2×2 ablation:
  actor_context_coupled (bool)  — distribute peers' posts/comments/votes
                                  to actors? false ⇒ each actor sees only
                                  its own history (and judge votes if
                                  visible). Default true.
  judge_votes_visible   (bool)  — push judge votes to actors? Default true.
                                  Override at launch with
                                  PRL_JUDGE_VOTES_VISIBLE=0 (or =false).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import aiohttp

from .agent import AgentResult, run_agent
from .prompts import apply_name_map, assign_session_names, get_config
from .tools import IdMap

PROJECT_DIR = Path(__file__).resolve().parent.parent

ANONYMOUS_VOTER = "Anonymous Scientist"
RUNNER_WAIT_TIMEOUT_S = 60
JUDGE_WAITDOWN_S = 120
DISTRIBUTION_DELAY_S = 0.1
TYPE_ORDER = {"post": 0, "comment": 1, "vote": 2}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _resolve_judge_visibility(cfg: dict) -> bool:
    """`PRL_JUDGE_VOTES_VISIBLE` overrides the config flag at launch.

    Lets us pick the 2×2 cell on the command line without editing
    config.json (e.g. `PRL_JUDGE_VOTES_VISIBLE=0` flips Parliament → BlindParliament).
    """
    env = os.environ.get("PRL_JUDGE_VOTES_VISIBLE")
    if env is None:
        return bool(cfg.get("judge_votes_visible", True))
    return env.strip().lower() not in ("0", "false", "no", "off", "")


# ── Types ─────────────────────────────────────────────────

class Cursors(NamedTuple):
    """Highest IDs we've already distributed in this session."""
    post: int = 0
    comment: int = 0
    vote: int = 0


class Fetched(NamedTuple):
    posts: list[dict]
    comments: list[dict]
    actor_votes: list[dict]
    judge_votes: list[dict]
    cursors: Cursors

    @property
    def has_discussion(self) -> bool:
        return bool(self.posts or self.comments)


# ── Sync helper for setup phase ───────────────────────────

def _api_sync(base_url: str, method: str, path: str, key: str) -> dict | list:
    req = urllib.request.Request(
        base_url + path,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        method=method,
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as e:
        print(f"  API error: {method} {path}: {e}")
        return {}


# ── Fetch & shape ─────────────────────────────────────────

def _build_score_lookup(all_posts: list[dict]) -> tuple[dict[int, int], dict[int, int]]:
    posts = {p["post_id"]: p.get("score", 0) for p in all_posts}
    comments = {c["comment_id"]: c.get("score", 0)
                for p in all_posts for c in p.get("comments", [])}
    return posts, comments


def _shape_posts(all_posts: list[dict], after: int) -> tuple[list[dict], int]:
    """Filter posts above the cursor, returning items + new max id."""
    out, new_max = [], after
    for p in all_posts:
        pid = p.get("post_id") or 0
        if pid > after:
            out.append({"type": "post", "id": pid,
                        "author": p.get("author", "?"),
                        "content": p.get("content", "")})
            if pid > new_max:
                new_max = pid
    return out, new_max


def _shape_comments(all_posts: list[dict], after: int) -> tuple[list[dict], int]:
    out, new_max = [], after
    for p in all_posts:
        for c in p.get("comments", []):
            cid = c.get("comment_id") or 0
            if cid > after:
                out.append({"type": "comment", "id": cid,
                            "post_id": p["post_id"],
                            "author": c.get("author", "?"),
                            "content": c.get("content", "")})
                if cid > new_max:
                    new_max = cid
    return out, new_max


def _shape_votes(all_votes: list[dict], after: int,
                 post_scores: dict[int, int], comment_scores: dict[int, int]
                 ) -> tuple[list[dict], list[dict], int]:
    actor_votes, judge_votes, new_max = [], [], after
    for v in all_votes:
        vid = v.get("vote_id") or 0
        if vid <= after:
            continue
        target_type = "post" if v.get("post_id") else "comment"
        target_id = v.get("post_id") or v.get("comment_id")
        score_lookup = post_scores if target_type == "post" else comment_scores
        item = {"type": "vote", "id": vid,
                "target_type": target_type, "target_id": target_id,
                "value": v.get("value", 0),
                "previous_value": v.get("previous_value"),
                "author": v.get("author", "?"),
                "target_score": score_lookup.get(target_id, 0)}
        (judge_votes if v.get("role") == "judge" else actor_votes).append(item)
        if vid > new_max:
            new_max = vid
    return actor_votes, judge_votes, new_max


async def _fetch_new_content(http: aiohttp.ClientSession, parliament_url: str,
                             admin_key: str, session_id: str,
                             cursors: Cursors) -> Fetched:
    """Pull all new posts/comments/votes since the given cursors."""
    headers = {"Authorization": f"Bearer {admin_key}",
               "Content-Type": "application/json"}

    async with http.get(f"{parliament_url}/admin/sessions/{session_id}/posts",
                        headers=headers) as resp:
        all_posts = await resp.json() if resp.status == 200 else []
        if resp.status != 200:
            print(f"  Warning: fetch posts failed ({resp.status}) "
                  f"for session {session_id[:8]}", flush=True)

    async with http.get(f"{parliament_url}/admin/sessions/{session_id}/votes",
                        headers=headers) as resp:
        all_votes = await resp.json() if resp.status == 200 else []
        if resp.status != 200:
            print(f"  Warning: fetch votes failed ({resp.status}) "
                  f"for session {session_id[:8]}", flush=True)

    posts, new_pid = _shape_posts(all_posts, cursors.post)
    comments, new_cid = _shape_comments(all_posts, cursors.comment)
    post_scores, comment_scores = _build_score_lookup(all_posts)
    actor_votes, judge_votes, new_vid = _shape_votes(
        all_votes, cursors.vote, post_scores, comment_scores)

    return Fetched(posts, comments, actor_votes, judge_votes,
                   Cursors(new_pid, new_cid, new_vid))


# ── Distribution ──────────────────────────────────────────

def _distribute(queues: dict[str, asyncio.Queue],
                tasks: dict[str, asyncio.Task],
                recipients: set[str], items: list[dict]) -> None:
    """Put `items` into each recipient's queue, filtering self-authored."""
    for name in recipients:
        if tasks[name].done():
            continue
        to_push = [i for i in items if i.get("author") != name]
        if to_push:
            to_push.sort(key=lambda x: (TYPE_ORDER.get(x["type"], 9), x["id"]))
            queues[name].put_nowait(to_push)


def _nudge_actors(queues: dict[str, asyncio.Queue], names: list[str],
                  coupled: bool = True) -> None:
    """Push a "do something" prompt when actors stall.

    Coupled mode: implies "the room is silent, break it" — references
    peers explicitly (the actor expects them).
    Independent mode: no peers exist; nudge is a pure self-prompt.
    """
    if coupled:
        msg = ("No new posts or comments from anyone. All scientists are "
               "waiting. Break the silence — post your next analysis step "
               "or summarize the final answer.")
    else:
        msg = ("No new feedback. Continue your derivation: submit the "
               "next reasoning step, or call leave if the chain is settled.")
    for n in names:
        queues[n].put_nowait(msg)


# ── Session lifecycle ─────────────────────────────────────

def _spawn_agent_task(http: aiohttp.ClientSession, agent: dict, role: str,
                      session: dict, ref_solution: str, parliament_url: str,
                      endpoint: str, model_name: str,
                      queues: dict[str, asyncio.Queue],
                      events: dict[str, asyncio.Event],
                      processing: set[str], id_map: IdMap,
                      max_rounds: int, llm_log_dir: Path | None,
                      discard_dir: Path | None) -> asyncio.Task:
    return asyncio.create_task(run_agent(
        name=agent["name"], role=role, api_key=agent["api_key"],
        session_id=session["session_id"], session_title=session["title"],
        reference_solution=ref_solution, parliament_url=parliament_url,
        llm_endpoint=endpoint, model_name=model_name,
        new_content_queue=queues[agent["name"]],
        submit_event=events[agent["name"]],
        processing=processing, http=http, id_map=id_map,
        max_rounds=max_rounds,
        llm_log_dir=llm_log_dir, discard_dir=discard_dir,
    ))


async def _wait_for_actor_event(actor_names: set[str],
                                events: dict[str, asyncio.Event],
                                tasks: dict[str, asyncio.Task]) -> None:
    pending = [events[n] for n in actor_names
               if not events[n].is_set() and not tasks[n].done()]
    if not pending:
        return
    waits = [asyncio.create_task(e.wait()) for e in pending]
    try:
        await asyncio.wait_for(
            asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED),
            timeout=RUNNER_WAIT_TIMEOUT_S)
    except asyncio.TimeoutError:
        pass
    for w in waits:
        w.cancel()


async def _close_session(http: aiohttp.ClientSession, parliament_url: str,
                         admin_key: str, sid: str) -> None:
    headers = {"Authorization": f"Bearer {admin_key}",
               "Content-Type": "application/json"}
    async with http.post(f"{parliament_url}/sessions/{sid}/close",
                         headers=headers) as resp:
        if resp.status >= 400:
            print(f"  Warning: failed to close session {sid[:8]}", flush=True)


def _print_session_summary(sid: str, gpu_port: str, dur: float,
                           results: list[AgentResult]) -> None:
    print(f"  [{_ts()}] Session {sid[:8]} done on :{gpu_port} "
          f"({dur:.0f}s)", flush=True)
    icons = {"session_end": "+", "max_rounds": "M", "left": "L",
             "context_overflow": "X", "exception": "E",
             "llm_errors": "!", "step_limit": "S", "no_tool": "N"}
    for r in results:
        icon = icons.get(r.exit_reason, "?")
        print(f"    [{icon}] {r.name:15s} {r.role:6s} "
              f"{r.exit_reason:15s} {r.rounds:2d}r {r.llm_calls:2d}llm "
              f"{r.posts_created}p {r.comments_created}c {r.votes_cast}v "
              f"{r.duration:.0f}s tok={r.total_prompt_tokens}+{r.total_completion_tokens} "
              f"fb={r.fallback_parses} err={r.api_errors},{r.llm_errors} "
              f"wait={r.wait_time:.0f}s", flush=True)


async def run_session(
    session: dict, session_details: dict,
    actors: list[dict], judges: list[dict],
    parliament_url: str, admin_key: str,
    endpoint: str, model_name: str, max_rounds: int,
    llm_log_dir: Path | None = None, discard_dir: Path | None = None,
) -> list[AgentResult]:
    """Run one session through the polling protocol."""
    sid = session["session_id"]
    ref_solution = session_details.get("reference_solution", "")
    gpu_port = endpoint.split(":")[2].split("/")[0]
    cfg = get_config()
    actor_context_coupled = bool(cfg.get("actor_context_coupled", True))
    judge_votes_visible = _resolve_judge_visibility(cfg)

    session_llm_dir = (llm_log_dir / sid[:8]) if llm_log_dir else None
    session_discard_dir = (discard_dir / sid[:8]) if discard_dir else None
    if session_llm_dir:
        session_llm_dir.mkdir(parents=True, exist_ok=True)
    if session_discard_dir:
        session_discard_dir.mkdir(parents=True, exist_ok=True)

    sess_t0 = time.time()
    print(f"\n  [{_ts()}] Session {sid[:8]} starting on :{gpu_port}", flush=True)

    actor_names = {a["name"] for a in actors}
    judge_names = {j["name"] for j in judges}
    queues = {n: asyncio.Queue() for n in actor_names | judge_names}
    events = {n: asyncio.Event() for n in actor_names | judge_names}
    actor_processing: set[str] = set(actor_names)
    judge_processing: set[str] = set()
    id_map = IdMap()
    # Per-session anonymization map (Scientist_N → real name). Applied to
    # every fetched item's `author` field before distribution so the LLM
    # never sees "Scientist_N" in either the prompt or any prior message.
    name_map = assign_session_names(sid)

    cohorts = (
        (actors, "actor", "", actor_processing),
        (judges, "judge", ref_solution, judge_processing),
    )
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(connector=connector) as http:
        tasks: dict[str, asyncio.Task] = {
            agent["name"]: _spawn_agent_task(
                http, agent, role, session, ref, parliament_url, endpoint,
                model_name, queues, events, processing, id_map,
                max_rounds, session_llm_dir, session_discard_dir)
            for cohort, role, ref, processing in cohorts
            for agent in cohort
        }

        cursors = Cursors()
        idle_rounds = 0
        round_num = 0

        while True:
            if all(t.done() for t in tasks.values()):
                break

            await _wait_for_actor_event(actor_names, events, tasks)
            for e in events.values():
                e.clear()

            fetched = await _fetch_new_content(
                http, parliament_url, admin_key, sid, cursors)

            if fetched.has_discussion:
                cursors = fetched.cursors
                id_map.localize_content(fetched.posts)
                id_map.localize_content(fetched.comments)
                id_map.localize_content(fetched.actor_votes)
                id_map.localize_content(fetched.judge_votes)
                apply_name_map(fetched.posts, name_map)
                apply_name_map(fetched.comments, name_map)
                apply_name_map(fetched.actor_votes, name_map)

                _distribute(queues, tasks, judge_names,
                            fetched.posts + fetched.comments)
                # Stagger actor distribution so judges queue first
                # (they'll start scoring while actors are still processing).
                await asyncio.sleep(DISTRIBUTION_DELAY_S)

                # Coupled mode (Parliament / BlindParliament): actors see
                # peers' posts, comments, and inter-actor votes.
                # Independent mode (Solo / BlindSolo): actors only see
                # judge feedback on their own steps; everything else stays
                # invisible (peers do not exist from this actor's POV).
                if actor_context_coupled:
                    actor_payload = (fetched.posts + fetched.comments
                                     + fetched.actor_votes)
                else:
                    actor_payload = []
                if judge_votes_visible:
                    actor_payload += [{**v, "author": ANONYMOUS_VOTER}
                                      for v in fetched.judge_votes]
                _distribute(queues, tasks, actor_names, actor_payload)
                # Independent mode: when nothing landed in the actor's
                # queue (BlindSolo, or Solo with no fresh judge votes
                # this round), nudge so the actor doesn't burn 60 s on
                # an empty queue waiting for peers that don't exist.
                if not actor_context_coupled and not actor_payload:
                    _nudge_actors(
                        queues,
                        [n for n in actor_names if not tasks[n].done()],
                        coupled=False)

                idle_rounds = 0
                print(f"  [{_ts()}] Session {sid[:8]} round={round_num} "
                      f"distributed {len(fetched.posts)}p "
                      f"{len(fetched.comments)}c "
                      f"{len(fetched.actor_votes)}av "
                      f"{len(fetched.judge_votes)}jv", flush=True)
            elif not actor_processing:
                idle_rounds += 1
                active = [n for n in actor_names if not tasks[n].done()]
                if idle_rounds <= 1 and active:
                    _nudge_actors(queues, active, coupled=actor_context_coupled)
                    print(f"  [{_ts()}] Session {sid[:8]} round={round_num} "
                          f"nudged actors (idle={idle_rounds})", flush=True)
                elif idle_rounds > 1:
                    if judge_processing:
                        print(f"  [{_ts()}] Session {sid[:8]} "
                              f"waiting for judges to finish", flush=True)
                        for _ in range(JUDGE_WAITDOWN_S):
                            if not judge_processing:
                                break
                            await asyncio.sleep(1)
                    break

            round_num += 1

        # Tell remaining agents to stop
        for q in queues.values():
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        await _close_session(http, parliament_url, admin_key, sid)

    results: list[AgentResult] = []
    for r in gathered:
        if isinstance(r, AgentResult):
            results.append(r)
        else:
            results.append(AgentResult(
                name="?", role="?", session_id=sid,
                exit_reason="exception", error=str(r)))

    _print_session_summary(sid, gpu_port, time.time() - sess_t0, results)

    if session_llm_dir:
        with open(session_llm_dir / "session_summary.json", "w") as f:
            json.dump({
                "session_id": sid,
                "title": session.get("title", ""),
                "gpu_port": gpu_port,
                "timestamp": datetime.now().isoformat(),
                "rounds": round_num,
                "agents": [asdict(r) for r in results],
            }, f, indent=2, ensure_ascii=False, default=str)

    return results


# ── Top-level experiment ──────────────────────────────────

def _summarize(results: list[AgentResult], duration: float,
               sessions_done: int, sessions: list[dict],
               parliament_url: str) -> dict:
    done_normally = sum(1 for r in results
                        if r.exit_reason in ("session_end", "max_rounds"))
    total = len(results)
    summary = {
        "agents_ok": done_normally,
        "agents_total": total,
        "posts": sum(r.posts_created for r in results),
        "comments": sum(r.comments_created for r in results),
        "votes": sum(r.votes_cast for r in results),
        "prompt_tokens": sum(r.total_prompt_tokens for r in results),
        "completion_tokens": sum(r.total_completion_tokens for r in results),
        "llm_errors": sum(r.llm_errors for r in results),
        "api_errors": sum(r.api_errors for r in results),
        "no_tool_responses": sum(r.no_tool_responses for r in results),
        "fallback_parses": sum(r.fallback_parses for r in results),
        "exit_reasons": dict(Counter(r.exit_reason for r in results)),
    }
    print(f"\n{'=' * 70}", flush=True)
    print(f"Experiment complete!", flush=True)
    print(f"  Duration:    {duration:.0f}s ({duration / 60:.1f} min)", flush=True)
    print(f"  Sessions:    {sessions_done}", flush=True)
    print(f"  Agents:      {done_normally}/{total} finished normally", flush=True)
    print(f"  Posts:       {summary['posts']}", flush=True)
    print(f"  Comments:    {summary['comments']}", flush=True)
    print(f"  Votes:       {summary['votes']}", flush=True)
    print(f"  Tokens:      {summary['prompt_tokens']} prompt + "
          f"{summary['completion_tokens']} completion", flush=True)
    print(f"  Errors:      {summary['llm_errors']} LLM, "
          f"{summary['api_errors']} API, "
          f"{summary['no_tool_responses']} no_tool, "
          f"{summary['fallback_parses']} fallback", flush=True)
    print(f"  Exit reasons:{summary['exit_reasons']}", flush=True)
    print(f"  View:        {parliament_url}", flush=True)
    print(f"{'=' * 70}", flush=True)
    return summary


async def run_experiment(
    parliament_url: str, admin_key: str, gpu_endpoints: list[str],
    sessions_per_gpu: int, num_actors: int, num_judges: int,
    model_name: str, max_rounds: int,
    output_path: str | None = None,
) -> int:
    """Run the full experiment. Returns 0 on success."""

    print(f"Harness starting", flush=True)
    print(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
          flush=True)
    print(f"  Parliament: {parliament_url}", flush=True)
    print(f"  GPUs:       {len(gpu_endpoints)}", flush=True)
    for i, ep in enumerate(gpu_endpoints):
        print(f"    [{i}] {ep}", flush=True)
    print(f"  Concurrency:{sessions_per_gpu}/GPU × {len(gpu_endpoints)} GPUs "
          f"= {sessions_per_gpu * len(gpu_endpoints)} parallel sessions",
          flush=True)
    print(f"  Agents:     {num_actors} actors + {num_judges} judges per session",
          flush=True)
    print(f"  Max rounds: {max_rounds} (actor only, judge unlimited)",
          flush=True)

    sessions = _api_sync(parliament_url, "GET", "/admin/sessions", admin_key)
    open_sessions = [s for s in sessions if s.get("status") == "open"]
    if not open_sessions:
        print("ERROR: No open sessions.")
        return 1

    all_users = _api_sync(parliament_url, "GET", "/admin/users", admin_key)
    actors_list = [u for u in all_users if u.get("role") == "actor"][:num_actors]
    judges_list = [u for u in all_users if u.get("role") == "judge"][:num_judges]
    if not actors_list or not judges_list:
        print(f"ERROR: No agents found ({len(actors_list)} actors, "
              f"{len(judges_list)} judges).")
        return 1

    session_details: dict[str, dict] = {}
    for s in open_sessions:
        session_details[s["session_id"]] = _api_sync(
            parliament_url, "GET",
            f"/admin/sessions/{s['session_id']}", admin_key)

    print(f"\n  {len(open_sessions)} sessions to process", flush=True)

    queue: asyncio.Queue[dict] = asyncio.Queue()
    for s in open_sessions:
        queue.put_nowait(s)

    llm_log_dir = discard_dir = None
    if output_path:
        run_dir = Path(output_path).parent
        llm_log_dir = run_dir / "llm_logs"
        discard_dir = run_dir / "discards"
        llm_log_dir.mkdir(parents=True, exist_ok=True)
        discard_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[AgentResult] = []
    results_lock = asyncio.Lock()
    sessions_done = 0

    async def gpu_slot(ep: str):
        nonlocal sessions_done
        while True:
            try:
                session = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            sid = session["session_id"]
            results = await run_session(
                session, session_details.get(sid, {}),
                actors_list, judges_list, parliament_url, admin_key,
                ep, model_name, max_rounds,
                llm_log_dir=llm_log_dir, discard_dir=discard_dir)
            sessions_done += 1
            print(f"  [{_ts()}] Progress: {sessions_done}/{len(open_sessions)}",
                  flush=True)
            async with results_lock:
                all_results.extend(results)

    t0 = time.time()
    slots = [gpu_slot(ep) for ep in gpu_endpoints
             for _ in range(sessions_per_gpu)]
    await asyncio.gather(*slots)
    duration = time.time() - t0

    summary = _summarize(all_results, duration, sessions_done,
                         open_sessions, parliament_url)

    if not output_path:
        output_path = str(PROJECT_DIR / "data" /
                          f"experiment_{datetime.now().strftime('%m%d_%H%M%S')}.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": "event-driven",
            "config": get_config(),
            "duration_seconds": round(duration, 1),
            "sessions": [s["session_id"] for s in open_sessions],
            "summary": summary,
            "results": [asdict(r) for r in all_results],
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results: {output_path}", flush=True)
    return 0
