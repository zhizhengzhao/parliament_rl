"""Global experiment scheduler — polling mode.

Pure polling: runner waits for all agents to finish their round (processing
set empties), then fetches new content and distributes. No event-based
signaling. Idle detection based on posts + comments only.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import aiohttp

from .agent import AgentResult, run_agent, get_config

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


def _api_sync(base_url: str, method: str, path: str, key: str) -> dict | list:
    url = base_url + path
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}
    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as e:
        print(f"  API error: {method} {path}: {e}")
        return {} if method != "GET" else []


ANONYMOUS_VOTER = "Anonymous Scientist"


async def _fetch_new_content(
    http: aiohttp.ClientSession,
    parliament_url: str,
    admin_key: str,
    session_id: str,
    after_post_id: int,
    after_comment_id: int,
    after_vote_id: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], int, int, int]:
    """Fetch all new posts, comments, and votes from Parliament DB.

    Returns four separate lists + updated high-water marks:
        posts, comments, actor_votes, judge_votes,
        max_post_id, max_comment_id, max_vote_id
    """
    headers = {"Authorization": f"Bearer {admin_key}",
               "Content-Type": "application/json"}

    async with http.get(
        f"{parliament_url}/admin/sessions/{session_id}/posts",
        headers=headers,
    ) as resp:
        all_posts = await resp.json() if resp.status == 200 else []

    posts, comments = [], []
    max_post_id = after_post_id
    max_comment_id = after_comment_id
    max_vote_id = after_vote_id

    for p in all_posts:
        pid = p.get("post_id") or 0
        if pid > after_post_id:
            posts.append({
                "type": "post", "id": pid,
                "author": p.get("author", "?"),
                "content": p.get("content", ""),
            })
            max_post_id = max(max_post_id, pid)
        for c in p.get("comments", []):
            cid = c.get("comment_id") or 0
            if cid > after_comment_id:
                comments.append({
                    "type": "comment", "id": cid, "post_id": pid,
                    "author": c.get("author", "?"),
                    "content": c.get("content", ""),
                })
                max_comment_id = max(max_comment_id, cid)

    async with http.get(
        f"{parliament_url}/admin/sessions/{session_id}/votes",
        headers=headers,
    ) as resp:
        all_votes = await resp.json() if resp.status == 200 else []

    actor_votes, judge_votes = [], []
    for v in all_votes:
        vid = v.get("vote_id") or 0
        if vid <= after_vote_id:
            continue
        target_type = "post" if v.get("post_id") else "comment"
        target_id = v.get("post_id") or v.get("comment_id")
        item = {
            "type": "vote", "id": vid,
            "target_type": target_type, "target_id": target_id,
            "value": v.get("value", 0),
            "previous_value": v.get("previous_value"),
            "author": v.get("author", "?"),
        }
        if v.get("role") == "judge":
            judge_votes.append(item)
        else:
            actor_votes.append(item)
        max_vote_id = max(max_vote_id, vid)

    return posts, comments, actor_votes, judge_votes, \
        max_post_id, max_comment_id, max_vote_id


async def run_session(
    session: dict,
    session_details: dict,
    actors: list[dict],
    judges: list[dict],
    parliament_url: str,
    admin_key: str,
    endpoint: str,
    model_name: str,
    max_rounds: int,
    timeout: float,
    llm_log_dir: Path | None = None,
    discard_dir: Path | None = None,
) -> list[AgentResult]:
    """Run one session using the polling protocol."""
    sid = session["session_id"]
    ref_solution = session_details.get("reference_solution", "")
    gpu_port = endpoint.split(":")[2].split("/")[0]

    session_llm_dir = None
    session_discard_dir = None
    if llm_log_dir:
        session_llm_dir = llm_log_dir / sid[:8]
        session_llm_dir.mkdir(parents=True, exist_ok=True)
    if discard_dir:
        session_discard_dir = discard_dir / sid[:8]
        session_discard_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n  [{ts}] Session {sid[:8]} starting on :{gpu_port}", flush=True)

    agent_queues: dict[str, asyncio.Queue] = {}
    all_names = [a["name"] for a in actors] + [j["name"] for j in judges]
    for name in all_names:
        agent_queues[name] = asyncio.Queue()

    actor_name_set = {a["name"] for a in actors}
    judge_name_set = {j["name"] for j in judges}
    processing: set[str] = set(actor_name_set)
    judge_votes_visible = get_config().get("judge_votes_visible", True)

    async with aiohttp.ClientSession() as http:
        agent_tasks = {}
        for a in actors:
            agent_tasks[a["name"]] = asyncio.create_task(run_agent(
                name=a["name"], role="actor", api_key=a["api_key"],
                session_id=sid, session_title=session["title"],
                reference_solution="",
                parliament_url=parliament_url, llm_endpoint=endpoint,
                model_name=model_name,
                new_content_queue=agent_queues[a["name"]],
                processing=processing,
                http=http, max_rounds=max_rounds, timeout=timeout,
                llm_log_dir=session_llm_dir,
                discard_dir=session_discard_dir,
            ))
        for j in judges:
            agent_tasks[j["name"]] = asyncio.create_task(run_agent(
                name=j["name"], role="judge", api_key=j["api_key"],
                session_id=sid, session_title=session["title"],
                reference_solution=ref_solution,
                parliament_url=parliament_url, llm_endpoint=endpoint,
                model_name=model_name,
                new_content_queue=agent_queues[j["name"]],
                processing=processing,
                http=http, max_rounds=max_rounds, timeout=timeout,
                llm_log_dir=session_llm_dir,
                discard_dir=session_discard_dir,
            ))

        last_post_id = 0
        last_comment_id = 0
        last_vote_id = 0
        idle_rounds = 0

        for round_num in range(max_rounds * 3):
            if all(t.done() for t in agent_tasks.values()):
                break

            # Poll: wait until processing is empty (all agents idle)
            for _ in range(60):
                if not processing:
                    break
                await asyncio.sleep(1)
            else:
                processing.clear()

            # All actors left → session ends
            if all(agent_tasks[n].done() for n in actor_name_set):
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}] Session {sid[:8]} "
                      f"all actors done, ending", flush=True)
                break

            # Fetch new content
            posts, comments, actor_votes, judge_votes, \
                new_pid, new_cid, new_vid = await _fetch_new_content(
                    http, parliament_url, admin_key, sid,
                    last_post_id, last_comment_id, last_vote_id)
            last_post_id = new_pid
            last_comment_id = new_cid
            last_vote_id = new_vid

            # Distribute to each agent
            for name in all_names:
                if agent_tasks[name].done():
                    continue
                if name in judge_name_set:
                    to_push = posts + comments
                else:
                    to_push = posts + comments + actor_votes
                    if judge_votes_visible:
                        anon = [{**v, "author": ANONYMOUS_VOTER}
                                for v in judge_votes]
                        to_push = to_push + anon
                to_push = [i for i in to_push
                           if i.get("author") != name]
                if to_push:
                    to_push.sort(key=lambda x: x["id"])
                    agent_queues[name].put_nowait(to_push)
                    processing.add(name)

            # Idle: only posts + comments count as discussion progress
            has_discussion = bool(posts or comments)

            if has_discussion:
                idle_rounds = 0
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}] Session {sid[:8]} round={round_num} "
                      f"distributed new content", flush=True)
            elif actor_votes or (judge_votes_visible and judge_votes):
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}] Session {sid[:8]} round={round_num} "
                      f"distributed votes only", flush=True)
            else:
                idle_rounds += 1
                active_actors = [n for n in actor_name_set
                                 if not agent_tasks[n].done()]
                if idle_rounds <= 2 and active_actors:
                    for aname in active_actors:
                        agent_queues[aname].put_nowait(
                            "No new posts or comments from anyone. "
                            "All scientists are waiting. Break the "
                            "silence \u2014 post your next analysis "
                            "step.")
                        processing.add(aname)
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{ts}] Session {sid[:8]} "
                          f"round={round_num} nudged actors "
                          f"(idle={idle_rounds})", flush=True)
                elif idle_rounds >= 3:
                    break

        # Signal remaining agents to stop
        for q in agent_queues.values():
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

        results = await asyncio.gather(*agent_tasks.values(),
                                       return_exceptions=True)

    session_results: list[AgentResult] = []
    for r in results:
        if isinstance(r, AgentResult):
            session_results.append(r)
        else:
            session_results.append(AgentResult(
                name="?", role="?", session_id=sid,
                exit_reason="exception", error=str(r)))

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] Session {sid[:8]} done on :{gpu_port}", flush=True)
    for r in session_results:
        icon = {"session_end": "+", "no_new_content": "+", "max_rounds": "M",
                "timeout": "T", "llm_errors": "!", "leave": "L",
                "step_limit": "S", "no_tool": "N",
                }.get(r.exit_reason, "?")
        print(f"    [{icon}] {r.name:15s} {r.role:6s} "
              f"{r.exit_reason:15s} {r.rounds:2d}r {r.llm_calls:2d}llm "
              f"{r.posts_created}p {r.comments_created}c {r.votes_cast}v "
              f"{r.duration:.0f}s", flush=True)

    return session_results


async def run_experiment(
    parliament_url: str,
    admin_key: str,
    gpu_endpoints: list[str],
    sessions_per_gpu: int,
    num_actors: int,
    num_judges: int,
    model_name: str,
    timeout: float,
    max_rounds: int,
    output_path: str | None = None,
) -> int:
    """Run the full experiment. Returns 0 on success."""

    print(f"Harness starting (polling mode)", flush=True)
    print(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Parliament: {parliament_url}", flush=True)
    print(f"  GPUs:       {len(gpu_endpoints)}", flush=True)
    for i, ep in enumerate(gpu_endpoints):
        print(f"    [{i}] {ep}", flush=True)
    print(f"  Concurrency:{sessions_per_gpu}/GPU × {len(gpu_endpoints)} GPUs "
          f"= {sessions_per_gpu * len(gpu_endpoints)} parallel sessions", flush=True)
    print(f"  Agents:     {num_actors} actors + {num_judges} judges per session", flush=True)
    print(f"  Max rounds: {max_rounds}  Timeout: {timeout}s", flush=True)

    sessions = _api_sync(parliament_url, "GET", "/admin/sessions", admin_key)
    open_sessions = [s for s in sessions if s.get("status") == "open"]
    if not open_sessions:
        print("ERROR: No open sessions.")
        return 1

    all_users = _api_sync(parliament_url, "GET", "/admin/users", admin_key)
    actors = [u for u in all_users if u.get("role") == "actor"][:num_actors]
    judges = [u for u in all_users if u.get("role") == "judge"][:num_judges]

    session_details: dict[str, dict] = {}
    for s in open_sessions:
        detail = _api_sync(parliament_url, "GET",
                           f"/admin/sessions/{s['session_id']}", admin_key)
        session_details[s["session_id"]] = detail

    print(f"\n  {len(open_sessions)} sessions to process", flush=True)

    queue: asyncio.Queue[dict] = asyncio.Queue()
    for s in open_sessions:
        queue.put_nowait(s)

    llm_log_dir = None
    discard_dir = None
    if output_path:
        run_dir = Path(output_path).parent
        llm_log_dir = run_dir / "llm_logs"
        llm_log_dir.mkdir(parents=True, exist_ok=True)
        discard_dir = run_dir / "discards"
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
            detail = session_details.get(sid, {})
            results = await run_session(
                session, detail, actors, judges,
                parliament_url, admin_key, ep, model_name,
                max_rounds, timeout,
                llm_log_dir=llm_log_dir, discard_dir=discard_dir)

            sessions_done += 1
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Progress: {sessions_done}/{len(open_sessions)}", flush=True)

            async with results_lock:
                all_results.extend(results)

    t0 = time.time()

    slots = []
    for ep in gpu_endpoints:
        for _ in range(sessions_per_gpu):
            slots.append(gpu_slot(ep))
    await asyncio.gather(*slots)

    duration = time.time() - t0

    done = sum(1 for r in all_results
               if r.exit_reason in ("session_end", "no_new_content",
                                    "max_rounds", "leave"))
    total = len(all_results)
    total_posts = sum(r.posts_created for r in all_results)
    total_comments = sum(r.comments_created for r in all_results)
    total_votes = sum(r.votes_cast for r in all_results)

    print(f"\n{'='*70}", flush=True)
    print(f"Experiment complete!", flush=True)
    print(f"  Duration:    {duration:.0f}s ({duration/60:.1f} min)", flush=True)
    print(f"  Sessions:    {sessions_done}", flush=True)
    print(f"  Agents:      {done}/{total} finished normally", flush=True)
    print(f"  Posts:       {total_posts}", flush=True)
    print(f"  Comments:    {total_comments}", flush=True)
    print(f"  Votes:       {total_votes}", flush=True)
    print(f"  View:        {parliament_url}", flush=True)
    print(f"{'='*70}", flush=True)

    if not output_path:
        output_path = str(PROJECT_DIR / "data" /
                          f"experiment_{datetime.now().strftime('%m%d_%H%M%S')}.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "mode": "polling",
        "config": get_config(),
        "duration_seconds": round(duration, 1),
        "sessions": [s["session_id"] for s in open_sessions],
        "results": [asdict(r) for r in all_results],
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results: {output_path}", flush=True)
    return 0
