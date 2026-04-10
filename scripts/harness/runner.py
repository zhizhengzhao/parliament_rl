"""Global experiment scheduler — event-driven polling mode.

Runner wakes on agent submit_event, immediately fetches and distributes
new content. Idle detection based on whether new posts/comments exist.
Two processing sets: processing (actors, or actors+judges in visible mode)
and judge_processing (judges in not-visible mode).
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
    """Fetch all new posts, comments, and votes from Parliament DB."""
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

    post_scores = {}
    comment_scores = {}
    for p in all_posts:
        post_scores[p.get("post_id")] = p.get("score", 0)
        for c in p.get("comments", []):
            comment_scores[c.get("comment_id")] = c.get("score", 0)

    actor_votes, judge_votes = [], []
    for v in all_votes:
        vid = v.get("vote_id") or 0
        if vid <= after_vote_id:
            continue
        target_type = "post" if v.get("post_id") else "comment"
        target_id = v.get("post_id") or v.get("comment_id")
        if target_type == "post":
            target_score = post_scores.get(target_id, 0)
        else:
            target_score = comment_scores.get(target_id, 0)
        item = {
            "type": "vote", "id": vid,
            "target_type": target_type, "target_id": target_id,
            "value": v.get("value", 0),
            "previous_value": v.get("previous_value"),
            "author": v.get("author", "?"),
            "target_score": target_score,
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
    agent_events: dict[str, asyncio.Event] = {}
    all_names = [a["name"] for a in actors] + [j["name"] for j in judges]
    for name in all_names:
        agent_queues[name] = asyncio.Queue()
        agent_events[name] = asyncio.Event()

    actor_name_set = {a["name"] for a in actors}
    judge_name_set = {j["name"] for j in judges}
    judge_votes_visible = get_config().get("judge_votes_visible", True)

    # Two processing sets:
    # processing: actors always; + judges if visible
    # judge_processing: judges if NOT visible (for end-of-session wait)
    processing: set[str] = set(actor_name_set)
    judge_processing: set[str] = set()

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
                submit_event=agent_events[a["name"]],
                processing=processing,
                http=http, max_rounds=max_rounds, timeout=timeout,
                llm_log_dir=session_llm_dir,
                discard_dir=session_discard_dir,
            ))
        for j in judges:
            jp = processing if judge_votes_visible else judge_processing
            agent_tasks[j["name"]] = asyncio.create_task(run_agent(
                name=j["name"], role="judge", api_key=j["api_key"],
                session_id=sid, session_title=session["title"],
                reference_solution=ref_solution,
                parliament_url=parliament_url, llm_endpoint=endpoint,
                model_name=model_name,
                new_content_queue=agent_queues[j["name"]],
                submit_event=agent_events[j["name"]],
                processing=jp,
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

            # Wait for any agent event (or 60s timeout)
            pending = [e for e in agent_events.values()
                        if not e.is_set()]
            if pending:
                waits = [asyncio.create_task(e.wait()) for e in pending]
                try:
                    await asyncio.wait_for(
                        asyncio.wait(waits,
                                     return_when=asyncio.FIRST_COMPLETED),
                        timeout=60)
                except asyncio.TimeoutError:
                    pass
                for w in waits:
                    w.cancel()

            for e in agent_events.values():
                e.clear()

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
                    type_order = {"post": 0, "comment": 1, "vote": 2}
                    to_push.sort(key=lambda x: (
                        type_order.get(x["type"], 9), x["id"]))
                    agent_queues[name].put_nowait(to_push)

            # Determine if there's meaningful content
            if judge_votes_visible:
                has_content = bool(posts or comments
                                   or actor_votes or judge_votes)
            else:
                has_content = bool(posts or comments or actor_votes)

            if has_content:
                idle_rounds = 0
                has_disc = bool(posts or comments)
                ts = datetime.now().strftime("%H:%M:%S")
                label = "new content" if has_disc else "votes only"
                print(f"  [{ts}] Session {sid[:8]} round={round_num} "
                      f"distributed {label}", flush=True)
            else:
                # No content — check processing
                if processing:
                    pass  # agents still working
                else:
                    idle_rounds += 1
                    active_actors = [n for n in actor_name_set
                                     if not agent_tasks[n].done()]
                    if idle_rounds <= 1 and active_actors:
                        for aname in active_actors:
                            agent_queues[aname].put_nowait(
                                "No new posts or comments from anyone. "
                                "All scientists are waiting. Break the "
                                "silence \u2014 post your next analysis "
                                "step or summarize the final answer.")
                        ts = datetime.now().strftime("%H:%M:%S")
                        print(f"  [{ts}] Session {sid[:8]} "
                              f"round={round_num} nudged actors "
                              f"(idle={idle_rounds})", flush=True)
                    elif idle_rounds > 1:
                        # Session ending — wait for judge_processing
                        if judge_processing:
                            ts = datetime.now().strftime("%H:%M:%S")
                            print(f"  [{ts}] Session {sid[:8]} "
                                  f"waiting for judges to finish",
                                  flush=True)
                            for _ in range(120):
                                if not judge_processing:
                                    break
                                await asyncio.sleep(1)
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
                "timeout": "T", "llm_errors": "!", "step_limit": "S",
                "no_tool": "N",
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
    actors_list = [u for u in all_users if u.get("role") == "actor"][:num_actors]
    judges_list = [u for u in all_users if u.get("role") == "judge"][:num_judges]

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
                session, detail, actors_list, judges_list,
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
                                    "max_rounds"))
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
