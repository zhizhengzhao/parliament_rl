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
                                  its own history (and judge votes on
                                  their own posts if visible). Default true.
  judge_votes_visible   (bool)  — push judge votes to actors? Default true.
                                  Override at launch with
                                  PRL_JUDGE_VOTES_VISIBLE=0 (or =false).

Per-actor isolation in solo (independent) cells:
  When ``actor_context_coupled=False`` each actor gets its own IdMap so
  ``P_1, P_2, …`` are *always* the actor's own posts in chronological
  order — no global P_id gaps that would reveal peers' existence.
  Judge-vote distribution is also filtered per actor: only votes whose
  target post was authored by *this* actor get pushed.  Coupled cells
  keep a single session-shared IdMap (peers' posts are visible, so
  shared global numbering is the natural choice).
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

def _build_score_lookup(all_votes: list[dict], judge_visible: bool
                        ) -> tuple[dict[int, int], dict[int, int]]:
    """Compute cumulative post / comment scores from the visible-vote pool.

    The server's stored `post.score` is the unconditional `SUM(value)`
    across **all** voters (actor + judge).  In ``judge_visible=False``
    cells (BlindParliament / BlindSolo) we cannot use it directly: the
    actor sees `current score of P_3: +X` on every vote event we
    distribute, and X must reflect only votes the actor was allowed to
    see.  Otherwise judges leak through the score field even when their
    own vote events are filtered out.

    We therefore re-aggregate from the raw votes here, applying the
    same visibility rule we use for ``_distribute``.
    """
    post_scores: dict[int, int] = {}
    comment_scores: dict[int, int] = {}
    for v in all_votes:
        if not judge_visible and v.get("role") == "judge":
            continue                                 # exclude from cumulative
        val = v.get("value", 0)
        if v.get("post_id"):
            pid = v["post_id"]
            post_scores[pid] = post_scores.get(pid, 0) + val
        elif v.get("comment_id"):
            cid = v["comment_id"]
            comment_scores[cid] = comment_scores.get(cid, 0) + val
    return post_scores, comment_scores


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
                             cursors: Cursors,
                             judge_visible: bool) -> Fetched:
    """Pull all new posts/comments/votes since the given cursors.

    ``judge_visible`` controls how cumulative scores attached to vote
    events are computed: when False, judges' votes are excluded from
    the running totals so the score field never reveals judge influence
    to actors in BlindParliament / BlindSolo.
    """
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
    # Visibility-aware score aggregation — see `_build_score_lookup`.
    post_scores, comment_scores = _build_score_lookup(all_votes, judge_visible)
    actor_votes, judge_votes, new_vid = _shape_votes(
        all_votes, cursors.vote, post_scores, comment_scores)

    return Fetched(posts, comments, actor_votes, judge_votes,
                   Cursors(new_pid, new_cid, new_vid))


# ── Distribution ──────────────────────────────────────────

def _distribute_shared(queues: dict[str, asyncio.Queue],
                       tasks: dict[str, asyncio.Task],
                       recipients: set[str], items: list[dict]) -> None:
    """Push pre-localized ``items`` (shared IdMap) to recipients.

    Used by the coupled path and the judges' fan-out, where every
    recipient sees the same global ID numbering.  Items must already
    have been ``id_map.localize_content``-converted before this call.
    """
    for name in recipients:
        if tasks[name].done():
            continue
        to_push = [i for i in items if i.get("author") != name]
        if to_push:
            to_push.sort(key=lambda x: (TYPE_ORDER.get(x["type"], 9), x["id"]))
            queues[name].put_nowait(to_push)


def _distribute_per_actor(queues: dict[str, asyncio.Queue],
                          tasks: dict[str, asyncio.Task],
                          actor_name: str,
                          actor_id_map: "IdMap",
                          name_map: dict[str, str],
                          payload_global: list[dict]) -> bool:
    """Solo-path: localize ``payload_global`` with this actor's own IdMap
    and push.  Returns True iff anything was pushed.

    Items in ``payload_global`` carry **global** IDs; we deep-copy then
    apply ``actor_id_map.localize_content`` so each actor's queue is
    numbered from their own P_1.  Self-authored items are filtered last.
    """
    if tasks[actor_name].done() or not payload_global:
        return False
    items = [{**i} for i in payload_global if i.get("author") != actor_name]
    if not items:
        return False
    actor_id_map.localize_content(items)
    apply_name_map(items, name_map)
    items.sort(key=lambda x: (TYPE_ORDER.get(x["type"], 9), x["id"]))
    queues[actor_name].put_nowait(items)
    return True


# ── Per-cell nudge messages ───────────────────────────────
#
# Each 2×2 cell needs a different "nothing happened" prompt because the
# actor's *expectation* of what could arrive differs:
#   A coupled+visible    : peers + judge votes can arrive  → talk silence
#   B coupled+blind      : peers can comment/post (no scores) → same as A
#   C solo+judge_visible : only anonymous scores can arrive → "no votes yet"
#   D solo+blind         : nothing can ever arrive          → pure self-prompt

_NUDGE_COUPLED = (
    "No new posts or comments from anyone. All scientists are waiting. "
    "Break the silence with your next move, or post the final move if "
    "the chain is settled."
)
_NUDGE_SOLO_JUDGE = (
    "No anonymous scores have arrived yet — they may or may not come. "
    "Continue with your next reasoning move, or call leave if the chain "
    "is settled."
)
_NUDGE_SOLO_BLIND = (
    "You are working alone — no external feedback will arrive. "
    "Continue your derivation independently, or call leave when the "
    "answer chain is settled."
)


def _nudge_actors(queues: dict[str, asyncio.Queue], names: list[str], *,
                  coupled: bool, judge_visible: bool) -> None:
    """Push a "do something" prompt when actors stall.

    Wording is cell-aware so the actor never receives a hint that
    contradicts what they could possibly observe at rollout time.
    """
    if coupled:
        msg = _NUDGE_COUPLED
    elif judge_visible:
        msg = _NUDGE_SOLO_JUDGE
    else:
        msg = _NUDGE_SOLO_BLIND
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
    """Wire up one agent's coroutine.  ``id_map`` should be that agent's
    own IdMap (per-actor in solo cells, session-shared in coupled)."""
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
        # ``r.rounds`` = total attempts; ``S/F`` = successful/failed split.
        # When F is large vs S the session was LLM-API-throttled, not
        # actor-decided.  This distinguishes "short session because
        # framework let actor leave early" (F=0) from "short session
        # because half the rounds errored out" (F≫0).
        print(f"    [{icon}] {r.name:15s} {r.role:6s} "
              f"{r.exit_reason:15s} {r.rounds:2d}r "
              f"(S{r.successful_rounds}/F{r.failed_rounds}) "
              f"{r.llm_calls:2d}llm "
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

    # Per-session anonymization map (Scientist_N → real name). Applied to
    # every fetched item's `author` field before distribution so the LLM
    # never sees "Scientist_N" in either the prompt or any prior message.
    # Seeded by `title` (= the problem text) rather than `session_id`, so
    # all 2×2 cells running the same question get the same roster —
    # removes persona/identity as a cell-comparison confounder.
    name_map = assign_session_names(session["title"])

    # IdMap topology depends on the cell:
    #   Coupled  → one shared IdMap so every actor sees the same global
    #              P_id (peers are visible, shared numbering is natural).
    #   Solo     → per-actor IdMap so each actor's queue is numbered
    #              from their own P_1, P_2, …  Without this, an actor
    #              who submits the 3rd global post would see P_3 with
    #              no preceding P_1 / P_2 in their context — leaking
    #              the existence of peers.
    #   Judges always share one IdMap among themselves (they read every
    #   post by design).
    if actor_context_coupled:
        shared_id_map = IdMap()
        actor_id_maps: dict[str, IdMap] = {n: shared_id_map for n in actor_names}
        judge_id_map: IdMap = shared_id_map
    else:
        actor_id_maps = {n: IdMap() for n in actor_names}
        judge_id_map = IdMap()

    # Track each actor's own posts (global IDs) so solo-cell judge-vote
    # filtering can keep only votes whose target post belongs to *this*
    # actor.  Coupled cells don't need this — peer posts are visible
    # there, so peer-targeted votes are fine.
    actor_post_ids: dict[str, set[int]] = {n: set() for n in actor_names}

    cohorts = (
        (actors, "actor", "", actor_processing),
        (judges, "judge", ref_solution, judge_processing),
    )
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(connector=connector) as http:
        tasks: dict[str, asyncio.Task] = {
            agent["name"]: _spawn_agent_task(
                http, agent, role, session, ref, parliament_url, endpoint,
                model_name, queues, events, processing,
                actor_id_maps[agent["name"]] if role == "actor" else judge_id_map,
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

            # Defensive prune: if an agent task crashed without
            # reaching its ``processing.discard(name)`` cleanup (e.g.
            # python_exec subprocess deadlock, asyncio gather race),
            # the name lingers in the processing set and the
            # ``not actor_processing`` idle-detection check below never
            # fires — the session then drags out to max_rounds with
            # no productive work.  ``task.done()`` is the asyncio
            # ground truth: a True result means the coroutine has
            # finished or raised, so the name has no business sitting
            # in ``processing`` anymore.
            stale_actors = {n for n in actor_processing if tasks[n].done()}
            stale_judges = {n for n in judge_processing if tasks[n].done()}
            actor_processing -= stale_actors
            judge_processing -= stale_judges

            fetched = await _fetch_new_content(
                http, parliament_url, admin_key, sid, cursors,
                judge_visible=judge_votes_visible)

            if fetched.has_discussion:
                cursors = fetched.cursors

                # Update per-actor own-post bookkeeping (global IDs)
                # before any distribution touches the items.
                for p in fetched.posts:
                    if p.get("author") in actor_post_ids:
                        actor_post_ids[p["author"]].add(p["id"])

                # ── Judges: shared IdMap, see every post + comment ──
                judge_items = [{**i} for i in fetched.posts + fetched.comments]
                judge_id_map.localize_content(judge_items)
                apply_name_map(judge_items, name_map)
                _distribute_shared(queues, tasks, judge_names, judge_items)

                # Stagger actor distribution so judges queue first
                # (they'll start scoring while actors are still processing).
                await asyncio.sleep(DISTRIBUTION_DELAY_S)

                # ── Actors: per-actor distribution ──
                # Coupled (Parliament/BlindParliament): actors see peers'
                # posts, comments, and inter-actor votes; one shared
                # IdMap means every actor sees the same P_id numbering.
                # Solo (Solo/BlindSolo): actors only see judge feedback
                # *on their own posts*; per-actor IdMap renumbers from
                # P_1 so peers stay invisible by construction.
                pushed_anything: dict[str, bool] = {n: False for n in actor_names}
                for actor_name in actor_names:
                    if tasks[actor_name].done():
                        continue
                    if actor_context_coupled:
                        payload = (list(fetched.posts) + list(fetched.comments)
                                   + list(fetched.actor_votes))
                        if judge_votes_visible:
                            payload += [{**v, "author": ANONYMOUS_VOTER}
                                        for v in fetched.judge_votes]
                    else:
                        # Solo cell: hide peers entirely.  Only push
                        # judge votes whose target post is authored by
                        # *this* actor (own_pids).  Self-authored items
                        # never need echoing — the actor already saw the
                        # local P_id from `submit`'s response, and our
                        # per-actor IdMap means that numbering is theirs.
                        own_pids = actor_post_ids[actor_name]
                        payload = []
                        if judge_votes_visible:
                            own_judge_votes = [
                                v for v in fetched.judge_votes
                                if v.get("target_type") == "post"
                                and v.get("target_id") in own_pids
                            ]
                            payload = [{**v, "author": ANONYMOUS_VOTER}
                                       for v in own_judge_votes]

                    pushed = _distribute_per_actor(
                        queues, tasks, actor_name,
                        actor_id_maps[actor_name], name_map, payload)
                    pushed_anything[actor_name] = pushed

                # Solo cell: nudge actors who got an empty queue this
                # round (e.g. BlindSolo always, Solo when no judge vote
                # landed on their post).  Coupled cells nudge later via
                # the idle-round path.
                if not actor_context_coupled:
                    silent = [n for n in actor_names
                              if not pushed_anything[n] and not tasks[n].done()]
                    if silent:
                        _nudge_actors(queues, silent,
                                      coupled=False,
                                      judge_visible=judge_votes_visible)

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
                    _nudge_actors(queues, active,
                                  coupled=actor_context_coupled,
                                  judge_visible=judge_votes_visible)
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
