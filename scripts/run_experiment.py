#!/usr/bin/env python3
"""Run a complete Parliament RL data collection experiment.

Each session is an independent data collection unit:
  - All agents start fresh with no memory from previous sessions
  - All actors + judges run concurrently within a session
  - When all actors leave, judges finish evaluating and leave too
  - Sessions can run in parallel across different forum topics

Usage:
    python scripts/run_experiment.py \
        --parliament-url http://localhost:8080 \
        --model-api http://localhost:8888/v1 \
        --agents 4 --judges 4 \
        --timeout 600 --parallel-sessions
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_DIR / "skills"


def api(base_url: str, method: str, path: str,
        key: str, body: dict | None = None) -> dict | list:
    url = base_url + path
    data = json.dumps(body).encode() if body else None
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        print(f"  API error {e.code} on {method} {path}: {body_text[:200]}")
        return {} if method != "GET" else []
    except Exception as e:
        print(f"  API connection error on {method} {path}: {e}")
        return {} if method != "GET" else []


def load_skill(role: str, skills_dir: Path | None = None) -> str:
    skills_dir = skills_dir or SKILLS_DIR
    skill_file = skills_dir / role / "SKILL.md"
    if not skill_file.exists():
        print(f"  WARNING: Skill file not found: {skill_file}")
        return ""
    return skill_file.read_text(encoding="utf-8")


def build_prompt(name: str, role: str, api_key: str, session_id: str,
                 session_title: str, parliament_url: str,
                 reference_solution: str, skill_content: str) -> str:
    """Build the full agent prompt: identity + problem + skill reference."""
    if role == "judge":
        context = (
            f"You are **{name}**, a silent judge on Science Parliament.\n"
            f"Server: {parliament_url} | Key: {api_key} | Session: {session_id}\n\n"
            f"**Problem:** {session_title}\n\n"
            f"**Reference solution (NEVER reveal):**\n{reference_solution}\n\n"
            f"Read the skill instructions below, then begin.\n\n"
        )
    else:
        context = (
            f"You are **{name}**, a scientist on Science Parliament.\n"
            f"Server: {parliament_url} | Key: {api_key} | Session: {session_id}\n\n"
            f"**Problem:** {session_title}\n\n"
            f"Read the skill instructions below, then begin.\n\n"
        )

    skill_with_urls = skill_content
    skill_with_urls = skill_with_urls.replace("URL/", f"{parliament_url}/")
    skill_with_urls = skill_with_urls.replace("URL", parliament_url)
    skill_with_urls = skill_with_urls.replace("SID", session_id)
    skill_with_urls = skill_with_urls.replace("KEY", api_key)

    return context + "---\n\n" + skill_with_urls


def clear_openclaw_sessions(openclaw_agent_id: str):
    """Clear cached OpenClaw sessions to ensure fresh agent state."""
    sessions_dir = Path.home() / ".openclaw" / "agents" / openclaw_agent_id / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir, ignore_errors=True)
        sessions_dir.mkdir(parents=True, exist_ok=True)


def setup_agent_slots(base_agent_id: str, num_slots: int) -> list[str]:
    """Create N temporary OpenClaw agent profiles for parallel execution.

    Each slot is a clone of base_agent_id with its own sessions directory
    to avoid file lock contention. All slots share the same workspace and
    model — they are identical agents, just isolated at the filesystem level.

    Call cleanup_agent_slots() after the experiment to erase all traces.
    """
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    config = json.loads(config_path.read_text())

    base_entry = None
    for entry in config["agents"]["list"]:
        if entry.get("id") == base_agent_id:
            base_entry = entry
            break
    if not base_entry:
        print(f"  WARNING: base agent '{base_agent_id}' not found in openclaw.json")
        return [base_agent_id] * num_slots

    slot_ids = []
    existing_ids = {e.get("id") for e in config["agents"]["list"]}

    for i in range(num_slots):
        slot_id = f"{base_agent_id}-slot-{i}"
        slot_ids.append(slot_id)
        if slot_id in existing_ids:
            continue
        slot_dir = Path.home() / ".openclaw" / "agents" / slot_id
        slot_dir.mkdir(parents=True, exist_ok=True)
        (slot_dir / "sessions").mkdir(exist_ok=True)
        config["agents"]["list"].append({
            "id": slot_id,
            "name": slot_id,
            "workspace": base_entry.get("workspace", ""),
            "agentDir": str(slot_dir / "agent"),
            "model": base_entry.get("model", ""),
        })
        existing_ids.add(slot_id)

    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"  Agent slots created: {num_slots}", flush=True)
    return slot_ids


def cleanup_agent_slots(base_agent_id: str):
    """Remove all temporary slot profiles and directories.

    Restores openclaw.json and ~/.openclaw/agents/ to pre-experiment state.
    """
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    config = json.loads(config_path.read_text())

    prefix = f"{base_agent_id}-slot-"
    config["agents"]["list"] = [
        e for e in config["agents"]["list"] if not e.get("id", "").startswith(prefix)
    ]
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    agents_dir = Path.home() / ".openclaw" / "agents"
    for slot_dir in agents_dir.glob(f"{prefix}*"):
        shutil.rmtree(slot_dir, ignore_errors=True)

    print(f"  Agent slots cleaned up", flush=True)


async def run_single_agent(openclaw_cmd: str, name: str, role: str,
                           prompt: str, session_id: str,
                           model_api: str, timeout: int,
                           openclaw_agent_id: str) -> dict:
    """Run one OpenClaw agent as a fresh process with unique session ID."""
    # Unique session key per agent per session — guarantees no memory leakage
    unique_key = f"{name}_{session_id}_{uuid.uuid4().hex[:8]}"
    cmd = [
        openclaw_cmd, "agent",
        "--agent", openclaw_agent_id,
        "--message", prompt,
        "--session-id", unique_key,
        "--local",
        "--json",
        "--timeout", str(timeout),
    ]

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] Starting {name} ({role}) session={session_id[:8]}...", flush=True)
    start = time.time()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "OPENAI_API_KEY": "not-needed",
                "OPENAI_BASE_URL": model_api,
            },
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout + 120
        )
    except asyncio.TimeoutError:
        duration = time.time() - start
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {name} timed out after {duration:.0f}s", flush=True)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return {"name": name, "role": role, "session_id": session_id,
                "status": "timeout", "duration": round(duration, 1)}
    except FileNotFoundError:
        print(f"  ERROR: openclaw not found: {openclaw_cmd}", flush=True)
        return {"name": name, "role": role, "session_id": session_id,
                "status": "error", "error": "openclaw not found", "duration": 0}
    except Exception as e:
        duration = time.time() - start
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {name} error: {e}", flush=True)
        return {"name": name, "role": role, "session_id": session_id,
                "status": "error", "error": str(e), "duration": round(duration, 1)}

    duration = time.time() - start
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {name} finished in {duration:.0f}s (exit={proc.returncode})", flush=True)

    stderr_text = stderr.decode(errors="replace")[-500:] if stderr else ""

    return {
        "name": name,
        "role": role,
        "session_id": session_id,
        "status": "done" if proc.returncode == 0 else "failed",
        "exit_code": proc.returncode,
        "duration": round(duration, 1),
        "stdout_len": len(stdout) if stdout else 0,
        "stderr_tail": stderr_text,
    }


async def run_forum(session_id: str, session_title: str,
                    reference_solution: str,
                    actors: list[dict], judges: list[dict],
                    parliament_url: str, openclaw_cmd: str,
                    model_api: str, timeout: int,
                    skills_dir: Path,
                    agent_slots: list[str]) -> list[dict]:
    """Run one complete forum discussion: all agents fresh, concurrent."""
    print(f"\n{'='*70}", flush=True)
    print(f"Forum: {session_title[:65]}", flush=True)
    print(f"Session: {session_id}", flush=True)
    print(f"Actors: {[a['name'] for a in actors]}", flush=True)
    print(f"Judges: {[j['name'] for j in judges]}", flush=True)
    print(f"{'='*70}", flush=True)

    actor_skill = load_skill("actor", skills_dir)
    judge_skill = load_skill("judge", skills_dir)

    if not actor_skill or not judge_skill:
        print("  FATAL: Skill file(s) missing", flush=True)
        return []

    for slot_id in agent_slots:
        clear_openclaw_sessions(slot_id)

    agent_specs = []
    for actor in actors:
        key = actor.get("api_key", "")
        if not key:
            continue
        prompt = build_prompt(
            actor["name"], "actor", key, session_id, session_title,
            parliament_url, "", actor_skill,
        )
        agent_specs.append((actor["name"], "actor", prompt))

    for judge in judges:
        key = judge.get("api_key", "")
        if not key:
            continue
        prompt = build_prompt(
            judge["name"], "judge", key, session_id, session_title,
            parliament_url, reference_solution, judge_skill,
        )
        agent_specs.append((judge["name"], "judge", prompt))

    if not agent_specs:
        print("  ERROR: No agents to run", flush=True)
        return []

    tasks = []
    for i, (name, role, prompt) in enumerate(agent_specs):
        slot_id = agent_slots[i % len(agent_slots)]
        tasks.append(run_single_agent(
            openclaw_cmd, name, role, prompt,
            session_id, model_api, timeout, slot_id,
        ))

    results = await asyncio.gather(*tasks)

    for slot_id in agent_slots:
        clear_openclaw_sessions(slot_id)

    print(f"\nResults for session {session_id}:", flush=True)
    for r in results:
        icon = {"done": "+", "failed": "X", "timeout": "T", "error": "!"}.get(r["status"], "?")
        print(f"  [{icon}] {r['name']:15s} {r['role']:6s} {r['status']:8s} {r.get('duration',0):.0f}s", flush=True)

    return list(results)


def preflight_checks(parliament_url: str, admin_key: str,
                     openclaw_cmd: str, model_api: str,
                     skills_dir: Path) -> bool:
    ok = True
    print("Preflight checks:")

    sessions = api(parliament_url, "GET", "/admin/sessions", admin_key)
    if sessions:
        print(f"  [OK] Parliament: {len(sessions)} sessions")
    else:
        print(f"  [FAIL] Parliament not reachable at {parliament_url}")
        ok = False

    for role in ["actor", "judge"]:
        p = skills_dir / role / "SKILL.md"
        if p.exists():
            print(f"  [OK] Skill: {role} ({p.stat().st_size} bytes)")
        else:
            print(f"  [FAIL] Skill missing: {p}")
            ok = False

    try:
        r = subprocess.run([openclaw_cmd, "--version"], capture_output=True, text=True, timeout=10)
        print(f"  [OK] OpenClaw: {r.stdout.strip()}")
    except FileNotFoundError:
        print(f"  [FAIL] OpenClaw not found: {openclaw_cmd}")
        ok = False
    except Exception as e:
        print(f"  [WARN] OpenClaw: {e}")

    try:
        req = urllib.request.Request(f"{model_api}/models", method="GET")
        resp = urllib.request.urlopen(req, timeout=10)
        models = json.loads(resp.read())
        ids = [m.get("id", "?") for m in models.get("data", [])]
        print(f"  [OK] Model API: {ids}")
    except Exception as e:
        print(f"  [WARN] Model API not reachable: {e}")

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Run Parliament RL Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--parliament-url", default="http://localhost:8080")
    parser.add_argument("--admin-key", default="sp_admin_parliament")
    parser.add_argument("--openclaw-cmd", default="openclaw")
    parser.add_argument("--model-api", default="http://localhost:8888/v1",
                        help="vLLM API URL (default: nginx LB on 8888)")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--judges", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Max seconds per agent per session (default: 600)")
    parser.add_argument("--skills-dir", default=None)
    parser.add_argument("--parallel-sessions", action="store_true",
                        help="Run all sessions in parallel (default: sequential)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--openclaw-agent", default="parliament-scientist")
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir) if args.skills_dir else SKILLS_DIR

    print(f"Parliament RL Data Collection", flush=True)
    print(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Parliament: {args.parliament_url}", flush=True)
    print(f"  Model API:  {args.model_api}", flush=True)
    print(f"  Agents:     {args.agents} actors + {args.judges} judges per session", flush=True)
    print(f"  Timeout:    {args.timeout}s per agent", flush=True)
    print(f"  Parallel:   {args.parallel_sessions}", flush=True)
    print(flush=True)

    if not args.skip_preflight:
        if not preflight_checks(args.parliament_url, args.admin_key,
                                args.openclaw_cmd, args.model_api, skills_dir):
            print("\nPreflight failed. Use --skip-preflight to override.")
            sys.exit(1)
        print()

    sessions = api(args.parliament_url, "GET", "/admin/sessions", args.admin_key)
    if not sessions:
        print("ERROR: No sessions. Start Parliament with --seed first.")
        sys.exit(1)

    open_sessions = [s for s in sessions if s.get("status") == "open"]
    if not open_sessions:
        print("ERROR: No open sessions.")
        sys.exit(1)

    all_users = api(args.parliament_url, "GET", "/admin/users", args.admin_key)
    if not all_users:
        print("ERROR: No users found.")
        sys.exit(1)

    actors = [u for u in all_users if u.get("role") == "actor"][:args.agents]
    judges = [u for u in all_users if u.get("role") == "judge"][:args.judges]

    missing = [u["name"] for u in actors + judges if not u.get("api_key")]
    if missing:
        print(f"ERROR: Missing api_key for: {missing}")
        sys.exit(1)

    agents_per_session = args.agents + args.judges
    num_parallel = len(open_sessions) if args.parallel_sessions else 1
    total_slots = agents_per_session * num_parallel
    agent_slots = setup_agent_slots(args.openclaw_agent, total_slots)

    print(f"\n{len(open_sessions)} sessions, {len(actors)} actors, {len(judges)} judges", flush=True)
    for u in actors:
        print(f"  Actor: {u['name']:15s} key={u['api_key'][:20]}...", flush=True)
    for u in judges:
        print(f"  Judge: {u['name']:15s} key={u['api_key'][:20]}...", flush=True)

    async def run_all():
        session_details = {}
        for s in open_sessions:
            detail = api(args.parliament_url, "GET",
                         f"/admin/sessions/{s['session_id']}", args.admin_key)
            session_details[s["session_id"]] = detail

        def make_forum_coro(session, slot_offset: int):
            sid = session["session_id"]
            detail = session_details.get(sid, {})
            session_slots = agent_slots[slot_offset:slot_offset + agents_per_session]
            return run_forum(
                session_id=sid,
                session_title=session["title"],
                reference_solution=detail.get("reference_solution", ""),
                actors=actors,
                judges=judges,
                parliament_url=args.parliament_url,
                openclaw_cmd=args.openclaw_cmd,
                model_api=args.model_api,
                timeout=args.timeout,
                skills_dir=skills_dir,
                agent_slots=session_slots,
            )

        if args.parallel_sessions:
            coros = [make_forum_coro(s, i * agents_per_session)
                     for i, s in enumerate(open_sessions)]
            return await asyncio.gather(*coros)
        else:
            results = []
            for s in open_sessions:
                r = await make_forum_coro(s, 0)
                results.append(r)
            return results

    t0 = time.time()
    all_results = asyncio.run(run_all())
    duration = time.time() - t0

    cleanup_agent_slots(args.openclaw_agent)

    flat = [r for forum in all_results for r in forum]
    done = sum(1 for r in flat if r.get("status") == "done")
    fail = sum(1 for r in flat if r.get("status") != "done")

    print(f"\n{'='*70}", flush=True)
    print(f"Experiment complete!", flush=True)
    print(f"  Duration:  {duration:.0f}s ({duration/60:.1f} min)", flush=True)
    print(f"  Agents:    {done} done, {fail} failed/timeout", flush=True)
    print(f"  Sessions:  {len(open_sessions)}", flush=True)
    print(f"  View:      {args.parliament_url}", flush=True)
    print(f"{'='*70}", flush=True)

    if args.output:
        output_path = args.output
    else:
        info = api(args.parliament_url, "GET", "/admin/info", args.admin_key)
        run_dir = info.get("run_dir") if isinstance(info, dict) else None
        if run_dir:
            output_path = os.path.join(run_dir, "experiment.json")
        else:
            output_path = str(PROJECT_DIR / "data" / f"experiment_{datetime.now().strftime('%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "duration_seconds": round(duration, 1),
        "sessions": [s["session_id"] for s in open_sessions],
        "results": flat,
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
