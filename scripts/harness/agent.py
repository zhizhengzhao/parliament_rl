"""Single agent loop — polling mode with N-second collection window.

Each agent runs in rounds. Within a round, the LLM can call python_exec
any number of times. The round ends when the agent calls submit or wait.
Between rounds, the agent waits for new content via a collection window:
poll queue every 1s, on first content start N-second timer, collect all
content during the window, then start next round.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from .tools import ToolExecutor, get_tools

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

COLLECT_WINDOW_S = 10


# ── Config loading ────────────────────────────────────────

def _find_latest_config() -> Path:
    cfg_root = PROJECT_DIR / "context_configs"
    versions = sorted(d for d in cfg_root.iterdir() if d.is_dir())
    if not versions:
        raise FileNotFoundError("No versions in context_configs/")
    return versions[-1]


_CFG: dict | None = None


def load_context_config(version: str | None = None) -> dict:
    if version:
        cfg_dir = PROJECT_DIR / "context_configs" / version
    else:
        cfg_dir = _find_latest_config()
    config = json.loads((cfg_dir / "config.json").read_text())
    config["_dir"] = str(cfg_dir)
    config["actor_prompt"] = (cfg_dir / "actor_prompt.txt").read_text()
    config["judge_prompt"] = (cfg_dir / "judge_prompt.txt").read_text()
    return config


def get_config() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = load_context_config()
    return _CFG


# ── Defaults from config ─────────────────────────────────

def _defaults() -> tuple:
    try:
        c = get_config()["agent"]
        return (c["max_rounds"], c["timeout_s"],
                c["max_consecutive_errors"], c["llm_timeout_s"])
    except Exception:
        return (20, 600, 3, 120)

MAX_ROUNDS, TIMEOUT_S, MAX_CONSECUTIVE_ERRORS, LLM_TIMEOUT_S = _defaults()


# ── Data classes ──────────────────────────────────────────

@dataclass
class AgentResult:
    name: str
    role: str
    session_id: str
    rounds: int = 0
    llm_calls: int = 0
    duration: float = 0
    exit_reason: str = ""
    error: str = ""
    posts_created: int = 0
    comments_created: int = 0
    votes_cast: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    fallback_parses: int = 0
    api_errors: int = 0
    llm_errors: int = 0
    no_tool_responses: int = 0
    wait_time: float = 0


# ── Prompt building ───────────────────────────────────────

def build_system_prompt(name: str, role: str, session_title: str,
                        reference_solution: str = "") -> str:
    cfg = get_config()
    if role == "judge":
        return cfg["judge_prompt"].format(
            name=name, session_title=session_title,
            reference_solution=reference_solution)
    return cfg["actor_prompt"].format(name=name, session_title=session_title)


def format_new_content(items: list[dict]) -> str:
    """Format new posts/comments/votes for injection into context."""
    if not items:
        return ""
    parts = []
    for item in items:
        if item["type"] == "post":
            parts.append(f'[P_{item["id"]}] by {item["author"]}:\n{item["content"]}')
        elif item["type"] == "comment":
            parts.append(f'[C_{item["id"]}] on P_{item["post_id"]} '
                         f'by {item["author"]}:\n{item["content"]}')
        elif item["type"] == "vote":
            tt = "P" if item["target_type"] == "post" else "C"
            direction = "positive" if item["value"] > 0 else "negative"
            score = item.get("target_score", 0)
            score_str = f'+{score}' if score > 0 else str(score)
            prev = item.get("previous_value")
            if prev is not None:
                prev_dir = "positive" if prev > 0 else "negative"
                parts.append(f'[V on {tt}_{item["target_id"]}] '
                             f'by {item["author"]}: changed {prev_dir} → '
                             f'{direction} vote, current score of '
                             f'{tt}_{item["target_id"]}: {score_str}')
            else:
                parts.append(f'[V on {tt}_{item["target_id"]}] '
                             f'by {item["author"]}: {direction} vote, '
                             f'current score of '
                             f'{tt}_{item["target_id"]}: {score_str}')
    return "\n\n".join(parts)


# ── LLM call logging ─────────────────────────────────────

def _save_llm_log(path: Path, entry: dict):
    """Append one LLM call record to a JSONL log file."""
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ── Fallback tool call parser ─────────────────────────────

def _parse_tool_call_from_content(content: str) -> dict | None:
    """Extract tool call from content when vLLM's parser misses it."""
    tc_match = re.search(
        r'<tool_call>(.*?)(?:</tool_call>|$)', content, re.DOTALL)
    if not tc_match:
        return None
    tc_block = tc_match.group(1)
    fn_match = re.search(r'<function=(\w+)>', tc_block)
    if not fn_match:
        return None
    fn_name = fn_match.group(1)
    args = {}
    for part in re.split(r'<parameter=', tc_block)[1:]:
        name_match = re.match(r'(\w+)', part)
        if not name_match:
            continue
        name = name_match.group(1)
        rest = part[len(name):]
        value = re.sub(r'^[=>]+\s*', '', rest)
        value = re.sub(r'\s*</parameter>.*', '', value, flags=re.DOTALL)
        value = value.strip().rstrip('>')
        if not value:
            continue
        try:
            args[name] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            try:
                args[name] = json.loads(value.rstrip(','))
            except (json.JSONDecodeError, ValueError):
                args[name] = value
    return {"name": fn_name, "arguments": args}


MAX_NO_TOOL_RETRIES = 3


def _save_discard_streak(discard_dir: Path, name: str, streak: list[dict]):
    """Save a streak of discarded no-tool responses, filed by streak length."""
    if not discard_dir or not streak:
        return
    n = len(streak)
    path = discard_dir / f"{name}_x{n}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(streak, ensure_ascii=False, default=str) + "\n")


# ── Collection window ─────────────────────────────────────

async def _collect_from_queue(
    queue: asyncio.Queue,
    window_s: float = COLLECT_WINDOW_S,
) -> list | str | None:
    """Wait for queue content, then collect for up to window_s seconds.

    Returns collected items (list[dict]), a nudge string, or None (session end).
    Ends early if a post or comment arrives.
    """
    # Phase 1: poll until queue has something
    try:
        first = await asyncio.wait_for(queue.get(), timeout=300)
    except asyncio.TimeoutError:
        return None
    if first is None:
        return None
    if isinstance(first, str):
        return first

    collected = list(first)
    has_post_or_comment = any(
        i["type"] in ("post", "comment") for i in collected)

    if has_post_or_comment:
        return collected

    # Phase 2: only votes so far — wait up to window_s for posts/comments
    deadline = time.time() + window_s
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            item = await asyncio.wait_for(
                queue.get(), timeout=min(remaining, 1.0))
        except asyncio.TimeoutError:
            continue
        if item is None:
            return None
        if isinstance(item, str):
            # Nudge during collection — treat as end of collection
            collected.append({"type": "nudge", "content": item})
            return collected
        collected.extend(item)
        if any(i["type"] in ("post", "comment") for i in item):
            break

    return collected


# ── Single agent round ────────────────────────────────────

async def run_agent_round(
    messages: list[dict[str, Any]],
    role: str,
    name: str,
    executor: ToolExecutor,
    llm_endpoint: str,
    model_name: str,
    http: aiohttp.ClientSession,
    result: AgentResult,
    llm_log_dir: Path | None = None,
    discard_dir: Path | None = None,
) -> dict | None:
    """Run one round: LLM loops on python_exec until it calls submit/wait.

    Returns:
        dict with submit args, or {"_action": "wait"}, or None on failure.
    """
    tools = get_tools(role)
    gpu_port = llm_endpoint.split(":")[2].split("/")[0]
    consecutive_errors = 0
    no_tool_streak: list[dict] = []

    for step in range(20):
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                m["content"] = None

        llm_start = time.time()
        context_msgs = len(messages)
        try:
            llm_response = await _call_llm(
                http, llm_endpoint, model_name, messages, tools)
        except Exception as e:
            llm_dur = time.time() - llm_start
            result.llm_errors += 1
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] {name} step={step} LLM ERROR {llm_dur:.1f}s "
                  f"ctx={context_msgs}msgs: {e}", flush=True)
            if llm_log_dir:
                _save_llm_log(llm_log_dir / f"{name}.jsonl", {
                    "round": result.rounds, "step": step,
                    "timestamp": datetime.now().isoformat(),
                    "duration_s": round(llm_dur, 2),
                    "status": "error",
                    "context_msgs": context_msgs,
                    "error": str(e),
                })
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                result.exit_reason = "llm_errors"
                result.error = str(e)
                return None
            await asyncio.sleep(2)
            continue

        llm_dur = time.time() - llm_start
        consecutive_errors = 0
        result.llm_calls += 1

        assistant_msg = llm_response["choices"][0]["message"]
        usage = llm_response.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        completion_tok = usage.get("completion_tokens", 0)
        result.total_prompt_tokens += prompt_tok
        result.total_completion_tokens += completion_tok
        tool_calls = assistant_msg.get("tool_calls")

        fallback_used = False
        if not tool_calls:
            content = assistant_msg.get("content") or ""
            parsed = _parse_tool_call_from_content(content)
            if parsed:
                tool_calls = [{
                    "id": f"fallback_{result.rounds}_{step}",
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": json.dumps(
                            parsed["arguments"], ensure_ascii=False),
                    },
                }]
                assistant_msg["tool_calls"] = tool_calls
                fallback_used = True
                result.fallback_parses += 1

        tool_names = [tc["function"]["name"] for tc in (tool_calls or [])]

        ts = datetime.now().strftime("%H:%M:%S")
        fb_tag = " [fallback]" if fallback_used else ""
        print(f"  [{ts}] {name} step={step} LLM {llm_dur:.1f}s "
              f"prompt={prompt_tok} comp={completion_tok} "
              f"ctx={context_msgs}msgs gpu=:{gpu_port} "
              f"tools={tool_names or 'no_tool'}{fb_tag}", flush=True)

        if not tool_calls:
            result.no_tool_responses += 1
            no_tool_streak.append({
                "round": result.rounds, "step": step,
                "timestamp": datetime.now().isoformat(),
                "response": llm_response,
            })
            if llm_log_dir:
                _save_llm_log(llm_log_dir / f"{name}.jsonl", {
                    "round": result.rounds, "step": step,
                    "timestamp": datetime.now().isoformat(),
                    "duration_s": round(llm_dur, 2),
                    "status": "no_tool",
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": completion_tok,
                    "context_msgs": context_msgs,
                    "content_preview": (assistant_msg.get("content") or "")[:500],
                })
            if len(no_tool_streak) >= MAX_NO_TOOL_RETRIES:
                _save_discard_streak(discard_dir, name, no_tool_streak)
                result.exit_reason = "no_tool"
                return None
            continue

        if no_tool_streak:
            _save_discard_streak(discard_dir, name, no_tool_streak)
            no_tool_streak = []

        if llm_log_dir:
            last_user_msg = None
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")
                    break
            _save_llm_log(llm_log_dir / f"{name}.jsonl", {
                "round": result.rounds, "step": step,
                "timestamp": datetime.now().isoformat(),
                "duration_s": round(llm_dur, 2),
                "status": "fallback" if fallback_used else "ok",
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "context_msgs": context_msgs,
                "tool_calls": tool_names,
                "last_user_message": (last_user_msg or "")[:1000],
                "response": llm_response,
            })

        _TOOL_PRIORITY = {"python_exec": 0, "submit": 1, "wait": 2}
        tool_calls = sorted(
            tool_calls,
            key=lambda tc: _TOOL_PRIORITY.get(tc["function"]["name"], 99),
        )
        assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        end_action = None
        end_args = None

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                fn_args = {}

            if fn_name == "submit":
                submit_result = await executor.execute_submit(fn_args, role)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(
                        submit_result, ensure_ascii=False),
                })
                if submit_result.get("post_id"):
                    result.posts_created += 1
                result.comments_created += len(
                    submit_result.get("comments", []))
                result.votes_cast += len(
                    submit_result.get("votes", []))
                result.api_errors += len(
                    submit_result.get("errors", []))
                if end_action is None:
                    end_action = "submit"
                    end_args = fn_args

            elif fn_name == "wait":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "Waiting for new content.",
                })
                if end_action is None:
                    end_action = "wait"

            elif fn_name == "python_exec":
                output = ToolExecutor.python_exec(fn_args.get("code", ""))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": output,
                })

            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": (f"Error: unknown tool '{fn_name}'. "
                                "Use python_exec, submit, or wait."),
                })

        if end_action == "submit":
            return end_args
        elif end_action == "wait":
            return {"_action": "wait"}

    result.exit_reason = "step_limit"
    return None


# ── Full agent lifecycle ──────────────────────────────────

async def run_agent(
    name: str,
    role: str,
    api_key: str,
    session_id: str,
    session_title: str,
    reference_solution: str,
    parliament_url: str,
    llm_endpoint: str,
    model_name: str,
    new_content_queue: asyncio.Queue,
    submit_event: asyncio.Event,
    processing: set[str],
    http: aiohttp.ClientSession,
    max_rounds: int = MAX_ROUNDS,
    timeout: float = TIMEOUT_S,
    llm_log_dir: Path | None = None,
    discard_dir: Path | None = None,
) -> AgentResult:
    """Run agent through multiple rounds of the polling protocol."""

    result = AgentResult(name=name, role=role, session_id=session_id)
    try:
        return await _run_agent_inner(
            name, role, api_key, session_id, session_title,
            reference_solution, parliament_url, llm_endpoint, model_name,
            new_content_queue, submit_event, processing, http,
            max_rounds, timeout, llm_log_dir, discard_dir, result)
    except Exception:
        result.exit_reason = "exception"
        result.error = traceback.format_exc()
        processing.discard(name)
        submit_event.set()
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {name} EXCEPTION:\n{result.error}", flush=True)
        return result


async def _run_agent_inner(
    name: str, role: str, api_key: str, session_id: str,
    session_title: str, reference_solution: str,
    parliament_url: str, llm_endpoint: str, model_name: str,
    new_content_queue: asyncio.Queue, submit_event: asyncio.Event,
    processing: set[str],
    http: aiohttp.ClientSession,
    max_rounds: int, timeout: float,
    llm_log_dir: Path | None, discard_dir: Path | None,
    result: AgentResult,
) -> AgentResult:
    executor = ToolExecutor(parliament_url, session_id, api_key, http)
    await executor.join()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(
            name, role, session_title, reference_solution)},
    ]

    start = time.time()

    for round_num in range(max_rounds):
        if time.time() - start > timeout:
            result.exit_reason = "timeout"
            break

        result.rounds = round_num + 1

        if round_num == 0:
            if role == "actor":
                processing.add(name)
                messages.append({
                    "role": "user",
                    "content": "Parliament is empty. No one has posted yet. Begin.",
                })
            else:
                wait_start = time.time()
                collected = await _collect_from_queue(new_content_queue)
                result.wait_time += time.time() - wait_start
                if collected is None:
                    result.exit_reason = "session_end"
                    break
                processing.add(name)
                if isinstance(collected, str):
                    messages.append({"role": "user", "content": collected})
                else:
                    items = [i for i in collected if i.get("type") != "nudge"]
                    text = format_new_content(items)
                    messages.append({
                        "role": "user",
                        "content": f"New content to evaluate:\n\n{text}",
                    })
        else:
            wait_start = time.time()
            collected = await _collect_from_queue(new_content_queue)
            result.wait_time += time.time() - wait_start
            if collected is None:
                result.exit_reason = "session_end"
                break
            processing.add(name)
            if isinstance(collected, str):
                messages.append({"role": "user", "content": collected})
            else:
                nudges = [i for i in collected
                          if i.get("type") == "nudge"]
                items = [i for i in collected
                         if i.get("type") != "nudge"]
                parts = []
                if items:
                    parts.append(format_new_content(items))
                for n in nudges:
                    parts.append(n["content"])
                messages.append({
                    "role": "user",
                    "content": "\n\n".join(parts) if parts
                    else "No new content.",
                })

        # Run the round
        round_result = await run_agent_round(
            messages, role, name, executor,
            llm_endpoint, model_name, http, result,
            llm_log_dir=llm_log_dir, discard_dir=discard_dir)

        processing.discard(name)
        submit_event.set()

        if round_result is None:
            if result.exit_reason:
                break
            continue

        if round_result.get("_action") == "wait":
            pass

    await executor.leave(result.exit_reason or "completed")
    processing.discard(name)
    result.duration = round(time.time() - start, 1)
    if not result.exit_reason:
        result.exit_reason = "max_rounds"
    return result


# ── LLM call ──────────────────────────────────────────────

async def _call_llm(
    http: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 4096,
    }
    async with http.post(
        f"{endpoint}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=LLM_TIMEOUT_S),
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"LLM API error {resp.status}: {text[:300]}")
        return await resp.json()
