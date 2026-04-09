"""Single agent loop — polling mode.

Each agent runs in rounds. Within a round, the LLM can call python_exec
any number of times. The round ends when the agent calls submit, wait,
or leave. Between rounds, the harness pushes new content.
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
        return (10, 600, 5, 120)

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
    """Format new posts/comments/votes for injection into context.

    Uses P_xxx for posts, C_xxx for comments, V for votes.
    """
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
            val = f'+{item["value"]}' if item["value"] > 0 else str(item["value"])
            prev = item.get("previous_value")
            if prev is not None:
                pv = f'+{prev}' if prev > 0 else str(prev)
                parts.append(f'[V on {tt}_{item["target_id"]}] '
                             f'by {item["author"]}: changed {pv} → {val}')
            else:
                parts.append(f'[V on {tt}_{item["target_id"]}] '
                             f'by {item["author"]}: {val}')
    return "\n\n".join(parts)


# ── LLM call logging ─────────────────────────────────────

def _save_llm_log(path: Path, entry: dict):
    """Append one LLM call record to a JSONL log file."""
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ── Fallback tool call parser ─────────────────────────────

def _parse_tool_call_from_content(content: str) -> dict | None:
    """Extract tool call from content when vLLM's parser misses it.

    Qwen models sometimes emit <tool_call><function=NAME><parameter=...>
    as plain text that vLLM's qwen3_coder parser fails to recognize.
    """
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


def _try_get_new_content(queue: asyncio.Queue,
                         messages: list[dict]) -> bool:
    """Non-blocking check for new content in the queue. Appends to messages
    if found. Returns True if session is ending (None received)."""
    try:
        item = queue.get_nowait()
    except asyncio.QueueEmpty:
        return False
    if item is None:
        queue.put_nowait(None)
        return True
    if isinstance(item, str):
        messages.append({"role": "user", "content": item})
    elif item:
        text = format_new_content(item)
        messages.append({"role": "user",
                         "content": f"New content:\n\n{text}"})
    return False


MAX_NO_TOOL_RETRIES = 3


def _save_discard_streak(discard_dir: Path, name: str, streak: list[dict]):
    """Save a streak of discarded no-tool responses, filed by streak length."""
    if not discard_dir or not streak:
        return
    n = len(streak)
    path = discard_dir / f"{name}_x{n}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(streak, ensure_ascii=False, default=str) + "\n")


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
    new_content_queue: asyncio.Queue,
    llm_log_dir: Path | None = None,
    discard_dir: Path | None = None,
) -> dict | None:
    """Run one round: LLM loops on python_exec until it calls submit/wait/leave.

    Returns:
        dict with submit args, or {"_action": "wait"/"leave"}, or None on failure.
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
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] {name} step={step} LLM ERROR {llm_dur:.1f}s "
                  f"ctx={context_msgs}msgs: {e}", flush=True)
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
        tool_calls = assistant_msg.get("tool_calls")

        # Fallback: parse tool calls from content when vLLM misses them
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

        tool_names = [tc["function"]["name"] for tc in (tool_calls or [])]

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {name} step={step} LLM {llm_dur:.1f}s "
              f"prompt={usage.get('prompt_tokens',0)} "
              f"ctx={context_msgs}msgs gpu=:{gpu_port} "
              f"tools={tool_names or 'no_tool'}", flush=True)

        if not tool_calls:
            # Buffer the discarded response
            no_tool_streak.append({
                "round": result.rounds, "step": step,
                "timestamp": datetime.now().isoformat(),
                "response": llm_response,
            })
            if len(no_tool_streak) >= MAX_NO_TOOL_RETRIES:
                _save_discard_streak(discard_dir, name, no_tool_streak)
                result.exit_reason = "no_tool"
                return None
            # Check for new content and resample
            if _try_get_new_content(new_content_queue, messages):
                _save_discard_streak(discard_dir, name, no_tool_streak)
                result.exit_reason = "session_end"
                return None
            continue

        # Tool call succeeded — flush any pending streak
        if no_tool_streak:
            _save_discard_streak(discard_dir, name, no_tool_streak)
            no_tool_streak = []

        # Log successful calls only
        if llm_log_dir:
            _save_llm_log(llm_log_dir / f"{name}.jsonl", {
                "round": result.rounds, "step": step,
                "timestamp": datetime.now().isoformat(),
                "duration_s": round(llm_dur, 2),
                "request": {
                    "model": model_name,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "max_tokens": 4096,
                },
                "response": llm_response,
            })

        messages.append(assistant_msg)

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
                return fn_args

            elif fn_name == "wait":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "Waiting for new content.",
                })
                return {"_action": "wait"}

            elif fn_name == "leave":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "Leaving session.",
                })
                return {"_action": "leave"}

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
                                "Use python_exec, submit, wait, or leave."),
                })

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
            new_content_queue, submit_event, http,
            max_rounds, timeout, llm_log_dir, discard_dir, result)
    except Exception:
        result.exit_reason = "exception"
        result.error = traceback.format_exc()
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {name} EXCEPTION:\n{result.error}", flush=True)
        return result


async def _run_agent_inner(
    name: str, role: str, api_key: str, session_id: str,
    session_title: str, reference_solution: str,
    parliament_url: str, llm_endpoint: str, model_name: str,
    new_content_queue: asyncio.Queue, submit_event: asyncio.Event,
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

        # Wait for new content (or start immediately on round 0 for actors)
        if round_num == 0:
            if role == "actor":
                messages.append({
                    "role": "user",
                    "content": "Parliament is empty. No one has posted yet. Begin.",
                })
            else:
                try:
                    new_items = await asyncio.wait_for(
                        new_content_queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    result.exit_reason = "no_new_content"
                    break
                if new_items is None:
                    result.exit_reason = "session_end"
                    break
                if isinstance(new_items, str):
                    messages.append({"role": "user", "content": new_items})
                else:
                    text = format_new_content(new_items)
                    messages.append({
                        "role": "user",
                        "content": f"New content to evaluate:\n\n{text}",
                    })
        else:
            try:
                new_items = await asyncio.wait_for(
                    new_content_queue.get(), timeout=300)
            except asyncio.TimeoutError:
                result.exit_reason = "no_new_content"
                break
            if new_items is None:
                result.exit_reason = "session_end"
                break
            if isinstance(new_items, str):
                messages.append({"role": "user", "content": new_items})
            elif new_items:
                text = format_new_content(new_items)
                messages.append({
                    "role": "user",
                    "content": f"New content since your last submission:\n\n{text}",
                })
            else:
                result.exit_reason = "no_new_content"
                break

        # Run the round
        round_result = await run_agent_round(
            messages, role, name, executor,
            llm_endpoint, model_name, http, result,
            new_content_queue, llm_log_dir=llm_log_dir,
            discard_dir=discard_dir)

        if round_result is None:
            if result.exit_reason:
                break
            continue

        action = round_result.get("_action")
        if action == "leave":
            result.exit_reason = "leave"
            break
        elif action == "wait":
            submit_event.set()
            # Fall through: next iteration waits for new content
        else:
            # Normal submit
            submit_event.set()

    await executor.leave(result.exit_reason or "completed")
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
