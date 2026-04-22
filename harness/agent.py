"""Single agent loop — event-driven polling mode.

Each agent runs in rounds. Within a round, the LLM can call python_exec
(plus `vote` in coupled mode) any number of times. The round ends when:
  - Actor (coupled)     calls submit or wait  →  set_event (wakes runner)
  - Actor (independent) calls submit or leave →  set_event (wakes runner)
                                                  `leave` also retires
                                                  the agent permanently
  - Judge               calls vote             →  no event (silent)

Between rounds the agent blocks on its queue. Items pushed by the
runner are batches of posts/comments/(anonymized)judge_votes for
coupled actors, or just (anonymized)judge_votes for independent actors;
a placeholder `str` may also be pushed to nudge a stalled actor.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from .prompts import build_system_prompt, format_new_content, get_config
from .tools import IdMap, ToolExecutor, get_tools, python_exec

# ── Per-round limits (overridable via context_config) ─────

STEP_LIMIT = 20
MAX_NO_TOOL_RETRIES = 3
MAX_TOKENS = 2048                 # one reasoning step rarely needs more


def _agent_defaults() -> tuple[int, int, int]:
    try:
        c = get_config()["agent"]
        return c["max_rounds"], c["max_consecutive_errors"], c["llm_timeout_s"]
    except Exception:
        return 30, 3, 120


MAX_ROUNDS, MAX_CONSECUTIVE_ERRORS, LLM_TIMEOUT_S = _agent_defaults()

_TOOL_PRIORITY = {"python_exec": 0, "vote": 1, "submit": 2,
                  "wait": 3, "leave": 3}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── Result record ─────────────────────────────────────────

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


# ── LLM call ──────────────────────────────────────────────

async def _call_llm(http: aiohttp.ClientSession, endpoint: str, model: str,
                    messages: list[dict], tools: list[dict]) -> dict:
    await asyncio.sleep(random.uniform(0, 0.5))
    payload = {
        "model": model, "messages": messages, "tools": tools,
        "tool_choice": "auto", "max_tokens": MAX_TOKENS,
    }
    async with http.post(
        f"{endpoint}/chat/completions", json=payload,
        timeout=aiohttp.ClientTimeout(connect=10, total=LLM_TIMEOUT_S),
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"LLM API error {resp.status}: {text[:300]}")
        return await resp.json()


# ── Fallback tool-call parser (when vLLM's parser misses) ─

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
_FN_NAME_RE = re.compile(r"<function=(\w+)>")
_PARAM_NAME_RE = re.compile(r"(\w+)")


def _parse_tool_call_from_content(content: str) -> dict | None:
    tc_match = _TOOL_CALL_RE.search(content)
    if not tc_match:
        return None
    block = tc_match.group(1)
    fn_match = _FN_NAME_RE.search(block)
    if not fn_match:
        return None

    args: dict[str, Any] = {}
    for part in re.split(r"<parameter=", block)[1:]:
        name_match = _PARAM_NAME_RE.match(part)
        if not name_match:
            continue
        name = name_match.group(1)
        rest = part[len(name):]
        value = re.sub(r"^[=>]+\s*", "", rest)
        value = re.sub(r"\s*</parameter>.*", "", value, flags=re.DOTALL)
        value = value.strip().rstrip(">")
        if not value:
            continue
        try:
            args[name] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            try:
                args[name] = json.loads(value.rstrip(","))
            except (json.JSONDecodeError, ValueError):
                args[name] = value
    return {"name": fn_match.group(1), "arguments": args}


# ── Per-agent logging ─────────────────────────────────────

class AgentLogger:
    """JSONL logging for one agent within one session.

    Writes append-only entries. Both directories are optional;
    when either is None the corresponding writes become no-ops.
    """

    def __init__(self, name: str, llm_log_dir: Path | None,
                 discard_dir: Path | None):
        self.name = name
        self._llm_path = (llm_log_dir / f"{name}.jsonl") if llm_log_dir else None
        self._discard_dir = discard_dir

    def _append(self, entry: dict) -> None:
        if self._llm_path is None:
            return
        with open(self._llm_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def llm(self, **fields: Any) -> None:
        fields.setdefault("timestamp", datetime.now().isoformat())
        self._append(fields)

    def discard_streak(self, streak: list[dict]) -> None:
        if not self._discard_dir or not streak:
            return
        path = self._discard_dir / f"{self.name}_x{len(streak)}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(streak, ensure_ascii=False, default=str) + "\n")


# ── Tool-call helpers ─────────────────────────────────────

def _strip_thinking(messages: list[dict]) -> None:
    """Drop assistant `content` once tool_calls were emitted (saves tokens)."""
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m["content"] = None


def _build_fallback_tool_call(parsed: dict, round_n: int, step: int) -> dict:
    return {
        "id": f"fallback_{round_n}_{step}",
        "type": "function",
        "function": {
            "name": parsed["name"],
            "arguments": json.dumps(parsed["arguments"], ensure_ascii=False),
        },
    }


def _last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


def _is_tool_error(content: str) -> bool:
    return (content.startswith("Error")
            or ('"errors": [' in content and '"errors": []' not in content))


def _trunc(value: Any, n: int = 500) -> str:
    return value[:n] if isinstance(value, str) else str(value)[:n]


# ── Tool dispatch ─────────────────────────────────────────

async def _dispatch_one_tool(tc: dict, executor: ToolExecutor, role: str,
                             result: AgentResult, messages: list[dict]
                             ) -> tuple[str | None, dict | None]:
    """Execute a single tool call, append its result message, return end_action.

    Returns:
      (None, None)             — round continues
      ("submit", submit_args)  — actor finished by submit
      ("wait", None)           — actor (coupled) finished by wait
      ("leave", None)          — actor (independent) retired permanently
      ("vote", None)           — judge finished by vote
    """
    fn_name = tc["function"]["name"]
    raw_args = tc["function"].get("arguments", "{}")
    try:
        fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except (json.JSONDecodeError, TypeError):
        messages.append({
            "role": "tool", "tool_call_id": tc["id"],
            "content": f"Error: could not parse arguments: {str(raw_args)[:200]}",
        })
        return None, None

    if fn_name == "python_exec":
        code = fn_args.get("code", "") if isinstance(fn_args, dict) else str(fn_args)
        output = await python_exec(code)
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})
        return None, None

    if fn_name == "vote":
        votes_arg = fn_args.get("votes", fn_args) if isinstance(fn_args, dict) else fn_args
        vote_result = await executor.execute_votes(votes_arg)
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps(vote_result, ensure_ascii=False)})
        result.votes_cast += len(vote_result.get("votes", []))
        result.api_errors += len(vote_result.get("errors", []))
        return ("vote", None) if role == "judge" else (None, None)

    if fn_name == "submit":
        if isinstance(fn_args, dict) and fn_args.get("votes"):
            v_result = await executor.execute_votes(fn_args.pop("votes"))
            result.votes_cast += len(v_result.get("votes", []))
            result.api_errors += len(v_result.get("errors", []))
        submit_result = await executor.execute_submit(fn_args)
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps(submit_result, ensure_ascii=False)})
        if submit_result.get("post_id"):
            result.posts_created += 1
        result.comments_created += len(submit_result.get("comments", []))
        result.api_errors += len(submit_result.get("errors", []))
        return "submit", fn_args

    if fn_name == "wait":
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": "Waiting for new content."})
        return "wait", None

    if fn_name == "leave":
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": "Left the session."})
        return "leave", None

    messages.append({
        "role": "tool", "tool_call_id": tc["id"],
        "content": (f"Error: unknown tool '{fn_name}'. "
                    "Available: python_exec, vote, submit, wait/leave."),
    })
    return None, None


# ── Queue wait ────────────────────────────────────────────

async def _wait_for_content(queue: asyncio.Queue) -> list | str | None:
    """Block for the first batch, then drain any extras into one round.

    While the agent was busy running its previous LLM call the runner may
    have distributed several batches. Merging them keeps the round count
    proportional to substantive activity.
    """
    first = await queue.get()
    if first is None:
        return None
    if isinstance(first, str):
        if queue.empty():
            return first
        first = None

    items = list(first) if first else []
    while not queue.empty():
        try:
            extra = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if extra is None:
            queue.put_nowait(None)
            break
        if isinstance(extra, list):
            items.extend(extra)
    return items if items else None


# ── Single round ──────────────────────────────────────────

async def run_agent_round(
    messages: list[dict[str, Any]], role: str, name: str,
    executor: ToolExecutor, llm_endpoint: str, model_name: str,
    http: aiohttp.ClientSession, result: AgentResult,
    logger: AgentLogger,
) -> dict | None:
    """Run one round: LLM loops until it calls a round-ending tool.

    Returns:
      submit args dict          — actor ended via submit
      {"_action": "wait"}       — actor ended via wait (coupled mode)
      {"_action": "leave"}      — actor ended via leave (independent mode)
      {"_action": "vote"}       — judge ended via vote
      None                      — failure (caller retries next round)
    """
    coupled = bool(get_config().get("actor_context_coupled", True))
    tools = get_tools(role, coupled=coupled)
    gpu_port = llm_endpoint.split(":")[2].split("/")[0]
    consecutive_errors = 0
    no_tool_streak: list[dict] = []

    for step in range(STEP_LIMIT):
        _strip_thinking(messages)
        ctx_msgs = len(messages)

        # 1) LLM call --------------------------------------------------
        llm_start = time.time()
        try:
            llm_response = await _call_llm(
                http, llm_endpoint, model_name, messages, tools)
            assistant_msg = llm_response["choices"][0]["message"]
        except Exception as e:
            llm_dur = time.time() - llm_start
            err = str(e)
            result.llm_errors += 1
            print(f"  [{_ts()}] {name} step={step} LLM ERROR {llm_dur:.1f}s "
                  f"ctx={ctx_msgs}msgs: {e}", flush=True)
            logger.llm(round=result.rounds, step=step,
                       duration_s=round(llm_dur, 2), status="error",
                       context_msgs=ctx_msgs, error=err, messages=messages)
            if "400" in err and "input_tokens" in err:
                result.exit_reason = "context_overflow"
                print(f"  [{_ts()}] {name} context overflow, retiring",
                      flush=True)
                return None
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                return None
            await asyncio.sleep(2)
            continue

        llm_dur = time.time() - llm_start
        consecutive_errors = 0
        result.llm_calls += 1
        usage = llm_response.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        completion_tok = usage.get("completion_tokens", 0)
        result.total_prompt_tokens += prompt_tok
        result.total_completion_tokens += completion_tok

        # 2) Tool-call extraction (with fallback parser) ---------------
        tool_calls = assistant_msg.get("tool_calls")
        fallback_used = False
        if not tool_calls:
            content = assistant_msg.get("content") or ""
            parsed = _parse_tool_call_from_content(content)
            if parsed:
                tool_calls = [_build_fallback_tool_call(
                    parsed, result.rounds, step)]
                assistant_msg["tool_calls"] = tool_calls
                fallback_used = True
                result.fallback_parses += 1

        tool_names = [tc["function"]["name"] for tc in (tool_calls or [])]
        fb_tag = " [fallback]" if fallback_used else ""
        print(f"  [{_ts()}] {name} step={step} LLM {llm_dur:.1f}s "
              f"prompt={prompt_tok} comp={completion_tok} ctx={ctx_msgs}msgs "
              f"gpu=:{gpu_port} tools={tool_names or 'no_tool'}{fb_tag}",
              flush=True)

        # 3) No-tool handling ------------------------------------------
        if not tool_calls:
            result.no_tool_responses += 1
            no_tool_streak.append({
                "round": result.rounds, "step": step,
                "timestamp": datetime.now().isoformat(),
                "response": llm_response,
            })
            logger.llm(round=result.rounds, step=step,
                       duration_s=round(llm_dur, 2), status="no_tool",
                       prompt_tokens=prompt_tok, completion_tokens=completion_tok,
                       context_msgs=ctx_msgs,
                       content_preview=(assistant_msg.get("content") or "")[:500])
            if len(no_tool_streak) >= MAX_NO_TOOL_RETRIES:
                logger.discard_streak(no_tool_streak)
                return None
            continue

        if no_tool_streak:
            logger.discard_streak(no_tool_streak)
            no_tool_streak = []

        logger.llm(round=result.rounds, step=step,
                   duration_s=round(llm_dur, 2),
                   status="fallback" if fallback_used else "ok",
                   prompt_tokens=prompt_tok, completion_tokens=completion_tok,
                   context_msgs=ctx_msgs, tool_calls=tool_names,
                   last_user_message=_last_user_message(messages)[:1000],
                   response=llm_response)

        # 4) Dispatch tools (sorted: python_exec → vote → submit → wait)
        tool_calls.sort(key=lambda tc: _TOOL_PRIORITY.get(
            tc["function"]["name"], 99))
        assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        end_action: str | None = None
        end_args: dict | None = None
        for tc in tool_calls:
            action, args = await _dispatch_one_tool(
                tc, executor, role, result, messages)
            tool_content = messages[-1].get("content", "")
            logger.llm(round=result.rounds, step=step,
                       status="tool_error" if _is_tool_error(tool_content) else "tool_ok",
                       tool=tc["function"]["name"],
                       arguments=_trunc(tc["function"].get("arguments", "")),
                       response=_trunc(tool_content, 1000))
            if action and end_action is None:
                end_action, end_args = action, args

        if end_action == "submit":
            return end_args
        if end_action == "wait":
            return {"_action": "wait"}
        if end_action == "leave":
            return {"_action": "leave"}
        if end_action == "vote":
            return {"_action": "vote"}

    return None


# ── Full agent lifecycle ──────────────────────────────────

async def run_agent(
    name: str, role: str, api_key: str, session_id: str,
    session_title: str, reference_solution: str, parliament_url: str,
    llm_endpoint: str, model_name: str,
    new_content_queue: asyncio.Queue, submit_event: asyncio.Event,
    processing: set[str], http: aiohttp.ClientSession, id_map: IdMap,
    max_rounds: int = MAX_ROUNDS,
    llm_log_dir: Path | None = None, discard_dir: Path | None = None,
) -> AgentResult:
    result = AgentResult(name=name, role=role, session_id=session_id)
    logger = AgentLogger(name, llm_log_dir, discard_dir)
    try:
        return await _run_agent_inner(
            name, role, api_key, session_id, session_title,
            reference_solution, parliament_url, llm_endpoint, model_name,
            new_content_queue, submit_event, processing, http, id_map,
            max_rounds, logger, result)
    except Exception:
        result.exit_reason = "exception"
        result.error = traceback.format_exc()
        processing.discard(name)
        if role != "judge":
            submit_event.set()
        print(f"  [{_ts()}] {name} EXCEPTION:\n{result.error}", flush=True)
        return result


async def _run_agent_inner(
    name: str, role: str, api_key: str, session_id: str,
    session_title: str, reference_solution: str, parliament_url: str,
    llm_endpoint: str, model_name: str,
    new_content_queue: asyncio.Queue, submit_event: asyncio.Event,
    processing: set[str], http: aiohttp.ClientSession, id_map: IdMap,
    max_rounds: int, logger: AgentLogger, result: AgentResult,
) -> AgentResult:
    executor = ToolExecutor(parliament_url, session_id, api_key, http,
                            id_map, role=role)
    await executor.join()

    messages: list[dict[str, Any]] = [{
        "role": "system",
        "content": build_system_prompt(name, role, session_title,
                                       reference_solution,
                                       session_id=session_id),
    }]

    coupled = bool(get_config().get("actor_context_coupled", True))
    round_0_prompt = ("Parliament is empty. No one has posted yet. Begin."
                      if coupled
                      else "You are starting from a blank slate. "
                           "Submit your first reasoning step.")

    start = time.time()
    round_num = 0
    while role != "actor" or round_num < max_rounds:
        result.rounds = round_num + 1

        if round_num == 0 and role == "actor":
            processing.add(name)
            messages.append({"role": "user", "content": round_0_prompt})
        else:
            wait_start = time.time()
            collected = await _wait_for_content(new_content_queue)
            result.wait_time += time.time() - wait_start
            if collected is None:
                result.exit_reason = "session_end"
                break
            processing.add(name)
            if isinstance(collected, str):
                messages.append({"role": "user", "content": collected})
            else:
                messages.append({
                    "role": "user",
                    "content": format_new_content(collected) or "No new content.",
                })

        round_result = await run_agent_round(
            messages, role, name, executor, llm_endpoint, model_name,
            http, result, logger)

        processing.discard(name)
        if role != "judge":
            submit_event.set()

        if round_result is None:
            if result.exit_reason == "context_overflow":
                break
            continue

        # Independent-mode `leave` retires the actor permanently:
        # don't loop back for new content, just wrap up.
        if isinstance(round_result, dict) and round_result.get("_action") == "leave":
            result.exit_reason = "left"
            break

        round_num += 1

    await executor.leave(result.exit_reason or "completed")
    processing.discard(name)
    result.duration = round(time.time() - start, 1)
    if not result.exit_reason:
        result.exit_reason = "max_rounds"
    return result
