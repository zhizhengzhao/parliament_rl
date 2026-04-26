"""Cell-agnostic answer extractor (the "secretary") for frame evaluation.

After a Parliament/Solo session finishes, the actors have produced a
sequence of posts (and, in coupled cells, comments). To turn that
discussion into a *single* answer comparable across cells, we apply a
fixed "secretary" agent whose prompt and weights are the **same**
regardless of which cell produced the discussion. This isolates
"which policy generates a more answer-bearing discussion" from
"which extractor is luckier".

Design constraints:

* The secretary is **stateless** w.r.t. cells. It receives only:
  problem text + ordered list of actor posts (+ optional comments).
  It does not see judge votes, judge identities, or peer vote scores.
* The secretary's model and prompt are **fixed**. We use the original
  base model (typically Qwen3.5-9B) — never an RL-trained policy —
  so the extractor cannot drift into favouring any one cell's style.
* The output format is **boxed**, regardless of question type:
  ``\\boxed{X}`` for multiple-choice (X ∈ {A, B, C, D}) or
  ``\\boxed{<expression>}`` for free-form. Parsers in this module
  handle both with the same `\\boxed{}` syntax.

This module is pure (no IO, no vLLM dependency) so it can be
unit-tested. The orchestrator in :mod:`eval.frame` plugs an actual
LLM call (or any callable) into :func:`extract_answers`.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

# Boxed-letter (MC) and boxed-expression (free-form) parsers — the LLM
# often writes several ``\boxed{}`` blocks while reasoning; the *last*
# one wins (model self-corrects).  Letter parser is case-insensitive
# (the LLM occasionally drops capitalisation; ``parse_letter`` then
# ``.upper()``-s the result so the public contract stays "A/B/C/D").
_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*([A-D])\s*\}", re.IGNORECASE)
_BOXED_ANY_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


# ── Prompt construction ───────────────────────────────────


SECRETARY_SYSTEM_PROMPT = (
    "You are a meticulous scientific secretary. A group of Scientists "
    "has been discussing a problem on a research forum. Your only job "
    "is to read their full discussion and report the single final "
    "answer they converged on (or, if they did not converge, the "
    "answer best supported by their reasoning).\n\n"
    "Be silent on process. Do not summarise the discussion, do not "
    "critique the reasoning, do not explain your choice. Output one "
    "and only one ``\\boxed{...}`` containing the final answer.\n\n"
    "For multiple-choice questions answer with ``\\boxed{A}``, "
    "``\\boxed{B}``, ``\\boxed{C}``, or ``\\boxed{D}``. For free-form "
    "questions place the answer (formula, number, or short phrase) "
    "inside the ``\\boxed{}``."
)


def build_user_prompt(problem: str, posts: list[dict],
                      include_comments: bool = True,
                      max_chars: int = 24000) -> str:
    """Assemble the user message: problem + linearised actor discussion.

    `posts` carries dicts with at least ``content`` and optionally
    ``comments`` (a list of ``{author, content}``). Items with empty
    content are dropped. Long discussions are truncated from the
    *front* (prefer the latter half — that's usually where the answer
    converges) at ``max_chars``.
    """
    sections: list[str] = ["## Problem", problem.strip(), "", "## Discussion"]

    for i, p in enumerate(posts, start=1):
        content = (p.get("content") or "").strip()
        if not content:
            continue
        sections.append(f"### Post P_{i}")
        sections.append(content)
        if include_comments:
            for c in p.get("comments", []) or []:
                cc = (c.get("content") or "").strip()
                if cc:
                    sections.append(f"  - Comment: {cc}")
        sections.append("")

    sections.append(
        "## Task\n"
        "Output exactly one `\\boxed{...}` containing the final answer.\n"
    )

    body = "\n".join(sections)
    if len(body) > max_chars:
        # Front-truncate so the *late* posts (where the answer usually
        # appears) survive. Add a marker so the secretary knows the
        # transcript was clipped.
        keep = body[-max_chars:]
        # Realign at a heading boundary so we don't start mid-post.
        m = re.search(r"### Post P_\d+", keep)
        if m:
            keep = keep[m.start():]
        body = ("## Problem\n" + problem.strip() + "\n\n"
                "## Discussion (truncated; only the latter portion shown)\n"
                + keep)
    return body


def build_chat_messages(problem: str, posts: list[dict],
                        include_comments: bool = True,
                        max_chars: int = 24000) -> list[dict]:
    """Convenience: package into the chat-template messages list."""
    return [
        {"role": "system", "content": SECRETARY_SYSTEM_PROMPT},
        {"role": "user",
         "content": build_user_prompt(problem, posts, include_comments,
                                      max_chars=max_chars)},
    ]


# ── Answer parsing ────────────────────────────────────────


def parse_letter(text: str) -> str | None:
    """Return the *last* ``\\boxed{A-D}`` letter, or None."""
    matches = _BOXED_LETTER_RE.findall(text)
    return matches[-1].upper() if matches else None


def parse_boxed(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}`` (free-form), or None."""
    matches = _BOXED_ANY_RE.findall(text)
    return matches[-1].strip() if matches else None


# ── Free-form answer normalisation ────────────────────────


_WHITESPACE_RE = re.compile(r"\s+")
_LATEX_TRIM_RE = re.compile(r"^[\$\\\[\(\s]+|[\$\\\]\)\s]+$")


def normalize_freeform(s: str) -> str:
    """Cheap canonicalisation for boxed free-form answers.

    Strips whitespace, surrounding $ / \\[ / \\( / brackets, and
    collapses internal whitespace. *Not* a CAS — sufficient for
    "do these two boxed strings look the same once you ignore
    formatting noise". Stronger comparisons (sympy, etc.) are out of
    scope for the secretary.
    """
    s = s.strip()
    s = _LATEX_TRIM_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s)
    return s.lower()


def equivalent_freeform(pred: str, gold: str) -> bool:
    """True iff normalised pred and gold are string-equal.

    We deliberately keep this conservative — better to under-report
    accuracy than to over-claim equivalence with a brittle CAS rule.
    """
    return normalize_freeform(pred) == normalize_freeform(gold)


# ── Orchestration helper ──────────────────────────────────


def extract_answers(
    items: Iterable[tuple[str, str, list[dict]]],
    llm_call: Callable[[list[dict]], str],
    *,
    parse_mode: str = "letter",
    include_comments: bool = True,
) -> list[dict]:
    """Apply the secretary to a stream of (item_id, problem, posts).

    `llm_call(messages)` runs one chat completion and returns the raw
    assistant string. `parse_mode`:

    * ``"letter"`` — multiple-choice: return ``\\boxed{A-D}`` letter.
    * ``"boxed"``  — free-form: return whatever is inside the last
      ``\\boxed{...}``.

    Returns a list of dicts with ``item_id``, ``raw_response``,
    ``parsed_answer``.
    """
    out: list[dict] = []
    for item_id, problem, posts in items:
        messages = build_chat_messages(problem, posts, include_comments)
        text = llm_call(messages)
        parsed = (parse_letter(text) if parse_mode == "letter"
                  else parse_boxed(text))
        out.append({
            "item_id": item_id,
            "raw_response": text,
            "parsed_answer": parsed,
        })
    return out
