"""Tests for eval/secretary.py — pure logic, no LLM call.

The secretary is the single most safety-critical eval component:
all four cells go through it and a parser bug would silently drag
every cell's accuracy down. These tests pin its behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from eval.secretary import (
    SECRETARY_SYSTEM_PROMPT,
    build_user_prompt,
    build_chat_messages,
    parse_letter,
    parse_boxed,
    normalize_freeform,
    equivalent_freeform,
    extract_answers,
)


# ── Prompt construction ─────────────────────────────────


def test_build_user_prompt_has_problem_and_posts():
    prompt = build_user_prompt(
        "What is 2+2?",
        [{"content": "The answer is 4."}, {"content": "Confirmed."}],
    )
    assert "What is 2+2?" in prompt
    assert "P_1" in prompt and "P_2" in prompt
    assert "The answer is 4." in prompt
    assert "Output exactly one `\\boxed{...}`" in prompt


def test_build_user_prompt_skips_empty_posts():
    prompt = build_user_prompt(
        "Q",
        [{"content": "first"}, {"content": ""}, {"content": "  "}, {"content": "second"}],
    )
    # Only two posts survive — the empty-content ones drop.
    assert prompt.count("### Post P_") == 2
    assert "first" in prompt and "second" in prompt


def test_build_user_prompt_includes_comments_when_asked():
    prompt = build_user_prompt(
        "Q", [{"content": "main", "comments": [{"content": "agree"}]}],
        include_comments=True,
    )
    assert "agree" in prompt
    prompt2 = build_user_prompt(
        "Q", [{"content": "main", "comments": [{"content": "agree"}]}],
        include_comments=False,
    )
    assert "agree" not in prompt2


def test_build_user_prompt_truncates_long_discussions():
    big = "X" * 30000
    prompt = build_user_prompt(
        "Short Q", [{"content": big}, {"content": "FINAL late post"}],
        max_chars=2000,
    )
    assert len(prompt) <= 2200  # max_chars + small problem header
    assert "FINAL late post" in prompt
    assert "Short Q" in prompt
    assert "truncated" in prompt.lower()


def test_build_chat_messages_packages_correctly():
    msgs = build_chat_messages("Q", [{"content": "a"}])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SECRETARY_SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert "Q" in msgs[1]["content"]


# ── Parsers ─────────────────────────────────────────────


def test_parse_letter_basic():
    assert parse_letter(r"the final answer is \boxed{C}") == "C"
    assert parse_letter(r"\boxed{a}") == "A"  # case-folded
    assert parse_letter("no boxed answer here") is None


def test_parse_letter_self_correct_takes_last():
    txt = (r"At first I thought \boxed{A}, but on review I see it must "
           r"be \boxed{B} actually.")
    assert parse_letter(txt) == "B"


def test_parse_letter_ignores_non_letter_boxes():
    # \boxed{42} is not an A-D letter; should not match.
    assert parse_letter(r"\boxed{42}") is None
    # Mixed: one letter, one non-letter — only the letter counts.
    assert parse_letter(r"\boxed{42} then \boxed{D}") == "D"


def test_parse_boxed_basic():
    assert parse_boxed(r"the answer is \boxed{42}") == "42"
    assert parse_boxed(r"\boxed{x^2 + 1}") == "x^2 + 1"
    assert parse_boxed("no answer") is None


def test_parse_boxed_self_correct_takes_last():
    assert parse_boxed(r"\boxed{first} then \boxed{second}") == "second"


# ── Equivalence ────────────────────────────────────────


def test_normalize_freeform_lowercases_and_collapses_whitespace():
    assert normalize_freeform("  Hello   World  ") == "hello world"
    assert normalize_freeform("X+Y") == "x+y"


def test_normalize_freeform_strips_outer_latex_delimiters():
    assert normalize_freeform("$42$") == "42"
    assert normalize_freeform(r"\(a\)") == "a"
    assert normalize_freeform(r"\[expr\]") == "expr"


def test_equivalent_freeform_is_conservative():
    assert equivalent_freeform("42", "42")
    assert equivalent_freeform("$42$", "42")
    # Spacing differences are NOT papered over — conservative.
    assert not equivalent_freeform("x + 1", "x+1")


# ── Orchestration helper ───────────────────────────────


def test_extract_answers_letter_mode():
    seen = []

    def llm(messages):
        seen.append(messages)
        return r"After analysis, \boxed{B}"

    items = [
        ("q1", "What is X?", [{"content": "post"}]),
        ("q2", "What is Y?", [{"content": "post"}]),
    ]
    out = extract_answers(items, llm, parse_mode="letter")
    assert len(out) == 2
    assert out[0]["item_id"] == "q1"
    assert out[0]["parsed_answer"] == "B"
    assert out[1]["item_id"] == "q2"
    assert out[1]["parsed_answer"] == "B"
    # Each item received its own messages list (not the same object reused).
    assert len(seen) == 2
    assert seen[0][1]["content"] != seen[1][1]["content"]


def test_extract_answers_boxed_mode_handles_no_match():
    items = [("q1", "Q", [{"content": "p"}])]
    out = extract_answers(items, lambda _msgs: "no boxed", parse_mode="boxed")
    assert out[0]["parsed_answer"] is None
    assert "no boxed" in out[0]["raw_response"]
