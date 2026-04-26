"""Tests for rl/vote_ref_detector.py — the actor-content vote-reference detector.

Strategy: heavy on **false-positive guards** (the failure mode we care
most about — incorrectly flagging "+2 multiplier" as a vote ref would
re-introduce noise in the training data form), with a thinner pass
over true positives to confirm coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl.vote_ref_detector import (
    detect_vote_refs,
    session_uses_vote_language,
    collect_referenced_post_ids,
    VoteRef,
    VoteRefResult,
)


# ── True positives ────────────────────────────────────────


def test_l1_score_keyword_with_post_and_value():
    r = detect_vote_refs("I see the score of P_3 is +2, let me build on it.")
    assert r.has_ref
    assert any(x.layer == "L1" and x.target_post_id == 3 and x.value == 2 for x in r.refs)


def test_l1_scored_with_post():
    r = detect_vote_refs("P_5 scored -1 last round, suggesting an error.")
    assert r.has_ref
    assert any(x.layer == "L1" and x.target_post_id == 5 and x.value == -1 for x in r.refs)


def test_l1_unicode_minus():
    # \u2212 is the typographic minus often inserted by LaTeX-aware writers.
    r = detect_vote_refs("P_2 was scored \u22122 by the panel.")
    assert r.has_ref
    assert any(x.layer == "L1" and x.value == -2 for x in r.refs)


def test_l1_value_in_window_but_no_match_required_value():
    # "score of P_3" with no nearby ±[1-3] still triggers as a ref —
    # value is None but target P_id is captured.
    r = detect_vote_refs("Looking at the score of P_3, I'd say the trend is concerning.")
    assert r.has_ref
    assert any(x.layer == "L1" and x.target_post_id == 3 and x.value is None for x in r.refs)


def test_l2_voted_with_value_no_post():
    r = detect_vote_refs("I voted +1 because the derivation looked sound.")
    assert r.has_ref
    assert any(x.layer == "L2" and x.value == 1 for x in r.refs)


def test_l2_cast_minus():
    r = detect_vote_refs("After consideration, I downvoted -2 on the last move.")
    # 'downvoted' is a vote keyword; -2 is in tail → L2
    assert r.has_ref
    assert any(x.layer == "L2" and x.value == -2 for x in r.refs)


def test_l3_anonymous_scientist():
    r = detect_vote_refs("Looks like Anonymous Scientist disagrees with this approach.")
    assert r.has_ref
    assert any(x.layer == "L3" for x in r.refs)


def test_l3_senior_reviewer():
    r = detect_vote_refs("The senior reviewer flagged this — let me reconsider.")
    assert r.has_ref
    assert any(x.layer == "L3" for x in r.refs)


def test_l3_hidden_reviewer():
    r = detect_vote_refs("A hidden reviewer's verdict suggests we drop this branch.")
    assert r.has_ref
    assert any(x.layer == "L3" for x in r.refs)


def test_l3_anonymous_reviewer_variant():
    r = detect_vote_refs("Anonymous reviewer rated this -3, that's a strong warning.")
    assert r.has_ref
    # Both L3 (anonymous reviewer) and L1/L2 (scored/voted+value) might fire.
    assert any(x.layer == "L3" for x in r.refs)


def test_l4_high_scoring_qualifier():
    r = detect_vote_refs("The high-scoring P_3 confirms the symmetry argument.")
    assert r.has_ref
    assert any(x.layer == "L4" and x.target_post_id == 3 for x in r.refs)


def test_l4_negative_scoring():
    r = detect_vote_refs("Avoid the negative-scoring approach from earlier.")
    assert r.has_ref
    assert any(x.layer == "L4" for x in r.refs)


def test_multiple_refs_in_one_content():
    text = ("First I notice the score of P_3 is +2, then Anonymous Scientist "
            "voted -1 on the high-scoring P_5.")
    r = detect_vote_refs(text)
    assert r.has_ref
    layers = {x.layer for x in r.refs}
    assert "L1" in layers       # score of P_3 is +2
    assert "L3" in layers       # Anonymous Scientist
    assert "L4" in layers       # high-scoring P_5


# ── False-positive guards (most important) ───────────────


def test_no_match_pure_math_with_signed_int():
    # "+2 multiplier" must NOT trigger — no score/vote keyword nearby.
    r = detect_vote_refs("Setting x = +2 in the equation, we get 4y - x = 0.")
    assert not r.has_ref


def test_no_match_anonymous_function():
    # "anonymous function" must NOT trigger — L3 requires scientist/reviewer/etc. after.
    r = detect_vote_refs("We define f as an anonymous function: f = lambda x: x**2.")
    assert not r.has_ref


def test_no_match_vote_keyword_no_value_no_post():
    # "I'll vote A" — vote keyword but no signed int + no P_id in proximity.
    r = detect_vote_refs("If forced, I'll vote on this issue once everyone is heard.")
    assert not r.has_ref


def test_no_match_score_keyword_no_post_in_window():
    # "scored a major win" — score keyword present but no P_id within proximity.
    r = detect_vote_refs("Our derivation scored a major win in simplifying the integrand.")
    assert not r.has_ref


def test_no_match_consensus_alone():
    # Old SCORE_META_RE would falsely fire on "consensus" — new detector must NOT.
    r = detect_vote_refs("We need consensus on the choice of coordinate system.")
    assert not r.has_ref


def test_no_match_pure_reasoning():
    r = detect_vote_refs("First, observe the symmetry between u and v under reflection.")
    assert not r.has_ref


def test_no_match_arithmetic_signed_far_from_keyword():
    # Score keyword and signed int both present but >> proximity_chars apart.
    text = ("Initial scoring rules of the contest give 1 point per correct step. "
            "Let's now compute the integral: substitute u = -2 sin(theta).")
    r = detect_vote_refs(text)
    # 'scoring' is far (>>80 chars) from '-2', no P_id in window of 'scoring' → no ref
    refs_l1 = [x for x in r.refs if x.layer == "L1"]
    refs_l2 = [x for x in r.refs if x.layer == "L2"]
    assert refs_l1 == [] and refs_l2 == []


def test_no_match_high_score_no_qualifier_form():
    # "the score is high" — different word order, our L4 only matches
    # the specific qualifier forms (high-scoring, etc.).
    r = detect_vote_refs("The score is high, but I'm skeptical.")
    # 'score' is a noun with no P_id nearby → no L1; no qualifier form → no L4.
    assert not any(x.layer in ("L1", "L4") for x in r.refs)


def test_no_match_p_id_alone():
    r = detect_vote_refs("I'll build on P_3 next round.")
    assert not r.has_ref


def test_no_match_signed_int_alone():
    r = detect_vote_refs("The eigenvalue is +2.")
    assert not r.has_ref


def test_no_match_vote_in_compound_word():
    # 'voter' / 'voting' keywords inside unrelated context.
    # 'voter turnout' should still match because 'voter' contains 'voted'? Actually
    # our regex uses (?:up|down)?vot(?:e|es|ed|ing) — 'voter' has 'vot' but not
    # the listed suffixes (e/es/ed/ing). So 'voter' doesn't match. Good.
    r = detect_vote_refs("In the model, each voter selects a candidate.")
    assert not r.has_ref


def test_no_match_compound_anonymous():
    # 'Anonymous tip' / 'anonymously' / 'anonymous donations' should not trigger.
    for phrase in ("an anonymous tip arrived",
                   "the donor wished to remain anonymous",
                   "she submitted anonymously"):
        r = detect_vote_refs(phrase)
        assert not r.has_ref, f"falsely flagged: {phrase!r}"


# ── Edge cases ───────────────────────────────────────────


def test_empty_string():
    r = detect_vote_refs("")
    assert not r.has_ref
    assert r.refs == []
    assert r.by_layer == {}


def test_very_long_text_with_far_apart_signals():
    # Score keyword at start, P_id at end (far apart) — no L1.
    text = "scored well. " + "padding " * 50 + " P_3 is interesting."
    r = detect_vote_refs(text)
    # Distance between 'scored' (pos ~6) and 'P_3' (pos ~360) >> 80.
    assert not any(x.layer == "L1" for x in r.refs)


def test_by_layer_count():
    text = ("score of P_1 is +1; score of P_2 is +2; "
            "Anonymous Scientist agreed; high-scoring P_3.")
    r = detect_vote_refs(text)
    assert r.by_layer.get("L1", 0) >= 2
    assert r.by_layer.get("L3", 0) >= 1
    assert r.by_layer.get("L4", 0) >= 1


def test_enable_layers_subset():
    # L4-only case: a "high-scoring" qualifier with NO nearby P_id —
    # L1 wouldn't fire (no P_id in keyword's window); only L4 catches it.
    text = "Let's avoid the high-scoring approach mentioned earlier."
    # Default enables L4 → fires.
    r_full = detect_vote_refs(text)
    assert r_full.has_ref
    assert any(x.layer == "L4" for x in r_full.refs)
    # Disable L4 → nothing fires.
    r_no_l4 = detect_vote_refs(text, enable_layers=("L1", "L2", "L3"))
    assert not r_no_l4.has_ref


def test_proximity_window_tight():
    text = "score of the matter is, once we reach P_3, we should reconsider."
    # 'score' to 'P_3': ~30 chars — within default 80 → should fire L1.
    r = detect_vote_refs(text)
    assert any(x.layer == "L1" for x in r.refs)
    # With proximity_chars=10, 'score' and 'P_3' are too far.
    r2 = detect_vote_refs(text, proximity_chars=10)
    assert not any(x.layer == "L1" for x in r2.refs)


# ── High-level wrappers ─────────────────────────────────


def test_session_uses_vote_language_yes():
    contents = [
        "First derivation: x = +2.",  # no vote ref
        "I noted the score of P_3 is +2.",  # L1
    ]
    assert session_uses_vote_language(contents)


def test_session_uses_vote_language_no():
    contents = [
        "We start with the symmetry argument.",
        "Setting x = +2 we get the result.",
        "P_3 will follow up.",
    ]
    assert not session_uses_vote_language(contents)


def test_collect_referenced_post_ids():
    contents = [
        "Score of P_3 is +1; also score of P_7 is -2.",
        "Anonymous Scientist gave a -3 (no P_id mentioned in this clause).",
        "The high-scoring P_12 settles it.",
    ]
    pids = collect_referenced_post_ids(contents)
    assert pids == {3, 7, 12}


def test_collect_referenced_post_ids_only_l1_default():
    # L3 (Anonymous Scientist) carries no target_post_id by design;
    # L2 (vote action without P_id) ditto.  Default enable=L1+L4.
    contents = ["Anonymous Scientist voted -2."]
    assert collect_referenced_post_ids(contents) == set()
