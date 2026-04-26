"""Vote-reference detector for actor-authored content.

Used by ``rl/extract.py`` to decide whether vote events must be inserted
into the training context for a given actor's view.  The Parliament
training pipeline rebuilds messages independently of the rollout PTY
sequence, so we have to maintain one invariant ourselves:

    user-side input must cover assistant-side action's references.

If an actor's post says "I see +2 on P_3" but the user message before
that turn carries no vote event for P_3, we have a *dangling reference*
("hallucinated" training data).  This detector tells the renderer
whether actor content has any such reference, so the renderer can
include just-enough vote events to ground them — without indiscriminately
inserting every vote (which would add redundant noise to the
training-data form).

Design principle: **high precision over recall**.  Two failure modes:

* **False positive** (insert vote events when actor doesn't really
  reference any): wastes context tokens; mild damage.
* **False negative** (miss a real reference): leaves a dangling
  reference; same kind of "hallucination" bug we are fixing.

Generally we err toward false negative — actor's incidental "+2
multiplier" or "anonymous function" should NOT trigger insertion, so
the lexicon is tight.

Detection layers (high → low confidence):

* **L1 — score-keyword + post-id proximity**.  A score/vote keyword
  (``score``, ``scored``, ``vote``, ``voted``, ``upvote``, …) and a
  ``P_<n>`` reference within ±N chars window.  Optionally captures
  the signed value if a ``±[1-3]`` is in the same window.
* **L2 — explicit vote action with a value**.  A vote keyword
  immediately followed by a signed integer (``voted +2``, ``cast a
  -3 vote``).  Doesn't need a P_id (vote action can omit the target).
* **L3 — anonymous voter identity reference**.  Lexicon match for
  ``Anonymous Scientist``, ``senior reviewer``, ``hidden reviewer``,
  etc.  These cover all variants in
  ``rl/extract.py:TEMPLATE_POOL["anonymous_voter"]``.
* **L4 — implicit score qualifier**.  ``high-scoring P_3`` /
  ``negative-scoring`` / etc.  Lower confidence than L1.

Output is a :class:`VoteRefResult` with the per-layer breakdown so
callers can decide which layers count.

Public surface:

* :func:`detect_vote_refs(content)` — single-string detector.
* :func:`session_uses_vote_language(actor_contents)` — drop-in
  replacement for ``extract.py``'s old ``SCORE_META_RE``-based check,
  strictly more precise.
* :func:`collect_referenced_post_ids(actor_contents)` — set of
  ``P_<n>`` ids actor content directly references with a vote/score
  keyword, useful for "insert just the votes that ground the refs".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# ── Lexicon ──────────────────────────────────────────────

# Score / vote action verbs and nouns. Word-boundary anchored.
# Covers: score, scored, scoring, vote, voted, votes, voting,
# upvote(d), downvote(d), upvoting, downvoting.
_SCORE_KW_RE = re.compile(
    r"(?ix)\b("
    r"  scor(?:e|es|ed|ing)"
    r"| (?:up|down)?vot(?:e|es|ed|ing)"
    r")\b"
)

# Signed integer in judge / actor range: ±1, ±2, ±3.  Disallow
# embedded-word matches ("part1") via lookbehind on \w/dot.
# Accept ASCII '-' and Unicode '−' (U+2212) as minus.
_SIGNED_INT_RE = re.compile(r"(?<![\w.])([+\-−])\s*([1-3])\b")

# Local post id reference: P_3, P_12 (digits only, word-boundary).
_POST_REF_RE = re.compile(r"\bP_(\d+)\b")

# Anonymous-voter identity reference. Crucially we require the
# qualifier word AFTER "anonymous" to discriminate from "anonymous
# function" / "anonymous tip" / etc.
_ANON_VOTER_RE = re.compile(
    r"(?ix)\b("
    r"  anonymous\s+(?:scientist|reviewer|senior|contributor|evaluator|voter|vote)"
    r"| senior\s+reviewer"
    r"| senior\s*\(\s*anonymous\s*\)"
    r"| hidden\s+reviewer"
    r"| unnamed\s+senior"
    r"| an?\s+anonymous\s+senior"
    r")\b"
)

# Implicit score qualifier — lower-confidence L4.
_SCORE_QUALIFIER_RE = re.compile(
    r"(?ix)\b("
    r"  high[\-\s]?scoring"
    r"| low[\-\s]?scoring"
    r"| negative[\-\s]?scoring"
    r"| positive[\-\s]?scoring"
    r"| top[\-\s]?scor(?:ed|ing)"
    r")\b"
)


# ── Result types ─────────────────────────────────────────


@dataclass
class VoteRef:
    """A single detected reference to vote/score info."""

    layer: str                            # 'L1' / 'L2' / 'L3' / 'L4'
    span: tuple[int, int]                 # match span in original text
    snippet: str                          # short context window
    target_post_id: int | None = None     # captured local P_id, if any
    value: int | None = None              # captured signed value, if any


@dataclass
class VoteRefResult:
    """Aggregated detector output for one piece of content."""

    has_ref: bool
    refs: list[VoteRef] = field(default_factory=list)
    by_layer: dict[str, int] = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────


def _snippet(text: str, lo: int, hi: int, pad: int = 20) -> str:
    """Return a short context window around ``text[lo:hi]``."""
    s = max(0, lo - pad)
    e = min(len(text), hi + pad)
    return text[s:e].replace("\n", " ").strip()


def _signed_int_in_window(text: str) -> int | None:
    """First ±[1-3] in ``text``, returned as signed int (or None)."""
    m = _SIGNED_INT_RE.search(text)
    if not m:
        return None
    sign = -1 if m.group(1) in "-−" else +1
    return sign * int(m.group(2))


# ── Core detector ───────────────────────────────────────


def detect_vote_refs(
    content: str,
    *,
    proximity_chars: int = 80,
    enable_layers: tuple[str, ...] = ("L1", "L2", "L3", "L4"),
) -> VoteRefResult:
    """Run all enabled layers over ``content``.

    ``proximity_chars`` controls the window radius for L1
    (score-keyword ↔ P_id proximity).  Default 80 chars (~12-15 words)
    is a comfortable single-clause window — tight enough that
    "I scored a major win earlier; later let's discuss P_3" doesn't
    falsely trigger, loose enough that "the score of P_3 is +2" does.
    """
    if not content:
        return VoteRefResult(has_ref=False)
    refs: list[VoteRef] = []

    # ── L1: score keyword + post-id proximity ──
    # For each keyword, prefer the *nearest P_id on the right* (matches
    # natural "score of P_3 is +2" word order); fall back to the
    # nearest P_id on the left within the window (matches reversed
    # "P_3 was scored +2" word order).  This keeps each keyword
    # associated with its own P_id, so two consecutive sentences
    # ("score of P_3 is +1; score of P_7 is -2") yield two refs with
    # the right targets.
    if "L1" in enable_layers:
        for kw_match in _SCORE_KW_RE.finditer(content):
            # Find target P_id: prefer right side (natural "score of P_3"
            # word order), fall back to left side ("P_3 was scored").
            right_lo = kw_match.end()
            right_hi = min(len(content), right_lo + proximity_chars)
            right_window = content[right_lo:right_hi]
            p_match = _POST_REF_RE.search(right_window)
            target: int | None
            if p_match:
                target = int(p_match.group(1))
            else:
                left_hi = kw_match.start()
                left_lo = max(0, left_hi - proximity_chars)
                left_window = content[left_lo:left_hi]
                # Right-most (= closest to keyword) match on the left.
                p_iter = list(_POST_REF_RE.finditer(left_window))
                if not p_iter:
                    continue
                target = int(p_iter[-1].group(1))
            # Value can be on either side of the keyword — scan the
            # full ±proximity window.  Examples:
            #   "P_3 was scored -1"  → P_id left, value right
            #   "score of P_3 is +2" → both right
            full_lo = max(0, kw_match.start() - proximity_chars)
            full_hi = min(len(content), kw_match.end() + proximity_chars)
            value = _signed_int_in_window(content[full_lo:full_hi])
            refs.append(VoteRef(
                layer="L1",
                span=(kw_match.start(), kw_match.end()),
                snippet=_snippet(content, kw_match.start(), kw_match.end()),
                target_post_id=target,
                value=value,
            ))

    # ── L2: explicit vote-action verb + signed value ──
    # Matches "voted +2" / "cast -1" / "scored a +3" without requiring
    # a P_id (vote actions sometimes elide the target in casual text).
    # Must NOT double-fire on text already covered by L1 — track spans.
    if "L2" in enable_layers:
        l1_spans = {(r.span[0], r.span[1]) for r in refs if r.layer == "L1"}
        for kw_match in _SCORE_KW_RE.finditer(content):
            if (kw_match.start(), kw_match.end()) in l1_spans:
                continue
            tail = content[kw_match.end(): kw_match.end() + 30]
            v_match = _SIGNED_INT_RE.search(tail)
            if v_match:
                sign = -1 if v_match.group(1) in "-−" else +1
                value = sign * int(v_match.group(2))
                refs.append(VoteRef(
                    layer="L2",
                    span=(kw_match.start(),
                          kw_match.end() + v_match.end()),
                    snippet=_snippet(content, kw_match.start(),
                                     kw_match.end() + v_match.end()),
                    value=value,
                ))

    # ── L3: anonymous voter identity reference ──
    if "L3" in enable_layers:
        for am in _ANON_VOTER_RE.finditer(content):
            refs.append(VoteRef(
                layer="L3",
                span=(am.start(), am.end()),
                snippet=_snippet(content, am.start(), am.end()),
            ))

    # ── L4: implicit score qualifier ──
    if "L4" in enable_layers:
        for qm in _SCORE_QUALIFIER_RE.finditer(content):
            # Try to attach a P_id if it's nearby.
            tail = content[qm.end(): qm.end() + 50]
            head = content[max(0, qm.start() - 20): qm.start()]
            target: int | None = None
            for p_match in _POST_REF_RE.finditer(tail):
                target = int(p_match.group(1))
                break
            if target is None:
                for p_match in _POST_REF_RE.finditer(head):
                    target = int(p_match.group(1))
                    break
            refs.append(VoteRef(
                layer="L4",
                span=(qm.start(), qm.end()),
                snippet=_snippet(content, qm.start(), qm.end()),
                target_post_id=target,
            ))

    by_layer: dict[str, int] = {}
    for r in refs:
        by_layer[r.layer] = by_layer.get(r.layer, 0) + 1
    return VoteRefResult(has_ref=bool(refs), refs=refs, by_layer=by_layer)


# ── Convenience wrappers ────────────────────────────────


def session_uses_vote_language(
    actor_contents: Iterable[str],
    *,
    enable_layers: tuple[str, ...] = ("L1", "L2", "L3", "L4"),
) -> bool:
    """Strictly more precise drop-in replacement for the legacy
    ``SCORE_META_RE``-based check used by ``rl/extract.py``.

    Returns True iff *any* actor-authored content in ``actor_contents``
    contains at least one vote/score reference.
    """
    return any(detect_vote_refs(c, enable_layers=enable_layers).has_ref
               for c in actor_contents)


def collect_referenced_post_ids(
    actor_contents: Iterable[str],
    *,
    enable_layers: tuple[str, ...] = ("L1", "L4"),
) -> set[int]:
    """Set of all local ``P_<n>`` ids that actor content directly ties
    to a vote/score keyword (L1) or score qualifier (L4).

    Default excludes L2 (no P_id captured) and L3 (anonymous identity
    has no P_id by design).  Useful for "insert only the vote events
    that ground these refs" mode.
    """
    out: set[int] = set()
    for c in actor_contents:
        for ref in detect_vote_refs(c, enable_layers=enable_layers).refs:
            if ref.target_post_id is not None:
                out.add(ref.target_post_id)
    return out
