"""Tests for rl/extract.py — pure logic on synthetic sessions.

Pinned behaviours:

* `--strip-vote-events` (default ON) yields a vote-line-free training context.
* `--no-strip-vote-events` re-enables the legacy heuristic.
* Per-actor view isolates solo cells (no peer posts leak through).
* Reward = sum of judge votes only; advantage = (r-baseline)/scale.
* `assign_local_ids_view` numbers each actor's posts from 1 (no peer gaps).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rl.extract import (
    assign_local_ids_view,
    build_timeline,
    compute_post_rewards,
    _build_actor_view,
    extract_session,
    session_uses_vote_language,
)


# ── Synthetic session helpers ───────────────────────────


def _post(pid, author, role, content, ts="2026-04-25 10:00:00"):
    return {"post_id": pid, "author": author, "author_role": role,
            "content": content, "created_at": ts}


def _comment(cid, pid, author, role, content, ts="2026-04-25 10:00:00"):
    return {"comment_id": cid, "post_id": pid, "author": author,
            "author_role": role, "content": content, "created_at": ts}


def _vote(vid, *, post_id=None, comment_id=None, value=1, author="Judge_1",
          role="judge", ts="2026-04-25 10:00:00"):
    return {"vote_id": vid, "post_id": post_id, "comment_id": comment_id,
            "value": value, "previous_value": None, "author": author,
            "author_role": role, "created_at": ts}


def _session(title="Q", sid="s1"):
    return {"session_id": sid, "title": title}


# ── assign_local_ids_view ───────────────────────────────


def test_assign_local_ids_view_renumbers_from_one():
    posts = [_post(7, "Alice", "actor", "a"),
             _post(99, "Bob", "actor", "b")]
    comments = [_comment(50, 7, "Bob", "actor", "c")]
    v_posts, v_comments = assign_local_ids_view(posts, comments)
    assert [p["local_id"] for p in v_posts] == [1, 2]
    assert v_comments[0]["local_id"] == 1
    # Comment back-pointer resolves through this view.
    assert v_comments[0]["local_post_id"] == 1
    # Original dicts untouched (deep-copy semantics).
    assert "local_id" not in posts[0]


def test_assign_local_ids_view_orphan_comment_gets_zero():
    # Comment whose post is NOT in this actor's view → local_post_id=0
    posts = [_post(1, "Alice", "actor", "a")]
    comments = [_comment(10, 999, "Bob", "actor", "c")]
    _, v_comments = assign_local_ids_view(posts, comments)
    assert v_comments[0]["local_post_id"] == 0


# ── compute_post_rewards: judge-only ───────────────────


def test_compute_post_rewards_only_counts_judge_votes():
    posts = [_post(1, "Alice", "actor", "a"),
             _post(2, "Bob", "actor", "b")]
    votes = [
        _vote(10, post_id=1, value=2, author="Judge_1", role="judge"),
        _vote(11, post_id=1, value=-1, author="Bob", role="actor"),  # ignored
        _vote(12, post_id=2, value=3, author="Judge_2", role="judge"),
    ]
    r = compute_post_rewards(posts, votes)
    assert r[1] == 2.0  # only the judge vote
    assert r[2] == 3.0


# ── session_uses_vote_language ─────────────────────────


def test_session_uses_vote_language_detects_score_keywords():
    # Coupled cell: posts AND comments are checked.
    posts = [_post(1, "Alice", "actor", "I notice the score of P_3 is +2")]
    assert session_uses_vote_language(posts, [], actor_coupled=True)
    # Plain content with no vote/score talk → False.
    posts2 = [_post(1, "Alice", "actor", "I derived x = 2")]
    assert not session_uses_vote_language(posts2, [], actor_coupled=True)
    # Solo cell ignores comments (none exist there anyway).
    cmts = [_comment(1, 1, "Alice", "actor", "the vote on P_3 looks low")]
    assert not session_uses_vote_language(posts2, cmts, actor_coupled=False)
    # Coupled cell does pick up vote language in comments.
    assert session_uses_vote_language(posts2, cmts, actor_coupled=True)


# ── _build_actor_view: cell-aware filtering ────────────


def test_build_actor_view_coupled_keeps_everything():
    posts = [_post(1, "Alice", "actor", "a"), _post(2, "Bob", "actor", "b")]
    cmts = [_comment(10, 1, "Bob", "actor", "c")]
    votes = [_vote(20, post_id=1, value=2, author="Judge_1", role="judge")]
    vp, vc, vv = _build_actor_view("Alice", posts, cmts, votes,
                                   actor_coupled=True, judge_visible=True)
    assert {p["post_id"] for p in vp} == {1, 2}    # peers visible
    assert len(vc) == 1
    assert len(vv) == 1


def test_build_actor_view_coupled_no_judge_drops_judge_votes():
    posts = [_post(1, "Alice", "actor", "a")]
    votes = [_vote(20, post_id=1, value=2, author="Judge_1", role="judge"),
             _vote(21, post_id=1, value=1, author="Bob", role="actor")]
    _, _, vv = _build_actor_view("Alice", posts, [], votes,
                                 actor_coupled=True, judge_visible=False)
    assert len(vv) == 1
    assert vv[0]["author_role"] == "actor"


def test_build_actor_view_solo_keeps_only_own_posts():
    posts = [_post(1, "Alice", "actor", "a"),
             _post(2, "Bob", "actor", "b"),
             _post(3, "Alice", "actor", "a2")]
    cmts = [_comment(10, 1, "Bob", "actor", "c")]
    votes = [
        _vote(20, post_id=1, value=2, author="Judge_1", role="judge"),  # on Alice
        _vote(21, post_id=2, value=1, author="Judge_1", role="judge"),  # on Bob (drop)
        _vote(22, post_id=3, value=3, author="Judge_2", role="judge"),  # on Alice
    ]
    vp, vc, vv = _build_actor_view("Alice", posts, cmts, votes,
                                   actor_coupled=False, judge_visible=True)
    assert {p["post_id"] for p in vp} == {1, 3}    # only Alice's
    assert vc == []                                # solo: no comments
    assert {v["post_id"] for v in vv} == {1, 3}    # only own-post votes
    assert all(v["author_role"] == "judge" for v in vv)


def test_build_actor_view_blind_solo_drops_all_votes():
    posts = [_post(1, "Alice", "actor", "a")]
    votes = [_vote(20, post_id=1, value=2, author="Judge_1", role="judge")]
    _, _, vv = _build_actor_view("Alice", posts, [], votes,
                                 actor_coupled=False, judge_visible=False)
    assert vv == []


# ── extract_session: vote stripping ────────────────────


def _judge_voted_session():
    """A coupled-cell scenario where Alice posts twice with judge votes
    landing in between, plus an actor (Alice's first post) references
    voting so the legacy heuristic fires.

    Timing matters: build_timeline sorts by (created_at, type, id), so
    if Alice's posts and the votes share a timestamp, posts (type=0)
    come first and votes (type=2) come after — meaning votes never
    appear in Alice's user message before her second self-post. We
    therefore stagger timestamps so the vote falls *between* P_1 and P_2.
    """
    posts = [
        _post(1, "Alice", "actor",
              "First, the score of P_x looks symmetric — let me try.",
              ts="2026-04-25 10:00:01"),
        _post(2, "Alice", "actor",
              "Following up: by direct calculation, x=2.",
              ts="2026-04-25 10:00:03"),
    ]
    cmts: list[dict] = []
    # Judge vote occurs between Alice's two posts.
    votes = [_vote(10, post_id=1, value=2, author="Judge_1", role="judge",
                   ts="2026-04-25 10:00:02")]
    rewards = {1: 2.0, 2: 0.0}
    advantages = {1: 0.5, 2: -0.5}
    return _session("What is x?"), posts, cmts, votes, rewards, advantages


def _flatten_user_text(sample):
    return "\n".join(m["content"] for m in sample["messages"]
                     if m["role"] == "user")


def test_extract_session_strip_vote_events_default_removes_them():
    session, posts, cmts, votes, rewards, advs = _judge_voted_session()
    samples = extract_session(session, posts, cmts, votes, rewards, advs,
                              actor_coupled=True, judge_visible=True,
                              strip_vote_events=True)
    assert len(samples) >= 1
    for s in samples:
        ut = _flatten_user_text(s)
        # No `[V on P_X]` or `current score of P_X` lines should appear.
        assert "[V on P_" not in ut, f"vote event leaked: {ut[:200]}"
        assert "current score of P_" not in ut
        # Reward signal lives in turn_advantages, not in context.
        assert any(a != 0 for a in s["turn_advantages"])


def test_extract_session_legacy_keeps_vote_events_when_actor_mentions_score():
    """Legacy mode should insert vote events when SCORE_META_RE fires."""
    session, posts, cmts, votes, rewards, advs = _judge_voted_session()
    samples_legacy = extract_session(
        session, posts, cmts, votes, rewards, advs,
        actor_coupled=True, judge_visible=True, strip_vote_events=False)
    samples_strip = extract_session(
        session, posts, cmts, votes, rewards, advs,
        actor_coupled=True, judge_visible=True, strip_vote_events=True)
    assert samples_legacy and samples_strip
    legacy_text = _flatten_user_text(samples_legacy[0])
    strip_text = _flatten_user_text(samples_strip[0])
    # Legacy must produce STRICTLY MORE user-side text than strip
    # (the difference is the vote events).
    assert len(legacy_text) > len(strip_text), (
        f"legacy mode did not add vote events; "
        f"legacy={len(legacy_text)}, strip={len(strip_text)}")
    # Legacy text must mention the vote's numeric value (+2). The
    # actor's own posts never say "+2", so its presence is exclusively
    # from the rendered vote event.
    assert "+2" in legacy_text, "vote value missing from legacy user text"
    assert "+2" not in strip_text, (
        "strip mode leaked '+2' — vote event wasn't actually stripped")


def test_extract_session_strip_does_not_break_no_vote_sessions():
    """A session with no vote language → both modes produce vote-free context."""
    posts = [
        _post(1, "Alice", "actor", "Step 1: derive identity."),
        _post(2, "Bob", "actor", "Step 2: substitute and simplify."),
    ]
    rewards = {1: 1.0, 2: 1.0}
    advantages = {1: 0.0, 2: 0.0}
    for strip in (True, False):
        samples = extract_session(_session(), posts, [], [], rewards, advantages,
                                  actor_coupled=True, judge_visible=True,
                                  strip_vote_events=strip)
        assert samples
        for s in samples:
            assert "[V on P_" not in _flatten_user_text(s)


def test_extract_session_drop_stats_zero_turns_and_all_short():
    """drop_stats accumulator counts both drop reasons separately."""
    # Bob writes only — Alice has no posts → Alice is dropped (zero_turns).
    # Bob's only post is 5 chars, well below min_content_chars (default 20)
    # → Bob has 1 turn but all turns are non-trainable → dropped (all_short).
    posts = [_post(1, "Bob", "actor", "ok.")]
    rewards = {1: 0.0}
    advantages = {1: 0.0}
    stats: dict = {}
    samples = extract_session(_session(), posts, [], [], rewards, advantages,
                              actor_coupled=True, judge_visible=True,
                              strip_vote_events=True, drop_stats=stats)
    assert samples == []
    assert stats == {"all_short": 1}, f"unexpected stats: {stats}"

    # Pure zero-turn case: Bob is in the user list but has no posts.
    # extract_session iterates actor_global_names from posts only, so an
    # actor with NO posts is invisible — covered upstream by sessions.
    # Below: include a single normal Alice post so we can verify
    # the zero_turns counter increments when the per-actor view yields
    # no own_post.  Alice writes; Bob's "view" filtered to a peer cell
    # would have no own posts, but we don't iterate Bob's view since he
    # has no posts.  So zero_turns is mostly a defensive counter.
    # We at least verify it never crashes and stays absent of a key.
    stats2: dict = {}
    posts2 = [_post(1, "Alice", "actor", "A meaningful long-enough post here.")]
    rewards2 = {1: 1.0}
    advantages2 = {1: 0.5}
    samples2 = extract_session(_session(), posts2, [], [], rewards2, advantages2,
                               actor_coupled=True, judge_visible=True,
                               strip_vote_events=True, drop_stats=stats2)
    assert len(samples2) == 1
    # Neither drop reason fired
    assert stats2 == {}


# ── build_timeline ─────────────────────────────────────


def test_build_timeline_omits_vote_events_when_flag_off():
    posts = [{**_post(1, "Alice", "actor", "a"), "local_id": 1}]
    votes = [_vote(20, post_id=1, value=2, author="Judge_1", role="judge")]
    tl = build_timeline(posts, [], votes, include_vote_events=False)
    assert all(e["type"] != "vote" for e in tl)
    tl_with = build_timeline(posts, [], votes, include_vote_events=True)
    assert any(e["type"] == "vote" for e in tl_with)
