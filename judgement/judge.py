"""
Science Parliament — Judge module.

After the parliament finishes discussing, the Judge reads the entire forum
record and synthesizes a final answer.

Usage (standalone):
    python judge.py <db_path> <question> [--choices "A|B|C|D"]

Programmatic:
    from judge import run_judge
    result = await run_judge(db_path, question, model)
    print(result["answer"])
"""

import asyncio
import json
import os
import re
import sqlite3
import sys

# ---------------------------------------------------------------------------
# Judge system prompts
# ---------------------------------------------------------------------------

_JUDGE_COMMON = """\
You are a senior scientist who has been silently observing a Science \
Parliament — a forum where a group of scientists collaborated over \
multiple rounds to solve a difficult problem.

You did not participate in the discussion. You have no stake in any \
particular answer. Your job is to read the entire record of the \
parliament's work and produce the definitive final answer.

The forum posts below are presented with their community scores \
(endorsements minus challenges). Higher-scored posts were considered \
more rigorous by the participating scientists. However, the majority \
is not always right — use your own judgment.

Each post and comment is dated (e.g. [2026-03-17], [2026-03-18]). \
Later dates often correct or refine earlier work. Pay special \
attention to later comments on early high-scored posts — they may \
contain important corrections.

Instructions:
- Read all posts and comments carefully.
- Identify the strongest line of reasoning in the discussion.
- If there are unresolved disagreements, evaluate both sides and \
  pick the one with better evidence.
- If critical errors were identified in highly-scored posts, account \
  for those corrections.
- Pay attention to the temporal flow: later work may supersede earlier claims.
- Synthesize a single, definitive answer."""

JUDGE_SYSTEM_PROMPT = _JUDGE_COMMON + """

You MUST end your response with this exact block:

<<<FINAL>>>
your final answer here
<<<END>>>

The answer between <<<FINAL>>> and <<<END>>> must be self-contained — \
someone reading only that block should get the complete answer.\
"""

JUDGE_SYSTEM_PROMPT_CHOICES = _JUDGE_COMMON + """

Select the single best answer from the provided choices.

You MUST end your response with this exact block:

<<<FINAL>>>
(X)
<<<END>>>

where X is the letter of the correct choice (A, B, C, or D). \
Nothing else should appear between the markers.\
"""


# ---------------------------------------------------------------------------
# Build the forum context for the Judge
# ---------------------------------------------------------------------------

_PARLIAMENT_START_DATE = "2026-03-17"

# Sort modes for judge context:
#   "time"  — chronological order (post_id ascending)
#   "score" — highest score first
JUDGE_SORT_MODE = "time"


def _load_round_map(db_path: str) -> list[dict] | None:
    rm_path = os.path.join(os.path.dirname(db_path), "round_map.json")
    if not os.path.exists(rm_path):
        return None
    try:
        with open(rm_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _id_to_date(item_id: int, boundaries: list[dict], key: str) -> str:
    """Map a post_id or comment_id to a date string via round_map."""
    from datetime import date, timedelta
    start = date.fromisoformat(_PARLIAMENT_START_DATE)
    for entry in boundaries:
        if item_id <= entry[key]:
            return str(start + timedelta(days=entry["round"] - 1))
    if boundaries:
        return str(start + timedelta(days=boundaries[-1]["round"] - 1))
    return _PARLIAMENT_START_DATE


def build_judge_context(
    db_path: str,
    question: str,
    choices: list[str] | None = None,
    sort_mode: str | None = None,
) -> str:
    """Read parliament.db and format the discussion for the Judge.

    Args:
        sort_mode: "time" (chronological) or "score" (highest score first).
                   Defaults to JUDGE_SORT_MODE.
    """
    if sort_mode is None:
        sort_mode = JUDGE_SORT_MODE

    conn = sqlite3.connect(db_path, timeout=3)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    order_clause = "p.post_id ASC" if sort_mode == "time" else "score DESC, p.post_id ASC"
    c.execute(f"""
        SELECT p.post_id, u.name AS author, p.content,
               (p.num_likes - p.num_dislikes) AS score
        FROM post p JOIN user u ON p.user_id = u.user_id
        ORDER BY {order_clause}
    """)
    posts = [dict(r) for r in c.fetchall()]

    cm_order = "cm.comment_id ASC" if sort_mode == "time" else "(cm.num_likes - cm.num_dislikes) DESC, cm.comment_id ASC"
    c.execute(f"""
        SELECT cm.comment_id, cm.post_id, u.name AS author, cm.content,
               (cm.num_likes - cm.num_dislikes) AS score
        FROM comment cm JOIN user u ON cm.user_id = u.user_id
        ORDER BY cm.post_id, {cm_order}
    """)
    comments_by_post: dict[int, list[dict]] = {}
    for r in c.fetchall():
        d = dict(r)
        comments_by_post.setdefault(d["post_id"], []).append(d)
    conn.close()

    round_map = _load_round_map(db_path)

    parts = ["THE PROBLEM:", question]

    if choices:
        parts.append("\nCHOICES:")
        for i, ch in enumerate(choices):
            parts.append(f"  ({chr(ord('A') + i)}) {ch}")

    sort_desc = "chronological order" if sort_mode == "time" else "community score, highest first"
    parts.append("\n" + "=" * 60)
    parts.append("PARLIAMENT DISCUSSION RECORD")
    parts.append(f"(Posts in {sort_desc}. Each post is dated.)")
    parts.append("=" * 60)

    for post in posts:
        if post["post_id"] == 1 and "Parliament is now in session" in (post["content"] or ""):
            continue
        score = post["score"] or 0
        date_tag = ""
        if round_map:
            date_tag = f" [{_id_to_date(post['post_id'], round_map, 'max_post_id')}]"
        parts.append(
            f"\n--- Post #{post['post_id']}{date_tag} "
            f"[score: {score:+d}] by {post['author']} ---"
        )
        parts.append(post["content"] or "")
        cmts = comments_by_post.get(post["post_id"], [])
        if cmts:
            parts.append("  Comments:")
            for cm in cmts:
                cm_date = ""
                if round_map:
                    cm_date = f"{_id_to_date(cm['comment_id'], round_map, 'max_comment_id')}, "
                parts.append(
                    f"  [{cm_date}{cm['score'] or 0:+d}] "
                    f"{cm['author']}: {cm['content'] or ''}"
                )

    parts.append("\n" + "=" * 60)
    parts.append("Based on the above discussion, provide your final answer.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Extract the answer from <<<FINAL>>>...<<<END>>> block
# ---------------------------------------------------------------------------

def extract_answer(response_text: str) -> str | None:
    """Parse the <<<FINAL>>>...<<<END>>> block from the Judge's response."""
    m = re.search(r"<<<FINAL>>>\s*(.+?)\s*<<<END>>>", response_text, re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Run the Judge
# ---------------------------------------------------------------------------

async def run_judge(
    db_path: str,
    question: str,
    model,
    choices: list[str] | None = None,
    output_dir: str | None = None,
) -> dict:
    """Invoke the Judge to synthesize a final answer from the parliament."""
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage

    system_prompt = JUDGE_SYSTEM_PROMPT_CHOICES if choices else JUDGE_SYSTEM_PROMPT
    user_message = build_judge_context(db_path, question, choices)

    agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Judge",
            content=system_prompt,
        ),
        model=model,
    )
    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content=user_message,
    )

    response = await agent.astep(user_msg)
    raw_text = response.msgs[0].content if response.msgs else ""
    answer = extract_answer(raw_text)

    result = {
        "answer": answer,
        "raw_response": raw_text,
        "system_prompt": system_prompt,
        "user_message": user_message,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "judge_response.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[Judge] Response saved to {path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main():
    if len(sys.argv) < 3:
        print("Usage: python judge.py <db_path> <question> [--choices 'A|B|C|D']")
        sys.exit(1)

    db_path = sys.argv[1]
    question = sys.argv[2]
    choices = None
    if "--choices" in sys.argv:
        idx = sys.argv.index("--choices")
        choices = sys.argv[idx + 1].split("|")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "parliament"))
    from session import create_model

    model = create_model()
    result = await run_judge(db_path, question, model, choices=choices)
    print(f"\n{'='*60}")
    print(f"JUDGE ANSWER: {result['answer']}")
    print(f"{'='*60}\n")
    print(result["raw_response"])


if __name__ == "__main__":
    asyncio.run(_main())
