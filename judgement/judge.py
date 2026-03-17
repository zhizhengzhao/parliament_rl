"""
Science Parliament — Judge module.

After the parliament finishes discussing, the Judge reads the entire forum
record and synthesizes a final answer.  The Judge is a senior scientist who
observed the discussion without participating — objective and comprehensive.

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
# Judge system prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a senior scientist who has been silently observing a Science \
Parliament — a forum where a group of scientists collaborated over \
multiple rounds to solve a difficult problem.

You did not participate in the discussion. You have no stake in any \
particular answer. Your job is to read the entire record of the \
parliament's work and produce the definitive final answer.

The forum posts below are sorted by community score (endorsements \
minus challenges). Higher-scored posts were considered more rigorous \
by the participating scientists. However, the majority is not always \
right — use your own judgment.

Instructions:
- Read all posts and comments carefully.
- Identify the strongest line of reasoning in the discussion.
- If there are unresolved disagreements, evaluate both sides and \
  pick the one with better evidence.
- If critical errors were identified in highly-scored posts, account \
  for those corrections.
- Synthesize a single, definitive answer.

You MUST end your response with exactly this line:

ANSWER: <your final answer>

The ANSWER line must be self-contained — someone reading only that \
line should get the complete answer without needing any other context.\
"""

JUDGE_SYSTEM_PROMPT_CHOICES = """\
You are a senior scientist who has been silently observing a Science \
Parliament — a forum where a group of scientists collaborated over \
multiple rounds to solve a difficult problem.

You did not participate in the discussion. You have no stake in any \
particular answer. Your job is to read the entire record of the \
parliament's work and select the correct answer from the given choices.

The forum posts below are sorted by community score (endorsements \
minus challenges). Higher-scored posts were considered more rigorous \
by the participating scientists. However, the majority is not always \
right — use your own judgment.

Instructions:
- Read all posts and comments carefully.
- Identify the strongest line of reasoning in the discussion.
- If there are unresolved disagreements, evaluate both sides and \
  pick the one with better evidence.
- If critical errors were identified in highly-scored posts, account \
  for those corrections.
- Select the single best answer from the provided choices.

You MUST end your response with exactly this line:

ANSWER: (X)

where X is the letter of the correct choice (e.g., A, B, C, or D).\
"""


# ---------------------------------------------------------------------------
# Build the forum context for the Judge
# ---------------------------------------------------------------------------

def build_judge_context(
    db_path: str,
    question: str,
    choices: list[str] | None = None,
) -> str:
    """Read parliament.db and format the discussion for the Judge.

    Returns a user-message string containing the question and all posts/comments
    sorted by score (highest first).
    """
    conn = sqlite3.connect(db_path, timeout=3)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Posts sorted by score descending
    c.execute("""
        SELECT p.post_id, u.name AS author, p.content,
               (p.num_likes - p.num_dislikes) AS score
        FROM post p JOIN user u ON p.user_id = u.user_id
        ORDER BY score DESC, p.post_id ASC
    """)
    posts = [dict(r) for r in c.fetchall()]

    # Comments grouped by post, sorted by score descending
    c.execute("""
        SELECT cm.post_id, u.name AS author, cm.content,
               (cm.num_likes - cm.num_dislikes) AS score
        FROM comment cm JOIN user u ON cm.user_id = u.user_id
        ORDER BY cm.post_id, (cm.num_likes - cm.num_dislikes) DESC, cm.comment_id ASC
    """)
    comments_by_post: dict[int, list[dict]] = {}
    for r in c.fetchall():
        d = dict(r)
        comments_by_post.setdefault(d["post_id"], []).append(d)

    conn.close()

    # --- Assemble context ---
    parts = [
        "THE PROBLEM:",
        question,
    ]

    if choices:
        parts.append("\nCHOICES:")
        for i, ch in enumerate(choices):
            letter = chr(ord("A") + i)
            parts.append(f"  ({letter}) {ch}")

    parts.append("\n" + "=" * 60)
    parts.append("PARLIAMENT DISCUSSION RECORD")
    parts.append("(Posts sorted by community score, highest first)")
    parts.append("=" * 60)

    # Skip the automated opening post (post_id 1, "The Science Parliament is now in session")
    for post in posts:
        if post["post_id"] == 1 and "Parliament is now in session" in (post["content"] or ""):
            continue

        score = post["score"] or 0
        parts.append(f"\n--- Post #{post['post_id']} [score: {score:+d}] by {post['author']} ---")
        parts.append(post["content"] or "")

        cmts = comments_by_post.get(post["post_id"], [])
        if cmts:
            parts.append("  Comments:")
            for cm in cmts:
                cm_score = cm["score"] or 0
                parts.append(f"  [{cm_score:+d}] {cm['author']}: {cm['content'] or ''}")

    parts.append("\n" + "=" * 60)
    parts.append("Based on the above discussion, provide your final answer.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Extract the ANSWER line from the Judge's response
# ---------------------------------------------------------------------------

def extract_answer(response_text: str) -> str | None:
    """Parse 'ANSWER: ...' from the Judge's response.

    Returns the answer string, or None if not found.
    """
    for line in reversed(response_text.strip().splitlines()):
        m = re.match(r"^\s*ANSWER\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


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
    """Invoke the Judge to synthesize a final answer from the parliament.

    Args:
        db_path:    Path to parliament.db.
        question:   The original problem statement.
        model:      A CAMEL ModelBackend (same one used for the parliament).
        choices:    Optional list of answer choices (for multiple-choice benchmarks).
        output_dir: If set, save judge_response.json here.

    Returns:
        dict with keys: raw_response, answer, system_prompt, user_message.
    """
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
# CLI entry point
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
