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

Each post and comment is dated. Later contributions often correct or \
refine earlier work. Pay special attention to later comments on \
high-scored early posts — they may contain important corrections. \
Posts marked [summarized] have been condensed; core reasoning is preserved.

The parliament may not have reached a definitive conclusion. That is \
fine — you are the final decision-maker. Even if the discussion is \
incomplete or inconclusive, you must still provide your best answer \
based on the evidence available, combined with your own expertise.

Instructions:
- Read all posts and comments carefully.
- Identify the strongest line of reasoning in the discussion.
- If there are unresolved disagreements, evaluate both sides and \
  pick the one with better evidence.
- If critical errors were identified in highly-scored posts, account \
  for those corrections.
- Synthesize a single, definitive answer.
- You MUST always provide an answer — never refuse or say the \
  discussion was insufficient."""

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
# Uses context.py's _get_all_posts and _format_post for consistency
# with what scientists see (including compressed posts if applicable).
# ---------------------------------------------------------------------------

JUDGE_SORT_MODE = "time"   # "time" or "score"


def build_judge_context(
    db_path: str,
    question: str,
    choices: list[str] | None = None,
    sort_mode: str | None = None,
) -> str:
    """Read parliament.db and format the discussion for the Judge."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "parliament"))
    import context as ctx

    if sort_mode is None:
        sort_mode = JUDGE_SORT_MODE

    output_dir = os.path.dirname(db_path)
    ctx._load_compressed(output_dir)
    round_map = ctx._load_round_map(output_dir)

    posts = ctx._get_all_posts(db_path)
    posts = [p for p in posts if not (
        p["post_id"] == 1 and "Parliament is now in session" in (p["content"] or "")
    )]

    if sort_mode == "score":
        posts.sort(key=lambda p: -(p["score"] or 0))

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
        is_compressed = post["post_id"] in ctx._compressed_posts
        parts.append("")
        parts.append(ctx._format_post(post, compressed=is_compressed, round_map=round_map))

    parts.append("\n" + "=" * 60)
    parts.append("Based on the above discussion, provide your final answer.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Extract the answer from <<<FINAL>>>...<<<END>>> block
# ---------------------------------------------------------------------------

def extract_answer(response_text: str) -> str | None:
    """Parse the FINAL...END block from the Judge's response.

    Tolerates variations like >>>FINAL>>>, **FINAL**, <<<FINAL>>>, etc.
    Falls back to finding the last (A)/(B)/(C)/(D) in the text.
    """
    m = re.search(r"[<>]{3}\s*FINAL\s*[<>]{3}\s*(.+?)\s*[<>]{3}\s*END\s*[<>]{3}", response_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\*{2,3}\s*FINAL\s*\*{2,3}\s*(.+?)\s*\*{2,3}\s*END\s*\*{2,3}", response_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    choices = re.findall(r"\(([A-D])\)", response_text)
    if choices:
        return f"({choices[-1]})"
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
    max_attempts: int = 2,
) -> dict:
    """Invoke the Judge to synthesize a final answer from the parliament.

    Retries up to max_attempts if the first attempt fails or returns no answer.
    """
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage

    system_prompt = JUDGE_SYSTEM_PROMPT_CHOICES if choices else JUDGE_SYSTEM_PROMPT
    user_message = build_judge_context(db_path, question, choices)

    raw_text = ""
    answer = None
    error_log = []

    for attempt in range(1, max_attempts + 1):
        try:
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

            if answer is not None:
                break

            error_log.append(f"attempt {attempt}: no <<<FINAL>>> marker found")
            print(f"[Judge] Attempt {attempt}: no answer marker, {'retrying' if attempt < max_attempts else 'giving up'}...")

        except Exception as e:
            error_log.append(f"attempt {attempt}: {e}")
            print(f"[Judge] Attempt {attempt} error: {e}")

    result = {
        "answer": answer,
        "raw_response": raw_text,
        "system_prompt": system_prompt,
        "user_message": user_message,
        "attempts": len(error_log) + (1 if answer else 0),
        "errors": error_log if error_log else None,
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
