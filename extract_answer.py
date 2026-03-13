"""
Extract the final answer from a Science Parliament discussion.

Strategy: find all \boxed{...} answers across posts and comments, then pick
the one with the highest community support (post score = likes - dislikes).
If tied, prefer the most recent occurrence.
"""

import re
import sqlite3


def find_boxed_answers(text: str) -> list[str]:
    r"""Extract all \boxed{...} answers from text."""
    pattern = r'\\boxed\{(.+?)\}'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        pattern_alt = r'\\boxed\s*\{(.+?)\}'
        matches = re.findall(pattern_alt, text, re.DOTALL)
    return [m.strip() for m in matches]


def extract_final_answer(db_path: str) -> str | None:
    """
    Extract the best answer from the parliament discussion.

    Scoring:
    - Each \boxed{} in a post gets score = post.likes - post.dislikes
    - Each \boxed{} in a comment gets score = comment.likes - comment.dislikes
    - Answers are grouped by normalized text
    - The group with highest total score wins
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    answer_scores: dict[str, float] = {}

    cursor.execute("""
        SELECT content, num_likes, num_dislikes, created_at
        FROM post ORDER BY post_id
    """)
    for content, likes, dislikes, created_at in cursor.fetchall():
        if content is None:
            continue
        score = (likes or 0) - (dislikes or 0)
        for ans in find_boxed_answers(content):
            key = normalize_answer(ans)
            answer_scores[key] = answer_scores.get(key, 0) + score + 1

    cursor.execute("""
        SELECT content, num_likes, num_dislikes
        FROM comment ORDER BY comment_id
    """)
    for content, likes, dislikes in cursor.fetchall():
        if content is None:
            continue
        score = (likes or 0) - (dislikes or 0)
        for ans in find_boxed_answers(content):
            key = normalize_answer(ans)
            answer_scores[key] = answer_scores.get(key, 0) + score + 1

    conn.close()

    if not answer_scores:
        return None

    best = max(answer_scores, key=answer_scores.get)
    return best


def normalize_answer(text: str) -> str:
    """Basic normalization for answer comparison."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_answer.py <db_path>")
        sys.exit(1)
    answer = extract_final_answer(sys.argv[1])
    print(f"Best answer: {answer}")
