#!/bin/bash
# Install Parliament skills into OpenClaw workspace.
# Skills are copied (not symlinked) because OpenClaw rejects external symlinks.
# run_experiment.py reads skills directly from this project, so you only need
# this script for manual `openclaw agent` debugging.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SKILLS_SRC="$PROJECT_DIR/skills"
WORKSPACE_SKILLS="${OPENCLAW_WORKSPACE:-$HOME/.openclaw/workspace}/skills"

mkdir -p "$WORKSPACE_SKILLS/parliament-actor"
mkdir -p "$WORKSPACE_SKILLS/parliament-judge"

cp "$SKILLS_SRC/actor/SKILL.md" "$WORKSPACE_SKILLS/parliament-actor/SKILL.md"
cp "$SKILLS_SRC/judge/SKILL.md" "$WORKSPACE_SKILLS/parliament-judge/SKILL.md"

echo "Installed skills to $WORKSPACE_SKILLS"
echo "  parliament-actor: $(wc -l < "$WORKSPACE_SKILLS/parliament-actor/SKILL.md") lines"
echo "  parliament-judge: $(wc -l < "$WORKSPACE_SKILLS/parliament-judge/SKILL.md") lines"
