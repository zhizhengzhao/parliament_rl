# 2026_4_13_v3

## Changes from v2
- **Persona differentiation**: Each Scientist and Judge gets a unique cognitive style (config.json personas). Scientist personas vary in thinking approach (methodical, strategic, skeptical, synthesizing). Judge personas vary in evaluation focus (precision, methodology, progress, completeness).
- **Actor vote range enforcement**: ToolExecutor now enforces ±1 for actors at code level (was only schema-enforced, allowing LLM hallucination to bypass)
- **Strict judge scoring**: Rewrote voting scale to eliminate grade inflation. Key change: contributions that don't meaningfully advance the solution are -1, not +1. "There is no neutral score."
- **Actor high-score awareness**: New "Reading the Scores" section in actor prompt — high-scoring posts signal correct reasoning, negative scores signal errors
- **Strict ending criteria**: Only push to end when 100% certain. "If you have even a trace of doubt, maintain your scientific rigor and keep the discussion going."

## Metrics
(To be filled after experiment)