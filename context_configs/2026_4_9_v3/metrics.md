# 2026_4_9_v3

## Changes from v2
- Added wait tool (actor only): wait for new content, ends round
- Added leave tool (actor only): leave session permanently, irreversible
- Removed follow/unfollow entirely (tools, prompts, executor)
- Removed vote value=0 (only +1/-1)
- Actor votes distributed to other actors (real author names)
- Judge votes anonymized as "Anonymous Scientist", controlled by judge_votes_visible switch
- Judges never see any votes (independent evaluation)
- Submit content preserved in context (no longer stripped)
- Fallback tool call parser: catches vLLM qwen3_coder parse failures
- No-tool resample: discard response, check for new content, retry (max 3)
- Pure polling: no events, runner waits for processing set to empty
- Idle detection based on posts + comments only (votes don't affect idle)
- All actors done → session ends immediately
- step_limit: 20 per round
- max_rounds: 20 per agent

## Metrics
(To be filled after experiment)
