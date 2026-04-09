# 2026_4_9_v3

## Changes from v2
- Added wait tool (actor only): wait for new content, ends round
- Added leave tool (actor only): leave session permanently, irreversible
- Removed follow/unfollow entirely (tools, prompts, executor)
- Removed vote value=0 (only +1/-1)
- Actor votes distributed to other agents (judge votes hidden)
- Submit content preserved in context (no longer stripped)
- Fallback tool call parser: catches vLLM qwen3_coder parse failures
- No-tool resample: discard response, check for new content, retry (max 3)
- Idle detection: only triggers when all active actors have finished their round
- All-wait nudge: when all actors wait with no new content, push "break the silence"
- step_limit reduced from 50 to 20
- max_rounds increased from 10 to 20

## Metrics
(To be filled after experiment)
