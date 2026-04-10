# 2026_4_10_v2

## Changes from v1
- Vote extracted from submit into independent tool
- Actor: vote does NOT end round (like python_exec)
- Judge: vote ENDS round but does NOT wake runner (no set_event)
- Only actors trigger runner wake (submit/wait set_event)
- Runner only distributes when posts/comments exist (not vote-only)
- Collection window removed — no more 10s vote-only wait
- Processing split: actor_processing + judge_processing (both modes identical)
- Idle detection: only posts/comments count (votes ignored)
- Two modes differ only in distribution content (visible adds anon judge_votes)

## Metrics
(To be filled after experiment)