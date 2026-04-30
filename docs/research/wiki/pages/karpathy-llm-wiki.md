# Karpathy LLM Wiki Pattern

## Source

- Primary: `.local_wiki/raw/karpathy-llm-wiki/`

## What Matters

- The core idea is compilation, not retrieval-only chat. Raw notes, papers, and
  docs should be turned into a maintained wiki that compounds over time.
- The wiki should be small, structured, and editable. The raw source archive
  remains available, but the working memory for future agents is the synthesized
  Markdown.
- The practical benefit for HoopSense is reduced repeated context-building.
  Research on tracking, segmentation, calibration, and evaluation can accumulate
  into stable pages instead of being rederived from search or chat transcripts.

## HoopSense Adaptation

- Keep raw external material in `.local_wiki/raw/` and keep it out of git.
- Keep synthesized pages in `docs/research/wiki/pages/` so decisions and claims
  are reviewable in the repo.
- Prefer primary sources first: official gists, papers, and repos.
- Record implementation implications, not just summaries.

## Immediate Use

- Track model-selection rationale for Grounding DINO, SAM, and YOLO.
- Track timing findings from Layer 1 profiling.
- Track identity and ball-tracking algorithm options with links to papers.
