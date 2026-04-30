# HoopSense Research Wiki

This is a repo-local research wiki in the spirit of Andrej Karpathy's April 2026
`llm-wiki` gist: raw external material is cached locally, then distilled into a
small set of maintained Markdown pages instead of being re-searched in chat.

## Layout

- `index.md`: top-level entrypoint.
- `pages/`: maintained synthesis pages.
- `.local_wiki/raw/`: git-ignored cache of fetched raw sources, PDFs, gists, and
  HTML snapshots.

## Workflow

1. Fetch a source into the local cache:

```bash
python tools/infra/research_wiki_ingest.py <url> --slug <stable-name>
```

2. Add or update a page in `docs/research/wiki/pages/` that:
   - links the raw cached source by slug
   - extracts the few claims that matter to HoopSense
   - records any implementation implications or open questions

3. Link the page from `index.md`.

## Current Scope

The initial seed focuses on:

- Karpathy's LLM Wiki pattern
- Grounding DINO
- SAM / SAM 2
- HoopSense Layer 1 timing, evaluation, and model-selection notes
