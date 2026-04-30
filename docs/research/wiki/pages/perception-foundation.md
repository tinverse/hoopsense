# Perception Foundation Sources

## Source Cache

- Grounding DINO paper: `.local_wiki/raw/grounding-dino-paper/`
- Segment Anything paper: `.local_wiki/raw/segment-anything-paper/`
- SAM 2 paper: `.local_wiki/raw/sam2-paper/`
- Grounding DINO repo: `.local_wiki/raw/grounding-dino-repo/`
- Segment Anything repo: `.local_wiki/raw/segment-anything-repo/`

## Current Read

### Grounding DINO

- Use it as an open-vocabulary proposal generator and scene prior builder.
- In HoopSense terms, that means court or hoop or referee or player-region
  proposals, not final runtime identity decisions.
- It is best used sparsely: bootstrap, discontinuity refresh, collapse-triggered
  regrounding, and labeling workflows.

### SAM and SAM 2

- Use segmentation as a refinement or recovery stage, not the always-on first
  detector for the whole clip.
- Good fits here are ball bootstrap, player recovery on ambiguous grounded
  regions, and high-precision manual labeling support.
- The papers and official repos support the staged architecture already emerging
  in Layer 1: detect or propose first, then refine or track.

## Open Questions To Capture Next

- Best primary-source basis for SAM 3 in the current branch/runtime.
- Which grounding prompts are reliable for youth basketball clips with partial
  court visibility.
- Whether the segmentation path should directly seed runtime tracker state or
  stay a review/bootstrap path until stronger evaluation exists.
