# Survey Papers

## Cached Surveys

- Sports video analysis survey: `.local_wiki/raw/sports-video-analysis-survey/`
- Sports action recognition survey: `.local_wiki/raw/sports-action-recognition-survey/`
- Multiple object tracking review: `.local_wiki/raw/mot-literature-review/`
- Person Re-ID survey: `.local_wiki/raw/person-reid-survey/`
- Occluded person Re-ID survey: `.local_wiki/raw/occluded-person-reid-survey/`
- Human motion trajectory prediction survey: `.local_wiki/raw/human-motion-trajectory-survey/`

## Why These Matter

### Sports video analysis survey

- This is the broadest domain overview.
- It helps frame HoopSense as a staged content-understanding system rather than
  only a detector stack.
- Useful for taxonomy, system decomposition, and identifying missing context
  layers like event and game-state reasoning.

### Sports action recognition survey

- This is the most directly relevant survey for the future Action Brain.
- It covers datasets, methods, and the team-sport-specific difficulty that one
  action often depends on multiple players and fast interactions.
- Useful when defining the Layer 1 to Action Brain handoff contract.

### Multiple object tracking review

- This is the background survey for data association, identity continuity, and
  evaluation metrics.
- Useful for reasoning about the difference between detector quality, online
  association, offline/global association, and occlusion handling.
- This should inform the MHT or tracklet-stitching design rather than be copied
  mechanically, since HoopSense has stronger court and game constraints than a
  generic MOT benchmark.

### Person Re-ID survey

- This is the general survey for appearance-based identity linking.
- Useful for feature-learning families, metric learning choices, and the limits
  of pure appearance cues.
- Important because HoopSense is not a standard multi-camera Re-ID problem; the
  survey is mainly useful for embedding and evidence-design ideas.

### Occluded person Re-ID survey

- This is the more relevant Re-ID survey for basketball because players are
  frequently partially visible, overlapped, or truncated.
- Useful for understanding what kinds of occlusion failure modes appearance
  models can and cannot handle.
- This should influence how much weight we give appearance evidence versus
  geometry, continuity, and team cues.

### Human motion trajectory prediction survey

- This is not a sports paper, but it is relevant to our planned physics and
  anticipation layers.
- Useful for framing short-gap prediction, keep-alive horizons, and contextual
  trajectory priors for players and ball-adjacent motion.
- It is likely more useful for designing evidence features than for directly
  adopting a large forecasting model in the MVP.

## Practical Reading Order

1. `sports-video-analysis-survey`
2. `mot-literature-review`
3. `occluded-person-reid-survey`
4. `person-reid-survey`
5. `sports-action-recognition-survey`
6. `human-motion-trajectory-survey`

## What I Did Not Cache Yet

- I did not find a comparably clean primary-source survey specifically for
  sports ball tracking that looked better than a set of recent task papers and
  baselines.
- For the ball problem, current strong task papers and benchmarks are likely
  more useful than an older broad survey, if one exists.
