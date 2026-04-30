# Layer 1 Timing Notes

## Current Findings

- The dominant fixed bug was repeated SAM3 model reload inside ROI refinement.
  That was removed by only loading the model once per refiner instance.
- OCR was an avoidable cost even on clips with zero plausible on-court players.
  Gating OCR to plausible detections and lazily constructing the reader removed
  it as a meaningful short-sample bottleneck.
- The remaining hot spots on the 49-frame debug clip are player tracking and
  ball fallback.

## Measured Runtime Clues

- `YOLO.predict()` for person detection on the short clip is materially cheaper
  than the default persistent tracker path.
- The default persistent tracker behaves like the slower BoT-SORT path.
- `tracker='bytetrack.yaml'` is much faster than the current default path, but
  it increases unique track IDs, which likely means more fragmentation.
- Ball fallback cost still scales with the number of player-neighborhood ROIs
  searched per frame even after batching.

## Next Timing Questions

- Whether Layer 1 should switch to ByteTrack for the proposal stage and leave
  identity continuity to the explicit stitcher or MHT layer.
- Whether ball fallback should be invoked less often or with tighter gating
  rather than only making each invocation cheaper.
- Whether the evaluation harness should emit runtime scorecards alongside
  perception-quality metrics for preset comparison.

## Runtime Ball Scheduler Benchmark

Clip: `data/raw_clips/debug/DSC_4959_sample_31_debug2s.mp4`

Runtime command family: `pipelines/inference.py` inside the Orin Docker wrapper,
using `--player-tracker-backend bytetrack.yaml`.

| Run | Wall sec | Observed | Predicted | Missing | Longest missing | Full-frame frames | ROI frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline every-frame full-frame | 23.129 | 1 | 4 | 44 | 31 | 49 | 0 |
| full-frame every 5, missing every 1, ROI 4 | 25.657 | 6 | 5 | 38 | 28 | 41 | 49 |
| full-frame every 5, missing every 5, ROI 4 | 24.303 | 6 | 5 | 38 | 28 | 9 | 49 |
| full-frame every 5, missing every 5, ROI 1 | 22.527 | 0 | 0 | 49 | 49 | 9 | 49 |

Takeaway:

- ROI search with four regions improved ball recall on this clip, but did not
  improve wall time yet.
- Throttling missing-ball full-frame scans from every frame to every five frames
  reduced full-frame ball calls from 41 to 9 without changing ball-state counts.
- One ROI per frame was faster than baseline, but lost the ball completely.
- The next runtime optimization should improve ROI selection or batch/cadence
  policy before treating ROI search as a speed win.

## Runtime Ball 60s Validation

Clip: `data/raw_clips/youth/DSC_4959_sample_31.mp4`

Runtime command: `pipelines/inference.py` inside the Orin Docker wrapper with
`--player-tracker-backend bytetrack.yaml --ball-full-frame-interval-frames 5
--ball-missing-full-frame-interval-frames 5 --ball-roi-search
--ball-roi-max-count 4`.

Artifact: `tmp_runs/l3_198_validation/intelligent_game_dna.jsonl`

| Run | Frames | Observed | Predicted | Missing | Longest missing | Full-frame frames | ROI frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runtime mode-aware ROI | 1440 | 31 | 51 | 1358 | 305 | 288 | 1440 |
| runtime airborne/provenance ROI | 1440 | 71 | 126 | 1243 | 208 | 288 | 1440 |
| runtime source-aware reacquisition | 1440 | 121 | 219 | 1100 | 130 | 288 | 1440 |
| existing Layer 1 artifact | 1440 | 141 | 161 | 1138 | 209 | n/a | n/a |

Mode-aware ROI smoke and runtime validation both execute, and the 60s run now
survives tiny edge ROIs after adding a crop-size guard before batched Ultralytics
ROI inference. The quality result is still poor: runtime ball recall is far
behind the existing Layer 1 artifact. The next ball-tracking work should close
that runtime/review gap by feeding the runtime path stronger ball seeds, likely
from the same SAM/Grounding-assisted bootstrap evidence already used in review,
before expecting physics-only smoothing to carry passes and shots.

Follow-up run: `tmp_runs/l3_215_validation/intelligent_game_dna.jsonl`

- Added rejected candidate provenance with bbox, source, score parts, and
  rejection reasons.
- Added ball-specific mask semantics: scene/court foreground is a bonus only,
  not a hard floor mask for airborne ball candidates.
- Added airborne player reacquisition and pose-derived shot/pass corridor ROIs.
- Observed-or-predicted coverage improved from 82 frames to 197 frames, and the
  longest missing run dropped from 305 to 208 frames.
- Accepted ball candidate sources: pose corridor 23, airborne player
  reacquisition 19, last-ball ROI 13, nearby-player ROI 8, full-frame 8.
- Remaining rejection reasons are dominated by `below_min_score` and
  `large_jump_from_last_ball`, so the next tuning target is a source-aware
  acceptance policy for airborne/corridor candidates rather than more blind ROI
  generation.

Source-aware acceptance run: `tmp_runs/l3_220_validation/intelligent_game_dna.jsonl`

- Lowered acceptance thresholds only for stale-miss candidates from
  `pose_shot_pass_corridor` and `airborne_player_reacquisition` when size and
  confidence remain plausible.
- Recorded raw versus adjusted continuity and the source-specific acceptance
  threshold in candidate provenance.
- Observed-or-predicted coverage improved from 197 frames to 340 frames, and
  longest missing run dropped from 208 to 130 frames.
- Accepted sources: pose corridor 45, airborne player reacquisition 40,
  nearby-player ROI 14, last-ball ROI 13, full-frame 9.
- This now exceeds the existing Layer 1 artifact on observed-or-predicted
  coverage, but still needs visual review because lower reacquisition
  thresholds can introduce false positives.

SAM3 missed-frame audit: `tmp_runs/l3_221_sam3_misses_30_40.json`

- Evaluated every runtime-missing ball frame in the weakest 30-40s window from
  the source-aware run.
- SAM3 returned at least one `basketball` detection in 205 of 221 missed
  runtime frames.
- Classification counts: `runtime_detector_absence` 199,
  `tracker_rejected_nearby_candidate` 6, `sam_no_detection` 16.
- The dominant failure is not tracker rejection; the runtime YOLO/ROI path often
  produces no plausible ball candidate where SAM3 can still find one.
- Refinement direction: use SAM3 as a sustained-miss reacquisition oracle and as
  a validator for weak runtime candidates, then hand off accepted seeds back to
  the physics/ROI tracker instead of running SAM3 every frame.

Frame-quality audit: `tmp_runs/l3_223_frame_quality_sample_31.json`

- Evaluated Laplacian variance, Tenengrad score, and phase-correlation global
  motion over the 60s `DSC_4959_sample_31` runtime output.
- Runtime: 1440 frames processed in the Orin Docker path in about 18.5 seconds.
- Quality buckets: `sharp` 1037, `soft` 180, `blurred` 151, `severe_blur` 72.
- Ball missing rates: `severe_blur` 0.8472, `sharp` 0.7956, `blurred` 0.6490,
  `soft` 0.6444.
- Interpretation: severe blur is a bad measurement condition, but the runtime
  ball detector is still missing heavily on sharp frames. Blur should downweight
  measurements and suppress overreaction, but it is not the primary explanation
  for the current ball-recall gap.

SAM3 short-clip comparison audit:

- Tool: `tools/review/labeller/compare_sam3_ball_to_runtime.py`.
- The comparison now matches runtime ball state against the nearest retained
  SAM3 basketball candidate, not only SAM3 rank 1, and records
  `best_sam_rank` plus the nearest SAM3 detection.
- `tmp_runs/l3_226_compare_debug2s_v2.json`: 49 frames compared, SAM3 positive
  on 31 frames, runtime positive on 12 frames. Counts: `sam_only` 20,
  `disagree` 11, `runtime_only` 1, `both_missing` 17.
- `tmp_runs/l3_226_compare_sample3_first60_v2.json`: 60 frames compared, SAM3
  positive on 34 frames, runtime positive on 51 frames. Counts:
  `disagree` 31, `runtime_only` 20, `sam_only` 3, `both_missing` 6.
- `tmp_runs/l3_226_compare_sample31_first60_v2.json`: 60 frames compared
  against the current Layer 1 artifact, SAM3 positive on 34 frames, runtime
  positive on 11 frames. Counts: `sam_only` 24, `disagree` 10,
  `runtime_only` 1, `both_missing` 25.
- The candidate-aware matcher did not materially change the classifications,
  so the disagreement is not primarily a top-k ranking issue.
- The useful conclusion is that SAM3 can find basketball-like candidates in
  many runtime-missing frames, but full-frame text-prompted SAM3 also needs
  target-court, possession, and trajectory context. Otherwise it can select a
  different visible basketball or a background/adjacent-court candidate than
  the runtime tracker.
- Runtime implication: use SAM3 as a sustained-miss reacquisition or audit
  source, but score SAM3 candidates through the same court-pose, player
  proximity, trajectory, and possession evidence model before handing a seed
  back to the fast ROI tracker.
