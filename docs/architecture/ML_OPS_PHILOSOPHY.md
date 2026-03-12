# HoopSense MLOps Philosophy

HoopSense is a data-and-reasoning system before it is a model system.

## 1. Data Is a Product Surface

Training data is not a disposable intermediate. It is a first-class product artifact.

For HoopSense this means:
- Oracle datasets must be inspectable and reproducible
- real-world validation data must be attributable and reviewable
- dataset quality must be discussed with the same rigor as model architecture

## 2. Contracts Beat Implicit Coupling

ML systems fail when feature semantics drift silently.

For HoopSense:
- `features_v2` is a contract, not a convenience
- possession context must have its own contract
- event and stat artifacts must not be inferred from undocumented assumptions

## 3. Models Are Narrow Tools

The Action Brain is not the product. It is one component in a larger basketball reasoning stack.

That means:
- local motion classification should stay narrow
- event attribution should remain auditable
- stats should be derived from attributed events, not directly from neural outputs

## 4. Reproducibility Is a Functional Requirement

If a training result cannot be reproduced, it cannot be trusted.

Every meaningful model artifact should be tied to:
- dataset version
- code revision
- configuration
- environment target
- evaluation results

## 5. Aggregate Accuracy Is Not Enough

A single metric can hide systematic failure.

HoopSense must evaluate by slices such as:
- action class
- camera angle
- ball visibility
- pose quality
- possession phase
- court zone
- deployment target

## 6. Promotion Must Be Explicit

The newest checkpoint is not automatically the best checkpoint.

Model artifacts should move through explicit states:
- `candidate`
- `validated`
- `default`
- `retired`

Promotion should require evidence, not optimism.

## 7. Edge and Cloud Are Different Products

Cloud training and Jetson runtime must be related, but not conflated.

We should:
- keep artifact parity where possible
- keep environment guidance separate where necessary
- never hide architecture or runtime differences under a single ambiguous "Docker works" claim

## 8. Human Signals Matter

HoopSense has access to strong human truth sources:
- referee signals
- reviewer adjudication
- curated examples

These should be incorporated as feedback and correction signals, not treated as external noise.

## 9. Failures Must Be Explainable

When HoopSense gets a basketball conclusion wrong, we should be able to localize the failure to one or more layers:
- perception failure
- pose/ball alignment failure
- Action Brain misclassification
- possession-state error
- event-attribution error
- stat-ledger bug

Good MLOps makes that diagnosis possible.
