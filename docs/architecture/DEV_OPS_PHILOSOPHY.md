# HoopSense DevOps Philosophy

## 1. Reproducibility Is the Default

If a developer or agent cannot recreate the environment, the build is not trustworthy.

For HoopSense this means:
- prefer declared environments over manual setup
- keep environment definitions in version control
- treat runtime assumptions as product constraints, not tribal knowledge

## 2. Guix First, Docker Second

Guix is the preferred mechanism for reproducible development and build inputs.

Docker is the fallback mechanism for:
- cloud packaging
- platform handoff
- vendor-runtime boundaries that Guix does not fully solve cleanly

Docker is not the philosophy. It is the escape hatch.

## 3. Cloud and Edge Are Different Operating Modes

A cloud/x86 training image and a Jetson/ARM64 runtime are not the same product surface.

We should:
- share source and contracts
- share artifact lineage where possible
- separate runtime assumptions where necessary

## 4. CI Should Prove Contracts, Not Just Run Commands

A good CI pipeline for HoopSense should prove:
- code still builds
- tests still pass
- feature and schema contracts remain stable
- docs and plan artifacts stay coherent

CI is not only for syntax checks. It is for contract enforcement.

## 5. Artifacts Need Explicit Ownership

The system should know what each artifact is for:
- development shell
- training image
- edge runtime guide
- model checkpoint
- dataset manifest
- deployment compatibility report

Implicit artifact meanings create operational drift.

## 6. Vendor Reality Must Be Named Clearly

Jetson/Orin GPU access depends on host-installed NVIDIA components.

That dependency should be documented explicitly rather than hidden under claims of full reproducibility. Where full Guix control stops, the documented vendor boundary begins.

## 7. Small, Verifiable Paths Beat Grand Unified Infrastructure

The repo should prefer:
- a working local Guix shell
- a working cloud training container
- a documented Orin runtime path

before trying to build an oversized deployment platform.
