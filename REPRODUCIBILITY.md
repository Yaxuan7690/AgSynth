# Reproducibility scope

This repository reproduces the AgSynth data-generation workflow: kernel extraction, visual design, programmatic rendering, reverse QA generation, stage audits, figure-dependency checks, and terminal dual-model verification.

It does not include the private or externally licensed seed resources, the released AgSynth-10K dataset, model-service credentials, GRPO training code, training infrastructure, or benchmark evaluation scripts. Consequently, a clean checkout can validate the code structure and run on user-provided seeds, but cannot independently reproduce the paper's dataset scale or reported training results.

Record the following when reporting a generation run:

- commit or release identifier;
- seed-data provenance and checksum;
- model identifiers and provider configuration;
- command-line arguments;
- accepted/rejected counts and audit logs;
- environment and dependency versions.
