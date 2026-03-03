# interspeech-2026-code
# TTS-PRISM: A Perceptual Reasoning and Interpretable Speech Model for Fine-Grained Diagnosis

**Note for Reviewers:** This anonymous repository provides the core code to verify our methodology during the double-blind review process. **The model weights (checkpoints) will be publicly released upon paper acceptance.**

## 1. Architecture Overview
![Architecture Diagram](arch_diagram.png)

## 2. Core Structure
- `inference_diagnostic.py`: Single-pass inference script for 12-dimensional scoring and rationale generation.
- `Scoring_Criteria.md`: The comprehensive textual definitions and quantitative rubrics of our 12-dimensional evaluation schema.
