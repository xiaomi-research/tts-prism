# TTS-PRISM: A Perceptual Reasoning and Interpretable Speech Model for Fine-Grained Diagnosis

| [🤗 HuggingFace](#-huggingface-link-placeholder) | [📄 Paper](#-paper-link-placeholder) |

## Introduction

While generative text-to-speech (TTS) models approach human-level quality, monolithic metrics fail to diagnose fine-grained acoustic artifacts or explain perceptual collapse. To address this, we propose TTS-PRISM, a multi-dimensional diagnostic framework for Mandarin. First, we establish a 12-dimensional schema spanning stability to advanced expressiveness. Second, we design a targeted synthesis pipeline with adversarial perturbations and expert anchors to build a high-quality diagnostic dataset. Third, schema-driven instruction tuning embeds explicit scoring criteria and reasoning into an efficient end-to-end model. Experiments on a 1,600-sample Gold Test Set show TTS-PRISM outperforms generalist models in human alignment. Profiling six TTS paradigms establishes intuitive diagnostic flags that reveal fine-grained capability differences.

## Architecture Overview

![Architecture Diagram](arch_diagram.png)

## Model Download

| Models | 🤗 Hugging Face |
| :--- | :--- |
| MiMo-Audio-Tokenizer | [XiaomiMiMo/MiMo-Audio-Tokenizer](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer) |
| TTS-PRISM-7B | [[待定：你的模型权重 HuggingFace 链接]](#) |

```bash
pip install huggingface-hub

hf download XiaomiMiMo/MiMo-Audio-Tokenizer --local-dir ./models/MiMo-Audio-Tokenizer
hf download [待定：你的 HF 仓库 ID] --local-dir [待定：你的本地保存路径]
```

## Getting Started

Spin up the inference diagnostic pipeline in minutes.

### Prerequisites (Linux)
* Python 3.12
* CUDA >= 12.0

### Installation

```bash
git clone [待定：你的 Github/Gitlab 仓库克隆链接]
cd [待定：你的仓库文件夹名称]
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1
```

> **Note**
> If the compilation of flash-attn takes too long, you can download the precompiled wheel and install it manually:
> * [Download Precompiled Wheel](#-placeholder-for-wheel-link)
> 
> ```bash
> pip install /path/to/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
> ```

## Core Structure

- `inference_diagnostic.py`: Single-pass inference script for 12-dimensional scoring and rationale generation.
- `Scoring_Criteria.md`: The comprehensive textual definitions and quantitative rubrics of our 12-dimensional evaluation schema.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
Copyright (c) 2026 Xiaomi Corporation.