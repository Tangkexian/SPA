# SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection

**Kexian Tang\*, Jiani Wang\*, Shaowen Wang, Kaifeng Lyu†**

Institute for Interdisciplinary Information Sciences, Tsinghua University

\* Equal contribution. † Corresponding author: klyu@mail.tsinghua.edu.cn


<p align="left">
  <a href='TBA'>
    <img src='https://img.shields.io/badge/Paper-TBA-brown?style=flat&logo=arXiv' alt='arXiv PDF'>
  </a>
</p>

## Overview

![SPA Overview](asserts/overview.png)
**SPA** (**S**caling **P**rompt-engineered **A**ugmentation) is a simple but tough-to-beat baseline. It uses a small set of carefully designed prompts to generate large-scale synthetic data for knowledge injection.

We evaluate SPA on three representative benchmarks: **SQuAD** (Wikipedia-based QA), **QuALITY** (long-document comprehension), and **MultiHop-RAG** (multi-hop reasoning). Through systematic comparisons, we show that despite its simplicity, **SPA** outperforms several strong and more complex baselines.

Our results suggest that, for knowledge injection, careful prompt design combined with straightforward large-scale augmentation can be surprisingly effective. We hope SPA can serve as a strong baseline for future studies in this area. 

## News

- **[2026-03-xx]** Our Paper is released.

## Method

SPA operates in three steps:

1. **Prompt Engineering** — We draw upon insights from cognitive science and educational psychology to design a set of 7 prompt templates based on effective human learning strategies, covering three levels of learning strategies:
   - *Concept Learning*: Key Concepts, Mind Map
   - *Critical Thinking*: Implications, QA with Critical Thinking (QA-ct)
   - *Generative Learning*: Case Studies, Discussions, Teacher-style
2. **Scaling** — Repeatedly prompt an LLM to rewrite the source content based on the set of prompts, progressively **scaling** the augmented corpus into a large-scale synthetic corpus.

3. **Training** — The target model is trained on the synthetic corpus via continued pretraining under the same experimental settings as prior work.


See the [Code](https://github.com/Tangkexian/SPA/blob/main/src/utils_tools/prompt_utils.py) for full prompt templates.


## Performance

SPA **consistently improves with scale** and achieves the highest accuracy at moderate-to-large token budgets across benchmarks.
<table>
  <tr>
    <td><b>SQuAD</b></td>
    <td><b>QuALITY</b></td>
  </tr>
  <tr>
    <td><img src="asserts/squad_scaling_curve.png" width="400"/></td>
    <td><img src="asserts/quality_scaling_curve.png" width="400"/></td>
  </tr>
</table>


See the [paper](TBA) for full results.

## Getting Started

### 1. Environment Setup

```bash
conda create -n spa python=3.12
conda activate spa
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your OpenAI API key (used for synthetic data generation):

```bash
OPENAI_API_KEY=your_api_key_here
```

### 2. Generate Synthetic Data

Choose the benchmark you want to run and execute the corresponding script:

**SQuAD** (Wikipedia-based QA):
```bash
bash scripts/make_squad_data.sh
```

**QuALITY** (Long-document comprehension):
```bash
bash scripts/make_quality_data.sh
```

**MultiHop-RAG** (Multi-hop reasoning):
```bash
bash scripts/make_mhrag_data.sh
```

### 3. Tokenize Data

After generation, tokenize the synthetic corpus to prepare it for training:

```bash
bash scripts/tokenize.sh
```

### 4. Train

Run continued pretraining on the tokenized synthetic corpus:

```bash
bash scripts/train.sh
```

---

## Citation

```bibtex
@article{
  TBA
}
```
