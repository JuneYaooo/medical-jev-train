<div align="center">

# medical-jev-4b

**A medical multiple-choice model that runs on your own GPU**

Give it a case and a few options. It picks one answer and tells you the probability of each option.

[![Model](https://img.shields.io/badge/🤗_Model-v1.2.0-yellow)](https://huggingface.co/JuneYao/medical-jev-4b/tree/v1.2.0)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-242,921_questions-yellow)](https://huggingface.co/datasets/JuneYao/medical-jev-4b-data)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![Not for clinical use](https://img.shields.io/badge/⚠️_Not_for_clinical_use-critical)](#what-not-to-use-it-for)

[简体中文](README.md) · **English**

[Example](#example) · [How Good Is It](#how-good-is-it) · [Quick Start](#quick-start) · [What Not to Use It For](#what-not-to-use-it-for) · [Technical Details](#technical-details)

</div>

## What Is This

Many medical questions can be written as multiple choice. What is the most likely diagnosis? Is this treatment contraindicated? Which test should come next?

medical-jev-4b is built for this kind of question. It only chooses from the options you give it and doesn't write long explanations, so it is fast and its results are easy for programs to read. It needs no paid online API and runs on a single GPU.

It is intended for **medical teaching question banks, AI model evaluation, and offline research**.

## Example

> **Case**: A patient has fever and cough. The influenza A PCR is positive and the COVID-19 PCR is negative.
>
> **Rule**: If influenza A is positive, choose influenza A. If COVID-19 is positive, choose COVID-19. If both are negative, choose insufficient evidence.

The model's actual output:

| Option | Probability |
| --- | ---: |
| **Influenza A (model's choice)** | **97.43%** |
| COVID-19 | 1.53% |
| Insufficient evidence | 1.04% |

The rule contains the answer, so this example only shows how the model is used. See [more examples (Chinese)](examples/README.md).

## How Good Is It

We tested it on an external set of medical questions isolated from the training data: **1,514 questions across 27 task types**.

| Model | Correct | Accuracy | Runs on |
| --- | ---: | ---: | --- |
| JEV 1.13 | 1,092 | 72.13% | Online API |
| DeepSeek V4.1 Flash | 1,076 | 71.07% | Online API |
| DeepSeek V4 Pro | 1,041 | 68.76% | Online API |
| **medical-jev-4b (this project)** | **1,038** | **68.56%** | **Local GPU** |
| Qwen3.8-27B | 1,008 | 66.58% | Online API |
| Qwen3.5-4B (original model before training) | 925 | 61.10% | Local GPU |

Training helped the model answer **113 more questions** correctly. On a separate 1,557-question held-out set, accuracy rose from 75.47% to **80.92%** (+5.46 points, 95% CI 3.47 to 7.60).

**Strong at**: choosing tests (87%), etiology (86%), diagnosis (85%), complications (81%), and medication choice (79%).

**Weak at**: pharmacogenomic recommendations (47%), triage (51%), and traditional Chinese medicine syndrome questions with 148 options (27%).

**Other domains**: Across 124 MMLU / CMMLU subjects, no subject got significantly worse after multiple-comparison correction, and non-medical subjects improved by 5.45 points on average. Part of the gain is therefore general test-taking ability rather than medical knowledge.

Keep three things in mind when reading this table:

- The 68.56% score requires the "many options" mode, which splits options into groups, keeps the 2 most likely from each, and compares them round by round. The default settings don't support 100 of the questions, which have too many options. Those count as wrong, and accuracy is **66.78%**.
- We looked at the external benchmark repeatedly during development and analyzed held-out errors before release, so neither is a strict blind test.
- Online and local models answer in different ways, so treat this table as a rough reference.

## Quick Start

You need an NVIDIA GPU and Python 3.12. We verified this on Linux with an RTX 5090 32 GB. The first run automatically downloads a base model of about 9.3 GB.

```bash
# 1. Create an environment and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install huggingface-hub==1.33.0

# 2. Download the model
hf download JuneYao/medical-jev-4b --revision v1.2.0 --local-dir medical-jev-4b
pip install -r medical-jev-4b/requirements.txt

# 3. Run the included examples
python medical-jev-4b/predict_choices.py --bundle medical-jev-4b \
  --input medical-jev-4b/examples/scenarios.jsonl
```

### Use Your Own Questions

Create `my_questions.jsonl` with **one question per line**. The question below is split across lines for readability, but in the file it must be on a single line:

```json
{
  "id": "question-id",
  "request": {
    "state": {"case": "Put the case or other material here"},
    "questions": {"decision": {
      "type": "choice",
      "instructions": "Write the decision rule here",
      "criteria": {"option_a": "Text of option A", "option_b": "Text of option B"}
    }}
  }
}
```

Then run:

```bash
python medical-jev-4b/predict_choices.py --bundle medical-jev-4b \
  --input my_questions.jsonl --output predictions.jsonl
```

In the results, `answer` is the chosen option and `probabilities` holds the probability of each option.

<details>
<summary><b>More usage: Python, low VRAM, more than 52 options</b></summary>

```python
import sys; sys.path.insert(0, "medical-jev-4b")
from jevtrain.inference import ChoicePredictor

model = ChoicePredictor("medical-jev-4b")
print(model.predict([request]))   # request is the "request" object in the JSON above
```

- Low VRAM: add `--batch-size 1`.
- Base model already downloaded: pass its path with `--base-model`.
- More than 52 options: use `scripts/predict_choices.py --tournament-above 52` from this repository. This feature is experimental.
- The weights use a custom format. Load them with the bundled `predict_choices.py` or `ChoicePredictor`, not PEFT.

</details>

## What Not to Use It For

> [!WARNING]
> **Do not use it for diagnosis, triage, or treatment decisions for real patients.** Its output cannot replace a doctor's judgment.

- **Do not use it for triage.** Its triage accuracy is only 51%, below the 55% before training. It rarely asks for clarification on first messages with limited information, getting 16 of 52 right, so it could miss emergencies.
- **Do not treat probabilities as real confidence.** A 97% output does not mean a 97% chance of being right, and it cannot be used as a risk score.
- **The training data was not individually reviewed by clinical experts.** The model may have learned errors in the data.
- **It does not explain itself.** It only returns options and probabilities, so you can't see the reasoning behind a choice.

## Technical Details

The sections below are for readers who want to understand the training method or reproduce the results.

<details>
<summary><b>How the model works</b></summary>

1. **Score instead of generating text.** Options are encoded as A, B, C…, and the model computes the probability that the next token is each letter. One forward pass gives the answer and full distribution, and it cannot output anything outside the options.
2. **Balance sampling by task.** The CMB source alone has 195,784 questions in the training pool, which would drown out small tasks. Each task is capped at 3,000 questions and general medical questions at 10,000, for 44,347 training questions.
3. **Leakage removal and strict scoring.** Before training, 307 rows that duplicate or closely match (5-gram Jaccard ≥ 0.8) the dev, held-out, or benchmark questions, or that have conflicting answers, are removed. Failed and unsupported questions count as wrong, and comparisons with base 4B use a paired per-question bootstrap.

<p align="center"><img src="docs/figures/method.svg" alt="Method overview: data construction and isolation, training and model selection, single-forward-pass scoring" width="100%"></p>

Figure labels are in Chinese. **a**, 77 public question sources are unified into multiple-choice JSON and mapped to 27 decision tasks. **b**, Qwen3.5-4B is frozen and a LoRA is trained on all 32 layers, with loss computed only on answer-letter logits. The dev set is evaluated every 500 steps and step 5,000 is selected. **c**, Inference reads and normalizes the letter probabilities. Above 52 options, it narrows them down in groups.

</details>

<details>
<summary><b>Full evaluation details</b></summary>

<p align="center"><img src="docs/figures/results.svg" alt="Evaluation results: overall and per-task accuracy on the external benchmark" width="100%"></p>

- **Benchmark**: [jev-healthcare-lab](https://github.com/JuneYaooo/jev-healthcare-lab) `medical_decision_v1`, isolated from training data but used to diagnose problems during development, so it is not a blind test.
- **Improvement from training**: Accuracy rose by 7.46 percentage points. Allowing for variation from the sample of questions, the gain is roughly 4.89 to 9.82 points (95% confidence interval).
- **Score with default settings**: The default supports up to 52 options, and questions with more count as wrong. This model then answers 1,011 correctly (66.78%) and the original model 907 (59.91%).
- **Are the probabilities accurate?** They are uncalibrated. The model's stated confidence differs from its actual accuracy by about 15 percentage points on average (ECE 0.146).
- **Held-out test**: On 1,557 questions, this model answers 1,260 correctly (80.92%) and base 4B answers 1,175 (75.47%), a difference of +5.46 percentage points, 95% CI [3.47, 7.60]. Duplicates and close matches of these questions were removed from training, but held-out errors were analyzed before release, so this is not a blind test.
- **Reference models**: API models answer by generation, DeepSeek runs with thinking off, and JEV 1.13 is an archived online service. Raw records are in `results/clinical-v1` of jev-healthcare-lab.

| Task | Base 4B | **This model** | DeepSeek Flash | DeepSeek Pro | JEV 1.13 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diagnosis choice | 82 | **85** | 91 | 91 | 89 |
| Complications | 77 | **81** | 89 | 89 | 93 |
| Examination choice | 81 | **87** | 81 | 74 | 88 |
| Etiology | 80 | **86** | 84 | 84 | 84 |
| Medication choice | 67 | **79** | 80 | 72 | 83 |
| Treatment choice | 67 | **75** | 82 | 74 | 78 |
| TCM pathogenesis | 64 | **74** | 76 | 79 | 67 |
| Clinical trial referral | 55 | **61** | 60 | 58 | 53 |
| Pharmacogenomic recommendation | 39 | **47** | 54 | 51 | 55 |
| Synthetic case diagnosis (DDXPlus) | 55 | **61** | 77 | 71 | 70 |
| Disease staging | 42 | **62** | 50 | 53 | 62 |
| Triage | 55 | **51** | 61 | 57 | 63 |
| TCM syndrome (148 options) | 18 | **27** | 37 | 30 | 38 |

Causes of the weak spots: triage was not part of training, and the training data has almost no examples of asking for clarification when information is missing; synthetic case diagnosis has 49 options, more than the 26 seen in training; TCM syndrome has 148 options and must be narrowed down in groups; pharmacogenomic recommendations depend on detailed guideline rules. All tasks are in the [per-task results (Chinese)](docs/EVALUATION.md#逐任务成绩).

</details>

<details>
<summary><b>Model specifications</b></summary>

| Item | Details |
| --- | --- |
| Base model | Qwen3.5-4B |
| Adapter | All-32-layer LoRA (rank 16 / alpha 32, step 5,000), about 130 MB |
| Weights SHA-256 | `130a4cc0…7e4bc3` |
| Tasks | 27 types, including diagnosis, etiology, complications, examination, treatment, medication, contraindications, pharmacogenomics, triage, message priority, nursing, nutrition, rehabilitation, and TCM syndrome / pathogenesis |
| Options | Training saw 2–26; the default supports up to 52; more require "many options" mode (experimental) |
| Input length | Up to 1,536 tokens in training; up to 8,192 tokens in use |
| Languages | Mainly Chinese, plus English, Japanese, Korean, Portuguese, Spanish, Turkish, and others |
| Deployment | FP16 on one GPU, with automatic base model version and hash checks |

</details>

<details>
<summary><b>Reproduce training</b></summary>

```bash
pip install -e . && pip install -r requirements-train.txt
hf download --repo-type dataset JuneYao/medical-jev-4b-data --local-dir work/data/release
# Also generate work/benchmark/medical_decision_v1.jsonl as described in docs/EVALUATION.md
python scripts/build_clean_data.py .
jev train --config configs/jev/release/train.json --output runs/train
jev eval  --config configs/jev/release/train.json --checkpoint runs/train/best.safetensors \
  --input work/benchmark/medical_decision_v1.jsonl --max-options 52 --max-tokens 8192 \
  --tournament-above 52 --batch-size 1 --output runs/bench
```

Training takes about 1.4 hours on an RTX 5090. The cleaning script reproduces this release's training and development files byte for byte. The README figures are generated by `python scripts/make_readme_figures.py`.

</details>

<details>
<summary><b>Project background</b></summary>

TypeSafe's JEV offers a structured judgment API: pass in context and typed questions, and get a judgment back. This project asks how far an open, local 4B model can go on the same multiple-choice decisions. The model directly accepts the request format of the [jev-healthcare-lab](https://github.com/JuneYaooo/jev-healthcare-lab) benchmark. This project is not affiliated with JEV or TypeSafe.

</details>

## Downloads and Docs

| Content | Link |
| --- | --- |
| Model | [Hugging Face](https://huggingface.co/JuneYao/medical-jev-4b/tree/v1.2.0) |
| Data (242,921 questions, 77 sources) | [Hugging Face](https://huggingface.co/datasets/JuneYao/medical-jev-4b-data) |
| Benchmark | [jev-healthcare-lab](https://github.com/JuneYaooo/jev-healthcare-lab) |
| Per-question results | [docs/results](docs/results/README.md) |
| Detailed docs (Chinese) | [Evaluation](docs/EVALUATION.md) · [Training](docs/TRAINING.md) · [Data](docs/DATA.md) · [Input limits](docs/INPUT_LIMIT_CHECK.md) · [Model card](docs/MODEL_CARD_V1.md) · [Release report](docs/RELEASE_V1.md) · [Changelog](CHANGELOG.md) |

## License

The code and model adapter are released under [Apache-2.0](LICENSE). The original notice for parts previously released under MIT is kept in [NOTICE](NOTICE). The Qwen3.5-4B base model is also under [Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE). Training data keeps each source's upstream license, so please check them before use.
