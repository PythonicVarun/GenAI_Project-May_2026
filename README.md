# Smart MCQ Solver

This project is a deep learning and generative AI system designed to solve Multiple Choice Questions (MCQs) from scientific and technical domains. It is developed as part of the Deep Learning & Generative AI course project for the Diploma Level of BS in Data Science and Applications (IIT Madras).

The system targets the **Smart MCQ Solver Challenge** on Kaggle, predicting the top 3 most likely correct choices (ordered by decreasing confidence) for each question, optimized for the Mean Average Precision at 3 (**MAP@3**) evaluation metric.

---

## Architecture Overview 🏗️

The project integrates two major modeling approaches to achieve high-performance MCQ solving:

### 1. Custom Local Model (Model 1)
Built from scratch in PyTorch under the [src/local/](src/local/) directory.
- **RAG Preprocessing:** A [TFIDFRetriever](src/local/retriever.py#L21-L107) indexes training queries and correct answers, retrieving the top $K$ relevant context strings.
- **Tokenization:** A custom [Vocabulary](src/local/vocab.py#L10-L58) is built from corpus frequencies to convert text to indices.
- **Network Pipeline:** [BiLSTMScorer](src/local/model.py#L24-L71) passes prompt-context-choice sequence embeddings through a bidirectional LSTM, scores outputs via [SelfAttention](src/local/model.py#L7-L21), and computes a logit score. The model optimizes cross-entropy loss over choice classifications.
- **Validation:** Utilizes a group-split validation based on core questions (via [extract_core_question](src/local/utils.py#L32-L41)) to prevent semantic leakage.

### 2. Fine-tuned LLM Model (Model 2)
Developed inside [notebooks/finetune/gemma4-finetune.ipynb](notebooks/finetune/gemma4-finetune.ipynb).
- **Fine-Tuning:** Uses LoRA (Low-Rank Adaptation) parameter-efficient tuning via Unsloth to adapt Gemma-2 9B on the multiple-choice datasets.
- **Accelerated Inference:** Integrates fast model quantization and prompt templates to format MCQs with supporting retriever texts.

---

## Directory Structure 📂

```
.
├── CHANGELOG.md                              # Tracked versions and updates
├── Project_Guideline.md                      # IITM project evaluation guideline
├── pyproject.toml                            # Project config and packages dependency
├── uv.lock                                   # Lockfile for reproducible builds
├── dataset/                                  # Folder holding competition data
│   ├── train.csv                             # Train dataset (2000 samples)
│   ├── test.csv                              # Test dataset (evaluation prompt questions)
│   └── sample_submission.csv                 # Submission template format
├── scripts/
│   └── deploy_to_kaggle.py                   # Deploys code and models to Kaggle
├── notebooks/
│   ├── milestone-1.ipynb                     # Milestone - 1
│   ├── milestone-2.ipynb                     # Milestone - 2
│   ├── milestone-3.ipynb                     # Milestone - 3
│   ├── milestone-4.ipynb                     # Milestone - 4
│   ├── kernel-metadata.json                  # Kaggle metadata configuration
│   ├── dl-24f2004142-notebook-t22026.ipynb   # Main inference and blending stacker notebook
│   └── finetune/
│       └── gemma4-finetune.ipynb             # Unsloth/LoRA Gemma fine-tuning code
└── src/
    └── local/                                # Local PyTorch scoring model codebase
        ├── __init__.py                       # Package entry, auto-loads environment variables
        ├── __main__.py                       # Command line argument parser entrypoint
        ├── config.py                         # Training/Inference config parameters
        ├── dataset.py                        # MCQDataset builder and collator
        ├── git_utils.py                      # Git status validator and commit tracker
        ├── inference.py                      # Batch predicting and logic exporter
        ├── model.py                          # BiLSTMScorer & SelfAttention architecture
        ├── retriever.py                      # TFIDFRetriever index generator
        ├── train.py                          # Training loops and WandB tracker
        ├── utils.py                          # Metric helpers (MAP@3) and text cleaning
        └── vocab.py                          # Word vocabulary dictionaries
```

---

## Installation & Setup 🛠️

Project dependencies and scripts are managed using `uv` to ensure fast and reproducible environments.

### 1. Sync dependencies
Run `uv sync` to automatically construct a virtual environment `.venv` and install required packages defined in [pyproject.toml](pyproject.toml):
```bash
uv sync
```

### 2. Configure Environment Variables
Create a `.env` file in the project root following the format in [.env.example](.env.example):
```env
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=24f2004142-t22026
WANDB_ENTITY=varunagnihotri
KAGGLE_API_TOKEN=your_kaggle_api_token
```

---

## Execution 🚀

Use the entrypoint module [src/local/__main__.py](src/local/__main__.py) to execute training, local predictions, or logit exports.

### Train Local Model
Trains [BiLSTMScorer](src/local/model.py#L24-L71), saves checkpoints and retriever states in `outputs/local/`, and runs logging to Weights & Biases:
```bash
uv run python -m src.local --mode train
```

### Predict Local Model
Executes local test-set inference and exports predicted CSV outputs:
```bash
uv run python -m src.local --mode predict
```

### Export Probabilities
Dumps prediction probability numpy arrays (used for stacking/ensembling):
```bash
uv run python -m src.local --mode export_probs
```

### Auto-Deploy to Kaggle
The deployment script packages model parameters (`local_model_best.pt`, `vocab.pkl`, `retriever.pkl`), zips source files, creates or registers model instances on the Kaggle Models API, and pushes the notebook runner via the Kaggle API:
```bash
uv run scripts/deploy_to_kaggle.py
```

---

## Developer 👨‍💻

- **Name:** Varun Agnihotri
- **GitHub:** [PythonicVarun](https://github.com/PythonicVarun)
