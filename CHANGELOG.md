# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Integrated Weights & Biases (`wandb`) in [train.py](src/local/train.py) for tracking training losses, validation loss, learning rate, and validation MAP@3 metrics.
- Added `python-dotenv` dependency and configured package initialization in [__init__.py](src/local/__init__.py) to load environment variables from `.env` files automatically.
- Added [.env.example](.env.example) configuration template containing key placeholders (`WANDB_API_KEY`, `WANDB_PROJECT`).

## [0.1.0] - 2026-05-30

### Added
- Created the initial MCQ solver codebase under `src/local/` (originally `src/local_model/` in commit [[28d1b1e](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/28d1b1ea2ff26e5caa03e28bdcba5b1317ac2107)]):
  - [model.py](src/local/model.py): Built the PyTorch `BiLSTMScorer` network architecture.
  - [train.py](src/local/train.py): Formulated the model training and evaluation loop.
  - [inference.py](src/local/inference.py): Handled the scoring and predicting steps.
  - [dataset.py](src/local/dataset.py): Configured multiple choice questions dataset loader and collator.
  - [retriever.py](src/local/retriever.py): Standardized document index construction using a TF-IDF retriever.
  - [vocab.py](src/local/vocab.py): Formed vocabulary map helper functions.
  - [config.py](src/local/config.py): Set model hyperparameters and path definitions.
  - [utils.py](src/local/utils.py): Wrote validation scoring utilities (e.g. Mean Average Precision MAP@3).
- Added dataset files under `dataset/` including `train.csv`, `test.csv`, and `sample_submission.csv` in commit [[5f20571](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/5f205710b5791160591724ec3fc80b55379e148b)].
- Configured pre-commit check hooks using [pre-commit](https://pre-commit.com/) in commit [[fe87013](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/fe87013c88a89c821e8062149bb00b6659991b90)].
- Added automated CI check workflow [ci.yml](.github/workflows/ci.yml) for `pyrefly` and `ruff` validation in commit [[e22c171](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/e22c1713c2843c7c9f54d4b7b8cdd6ae8ed562dc)].

### Changed
- Refactored `src/local_model/` module directory name to `src/local/` for clarity and typing ergonomics in commit [[800c676](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/800c6761a25c32b5e7d14dfebc10966afabb5999)].
- Upgraded pre-commit formatting and check setups to replace `isort` and `black` formatters with modern `ruff` check-ins in commit [[61f0ca7](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/61f0ca71da8536faa858a28ddca926061fb2fc35)].
- Renamed CI workflow checking job name from `pyrefly-check` to `check` and optimized authorization flags in commit [[38fb5f8](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/38fb5f8007d10b5eeb6a27fac4e14db1c156bc7c)].
- Formatted and linted existing script code for syntax and style standards in commit [[45c16b9](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/45c16b9f17139f7fdf6a1a72e101e54e51fc674f)].

### Fixed
- Pinned `uv` setup action version to v8.1.0 in automated CI config in commit [[9a0e3db](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/9a0e3db8bd820f65ece5080816d7cab75b06f5be)].
- Corrected project definitions and setup dependency groups inside [pyproject.toml](pyproject.toml) in commit [[420697a](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/420697a6ad18fbc2eff0d3b4328a01baf09b7d44)].

## [0.0.1] - 2026-05-27

### Added
- Initialized the Git repository and configuration boilerplate in commit [[e1093e8](https://github.com/PythonicVarun/GenAI_Project-May_2026/commit/e1093e84d9e32c8b96070d11857b2eb2802a0182)].
- Configured base Python dependencies, `.gitignore`, `.python-version`, and `LICENSE`.
