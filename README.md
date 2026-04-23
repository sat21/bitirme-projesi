# Bitirme Projesi

This repository contains the full graduation-project workspace for tomato disease classification, model training, deployment artifacts, and Android integration.

## Repository Layout

- `shufflenet-v2-tensorflow/`: Training, evaluation, conversion, and deployment scripts.
- `tomatech-android/`: Android app that runs on-device inference.
- `tomato/`: Tomato image dataset used by training workflows.
- `checkpoints_tomato_*`: Saved checkpoints and evaluation outputs for multiple experiments.
- `teknofest_2026_on_degerlendirme/`: Reporting and review materials.

## Quick Start (Linux)

1. Create and activate a virtual environment.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

2. Install project dependencies.

~~~bash
pip install --upgrade pip
pip install -r requirements.txt
~~~

3. Run training or evaluation scripts from the relevant subfolder.

~~~bash
cd shufflenet-v2-tensorflow
python train_tomato_1_5x_aug.py
~~~

## Git Workflow

- Main branch: `main`
- Prefer feature branches for risky changes.
- Use small commits and push often.
- For shared history cleanup, prefer `git revert` over destructive rewrites.

## Large File Policy

This repository intentionally tracks important model outputs and dataset assets required for reproducibility.

The `.gitignore` file blocks local environments, migration backups, and future generated runtime/log/cache artifacts so new commits stay cleaner.

## Current Remote

- GitHub: https://github.com/sat21/bitirme-projesi
