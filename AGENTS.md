# AGENTS.md

## Project overview

This is an educational OCR project for CAPTCHA recognition.

Final deliverable: Jupyter notebook.

## Main project requirements

Required architecture:
- FCNN
- Bi-LSTM
- Cross-Entropy loss

Do not use:
- torchvision.models
- pretrained networks

## Required notebook structure

The notebook must contain the following sections:

1. Data preparation
2. Model creation and training
3. CER evaluation
4. Error analysis
5. Conclusions

## Data preparation requirements

Dataset must:
- inherit torch.utils.data.Dataset
- load images with Pillow
- extract labels from filenames

Train/test split:
- 80/20
- reproducible random seed

## Metrics

Required metric:
- Character Error Rate (CER)

## Error analysis

Show worst predictions and discuss:
- possible causes
- possible improvements

## Codex working rules

Before coding:

1. Inspect existing files.
2. Explain plan briefly.
3. Make minimal changes.
4. Run checks when possible.
5. Summarize changes.
