# Hate Speech Detector

[![Quality checks](https://github.com/Jithendra-coder/Hate-Detection-System/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/Jithendra-coder/Hate-Detection-System/actions/workflows/quality.yml)

A local Chrome extension and Flask API for user-triggered screening of page text with an LSTM classifier. The output is advisory, not a moderation decision. Expect false positives and false negatives, especially with short, contextual, or non-English text.

## Local setup

1. Download the model archive from [Google Drive](https://drive.google.com/file/d/18X6Ee_BfQC-kVhbBsLCSSKqOSjclI0zJ/view?usp=drive_link) and place `best_model.keras` in `Hate/backend/models/`.
2. Place the matching `tokenizer.pkl` in `Hate/`. The checked-in `Hate/labels.pkl` is used by the API.
3. Install `Hate/training/requirements.txt`, then run `python Hate/backend/app.py` from the repository root. The API listens on `http://localhost:8000`.
4. Load the `Hate/extension` directory as an unpacked extension from `chrome://extensions`.

The extension asks for access to all sites so its content script can work on arbitrary pages. It does not scan automatically by default; use the extension’s **Scan** button, or explicitly enable automatic scanning in its stored settings. Scanned page text is sent to the local API at `localhost:8000`. Feedback text is appended to `Hate/backend/feedback.jsonl`.

## Train a model

Run `python Hate/training/train_model.py --dataset Hate/training/hate_speech_dataset.csv`. The trainer writes the serving model and tokenizer to the paths used by the API. The dataset is included in this repository.

## Dataset and model status

The included `Hate/training/hate_speech_dataset.csv` matches the upstream `data/labeled_data.csv` from [Davidson et al.’s hate-speech and offensive-language dataset](https://github.com/t-davidson/hate-speech-and-offensive-language) byte-for-byte. The upstream repository is MIT-licensed; retain its attribution and license when redistributing the dataset. See the [2017 paper](https://doi.org/10.1145/2998181.2998183).

The dataset labels class `0` as hate speech, class `1` as offensive language, and class `2` as neither. The training code now maps class `0` to `HATE` and classes `1` and `2` to `NOT_HATE`; synthetic hate examples use the same positive label.

The checked-in training report records 94.26% accuracy but 0 precision, 0 recall, and 0 F1, so it does not demonstrate a useful classifier. The linked Google Drive model has not been regenerated and evaluated with the corrected label mapping. Treat it as an unvalidated prototype and do not use predictions for consequential decisions.
