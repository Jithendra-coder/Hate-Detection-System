# Hate Speech Detector

A local Chrome extension and Flask API that send selected page text to an LSTM classifier and highlight likely matches. The model is a screening aid, not a moderation decision: expect false positives and false negatives, especially with short, contextual, or non-English text.

## Local setup

1. Download the model archive from [Google Drive](https://drive.google.com/file/d/18X6Ee_BfQC-kVhbBsLCSSKqOSjclI0zJ/view?usp=drive_link) and place `best_model.keras` in `Hate/backend/models/`.
2. Place the matching `tokenizer.pkl` in `Hate/`. The checked-in `Hate/labels.pkl` is used by the API.
3. Install `Hate/training/requirements.txt`, then run `python Hate/backend/app.py` from the repository root. The API listens on `http://localhost:8000`.
4. Load the `Hate/extension` directory as an unpacked extension from `chrome://extensions`.

The extension asks for access to all sites so its content script can work on arbitrary pages. It does not scan automatically by default; use the extension’s **Scan** button, or explicitly enable automatic scanning in its stored settings. Submitted text is sent to the local API at `localhost:8000`; feedback text is appended to `Hate/backend/feedback.jsonl`.

## Train a model

Run `python Hate/training/train_model.py --dataset path/to/hate_speech_dataset.csv`. The trainer writes the serving model and tokenizer to the paths used by the API. Training data is not included in this repository.
