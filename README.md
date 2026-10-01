# Fake News Detection Using Machine Learning

Text-classification project that predicts whether a news article is real or fake from its text — built as the final project for **Machine Learning** (M.S. Computer Science, University of Central Missouri).

## The problem

Fake news spreads fast and erodes trust in media and institutions. This project builds and compares supervised ML classifiers that flag misinformation from article text, helping prioritize content for human review.

## Dataset

[Fake News Detection dataset](https://www.kaggle.com/datasets) (Ahmed, Traore & Saad, 2017) — **21,417 real** and **23,481 fake** news articles (~45K total), split 80/20 for train/test.

## Pipeline

1. **Text cleaning** — punctuation and stopword removal
2. **Tokenization** — splitting text into meaningful units
3. **Feature extraction** — TF-IDF vectorization to quantify word importance
4. **Modeling** — trained and compared four classifiers with scikit-learn:
   - Decision Tree (CART)
   - Logistic Regression (decision threshold tuned to 0.6)
   - Random Forest (majority-vote ensemble)
   - Support Vector Machine
5. **Evaluation** — accuracy on the held-out 20% test set

## Results

In the original course project, run on the full ~45K-article dataset:

| Model | Test accuracy |
|---|---|
| Logistic Regression (tuned) | **~94%** |
| Random Forest | strong baseline |
| Decision Tree | baseline |
| SVM | evaluated |

Logistic Regression with a tuned 0.6 decision threshold performed best at ~94% accuracy on the held-out 20% test set, as reported in the course presentation.

> **Note on reproducibility:** the Kaggle dataset isn't included in this repo. The notebook loads `Fake.csv` / `True.csv` from `data/` if you add them; otherwise it runs a clearly-labeled synthetic sample so the full pipeline (cleaning → TF-IDF → modeling → evaluation) still executes end-to-end. The synthetic sample is trivially separable, so its scores are a pipeline demo — not comparable to the ~94% course result above.

## Future work

- Fine-tuned transformer models (BERT/GPT) for deeper language understanding
- Larger, more diverse training data across writing styles and topics
- Real-time detection for social-media streams

## Tech stack

Python · scikit-learn · pandas · NumPy · matplotlib

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/fake_news_detection.ipynb
```

Download the Kaggle dataset into `data/` first (see notebook for the expected filename).

## Project structure

```
├── notebooks/fake_news_detection.ipynb  # full pipeline: EDA → preprocessing → modeling → evaluation
├── src/preprocessing.py                 # text cleaning / tokenization / TF-IDF helpers
├── requirements.txt
└── data/                                # dataset goes here (not committed)
```

## Team

Built by a team of 4 (Pavankalyan Prasadam, Ganesh Kumar Korra, Lenin Varma Nallapu, Anil Kumar Rayavarapu) as the ML final project.
