# Smart Product Recommendation System

1st Year • Problem 01 — Machine Learning product recommendations, as a **Streamlit web app** ready for Streamlit Community Cloud.

The app cleans historical customer–product data, trains a Random Forest to predict purchase likelihood, evaluates the model, then ranks the catalogue and shows a **personalized Top 5**.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first launch trains the model (about a minute). Later reloads reuse the cached pipeline.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick the repo and branch.
4. Set **Main file path** to `streamlit_app.py`.
5. Deploy. Python version comes from `runtime.txt` (`python-3.12`).

No secrets or extra config are required. Historical CSVs in `data/` ship with the repo so the cloud app can train on first boot.

## App pages

| Tab | What judges see |
| --- | --- |
| Recommend | Customer ID or new-customer details → Top 5 cards, probability chart, CSV download |
| Data pipeline | Cleaning stats, missing/duplicate handling, sample rows, charts |
| Model performance | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC, feature importance |
| How it works | End-to-end pipeline explanation |

## CLI (optional)

```bash
python main.py --customer C001 --no-interactive
python main.py --new --category Sports --purchases 4 --budget 1800 --no-interactive
```
