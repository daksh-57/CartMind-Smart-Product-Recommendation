"""Train and evaluate a classifier that predicts purchase likelihood."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import FEATURE_COLUMNS, FeatureEngineer, model_matrix
from .preprocess import clean_interactions
from .recommend import CustomerProfile, profile_from_history, profile_from_details, recommend_by_popularity, recommend_top_n

NUMERIC_FEATURES = FEATURE_COLUMNS
CATEGORICAL_FEATURES = ["product_category"]


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=2,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train_and_evaluate(raw: pd.DataFrame) -> tuple[Pipeline, dict, pd.DataFrame, FeatureEngineer]:
    cleaned = clean_interactions(raw)

    # Split into train/test BEFORE feature engineering to prevent leakage.
    train_df, test_df = train_test_split(
        cleaned,
        test_size=0.2,
        random_state=42,
        stratify=cleaned["will_purchase"] if "will_purchase" in cleaned.columns else None,
    )

    # Fit FeatureEngineer on training data only
    feature_engineer = FeatureEngineer()
    featured_train = feature_engineer.fit_transform(train_df)
    featured_test = feature_engineer.transform(test_df)

    X_train, y_train = model_matrix(featured_train)
    X_test, y_test = model_matrix(featured_test)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    feature_names = list(NUMERIC_FEATURES) + list(
        encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    )
    importances = pipeline.named_steps["model"].feature_importances_
    importance_rows = sorted(
        [
            {"feature": name, "importance": float(score)}
            for name, score in zip(feature_names, importances)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )[:12]

    # Compute recommendation metrics on test set
    rec_metrics = _compute_recommendation_metrics(pipeline, feature_engineer, featured_train, featured_test, test_df)

    metrics = {
        "rows_raw": int(len(raw)),
        "rows_cleaned": int(len(cleaned)),
        "duplicates_removed": int(cleaned.attrs.get("duplicates_removed", 0)),
        "missing_cells": int(cleaned.attrs.get("missing_cells", 0)),
        "n_customers": int(featured_train["customer_id"].nunique()),
        "n_products": int(featured_train["product_id"].nunique()),
        "positive_rate": float(y_train.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "roc_fpr": [float(v) for v in fpr],
        "roc_tpr": [float(v) for v in tpr],
        "feature_importance": importance_rows,
        **rec_metrics,
    }
    return pipeline, metrics, featured_train, feature_engineer


def _compute_recommendation_metrics(
    pipeline: Pipeline,
    feature_engineer: FeatureEngineer,
    featured_train: pd.DataFrame,
    featured_test: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Compute recommendation-specific metrics: Precision@5, Recall@5, NDCG@5, Catalog Coverage."""
    from collections import defaultdict
    
    # Get all unique customers in test set
    test_customers = test_df["customer_id"].unique()
    
    # Collect test purchases for each customer
    test_purchases = defaultdict(set)
    for cust_id in test_customers:
        cust_test = test_df[test_df["customer_id"] == cust_id]
        purchased = cust_test[cust_test["will_purchase"] == 1]["product_id"].tolist()
        test_purchases[cust_id] = set(purchased)
    
    # Get full catalog for coverage calculation
    catalog = featured_test[["product_id", "product_name", "product_category", "product_price"]].drop_duplicates()
    total_catalog_size = catalog["product_id"].nunique()
    
    # ML recommendations
    ml_precision_at_5 = []
    ml_recall_at_5 = []
    ml_ndcg_at_5 = []
    ml_recommended_products = set()
    
    # Popularity baseline recommendations
    pop_precision_at_5 = []
    pop_recall_at_5 = []
    pop_ndcg_at_5 = []
    pop_recommended_products = set()
    
    for cust_id in test_customers:
        try:
            # Create profile for this customer
            if cust_id in feature_engineer._customer_avg_price:
                profile = profile_from_history(cust_id, featured_train)
            else:
                # Use new customer defaults
                profile = profile_from_details(
                    previous_purchases=0,
                    favourite_category=feature_engineer.global_fallback_category,
                    budget=feature_engineer.global_median_price,
                    featured=featured_train,
                    feature_engineer=feature_engineer,
                )
            
            # Get recommended products from ML model
            ml_recs = recommend_top_n(pipeline, profile, catalog, featured_train, top_n=5)
            ml_rec_ids = set(ml_recs["product_id"].tolist())
            ml_recommended_products.update(ml_rec_ids)
            
            # Get recommended products from popularity baseline
            pop_recs = recommend_by_popularity(featured_train, top_n=5)
            pop_rec_ids = set(pop_recs["product_id"].tolist())
            pop_recommended_products.update(pop_rec_ids)
            
            # Relevant items for this customer
            relevant = test_purchases.get(cust_id, set())
            
            if relevant:
                # Precision@5
                ml_precision_at_5.append(len(ml_rec_ids & relevant) / 5)
                pop_precision_at_5.append(len(pop_rec_ids & relevant) / 5)

                # Recall@5
                ml_recall_at_5.append(len(ml_rec_ids & relevant) / len(relevant))
                pop_recall_at_5.append(len(pop_rec_ids & relevant) / len(relevant))

                # NDCG@5
                ml_ndcg_at_5.append(_ndcg_at_k(ml_rec_ids, relevant, k=5))
                pop_ndcg_at_5.append(_ndcg_at_k(pop_rec_ids, relevant, k=5))
        except Exception as e:
            print(f"Warning: Failed to compute metrics for customer {cust_id}: {e}")
            continue
    
    # Catalog coverage
    ml_coverage = len(ml_recommended_products) / total_catalog_size if total_catalog_size > 0 else 0
    pop_coverage = len(pop_recommended_products) / total_catalog_size if total_catalog_size > 0 else 0
    
    print(f"Computing recommendation metrics: {len(ml_precision_at_5)} customers with recommendations")
    return {
        "precision_at_5_ml": float(np.mean(ml_precision_at_5)) if ml_precision_at_5 else 0.0,
        "recall_at_5_ml": float(np.mean(ml_recall_at_5)) if ml_recall_at_5 else 0.0,
        "ndcg_at_5_ml": float(np.mean(ml_ndcg_at_5)) if ml_ndcg_at_5 else 0.0,
        "catalog_coverage_ml": float(ml_coverage),
        "precision_at_5_baseline": float(np.mean(pop_precision_at_5)) if pop_precision_at_5 else 0.0,
        "recall_at_5_baseline": float(np.mean(pop_recall_at_5)) if pop_recall_at_5 else 0.0,
        "ndcg_at_5_baseline": float(np.mean(pop_ndcg_at_5)) if pop_ndcg_at_5 else 0.0,
        "catalog_coverage_baseline": float(pop_coverage),
        "n_test_customers": len(test_customers),
    }


def _ndcg_at_k(recommended: set, relevant: set, k: int) -> float:
    """Calculate NDCG@k for a single customer."""
    if not relevant:
        return 0.0
    
    # DCG@k
    dcg = 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0
    
    # Ideal DCG@k (all relevant items at top positions)
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def save_artifacts(
    pipeline: Pipeline,
    featured: pd.DataFrame,
    metrics: dict,
    artifacts_dir: Path,
    feature_engineer: Optional[FeatureEngineer] = None,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifacts_dir / "purchase_model.joblib")
    featured.to_csv(artifacts_dir / "featured_interactions.csv", index=False)
    joblib.dump(metrics, artifacts_dir / "metrics.joblib")
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if feature_engineer is not None:
        joblib.dump(feature_engineer, artifacts_dir / "feature_engineer.joblib")
