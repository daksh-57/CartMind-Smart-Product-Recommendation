"""Tests for the FeatureEngineer class with anti-leakage guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, FeatureEngineer, model_matrix


@pytest.fixture
def sample_train_data() -> pd.DataFrame:
    """Create a small training dataset."""
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C002", "C002", "C003"],
            "product_id": ["P001", "P002", "P001", "P003", "P002"],
            "product_category": ["Electronics", "Clothing", "Electronics", "Home", "Clothing"],
            "product_price": [5000.0, 1500.0, 5000.0, 3000.0, 1500.0],
            "times_viewed": [10, 5, 8, 12, 3],
            "times_added_to_cart": [3, 2, 2, 4, 1],
            "times_purchased": [1, 0, 0, 1, 0],
            "previous_purchases": [2, 1, 2, 5, 1],
        }
    )


@pytest.fixture
def sample_test_data() -> pd.DataFrame:
    """Create test data with some unseen customers and products."""
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C999", "C003", "C999"],
            "product_id": ["P001", "P999", "P999", "P999"],  # P999 is unseen
            "product_category": ["Electronics", "Clothing", "Electronics", "Sports"],
            "product_price": [6000.0, 2000.0, 4000.0, 1000.0],
            "times_viewed": [7, 15, 6, 20],
            "times_added_to_cart": [2, 5, 1, 8],
            "times_purchased": [0, 1, 0, 0],
            "previous_purchases": [3, 0, 3, 0],
        }
    )


def test_feature_engineer_initialization() -> None:
    """FeatureEngineer should initialize with default values."""
    fe = FeatureEngineer()
    assert not fe._fitted
    assert fe.global_median_price == 0.0
    assert fe.global_fallback_category == "Unknown"


def test_feature_engineer_fit(sample_train_data: pd.DataFrame) -> None:
    """Fit should learn statistics from training data."""
    fe = FeatureEngineer()
    result = fe.fit(sample_train_data)

    assert result is fe
    assert fe._fitted
    assert fe.global_median_price == 3000.0
    assert fe.global_purchase_rate == pytest.approx(0.4)


def test_feature_engineer_fit_empty_raises() -> None:
    """Fit should raise on empty DataFrame."""
    fe = FeatureEngineer()
    with pytest.raises(ValueError, match="empty DataFrame"):
        fe.fit(pd.DataFrame())


def test_feature_engineer_transform_before_fit_raises() -> None:
    """Transform should raise if called before fit."""
    fe = FeatureEngineer()
    df = pd.DataFrame({"customer_id": ["C001"], "product_id": ["P001"]})
    with pytest.raises(RuntimeError, match="must be fitted"):
        fe.transform(df)


def test_feature_engineer_transform_adds_features(sample_train_data: pd.DataFrame) -> None:
    """Transform should add all engineered features."""
    fe = FeatureEngineer()
    featured = fe.fit_transform(sample_train_data)

    expected_cols = {
        "view_to_cart_rate",
        "engagement_score",
        "is_expensive",
        "price_log",
        "customer_avg_price",
        "customer_total_views",
        "customer_total_carts",
        "customer_product_count",
        "matches_favourite_category",
        "product_avg_views",
        "product_avg_carts",
        "product_popularity",
        "favourite_category",
    }
    assert expected_cols.issubset(set(featured.columns.tolist()))


def test_feature_engineer_antileakage(sample_train_data: pd.DataFrame, sample_test_data: pd.DataFrame) -> None:
    """Test data should use training statistics, not test statistics."""
    fe = FeatureEngineer()
    featured_train = fe.fit_transform(sample_train_data)
    featured_test = fe.transform(sample_test_data)

    c999_row = featured_test[featured_test["customer_id"] == "C999"].iloc[0]
    # Unseen customer should get fallback values from training
    assert c999_row["customer_avg_price"] == featured_train["product_price"].median()
    assert c999_row["customer_total_views"] == 0.0
    # Unseen product should get global median from training
    assert c999_row["product_avg_views"] == featured_train["times_viewed"].median()
    assert c999_row["product_avg_carts"] == featured_train["times_added_to_cart"].median()
    assert c999_row["product_popularity"] == featured_train["times_purchased"].mean()


def test_feature_engineer_properties(sample_train_data: pd.DataFrame) -> None:
    """FeatureEngineer should expose learned properties."""
    fe = FeatureEngineer().fit(sample_train_data)
    assert isinstance(fe.global_median_price, float)
    assert isinstance(fe.global_median_views, float)
    assert isinstance(fe.global_median_carts, float)
    assert isinstance(fe.global_purchase_rate, float)
    assert isinstance(fe.global_fallback_category, str)


def test_model_matrix_with_label() -> None:
    """model_matrix should return X and y when will_purchase column exists."""
    fe = FeatureEngineer()
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
            "product_id": ["P001", "P002"],
            "product_category": ["Electronics", "Clothing"],
            "product_price": [5000.0, 1500.0],
            "times_viewed": [10, 5],
            "times_added_to_cart": [3, 2],
            "times_purchased": [1, 0],
            "previous_purchases": [2, 1],
        }
    )
    featured = fe.fit_transform(df)
    featured["will_purchase"] = featured["times_purchased"] > 0
    X, y = model_matrix(featured)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(featured)
    assert list(X.columns) == FEATURE_COLUMNS + ["product_category"]
    assert set(y.unique()) <= {0, 1}


def test_feature_columns_consistency(sample_train_data: pd.DataFrame) -> None:
    """FEATURE_COLUMNS should match what transform produces."""
    fe = FeatureEngineer()
    featured = fe.fit_transform(sample_train_data)

    for col in FEATURE_COLUMNS:
        assert col in featured.columns, f"Missing feature column: {col}"
