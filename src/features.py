"""Feature engineering for purchase-likelihood prediction.

The purchase count of the *current* product is used only as the label.
Using it as an input feature would leak the answer to the model.

FeatureEngineer follows a fit/transform pattern to prevent data leakage.
All aggregate statistics are learned from training data only and applied
to any dataset via transform().
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd


class FeatureEngineer:
    """Fit on training data, transform any dataset using training statistics.

    All aggregate statistics (customer-level and product-level) are computed
    from the training split only.  Unseen customers or products receive
    sensible global fallback values learned during fit().

    Key anti-leakage guarantees:
    - Customer stats (avg_price, total_views, etc.) are computed only on training data
    - Product popularity is computed from training 'times_purchased' only
    - Global thresholds (median_price, etc.) come from training only
    """

    def __init__(self) -> None:
        # Learned customer-level statistics as dicts for fast lookup
        self._customer_avg_price: dict[str, float] = {}
        self._customer_total_views: dict[str, float] = {}
        self._customer_total_carts: dict[str, float] = {}
        self._customer_product_count: dict[str, int] = {}
        self._favourite_category: dict[str, str] = {}

        # Learned product-level statistics as dicts for fast lookup
        self._product_avg_views: dict[str, float] = {}
        self._product_avg_carts: dict[str, float] = {}
        self._product_popularity: dict[str, float] = {}

        # Global fallback values (learned from training data)
        self._global_median_price: float = 0.0
        self._global_median_views: float = 0.0
        self._global_median_carts: float = 0.0
        self._global_purchase_rate: float = 0.0
        self._global_fallback_category: str = "Unknown"
        self._global_avg_customer_product_count: float = 0.0

        self._fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Learn all statistics from *training* data only."""
        if df.empty:
            raise ValueError("Cannot fit FeatureEngineer on empty DataFrame")

        # --- Customer-level aggregates ---
        cust = (
            df.groupby("customer_id")
            .agg(
                customer_avg_price=("product_price", "mean"),
                customer_total_views=("times_viewed", "sum"),
                customer_total_carts=("times_added_to_cart", "sum"),
                customer_product_count=("product_id", "nunique"),
            )
            .reset_index()
        )
        self._customer_avg_price = dict(zip(cust["customer_id"], cust["customer_avg_price"]))
        self._customer_total_views = dict(zip(cust["customer_id"], cust["customer_total_views"]))
        self._customer_total_carts = dict(zip(cust["customer_id"], cust["customer_total_carts"]))
        self._customer_product_count = dict(zip(cust["customer_id"], cust["customer_product_count"]))

        # Favourite category per customer (most frequent category in training)
        fav = (
            df.groupby(["customer_id", "product_category"])
            .size()
            .reset_index(name="cat_count")
            .sort_values(["customer_id", "cat_count"], ascending=[True, False])
            .drop_duplicates("customer_id")
            .rename(columns={"product_category": "favourite_category"})
            [["customer_id", "favourite_category"]]
        )
        self._favourite_category = dict(zip(fav["customer_id"], fav["favourite_category"]))

        # --- Product-level aggregates ---
        prod = (
            df.groupby("product_id")
            .agg(
                product_avg_views=("times_viewed", "mean"),
                product_avg_carts=("times_added_to_cart", "mean"),
                product_popularity=("times_purchased", "mean"),
            )
            .reset_index()
        )
        self._product_avg_views = dict(zip(prod["product_id"], prod["product_avg_views"]))
        self._product_avg_carts = dict(zip(prod["product_id"], prod["product_avg_carts"]))
        self._product_popularity = dict(zip(prod["product_id"], prod["product_popularity"]))

        # --- Global fallbacks ---
        self._global_median_price = float(df["product_price"].median())
        self._global_median_views = float(df["times_viewed"].median())
        self._global_median_carts = float(df["times_added_to_cart"].median())
        self._global_purchase_rate = float(df["times_purchased"].mean())
        self._global_avg_customer_product_count = float(cust["customer_product_count"].mean())

        # Fallback category = most common category in training
        if not df["product_category"].mode().empty:
            self._global_fallback_category = str(df["product_category"].mode().iloc[0])
        else:
            self._global_fallback_category = "Unknown"

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned statistics to produce engineered features.

        The input DataFrame must contain the raw columns:
        customer_id, product_id, product_category, product_price,
        times_viewed, times_added_to_cart, previous_purchases

        Returns a DataFrame with all engineered features plus original columns.
        """
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform()")

        data = df.copy()

        # --- Derived features (row-wise only - no leakage) ---
        data["view_to_cart_rate"] = data["times_added_to_cart"] / data["times_viewed"].clip(lower=1)
        data["engagement_score"] = (
            0.4 * data["times_viewed"] + 1.2 * data["times_added_to_cart"]
        )
        data["is_expensive"] = (data["product_price"] > self._global_median_price).astype(int)
        data["price_log"] = (data["product_price"] + 1).apply(lambda x: math.log(x))

        # --- Customer-level features (with fallback for unseen customers) ---
        data["customer_avg_price"] = data["customer_id"].map(self._customer_avg_price).fillna(self._global_median_price)
        data["customer_total_views"] = data["customer_id"].map(self._customer_total_views).fillna(0.0)
        data["customer_total_carts"] = data["customer_id"].map(self._customer_total_carts).fillna(0.0)
        data["customer_product_count"] = data["customer_id"].map(self._customer_product_count).fillna(self._global_avg_customer_product_count).astype(int)

        # Favourite category and match indicator
        data["favourite_category"] = data["customer_id"].map(self._favourite_category).fillna(self._global_fallback_category)
        data["matches_favourite_category"] = (
            data["product_category"] == data["favourite_category"]
        ).astype(int)

        # --- Product-level features (with fallback for unseen products) ---
        data["product_avg_views"] = data["product_id"].map(self._product_avg_views).fillna(self._global_median_views)
        data["product_avg_carts"] = data["product_id"].map(self._product_avg_carts).fillna(self._global_median_carts)
        data["product_popularity"] = data["product_id"].map(self._product_popularity).fillna(self._global_purchase_rate)

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Properties for recommendation pipeline
    # ------------------------------------------------------------------
    @property
    def global_median_price(self) -> float:
        return self._global_median_price

    @property
    def global_median_views(self) -> float:
        return self._global_median_views

    @property
    def global_median_carts(self) -> float:
        return self._global_median_carts

    @property
    def global_purchase_rate(self) -> float:
        return self._global_purchase_rate

    @property
    def global_fallback_category(self) -> str:
        return self._global_fallback_category

    @property
    def global_avg_customer_product_count(self) -> float:
        return self._global_avg_customer_product_count


# Feature columns used by the model (must match what transform() produces)
FEATURE_COLUMNS = [
    "product_price",
    "price_log",
    "times_viewed",
    "times_added_to_cart",
    "previous_purchases",
    "view_to_cart_rate",
    "engagement_score",
    "is_expensive",
    "customer_avg_price",
    "customer_total_views",
    "customer_total_carts",
    "customer_product_count",
    "matches_favourite_category",
    "product_avg_views",
    "product_avg_carts",
    "product_popularity",
]


def model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y from a featured DataFrame."""
    X = df[FEATURE_COLUMNS + ["product_category"]].copy()
    y = df["will_purchase"].copy()
    return X, y
