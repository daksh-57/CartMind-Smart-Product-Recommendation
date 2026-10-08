"""Rank products by predicted purchase probability for a customer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .features import FEATURE_COLUMNS

if TYPE_CHECKING:
    from .features import FeatureEngineer


@dataclass
class CustomerProfile:
    customer_id: str
    previous_purchases: int
    favourite_category: str
    customer_avg_price: float
    customer_total_views: float
    customer_total_carts: float
    customer_product_count: int
    known_interactions: pd.DataFrame


def profile_from_history(customer_id: str, featured: pd.DataFrame) -> CustomerProfile:
    history = featured[featured["customer_id"] == customer_id]
    if history.empty:
        raise KeyError(f"Unknown customer ID: {customer_id}")

    favourite = history["favourite_category"].mode()
    favourite_category = favourite.iloc[0] if not favourite.empty else history["product_category"].mode().iloc[0]
    return CustomerProfile(
        customer_id=customer_id,
        previous_purchases=int(history["previous_purchases"].median()),
        favourite_category=str(favourite_category),
        customer_avg_price=float(history["customer_avg_price"].iloc[0]),
        customer_total_views=float(history["customer_total_views"].iloc[0]),
        customer_total_carts=float(history["customer_total_carts"].iloc[0]),
        customer_product_count=int(history["customer_product_count"].iloc[0]),
        known_interactions=history,
    )


def profile_from_details(
    previous_purchases: int,
    favourite_category: str,
    budget: float,
    featured: pd.DataFrame,
    feature_engineer: Optional["FeatureEngineer"] = None,
) -> CustomerProfile:
    median_price = float(featured["product_price"].median())
    
    # Use learned defaults from feature_engineer if available, otherwise fall back to training stats
    if feature_engineer is not None and feature_engineer._fitted:
        default_views = float(feature_engineer.global_median_views)
        default_carts = float(feature_engineer.global_median_carts)
        default_count = float(feature_engineer.global_avg_customer_product_count)
    else:
        # Compute from featured data
        default_views = float(featured["times_viewed"].median()) if "times_viewed" in featured.columns else 5.0
        default_carts = float(featured["times_added_to_cart"].median()) if "times_added_to_cart" in featured.columns else 2.0
        default_count = float(featured["customer_product_count"].median()) if "customer_product_count" in featured.columns else 3.0
    
    return CustomerProfile(
        customer_id="NEW_CUSTOMER",
        previous_purchases=int(max(previous_purchases, 0)),
        favourite_category=favourite_category,
        customer_avg_price=float(budget if budget > 0 else median_price),
        customer_total_views=default_views,
        customer_total_carts=default_carts,
        customer_product_count=int(default_count),
        known_interactions=featured.iloc[0:0].copy(),
    )


def _candidate_frame(profile: CustomerProfile, catalog: pd.DataFrame, featured: pd.DataFrame) -> pd.DataFrame:
    product_stats = featured.groupby("product_id", as_index=False).agg(
        product_avg_views=("product_avg_views", "mean"),
        product_avg_carts=("product_avg_carts", "mean"),
        product_popularity=("product_popularity", "mean"),
    )
    merged = catalog.merge(product_stats, on="product_id", how="left")
    for col, default in [
        ("product_avg_views", featured["times_viewed"].median()),
        ("product_avg_carts", featured["times_added_to_cart"].median()),
        ("product_popularity", featured["will_purchase"].mean()),
    ]:
        merged[col] = merged[col].fillna(default)

    already_bought = set()
    if not profile.known_interactions.empty:
        bought = profile.known_interactions[profile.known_interactions["will_purchase"] == 1]
        already_bought = set(bought["product_id"].tolist())

    candidates = merged[~merged["product_id"].isin(already_bought)].copy()
    if candidates.empty:
        candidates = merged.copy()

    # Build interaction lookup - only use actual observed interactions
    interaction_lookup = {}
    if not profile.known_interactions.empty:
        interaction_lookup = (
            profile.known_interactions.sort_values("times_viewed")
            .drop_duplicates("product_id", keep="last")
            .set_index("product_id")[["times_viewed", "times_added_to_cart"]]
            .to_dict("index")
        )

    # Use actual views/carts from history where available, otherwise use product averages directly
    # NO arbitrary multipliers - let the model learn the importance
    views = []
    carts = []
    for product_id in candidates["product_id"]:
        known = interaction_lookup.get(product_id)
        if known:
            views.append(int(known["times_viewed"]))
            carts.append(int(known["times_added_to_cart"]))
        else:
            # Use raw product statistics (no heuristic multipliers)
            views.append(int(candidates[candidates["product_id"] == product_id]["product_avg_views"].iloc[0]))
            carts.append(int(candidates[candidates["product_id"] == product_id]["product_avg_carts"].iloc[0]))

    candidates["times_viewed"] = views
    candidates["times_added_to_cart"] = carts
    candidates["previous_purchases"] = profile.previous_purchases
    candidates["view_to_cart_rate"] = candidates["times_added_to_cart"] / candidates["times_viewed"].clip(lower=1)
    candidates["engagement_score"] = 0.4 * candidates["times_viewed"] + 1.2 * candidates["times_added_to_cart"]
    candidates["price_log"] = np.log(candidates["product_price"] + 1)
    candidates["is_expensive"] = (
        candidates["product_price"] > featured["product_price"].median()
    ).astype(int)
    candidates["customer_avg_price"] = profile.customer_avg_price
    candidates["customer_total_views"] = profile.customer_total_views
    candidates["customer_total_carts"] = profile.customer_total_carts
    candidates["customer_product_count"] = profile.customer_product_count
    candidates["matches_favourite_category"] = (
        candidates["product_category"] == profile.favourite_category
    ).astype(int)
    return candidates


def generate_explanation(
    row: pd.Series,
    profile: CustomerProfile,
    featured: pd.DataFrame,
) -> str:
    """Generate a deterministic explanation for a recommendation based on available features."""
    reasons = []
    
    # Check engagement signals
    if row["times_viewed"] >= featured["times_viewed"].median() * 1.2:
        reasons.append("Strong customer engagement")
    elif row["times_added_to_cart"] >= featured["times_added_to_cart"].median():
        reasons.append("High cart interest")
    
    # Check category match
    if row["matches_favourite_category"] == 1:
        reasons.append("Matches your preferred category")
    
    # Check popularity
    if row["product_popularity"] >= featured["product_popularity"].median():
        reasons.append("Popular product in this category")
    
    # Check price fit
    if abs(row["product_price"] - profile.customer_avg_price) < profile.customer_avg_price * 0.5:
        reasons.append("Fits your budget range")
    
    # Check price relative to median
    if row["product_price"] <= featured["product_price"].median() * 0.7:
        reasons.append("Affordable option")
    
    # Default explanation if none of the above
    if not reasons:
        reasons.append("High predicted purchase likelihood")
    
    return "; ".join(reasons[:2])  # Limit to 2 explanations max


def recommend_top_n(
    model: Pipeline,
    profile: CustomerProfile,
    catalog: pd.DataFrame,
    featured: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    candidates = _candidate_frame(profile, catalog, featured)
    X = candidates[FEATURE_COLUMNS + ["product_category"]]
    probabilities = model.predict_proba(X)[:, 1]
    ranked = candidates.copy()
    ranked["purchase_probability"] = probabilities
    ranked = ranked.sort_values("purchase_probability", ascending=False)
    top_recommendations = ranked.head(top_n)
    
    # Add explanations
    explanations = []
    for _, row in top_recommendations.iterrows():
        explanations.append(generate_explanation(row, profile, featured))
    top_recommendations["explanation"] = explanations
    
    return top_recommendations[
        [
            "product_id",
            "product_name",
            "product_category",
            "product_price",
            "purchase_probability",
            "explanation",
        ]
    ].reset_index(drop=True)


def recommend_by_popularity(
    featured: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Popularity baseline: recommend top products based on training data purchase rate only.
    
    This is a simple non-ML baseline that does NOT use test-set information.
    """
    # Compute product popularity from TRAINING data only
    product_pop = featured.groupby("product_id").agg(
        product_name=("product_name", "first"),
        product_category=("product_category", "first"),
        product_price=("product_price", "first"),
        popularity=("will_purchase", "mean"),
    ).sort_values("popularity", ascending=False)
    
    top_products = product_pop.head(top_n)
    
    return top_products.reset_index()[
        [
            "product_id",
            "product_name",
            "product_category",
            "product_price",
            "popularity",
        ]
    ].rename(columns={"popularity": "purchase_probability"})
