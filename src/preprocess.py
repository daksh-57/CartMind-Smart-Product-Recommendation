"""Clean historical interaction data: missing values, duplicates, invalid counts."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "customer_id",
    "product_id",
    "product_category",
    "product_price",
    "times_viewed",
    "times_added_to_cart",
    "times_purchased",
    "previous_purchases",
]


def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    missing_cells = int(data.isna().sum().sum())
    before = len(data)
    data = data.drop_duplicates()
    duplicates_removed = before - len(data)

    data["customer_id"] = data["customer_id"].astype(str).str.strip()
    data["product_id"] = data["product_id"].astype(str).str.strip()
    if "product_name" in data.columns:
        data["product_name"] = data["product_name"].astype(str).str.strip()

    numeric_cols = [
        "product_price",
        "times_viewed",
        "times_added_to_cart",
        "times_purchased",
        "previous_purchases",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Category: fill with the product's most common category, else overall mode.
    product_category_mode = (
        data.dropna(subset=["product_category"])
        .groupby("product_id")["product_category"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
    )
    data["product_category"] = data["product_category"].fillna(
        data["product_id"].map(product_category_mode)
    )
    if data["product_category"].isna().any():
        overall = data["product_category"].mode()
        fill_value = overall.iloc[0] if not overall.empty else "Unknown"
        data["product_category"] = data["product_category"].fillna(fill_value)

    # Numeric: median by product, then global median.
    for col in ["product_price", "times_viewed", "times_added_to_cart", "previous_purchases"]:
        data[col] = data[col].fillna(data.groupby("product_id")[col].transform("median"))
        data[col] = data[col].fillna(data[col].median())

    data["times_purchased"] = data["times_purchased"].fillna(0)

    # Create purchase label: has the customer purchased this product?
    data["will_purchase"] = (data["times_purchased"] > 0).astype(int)

    data = data.dropna(subset=["customer_id", "product_id"])
    data = data[(data["product_price"] > 0) & (data["times_viewed"] >= 0)]
    data["times_viewed"] = data["times_viewed"].clip(lower=0).round().astype(int)
    data["times_added_to_cart"] = data["times_added_to_cart"].clip(lower=0).round().astype(int)
    data["times_purchased"] = data["times_purchased"].clip(lower=0).round().astype(int)
    data["previous_purchases"] = data["previous_purchases"].clip(lower=0).round().astype(int)

    # Cart adds cannot exceed views in this simplified model of browsing.
    data["times_added_to_cart"] = data[["times_added_to_cart", "times_viewed"]].min(axis=1)

    data.attrs["duplicates_removed"] = duplicates_removed
    data.attrs["missing_cells"] = missing_cells
    return data.reset_index(drop=True)
