"""Generate realistic historical customer-product interaction data.

The dataset includes missing values and duplicate rows so the cleaning
pipeline can be demonstrated as required by the problem statement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["Electronics", "Clothing", "Home", "Beauty", "Sports", "Books"]

PRODUCT_NAMES = {
    "Electronics": [
        "Wireless Earbuds",
        "Smartphone",
        "Laptop",
        "Power Bank",
        "Bluetooth Speaker",
        "Smartwatch",
        "USB-C Hub",
        "Webcam",
        "Tablet",
        "Gaming Mouse",
        "Keyboard",
        "Monitor",
        "Headphones",
        "Portable SSD",
    ],
    "Clothing": [
        "Cotton T-Shirt",
        "Denim Jeans",
        "Hoodie",
        "Sneakers",
        "Winter Jacket",
        "Casual Shirt",
        "Sports Shorts",
        "Wool Sweater",
        "Cap",
        "Socks Pack",
        "Belt",
        "Raincoat",
    ],
    "Home": [
        "Table Lamp",
        "Bedsheet Set",
        "Coffee Maker",
        "Vacuum Cleaner",
        "Cookware Set",
        "Cushion Cover",
        "Wall Clock",
        "Storage Box",
        "Air Fryer",
        "Water Bottle",
        "Desk Organizer",
        "LED Strip",
    ],
    "Beauty": [
        "Face Wash",
        "Moisturizer",
        "Lipstick",
        "Sunscreen",
        "Hair Serum",
        "Perfume",
        "Shampoo",
        "Face Mask",
        "Nail Kit",
        "Body Lotion",
    ],
    "Sports": [
        "Yoga Mat",
        "Dumbbells",
        "Football",
        "Tennis Racket",
        "Skipping Rope",
        "Gym Bag",
        "Resistance Bands",
        "Cycling Helmet",
        "Running Shoes",
        "Water Bladder",
    ],
    "Books": [
        "Python Basics",
        "Data Science Intro",
        "Mystery Novel",
        "Self Help Guide",
        "Cookbook",
        "History of AI",
        "Travel Diary",
        "Short Stories",
        "Kids Picture Book",
        "Business Strategy",
        "Poetry Collection",
        "Science Comics",
    ],
}

PRICE_RANGES = {
    "Electronics": (799, 89999),
    "Clothing": (199, 4999),
    "Home": (249, 14999),
    "Beauty": (99, 2499),
    "Sports": (149, 7999),
    "Books": (149, 1299),
}


def _build_product_catalog(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    product_id = 1
    for category, names in PRODUCT_NAMES.items():
        low, high = PRICE_RANGES[category]
        for name in names:
            price = round(float(rng.uniform(low, high)), 2)
            rows.append(
                {
                    "product_id": f"P{product_id:03d}",
                    "product_name": name,
                    "product_category": category,
                    "product_price": price,
                }
            )
            product_id += 1
    return pd.DataFrame(rows)


def generate_dataset(
    n_customers: int = 220,
    n_interactions: int = 6500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    products = _build_product_catalog(rng)

    category_list = products["product_category"].unique()
    customers = []
    for i in range(1, n_customers + 1):
        preferred = rng.choice(category_list)
        previous_purchases = int(rng.integers(0, 18))
        customers.append(
            {
                "customer_id": f"C{i:03d}",
                "preferred_category": preferred,
                "previous_purchases": previous_purchases,
                "price_sensitivity": float(rng.uniform(0.35, 1.0)),
                # Latent engagement propensity (0-1 scale)
                "engagement": float(rng.beta(2, 2)),
                # Latent product popularity preference
                "product_popularity": float(rng.uniform(0.3, 0.8)),
            }
        )
    customers_df = pd.DataFrame(customers)

    interactions = []
    for _ in range(n_interactions):
        customer = customers_df.iloc[int(rng.integers(0, len(customers_df)))]
        product = products.iloc[int(rng.integers(0, len(products)))]

        same_category = product["product_category"] == customer["preferred_category"]
        views = int(rng.integers(1, 18) + (4 if same_category else 0))
        cart_base = views * (0.45 if same_category else 0.22) * customer["price_sensitivity"]
        added_to_cart = int(max(0, rng.poisson(cart_base)))
        added_to_cart = min(added_to_cart, views)

        # Latent product popularity (some products are naturally more popular)
        product_popularity = customer.get("product_popularity", 0.5)

        # Latent customer engagement propensity
        engagement = customer.get("engagement", 0.5)

        price_penalty = min(product["product_price"] / 25000.0, 0.6)

        # Purchase logit-style score: target base rate ~15%, with modifiers
        purchase_score = (
            0.03                      # base purchase probability
            + 0.05 * same_category    # category match bonus
            + 0.01 * views            # browsing depth
            + 0.03 * added_to_cart    # cart intent
            + 0.008 * customer["previous_purchases"]  # loyalty
            + 0.03 * engagement       # inherent engagement
            + 0.02 * product_popularity  # product appeal
            - 0.12 * price_penalty    # price sensitivity
        )
        purchased = int(rng.random() < float(np.clip(purchase_score, 0.015, 0.65)))
        times_purchased = int(purchased * rng.integers(1, 4)) if purchased else 0

        interactions.append(
            {
                "customer_id": customer["customer_id"],
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "product_category": product["product_category"],
                "product_price": product["product_price"],
                "times_viewed": views,
                "times_added_to_cart": added_to_cart,
                "times_purchased": times_purchased,
                "previous_purchases": customer["previous_purchases"],
            }
        )

    data = pd.DataFrame(interactions)

    # Inject missing values (~3%) so cleaning is required.
    n_missing = max(1, int(0.03 * len(data)))
    for col in ["product_price", "times_viewed", "times_added_to_cart", "previous_purchases"]:
        idx = rng.choice(data.index, size=n_missing, replace=False)
        data.loc[idx, col] = np.nan

    cat_idx = rng.choice(data.index, size=max(1, n_missing // 2), replace=False)
    data.loc[cat_idx, "product_category"] = np.nan

    # Inject duplicate records.
    duplicates = data.sample(n=max(20, int(0.02 * len(data))), random_state=seed)
    data = pd.concat([data, duplicates], ignore_index=True)
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return data, products


def save_dataset(output_dir: Path, seed: int = 42) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions, products = generate_dataset(seed=seed)
    interactions_path = output_dir / "historical_interactions.csv"
    products_path = output_dir / "product_catalog.csv"
    interactions.to_csv(interactions_path, index=False)
    products.to_csv(products_path, index=False)
    return interactions_path, products_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate historical shopping data.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    interactions_path, products_path = save_dataset(root / "data", seed=args.seed)
    print(f"Saved interactions -> {interactions_path}")
    print(f"Saved catalog      -> {products_path}")


if __name__ == "__main__":
    main()
