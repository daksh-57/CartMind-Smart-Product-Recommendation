"""Smart Product Recommendation System — 1st Year Problem 01.

Run:
    python main.py
    python main.py --customer C001
    python main.py --new --category Electronics --purchases 3 --budget 2500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from data.generate_dataset import save_dataset
from src.recommend import (
    profile_from_details,
    profile_from_history,
    recommend_top_n,
)
from src.train import save_artifacts, train_and_evaluate

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
INTERACTIONS_PATH = DATA_DIR / "historical_interactions.csv"
CATALOG_PATH = DATA_DIR / "product_catalog.csv"
MODEL_PATH = ARTIFACTS / "purchase_model.joblib"
FEATURED_PATH = ARTIFACTS / "featured_interactions.csv"


def ensure_data() -> None:
    if not INTERACTIONS_PATH.exists() or not CATALOG_PATH.exists():
        print("Generating historical customer-product data...")
        save_dataset(DATA_DIR)


def train_if_needed(force: bool = False) -> tuple[object, dict, pd.DataFrame, object]:
    if not force and MODEL_PATH.exists() and FEATURED_PATH.exists():
        model = joblib.load(MODEL_PATH)
        featured = pd.read_csv(FEATURED_PATH)
        metrics = joblib.load(ARTIFACTS / "metrics.joblib") if (ARTIFACTS / "metrics.joblib").exists() else {}
        feature_engineer = joblib.load(ARTIFACTS / "feature_engineer.joblib") if (ARTIFACTS / "feature_engineer.joblib").exists() else None
        return model, metrics, featured, feature_engineer

    print("Cleaning data, engineering features, and training the model...")
    raw = pd.read_csv(INTERACTIONS_PATH)
    model, metrics, featured, feature_engineer = train_and_evaluate(raw)
    save_artifacts(model, featured, metrics, ARTIFACTS, feature_engineer)
    return model, metrics, featured, feature_engineer


def print_metrics(metrics: dict) -> None:
    if not metrics:
        return
    print("\n=== Data processing ===")
    print(f"Raw rows              : {metrics.get('rows_raw')}")
    print(f"After cleaning        : {metrics.get('rows_cleaned')}")
    print(f"Duplicates removed    : {metrics.get('duplicates_removed')}")
    print(f"Purchase rate         : {metrics.get('positive_rate', 0):.2%}")
    print("\n=== Model evaluation (held-out 20% test set) ===")
    print(f"Accuracy              : {metrics.get('accuracy', 0):.4f}")
    print(f"Precision             : {metrics.get('precision', 0):.4f}")
    print(f"Recall                : {metrics.get('recall', 0):.4f}")
    print(f"F1-score              : {metrics.get('f1', 0):.4f}")
    print(f"ROC-AUC               : {metrics.get('roc_auc', 0):.4f}")
    print("Confusion matrix [TN FP / FN TP]:")
    print(metrics.get("confusion_matrix"))
    print("\nClassification report:")
    print(metrics.get("classification_report", ""))


def print_recommendations(customer_label: str, ranked: pd.DataFrame) -> None:
    print(f"\n=== Top 5 recommended products for {customer_label} ===")
    display = ranked.copy()
    display["purchase_probability"] = display["purchase_probability"].map(lambda p: f"{p:.1%}")
    display["product_price"] = display["product_price"].map(lambda p: f"Rs.{p:,.2f}")
    print(display.to_string(index=False))


def interactive_loop(model, featured: pd.DataFrame, catalog: pd.DataFrame) -> None:
    categories = sorted(catalog["product_category"].unique())
    sample_ids = ", ".join(sorted(featured["customer_id"].unique())[:8])
    print("\nPersonalized recommendations")
    print("Type an existing Customer ID, 'new' for a new customer, or 'quit'.")
    print(f"Example IDs: {sample_ids}")

    while True:
        choice = input("\nCustomer ID / new / quit: ").strip()
        if choice.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            return
        try:
            if choice.lower() == "new":
                print(f"Categories: {', '.join(categories)}")
                category = input("Preferred category: ").strip() or categories[0]
                purchases = int(input("Previous purchases (number): ").strip() or "2")
                budget = float(input("Typical budget (Rs.): ").strip() or "2000")
                profile = profile_from_details(purchases, category, budget, featured, feature_engineer=feature_engineer)
                label = f"new customer (likes {category}, budget Rs.{budget:.0f})"
            else:
                profile = profile_from_history(choice.upper(), featured)
                label = profile.customer_id
            ranked = recommend_top_n(model, profile, catalog, featured, top_n=5)
            print_recommendations(label, ranked)
        except (KeyError, ValueError) as exc:
            print(f"Could not generate recommendations: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Product Recommendation System")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--customer", type=str, help="Existing customer ID, e.g. C001")
    parser.add_argument("--new", action="store_true", help="Recommend for a new customer")
    parser.add_argument("--category", type=str, default="Electronics")
    parser.add_argument("--purchases", type=int, default=3)
    parser.add_argument("--budget", type=float, default=2500)
    parser.add_argument("--no-interactive", action="store_true", help="Skip the prompt loop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data()
    model, metrics, featured, feature_engineer = train_if_needed(force=args.retrain)
    catalog = pd.read_csv(CATALOG_PATH)
    print_metrics(metrics)

    if args.customer:
        profile = profile_from_history(args.customer.upper(), featured)
        ranked = recommend_top_n(model, profile, catalog, featured)
        print_recommendations(profile.customer_id, ranked)
    elif args.new:
        profile = profile_from_details(args.purchases, args.category, args.budget, featured, feature_engineer=feature_engineer)
        ranked = recommend_top_n(model, profile, catalog, featured)
        print_recommendations(
            f"new customer (likes {args.category}, budget Rs.{args.budget:.0f})",
            ranked,
        )
    elif not args.no_interactive:
        demo_id = sorted(featured["customer_id"].unique())[0]
        profile = profile_from_history(demo_id, featured)
        ranked = recommend_top_n(model, profile, catalog, featured)
        print_recommendations(f"{demo_id} (demo)", ranked)
        try:
            interactive_loop(model, featured, catalog)
        except EOFError:
            print("\nNo interactive input available.")


if __name__ == "__main__":
    main()
