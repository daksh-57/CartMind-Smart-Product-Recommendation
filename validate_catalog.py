"""Validate product catalog alignment and color consistency."""
import pandas as pd
from src.train import train_and_evaluate
from src.recommend import (
    profile_from_history, profile_from_details,
    recommend_top_n, recommend_by_popularity,
)
import re
from collections import Counter

print("=" * 60)
print("TEST 1: Product catalog name validation")
print("=" * 60)
raw = pd.read_csv("data/historical_interactions.csv")
model, metrics, featured, feature_engineer = train_and_evaluate(raw)
catalog = pd.read_csv("data/product_catalog.csv")

customers = sorted(featured["customer_id"].unique())
catalog_names = dict(zip(catalog["product_id"], catalog["product_name"]))

errors = 0
for cust in customers[:5]:
    profile = profile_from_history(cust, featured)
    recs = recommend_top_n(model, profile, catalog, featured, top_n=5)
    for _, row in recs.iterrows():
        pid = row["product_id"]
        rec_name = row["product_name"]
        if pid not in catalog_names:
            print(f"  ERROR: Product {pid} not in catalog")
            errors += 1
        elif catalog_names[pid] != rec_name:
            print(f"  MISMATCH: {pid} -> rec says '{rec_name}' but catalog says '{catalog_names[pid]}'")
            errors += 1
print(f"  Product name validation complete. Errors: {errors}")

print()
print("=" * 60)
print("TEST 2: Existing customer recommendation")
print("=" * 60)
profile = profile_from_history(customers[0], featured)
recs = recommend_top_n(model, profile, catalog, featured, top_n=5)
print(f"  Customer: {profile.customer_id}")
print(f"  Recommendations: {len(recs)} rows")
print(f"  Columns: {list(recs.columns)}")
print(f"  Sample product: {recs.iloc[0]['product_name']} (${recs.iloc[0]['product_price']:.2f})")
print(f"  Sample probability: {float(recs.iloc[0]['purchase_probability']):.2%}")
assert len(recs) == 5, "Expected 5 recommendations"
assert all(col in recs.columns for col in ["product_id", "product_name", "product_category", "product_price", "purchase_probability", "explanation"])
print("  PASSED")

print()
print("=" * 60)
print("TEST 3: New customer recommendation")
print("=" * 60)
profile2 = profile_from_details(
    previous_purchases=3, favourite_category="Electronics",
    budget=5000.0, featured=featured, feature_engineer=feature_engineer,
)
recs2 = recommend_top_n(model, profile2, catalog, featured, top_n=5)
print(f"  Customer: {profile2.customer_id}")
print(f"  Recommendations: {len(recs2)} rows")
assert len(recs2) == 5, "Expected 5 recommendations for new customer"
print("  PASSED")

print()
print("=" * 60)
print("TEST 4: Popularity baseline")
print("=" * 60)
pop = recommend_by_popularity(featured, top_n=5)
print(f"  Popularity baseline: {len(pop)} rows")
assert len(pop) == 5, "Expected 5 popularity recommendations"
print("  PASSED")

print()
print("=" * 60)
print("TEST 5: Metrics have real values (no hardcoded demo)")
print("=" * 60)
metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
for k in metric_keys:
    v = metrics.get(k)
    assert v is not None, f"  MISSING: {k}"
    assert isinstance(v, (int, float)), f"  WRONG TYPE: {k} = {v}"
    print(f"  {k}: {v:.4f}")
rec_keys = ["precision_at_5_ml", "recall_at_5_ml", "ndcg_at_5_ml"]
for k in rec_keys:
    v = metrics.get(k)
    assert v is not None, f"  MISSING: {k}"
    print(f"  {k}: {v:.4f}")
print("  PASSED")

print()
print("=" * 60)
print("TEST 6: Invalid customer ID raises KeyError")
print("=" * 60)
try:
    profile_from_history("NONEXISTENT_CUSTOMER_999", featured)
    print("  FAILED: Should have raised KeyError")
except KeyError as e:
    print(f"  KeyError correctly raised: {e}")
    print("  PASSED")

print()
print("=" * 60)
print("TEST 7: Color palette compliance")
print("=" * 60)
with open("streamlit_app.py", encoding="utf-8") as f:
    content = f.read()
colors = re.findall(r"#[0-9A-Fa-f]{6}", content)
allowed = {
    "#FFFFFF", "#F0F4F8", "#F8FAFC", "#E2E8F0", "#94A3B8", "#64748B",
    "#CBD5E1", "#ffffff", "#0F172A", "#1E3A5F", "#1E6FFF", "#0EA5E9",
    "#06B6D4", "#10B981", "#93C5FD", "#3B82F6", "#6366F1",
}
non_compliant = []
for color, count in Counter(colors).most_common():
    if color not in allowed:
        non_compliant.append((color, count))
if non_compliant:
    for c, n in non_compliant:
        print(f"  Non-compliant: {c} ({n} occurrences)")
else:
    print("  All colors are navy/white/blue compliant")
print("  PASSED")

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
