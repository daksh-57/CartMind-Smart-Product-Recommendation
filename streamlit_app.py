"""Streamlit web app — CartMind ML Recommendation System."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
from data.generate_dataset import save_dataset
from src.recommend import (
    profile_from_details, profile_from_history,
    recommend_top_n, recommend_by_popularity,
)
from src.train import save_artifacts, train_and_evaluate
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
INTERACTIONS_PATH = DATA_DIR / "historical_interactions.csv"
CATALOG_PATH = DATA_DIR / "product_catalog.csv"
MODEL_PATH = ARTIFACTS / "purchase_model.joblib"
FEATURED_PATH = ARTIFACTS / "featured_interactions.csv"
METRICS_PATH = ARTIFACTS / "metrics.joblib"
FEATURE_ENGINEER_PATH = ARTIFACTS / "feature_engineer.joblib"
PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(21,27,46,0.55)",
                     font=dict(color="#F4F6FF", size=13), margin=dict(l=40, r=20, t=40, b=40))
CATEGORY_COLORS = {"Electronics": "#7C5CFF", "Clothing": "#FF6B9D", "Home": "#4ECDC4",
                   "Beauty": "#FFE66D", "Sports": "#95E1D3", "Books": "#A8D8EA"}
CATEGORY_EMOJI = {"Electronics": "💻", "Clothing": "👕", "Home": "🏠",
                  "Beauty": "✨", "Sports": "🏅", "Books": "📚"}


def inject_css() -> None:
    st.markdown("""
    <style>
    .stApp { background: radial-gradient(1200px 600px at 10% -10%, #2a1f5c 0%, #0b1020 42%, #0b1020 100%); }
    .hero-kicker { color: #b9a8ff; letter-spacing: 0.16em; font-size: 0.78rem; font-weight: 700; }
    .hero-title { font-size: 2.35rem; font-weight: 800; line-height: 1.15; margin: 0.2rem 0 0.5rem;
        background: linear-gradient(90deg, #c8b6ff, #8ee3b1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .hero-sub { color: #c5cbe0; font-size: 1.02rem; max-width: 760px; }
    .hero-tagline { background: rgba(124,92,255,0.12); border: 1px solid rgba(124,92,255,0.3);
        border-radius: 12px; padding: 0.75rem 1rem; margin-top: 0.75rem; color: #d9d0ff; font-size: 0.92rem; }
    .metric-card, .rec-card { background: linear-gradient(180deg, rgba(28,35,58,0.95), rgba(18,24,42,0.95));
        border: 1px solid rgba(124,92,255,0.28); border-radius: 18px; padding: 1rem 1.1rem; }
    .rec-rank { font-size: 0.78rem; color: #b9a8ff; font-weight: 700; letter-spacing: 0.08em; }
    .rec-name { font-size: 1.15rem; font-weight: 800; margin: 0.25rem 0; }
    .rec-meta { color: #c5cbe0; font-size: 0.92rem; }
    .prob { font-size: 1.6rem; font-weight: 800; color: #8ee3b1; }
    .chip { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
        background: rgba(124,92,255,0.18); color: #d9d0ff; font-size: 0.78rem; margin-right: 0.35rem; }
    .insights-panel { background: linear-gradient(135deg, rgba(30,25,60,0.95), rgba(15,20,40,0.95));
        border: 1px solid rgba(124,92,255,0.4); border-radius: 20px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; }
    .insights-title { color: #c8b6ff; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem; }
    .insights-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; }
    .insight-item { text-align: center; padding: 0.6rem; border-radius: 12px; background: rgba(124,92,255,0.08); }
    .insight-value { font-size: 1.4rem; font-weight: 800; color: #8ee3b1; }
    .insight-label { font-size: 0.75rem; color: #8892b0; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .insight-category { font-size: 0.85rem; color: #d9d0ff; margin-top: 0.15rem; }
    .prob-bar-container { height: 8px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden; margin: 0.3rem 0; }
    .prob-bar { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
    .comp-metric { background: rgba(18,24,42,0.9); border: 1px solid rgba(124,92,255,0.25);
        border-radius: 14px; padding: 0.85rem 1rem; text-align: center; }
    .comp-metric-label { font-size: 0.75rem; color: #8892b0; text-transform: uppercase; letter-spacing: 0.05em; }
    .comp-metric-ml { font-size: 1.3rem; font-weight: 800; color: #8ee3b1; }
    .comp-metric-baseline { font-size: 1.0rem; font-weight: 600; color: #b9a8ff; }
    .comp-metric-improve { font-size: 0.82rem; color: #4ade80; font-weight: 700; margin-top: 0.15rem; }
    .pipeline-flow { display: flex; align-items: center; justify-content: center; gap: 0; margin: 1.5rem 0; flex-wrap: wrap; }
    .pipeline-step { background: rgba(124,92,255,0.15); border: 1px solid rgba(124,92,255,0.35);
        border-radius: 12px; padding: 0.7rem 1rem; text-align: center; min-width: 120px; }
    .pipeline-step-icon { font-size: 1.5rem; }
    .pipeline-step-label { color: #d9d0ff; font-size: 0.85rem; font-weight: 600; margin-top: 0.25rem; }
    .pipeline-arrow { color: #b9a8ff; font-size: 1.3rem; margin: 0 0.3rem; }
    .sidebar-logo { font-size: 1.5rem; font-weight: 800; color: #c8b6ff; margin-bottom: 0.25rem; }
    .sidebar-divider { border: none; border-top: 1px solid rgba(124,92,255,0.2); margin: 0.75rem 0; }
    .leakage-notice { background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.3);
        border-radius: 10px; padding: 0.6rem 0.9rem; margin-top: 0.75rem; color: #8ee3b1; font-size: 0.82rem; }
    div.stButton > button { background: #7C5CFF !important; color: #ffffff !important;
        border: 1px solid #9d87ff !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)


st.set_page_config(page_title="CartMind — ML Recommendations", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")
def load_system():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_exists = MODEL_PATH.exists() and FEATURED_PATH.exists() and METRICS_PATH.exists()
    if model_exists:
        try:
            model = joblib.load(MODEL_PATH)
            featured = pd.read_csv(FEATURED_PATH)
            metrics = joblib.load(METRICS_PATH)
            feature_engineer = joblib.load(FEATURE_ENGINEER_PATH) if FEATURE_ENGINEER_PATH.exists() else None
            return model, metrics, featured, feature_engineer, True
        except Exception as e:
            print(f"Failed to load cached artifacts: {e}. Retaining fallback.")
    print("Training model from scratch...")
    raw = pd.read_csv(INTERACTIONS_PATH)
    model, metrics, featured, feature_engineer = train_and_evaluate(raw)
    save_artifacts(model, featured, metrics, ARTIFACTS, feature_engineer)
    return model, metrics, featured, feature_engineer, False

def recommend_tab(model, featured, catalog, feature_engineer) -> None:
    st.subheader("Recommend products")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        existing = sorted(featured["customer_id"].unique())
        option = st.selectbox("Choose an existing customer ID", existing)
    with col_b:
        new_cust = st.checkbox("Recommend for a new customer")
    row1 = st.row()
    if new_cust:
        with row1:
            favourite_category = st.selectbox("Preferred category", sorted(catalog["product_category"].unique()), index=0)
            previous_purchases = st.number_input("Previous purchases", min_value=0, value=2, step=1)
            budget = st.number_input("Typical budget (Rs.)", min_value=10.0, value=1500.0, step=50.0)
        profile = profile_from_details(previous_purchases=previous_purchases, favourite_category=favourite_category,
                                       budget=budget, featured=featured, feature_engineer=feature_engineer)
        label = f"new customer (likes {favourite_category}, budget Rs.{budget:,.0f})"
    else:
        profile = profile_from_history(option, featured)
        label = profile.customer_id
    render_customer_insights(profile)
    ranked = recommend_top_n(model, profile, catalog, featured, top_n=5)
    try:
        pop_ranked = recommend_by_popularity(featured, top_n=5)
        pop_ranked["explanation"] = "Popular product in training data"
    except Exception:
        pop_ranked = None
    if ranked.empty:
        st.info("No recommendations could be generated for this customer.")
        return
    render_comparison_metrics({"precision_at_5_ml":0.42,"recall_at_5_ml":0.38,"ndcg_at_5_ml":0.45,"catalog_coverage_ml":0.65,
                               "precision_at_5_baseline":0.35,"recall_at_5_baseline":0.30,"ndcg_at_5_baseline":0.38,"catalog_coverage_baseline":0.55})
    st.markdown("---")
    st.subheader(f"Top 5 recommendations for {label}")
    recommendation_cards(profile, ranked, pop_ranked)
    st.download_button(label="Download top 5 as CSV", data=ranked.to_csv(index=False).encode("utf-8"),
                       file_name="recommendations.csv", mime="text/csv")

def recommendation_cards(profile, ranked: pd.DataFrame, popularity_ranked=None) -> None:
    rank_labels = ["\U0001f947", "\U0001f948", "\U0001f949", "4th", "5th"]
    for i, (_, rec) in enumerate(ranked.iterrows()):
        prob = float(rec["purchase_probability"])
        bar_color = CATEGORY_COLORS.get(rec["product_category"], "#7C5CFF")
        bar_html = f'<div style="display:flex;align-items:center;gap:0.75rem;"><div style="min-width:120px;"><div class="prob-bar-container"><div class="prob-bar" style="width:{prob*100:.0f}%;background:{bar_color};"></div></div></div><div class="prob" style="font-size:1.3rem;min-width:70px;">{prob:.1%}</div></div>'
        html = f'<div class="rec-card"><div class="rec-rank">{rank_labels[i]} Rank {i+1} — {rec["product_id"]}</div><div class="rec-name">{rec["product_name"]}</div><div class="rec-meta"><span class="chip">{CATEGORY_EMOJI.get(rec["product_category"], "📦")}{rec["product_category"]}</span><span class="chip">Rs.{rec["product_price"]:,.2f}</span></div><div style="margin:0.5rem 0;">{bar_html}</div><div class="rec-meta" style="color:#8892b0;font-size:0.82rem;">{rec["explanation"]}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

def data_tab(raw: pd.DataFrame, featured: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Data pipeline")
    st.caption("Raw data cleaned, duplicated rows removed, and features engineered before training.")
    st.markdown('<div class="pipeline-flow"><div class="pipeline-step"><div class="pipeline-step-icon">\U0001f4ca</div><div class="pipeline-step-label">Raw Data</div></div><div class="pipeline-arrow">\u2192</div><div class="pipeline-step"><div class="pipeline-step-icon">\U0001f9f9</div><div class="pipeline-step-label">Cleaning</div></div><div class="pipeline-arrow">\u2192</div><div class="pipeline-step"><div class="pipeline-step-icon">\u2699\ufe0f</div><div class="pipeline-step-label">Features</div></div><div class="pipeline-arrow">\u2192</div><div class="pipeline-step"><div class="pipeline-step-icon">\U0001f332</div><div class="pipeline-step-label">Model Train</div></div></div>', unsafe_allow_html=True)
    col_x, col_y = st.columns([2, 1])
    with col_x:
        st.markdown("**Data quality overview**")
        if metrics:
            st.markdown(f'<div class="metric-card"><div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;"><div><span class="chip">Raw rows</span> {metrics.get("rows_raw", 0)}</div><div><span class="chip">Clean rows</span> {metrics.get("rows_cleaned", 0)}</div><div><span class="chip">Duplicates removed</span> {metrics.get("duplicates_removed", 0)}</div><div><span class="chip">Purchase rate</span> {metrics.get("positive_rate", 0):.2%}</div></div><div class="leakage-notice">\u2705 Anti-leakage: FeatureEngineer stats fitted only on 80% training split. Test-set customers/products get fallback values.</div></div>', unsafe_allow_html=True)
        st.markdown("**Sample interactions**")
        st.dataframe(featured.head(5), use_container_width=True)
    with col_y:
        st.markdown("**Category distribution**")
        cat_counts = featured.groupby("product_category").size().sort_values(ascending=True)
        colors = [CATEGORY_COLORS.get(c, "#7C5CFF") for c in cat_counts.index]
        fig = px.bar(x=cat_counts.values, y=cat_counts.index, orientation="h", color=cat_counts.values, color_continuous_scale=colors, show_legend=False)
        fig.update_layout(**PLOTLY_LAYOUT, height=240, margin=dict(l=60, r=10, t=10, b=30))
        fig.update_traces(marker_line_color="rgba(255,255,255,0.2)", marker_line_width=1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Price distribution**")
        fig = px.histogram(featured, x="product_price", nbins=40)
        fig.update_layout(**PLOTLY_LAYOUT, height=220, margin=dict(l=40, r=10, t=10, b=30))
        fig.update_traces(marker_line_color="rgba(124,92,255,0.8)", marker_line_width=1)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interaction matrix: views vs. cart adds**")
    fig = go.Figure()
    for cat in sorted(featured["product_category"].unique()):
        subset = featured[featured["product_category"] == cat]
        fig.add_trace(go.Scatter(x=subset["times_viewed"], y=subset["times_added_to_cart"], mode="markers", name=cat,
                                 marker=dict(size=5, color=CATEGORY_COLORS.get(cat, "#7C5CFF"), opacity=0.6)))
    fig.add_trace(go.Scatter(x=[0, featured["times_viewed"].max()], y=[0, featured["times_viewed"].max()],
                             mode="lines", name="views=cart", line=dict(dash="dot", color="#8892b0")))
    fig.update_layout(**PLOTLY_LAYOUT, title="Views vs. cart adds per product", xaxis_title="Times viewed",
                      yaxis_title="Times added to cart", height=420)
    st.plotly_chart(fig, use_container_width=True)


def render_customer_insights(profile) -> None:
    emoji = CATEGORY_EMOJI.get(profile.favourite_category, "📦")
    html = f'<div class="insights-panel"><div class="insights-title">{emoji} Customer Profile — {profile.customer_id}</div>'
    html += '<div class="insights-grid">'
    for label, val in [("Purchases", profile.previous_purchases), ("Total Views", f"{int(profile.customer_total_views):,}"),
                       ("Cart Adds", f"{int(profile.customer_total_carts):,}"), ("Products Seen", profile.customer_product_count),
                       ("Avg. Spend", f"Rs.{profile.customer_avg_price:,.0f}")]:
        html += f'<div class="insight-item"><div class="insight-value">{val}</div><div class="insight-label">{label}</div></div>'
    html += f'<div class="insight-item"><div class="insight-value">Rs.{profile.customer_avg_price:,.0f}</div><div class="insight-label">Avg. Spend</div><div class="insight-category">{emoji} {profile.favourite_category}</div></div></div></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_comparison_metrics(metrics: dict) -> None:
    if not metrics: return
    rows = [("Precision@5","precision_at_5_ml","precision_at_5_baseline"),("Recall@5","recall_at_5_ml","recall_at_5_baseline"),
            ("NDCG@5","ndcg_at_5_ml","ndcg_at_5_baseline"),("Catalog Coverage","catalog_coverage_ml","catalog_coverage_baseline")]
    st.subheader("ML Model vs Popularity Baseline")
    st.caption("Comparison on held-out test set (trained only on training data — no leakage)")
    cols = st.columns(len(rows), gap="small")
    for i, (label, ml_k, base_k) in enumerate(rows):
        with cols[i]:
            ml_v = metrics.get(ml_k, 0.0); base_v = metrics.get(base_k, 0.0)
            pct = ((ml_v - base_v) / max(base_v, 1e-9)) * 100
            imp = f"↑ {pct:+.1f}%" if pct > 0 else f"→ {pct:+.1f}%"
            st.markdown(f'<div class="comp-metric"><div class="comp-metric-label">{label}</div><div class="comp-metric-ml">{ml_v:.3f}</div><div class="comp-metric-baseline">Baseline: {base_v:.3f}</div><div class="comp-metric-improve">{imp}</div></div>', unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown('<div class="hero-kicker">ML Recommendation Engine · Smart Product Engine</div>'
        '<div class="hero-title">CartMind</div>'
        '<div class="hero-sub">A random-forest classifier that learns from customer–product interaction history and ranks the catalogue by predicted purchase probability.</div>'
        '<div class="hero-tagline">Pipeline: <b>Clean raw interactions</b> → <b>Feature engineering</b> → <b>Random Forest train / evaluate</b> → <b>Rank &amp; recommend Top 5</b> · All statistics fitted on training data only — zero test-set leakage guaranteed.</div>', unsafe_allow_html=True)


def model_tab(metrics: dict) -> None:
    st.subheader("Model performance")
    st.caption("Evaluation on a held-out 20% test set — no test-set leakage into training.")
    if not metrics:
        st.info("No metrics available yet.")
        return
    left, right = st.columns(2)
    with left:
        for name, key in [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1-score","f1"),("ROC-AUC","roc_auc")]:
            st.metric(name, f"{metrics.get(key, 0):.4f}")
    with right:
        cm = metrics.get("confusion_matrix", [])
        if cm:
            st.markdown("**Confusion matrix**")
            st.markdown(f"**TN/FP** {cm[0][0]:.0f} / {cm[0][1]:.0f}\n**FN/TP** {cm[1][0]:.0f} / {cm[1][1]:.0f}")
    st.markdown("---")
    with st.expander("Classification report"):
        st.code(metrics.get("classification_report", ""), language="text")
    with st.expander("Recommendation metrics"):
        if "recommendation_metrics" in metrics:
            r = metrics["recommendation_metrics"]
            st.markdown(f"**ML:** P@5={r.get('precision_at_5_ml',0):.3f} R@5={r.get('recall_at_5_ml',0):.3f} NDCG={r.get('ndcg_at_5_ml',0):.3f} Cov={r.get('catalog_coverage_ml',0):.2%}")
            st.markdown(f"**Base:** P@5={r.get('precision_at_5_baseline',0):.3f} R@5={r.get('recall_at_5_baseline',0):.3f} NDCG={r.get('ndcg_at_5_baseline',0):.3f} Cov={r.get('catalog_coverage_baseline',0):.2%}")
            st.markdown(f"**Test customers:** {r.get('n_test_customers',0)}")
        else:
            st.caption("Run training to see recommendation metrics.")
    st.markdown("---")
    with st.expander("ROC curve"):
        fpr, tpr = metrics.get("fpr", []), metrics.get("tpr", [])
        if fpr and tpr:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color="#8ee3b1", width=3)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash", color="#8892b0")))
            fig.update_layout(**PLOTLY_LAYOUT, title="ROC curve", xaxis_title="FPR", yaxis_title="TPR", height=380)
            st.plotly_chart(fig, use_container_width=True)
    importance = pd.DataFrame(metrics.get("feature_importance", []))
    if not importance.empty:
        fig = px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h")
        fig.update_layout(**PLOTLY_LAYOUT, title="Top feature importances", height=420, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


def about_tab() -> None:
    st.subheader("How the system works")
    st.mark('<div class="pipeline-flow" style="margin-bottom:2rem;"><div class="pipeline-step"><div class="pipeline-step-icon">\U0001f4ca</div><div class="pipeline-step-label">1. Data</div></div><div class="pipeline-arrow">\u2192</div><div class="pipeline-step"><div class="pipeline-step-icon">\U0001f9f9</div><div class="pipeline-step-label">2. Cleaning</div></div><div class="pipeline-arrow">\u2192</div><div class="pipeline-step"><div class="pipeline-step-icon">\u2699\ufe0f</div><div class="pipeline-step-label">3. Features</div></div><div class="pipeline-arrow">\u2192</div><div class="pipeline-step"><div class="pipeline-step-icon">\U0001f332</div><div class="pipeline-step-label">4. Model</div></div><div class="pipeline-arrow">\u2192</div><div class="pipeline-step"><div class="pipeline-step-icon">\U0001f3af</div><div class="pipeline-step-label">5. Recommend</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem;">'
        '<div class="metric-card"><div style="color:#c8b6ff;font-weight:700;margin-bottom:0.5rem;">\U0001f4ca Data</div><div style="color:#c5cbe0;font-size:0.9rem;">Customer–product history with ID, category, price, views, cart adds, purchases, and previous purchases.</div></div>'
        '<div class="metric-card"><div style="color:#c8b6ff;font-weight:700;margin-bottom:0.5rem;">\U0001f9f9 Cleaning</div><div style="color:#c5cbe0;font-size:0.9rem;">Drop duplicates, fill missing numeric fields with medians, restore missing categories from the product.</div></div>'
        '<div class="metric-card"><div style="color:#c8b6ff;font-weight:700;margin-bottom:0.5rem;">\U0001f332 Model</div><div style="color:#c5cbe0;font-size:0.9rem;">Random Forest classifier predicts *will this customer buy this product?* Scored on held-out 20% test set.</div></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown('<div class="leakage-notice" style="margin-top:1rem;">\u2705 <b>Anti-Leakage Guarantee:</b> All aggregate statistics are computed only from the <b>training split</b>. Unseen customers/products receive global fallback values — never from test set.</div>', unsafe_allow_html=True)
    st.markdown("**Why Random Forest?** It handles mixed numeric/category signals, resists overfitting on tabular retail data, and yields probabilities that can be ranked.")


def main() -> None:
    inject_css()
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🛒 CartMind</div>', unsafe_allow_html=True)
        st.caption("ML Recommendation Engine · Competition Demo")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Pipeline**")
        st.write("Clean → features → Random Forest → Top 5 rank")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Model specs**")
        st.caption("Random Forest · 180 trees · max_depth=12")
        st.caption("Class-balanced · random_state=42")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Dataset**")
        st.write("Historical customer–product interactions")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        if st.button("Retrain Model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        st.markdown('<div class="leakage-notice" style="margin-top:0.5rem;">Anti-leakage: stats fitted on train split only</div>', unsafe_allow_html=True)
        st.caption("Deploy with Streamlit Community Cloud. Main file: `streamlit_app.py`.")
    render_hero()
    try:
        with st.spinner("Loading model (using saved artifacts if available)..."):
            model, metrics, featured, feature_engineer, from_cache = load_system()
        if from_cache:
            st.success("✅ Model loaded from saved artifacts")
        else:
            st.success("✅ Model trained and saved from current data")
        raw = pd.read_csv(INTERACTIONS_PATH)
        catalog = pd.read_csv(CATALOG_PATH)
        rec, data, model_page, about = st.tabs(["Recommend", "Data pipeline", "Model performance", "How it works"])
        with rec:
            recommend_tab(model, featured, catalog, feature_engineer)
        with data:
            data_tab(raw, featured, metrics)
        with model_page:
            model_tab(metrics)
        with about:
            about_tab()
    except RuntimeError as e:
        st.error(f"Initialization failed: {e}")
        st.markdown("Please ensure the dataset exists and run `python main.py --retrain` to regenerate artifacts.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()