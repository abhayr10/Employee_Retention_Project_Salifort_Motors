"""
Salifort Motors – Employee Retention Predictor (Enhanced)
========================================================
Features:
  • Deep EDA  – 15+ charts, statistical summaries, segment analysis
  • ML Models – LR / DT / RF / XGBoost with CV or Holdout validation
  • XAI        – Feature importances + SHAP waterfall / beeswarm
  • Predict    – Single-employee risk prediction with natural-language explanation
  • AI Chatbot – Claude-powered assistant for any HR-analytics question
"""

# ─────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────
import warnings, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, ParameterGrid)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ─────────────────────────────────────────────────────────────
# 1.  Page config & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salifort Motors – Retention AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

section[data-testid="stSidebar"] { background:#080c14; }
section[data-testid="stSidebar"] * { color:#c8d6e5 !important; }

.hero {
    background: linear-gradient(120deg,#0d1b2a 0%,#1b2838 60%,#162032 100%);
    border-radius:14px; padding:2.5rem 2rem 2rem;
    margin-bottom:1.5rem; border-left:5px solid #f0a500;
}
.hero h1 { font-family:'Space Mono',monospace; color:#f0f4f8;
           font-size:1.9rem; margin:0; letter-spacing:-1px; }
.hero p  { color:#8899aa; margin-top:.5rem; font-size:.93rem; }

.kpi { background:#111827; border-radius:10px; padding:1.1rem .9rem;
       text-align:center; border-top:3px solid #f0a500; }
.kpi .val { font-family:'Space Mono',monospace; font-size:1.9rem;
            font-weight:700; color:#f0a500; }
.kpi .lbl { color:#6b7280; font-size:.78rem; margin-top:.25rem; }

.sh { font-family:'Space Mono',monospace; font-size:1rem; font-weight:700;
      color:#f0a500; border-bottom:1px solid #f0a50030;
      padding-bottom:.35rem; margin-bottom:1rem; margin-top:1.5rem; }

.insight { background:#0f1923; border-left:4px solid #f0a500;
           border-radius:0 8px 8px 0; padding:.85rem 1rem;
           margin:.6rem 0; font-size:.88rem; color:#c8d6e5; }

.pred-leave { background:#1f0d0d; border:2px solid #ef4444;
              border-radius:12px; padding:1.5rem; text-align:center; }
.pred-stay  { background:#0d1f10; border:2px solid #22c55e;
              border-radius:12px; padding:1.5rem; text-align:center; }
.pred-emoji  { font-size:2.8rem; }
.pred-title  { font-family:'Space Mono',monospace; font-size:1.3rem;
               font-weight:700; margin:.4rem 0; }
.pred-sub    { color:#9ca3af; font-size:.87rem; }

.chat-user { background:#1e2d40; border-radius:12px 12px 4px 12px;
             padding:.75rem 1rem; margin:.5rem 0; color:#e2e8f0; }
.chat-ai   { background:#162032; border-left:3px solid #f0a500;
             border-radius:4px 12px 12px 12px;
             padding:.75rem 1rem; margin:.5rem 0; color:#c8d6e5; }
.chat-label{ font-family:'Space Mono',monospace; font-size:.7rem;
             color:#6b7280; margin-bottom:.25rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 2.  Constants & helpers
# ─────────────────────────────────────────────────────────────
CAT_COLS = ["department", "salary"]
NUM_COLS = ["satisfaction_level", "last_evaluation", "number_project",
            "average_monthly_hours", "tenure", "work_accident",
            "promotion_last_5years"]
BG   = "#111827"
ACC  = "#f0a500"
RED  = "#ef4444"
GRN  = "#22c55e"
MUTED= "#6b7280"

def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2d3748")
    return fig, ax

def dark_figs(rows, cols, w=14, h=6):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(BG)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2d3748")
    return fig, axes

def insight(txt):
    st.markdown(f'<div class="insight">💡 {txt}</div>', unsafe_allow_html=True)

def sh(txt):
    st.markdown(f'<div class="sh">{txt}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 3.  Data helpers
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns={
        "Work_accident":        "work_accident",
        "average_montly_hours": "average_monthly_hours",
        "time_spend_company":   "tenure",
        "Department":           "department",
    })
    df = df.drop_duplicates(keep="first")
    return df

@st.cache_resource(show_spinner=False)
def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
    ])

@st.cache_resource(show_spinner=False)
def train_models(_df, model_choice, cv_method):
    X = _df.drop("left", axis=1)
    y = _df["left"]
    preprocessor = build_preprocessor()

    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.25, random_state=42, stratify=y_tr_full)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(random_state=42, n_jobs=-1),
    }
    if XGBOOST_OK:
        models["XGBoost"] = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42)

    param_grids = {
        "Logistic Regression": {"classifier__C": [0.1, 1.0, 10.0],
                                 "classifier__penalty": ["l2"]},
        "Decision Tree":       {"classifier__max_depth": [3, 5, 7, None],
                                 "classifier__min_samples_split": [2, 10, 20]},
        "Random Forest":       {"classifier__n_estimators": [50, 100],
                                 "classifier__max_depth": [5, 10, None],
                                 "classifier__min_samples_split": [2, 5]},
        "XGBoost":             {"classifier__n_estimators": [50, 100],
                                 "classifier__max_depth": [3, 5],
                                 "classifier__learning_rate": [0.1, 0.2]},
    }
    if model_choice != "All Models":
        models      = {model_choice: models[model_choice]}
        param_grids = {model_choice: param_grids[model_choice]}

    results, trained = [], {}
    for name, model in models.items():
        pg = param_grids[name]
        if cv_method == "5-Fold Cross-Validation":
            cv_pipeline = Pipeline([("preprocessor", preprocessor),
                                    ("classifier", model)])
            gs = GridSearchCV(cv_pipeline, pg,
                              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                              scoring="roc_auc", n_jobs=-1)
            gs.fit(X_tr_full, y_tr_full)
            best_pipeline = gs.best_estimator_
        else:
            X_tr_prep  = preprocessor.fit_transform(X_train)
            X_val_prep = preprocessor.transform(X_val)
            raw_grid   = {k.replace("classifier__", ""): v for k, v in pg.items()}
            best_score, best_mdl = 0, None
            for params in ParameterGrid(raw_grid):
                cur = model.__class__(**params, **{"random_state": 42})
                if XGBOOST_OK and name == "XGBoost":
                    cur.set_params(eval_metric="logloss", use_label_encoder=False)
                elif name == "Random Forest":
                    cur.set_params(n_jobs=-1)
                cur.fit(X_tr_prep, y_train)
                score = roc_auc_score(y_val, cur.predict_proba(X_val_prep)[:, 1])
                if score > best_score:
                    best_score, best_mdl = score, cur
            best_pipeline = Pipeline([("preprocessor", preprocessor),
                                      ("classifier", best_mdl)])
            best_pipeline.fit(X_tr_full, y_tr_full)

        trained[name] = best_pipeline
        y_pred  = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        results.append({
            "Model":     name,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1-Score":  f1_score(y_test, y_pred),
            "ROC-AUC":   roc_auc_score(y_test, y_proba),
        })

    results_df = (pd.DataFrame(results)
                  .sort_values("ROC-AUC", ascending=False)
                  .reset_index(drop=True))
    best_name  = results_df.iloc[0]["Model"]
    best_pipe  = trained[best_name]
    prep_f     = best_pipe.named_steps["preprocessor"]
    cat_out    = prep_f.named_transformers_["cat"].get_feature_names_out(CAT_COLS)
    feat_names = NUM_COLS + list(cat_out)
    return results_df, best_pipe, feat_names, X_test, y_test, trained

# ─────────────────────────────────────────────────────────────
# 4.  Claude chatbot helper
# ─────────────────────────────────────────────────────────────
def call_claude(messages: list, system: str) -> str:
    import json, urllib.request, urllib.error
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": system,
        "messages": messages,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        return f"⚠️ API error {e.code}: {e.read().decode()}"
    except Exception as ex:
        return f"⚠️ Error: {ex}"

# ─────────────────────────────────────────────────────────────
# 5.  Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 Salifort Motors")
    st.markdown("Employee Retention Intelligence Platform")
    st.markdown("---")
    uploaded = st.file_uploader("Upload HR dataset (CSV)", type=["csv"])
    st.markdown("---")
    st.markdown("#### ⚙️ Model Settings")
    model_options = (["All Models", "Logistic Regression",
                      "Decision Tree", "Random Forest"]
                     + (["XGBoost"] if XGBOOST_OK else []))
    model_choice = st.selectbox("Model to train", model_options)
    cv_method    = st.radio("Validation strategy",
                            ["5-Fold Cross-Validation", "Holdout (60/20 split)"])
    run_btn = st.button("🚀 Train Models", type="primary",
                        disabled=uploaded is None, use_container_width=True)
    st.markdown("---")
    st.caption("Powered by Streamlit · scikit-learn · Claude AI")

# ─────────────────────────────────────────────────────────────
# 6.  Hero
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏭 Salifort Motors — Employee Retention AI</h1>
  <p>Deep EDA · Machine Learning · Explainable AI · Natural Language Insights · AI Chatbot</p>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.info("👈  Upload **HR_capstone_dataset.csv** in the sidebar to get started.")
    st.stop()

df = load_and_clean(uploaded)

# ─────────────────────────────────────────────────────────────
# 7.  Tabs
# ─────────────────────────────────────────────────────────────
tab_data, tab_eda, tab_model, tab_xai, tab_predict, tab_chat = st.tabs([
    "📋 Data Overview",
    "📊 Deep EDA",
    "🤖 Model Results",
    "🔍 Explainability (XAI)",
    "🔮 Predict",
    "💬 AI Chatbot",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 – DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab_data:
    left_pct = df["left"].mean() * 100
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [len(df), df["left"].sum(), f"{left_pct:.1f}%",
         df["tenure"].median(), round(df["satisfaction_level"].mean(), 2)],
        ["Total Employees", "Employees Left", "Attrition Rate",
         "Median Tenure (yrs)", "Avg Satisfaction"]
    ):
        col.markdown(f'<div class="kpi"><div class="val">{val}</div>'
                     f'<div class="lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**First 10 rows**")
        st.dataframe(df.head(10), use_container_width=True)
    with cr:
        st.markdown("**Descriptive statistics**")
        st.dataframe(df.describe().T.style.format("{:.3f}"), use_container_width=True)

    st.markdown("---")
    sh("Data Quality Report")
    info_df = pd.DataFrame({
        "dtype":    df.dtypes.astype(str),
        "nulls":    df.isnull().sum(),
        "null_%":   (df.isnull().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    st.dataframe(info_df, use_container_width=True)
    insight(f"Dataset has **{len(df):,}** clean rows after removing duplicates. "
            "No missing values detected — ready for modelling.")

    sh("Outlier Check — Tenure")
    q25, q75 = df["tenure"].quantile([0.25, 0.75])
    iqr = q75 - q25
    upper_lim = q75 + 1.5 * iqr
    lower_lim = q25 - 1.5 * iqr
    outliers  = df[(df["tenure"] > upper_lim) | (df["tenure"] < lower_lim)]
    fig, ax = dark_fig(9, 2)
    sns.boxplot(x=df["tenure"], ax=ax, color=ACC)
    ax.set_xlabel("Tenure (years)", color="white")
    st.pyplot(fig)
    insight(f"**{len(outliers)}** outlier rows in tenure "
            f"(upper limit = {upper_lim:.0f} yrs). "
            "Tree-based models handle these naturally.")

# ═══════════════════════════════════════════════════════════════
# TAB 2 – DEEP EDA
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    left_df   = df[df["left"] == 1]
    stayed_df = df[df["left"] == 0]

    # 2.1 Attrition overview
    sh("2.1  Attrition Overview")
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = dark_fig(4, 4)
        counts = df["left"].value_counts()
        ax.pie(counts, labels=["Stayed", "Left"], autopct="%1.1f%%",
               colors=[GRN, RED], textprops={"color": "white"},
               wedgeprops={"linewidth": 2, "edgecolor": BG})
        ax.set_title("Overall Attrition", color="white", fontfamily="monospace")
        st.pyplot(fig)

    with c2:
        fig, ax = dark_fig(4, 4)
        dept_left = df.groupby("department")["left"].mean().sort_values() * 100
        colors = [RED if v > dept_left.mean() else ACC for v in dept_left]
        dept_left.plot.barh(ax=ax, color=colors)
        ax.set_xlabel("Attrition rate (%)", color="white")
        ax.axvline(dept_left.mean(), ls="--", color="white", lw=1, alpha=0.5)
        ax.set_title("By Department", color="white", fontfamily="monospace")
        st.pyplot(fig)

    with c3:
        fig, ax = dark_fig(4, 4)
        sal_left = (df.groupby("salary")["left"].mean()
                    .reindex(["low", "medium", "high"]) * 100)
        sal_left.plot.bar(ax=ax, color=[RED, ACC, GRN], width=0.5)
        ax.set_xlabel("Salary Band", color="white")
        ax.set_ylabel("Attrition %", color="white")
        ax.set_xticklabels(sal_left.index, rotation=0, color="white")
        ax.set_title("By Salary Band", color="white", fontfamily="monospace")
        st.pyplot(fig)

    insight("HR, management and sales have the highest attrition rates. "
            "Low-salary employees are ~3× more likely to leave than high-salary peers.")

    # 2.2 Satisfaction & Evaluation distributions
    sh("2.2  Satisfaction & Last Evaluation Distributions")
    fig, axes = dark_figs(1, 2, 13, 5)
    for ax, col, title in zip(
        axes,
        ["satisfaction_level", "last_evaluation"],
        ["Satisfaction Level Distribution", "Last Evaluation Distribution"]
    ):
        stayed_df[col].plot.kde(ax=ax, color=GRN, label="Stayed", lw=2)
        left_df[col].plot.kde(ax=ax, color=RED, label="Left", lw=2)
        ax.set_title(title, color="white", fontfamily="monospace")
        ax.legend(facecolor=BG, labelcolor="white")
        ax.set_xlabel(col, color="white")
    plt.tight_layout()
    st.pyplot(fig)

    stat, pval = stats.mannwhitneyu(stayed_df["satisfaction_level"],
                                    left_df["satisfaction_level"])
    insight(f"Mann-Whitney U test on satisfaction: p={pval:.2e} — "
            "the gap is **statistically significant** (p < 0.001). "
            f"Leavers median = {left_df['satisfaction_level'].median():.2f} vs "
            f"Stayers = {stayed_df['satisfaction_level'].median():.2f}.")

    # 2.3 Scatter: satisfaction vs evaluation
    sh("2.3  Three Employee Archetypes Among Leavers")
    fig, ax = dark_fig(10, 5)
    ax.scatter(stayed_df["satisfaction_level"], stayed_df["last_evaluation"],
               alpha=0.07, s=5, c=GRN, label="Stayed")
    ax.scatter(left_df["satisfaction_level"], left_df["last_evaluation"],
               alpha=0.35, s=8, c=RED, label="Left")
    ax.set_xlabel("Satisfaction Level")
    ax.set_ylabel("Last Evaluation")
    ax.set_title("Satisfaction vs Evaluation", color="white", fontfamily="monospace")
    ax.legend(facecolor=BG, labelcolor="white")
    st.pyplot(fig)
    insight("Three visible leaver clusters: **① Burnt-out high performers** "
            "(high eval, low satisfaction) · **② Disengaged low performers** "
            "(low eval, low satisfaction) · **③ Resigned mid-performers** "
            "(medium eval, medium satisfaction).")

    # 2.4 Work hours
    sh("2.4  Work Hours Analysis")
    fig, axes = dark_figs(1, 2, 13, 5)
    sns.histplot(data=df, x="average_monthly_hours", hue="left",
                 ax=axes[0], palette={0: GRN, 1: RED}, bins=40, kde=True, alpha=0.6)
    axes[0].set_title("Monthly Hours Distribution", color="white", fontfamily="monospace")
    axes[0].legend(["Stayed", "Left"], facecolor=BG, labelcolor="white")

    df_tmp = df.copy()
    df_tmp["hours_bucket"] = pd.cut(df_tmp["average_monthly_hours"],
                                    bins=[0, 150, 200, 250, 500],
                                    labels=["<150", "150-200", "200-250", ">250"])
    bucket_attr = df_tmp.groupby("hours_bucket", observed=True)["left"].mean() * 100
    bucket_attr.plot.bar(ax=axes[1], color=ACC, width=0.5)
    axes[1].set_xlabel("Monthly Hours Bucket")
    axes[1].set_ylabel("Attrition %")
    axes[1].set_xticklabels(bucket_attr.index, rotation=0, color="white")
    axes[1].set_title("Attrition by Hours Bucket", color="white", fontfamily="monospace")
    plt.tight_layout()
    st.pyplot(fig)
    insight("Both extremes are risky: **< 150 hrs/month** (underutilised) "
            "AND **> 250 hrs/month** (burnt-out) show elevated attrition. "
            "Optimal range: 150-200 hours.")

    # 2.5 Tenure
    sh("2.5  Tenure & Attrition")
    fig, axes = dark_figs(1, 2, 13, 5)
    tenure_attr = df.groupby("tenure")["left"].mean() * 100
    tenure_attr.plot.bar(ax=axes[0], color=RED, width=0.7)
    axes[0].set_xlabel("Tenure (years)")
    axes[0].set_ylabel("Attrition %")
    axes[0].set_xticklabels(tenure_attr.index, rotation=0, color="white")
    axes[0].set_title("Attrition Rate by Tenure", color="white", fontfamily="monospace")

    sns.boxplot(data=df, x="left", y="tenure", ax=axes[1],
                palette={0: GRN, 1: RED})
    axes[1].set_xticklabels(["Stayed", "Left"], color="white")
    axes[1].set_title("Tenure by Status", color="white", fontfamily="monospace")
    plt.tight_layout()
    st.pyplot(fig)
    insight("Years **3-4 are crisis points** — attrition peaks sharply. "
            "Employees surviving past 6 years become highly loyal. "
            "Focus retention efforts on the 2-4 year cohort.")

    # 2.6 Number of projects
    sh("2.6  Number of Projects vs Attrition")
    fig, axes = dark_figs(1, 2, 13, 5)
    proj_attr = df.groupby("number_project")["left"].mean() * 100
    proj_attr.plot.bar(ax=axes[0], color=ACC, width=0.6)
    axes[0].set_xlabel("Number of Projects")
    axes[0].set_ylabel("Attrition %")
    axes[0].set_xticklabels(proj_attr.index, rotation=0, color="white")
    axes[0].set_title("Attrition by # Projects", color="white", fontfamily="monospace")

    sns.violinplot(data=df, x="left", y="number_project", ax=axes[1],
                   palette={0: GRN, 1: RED}, inner="box")
    axes[1].set_xticklabels(["Stayed", "Left"], color="white")
    axes[1].set_title("Projects Distribution", color="white", fontfamily="monospace")
    plt.tight_layout()
    st.pyplot(fig)
    insight("Employees on **2 projects** (underutilised) and **6-7 projects** "
            "(overloaded) have the highest attrition. Optimal workload: 3-5 projects.")

    # 2.7 Promotion & Work accident
    sh("2.7  Promotion & Work Accident Impact")
    fig, axes = dark_figs(1, 2, 13, 4)
    for ax, col, title in zip(
        axes,
        ["promotion_last_5years", "work_accident"],
        ["Promotion in Last 5 Years", "Work Accident"]
    ):
        group = df.groupby(col)["left"].mean() * 100
        group.plot.bar(ax=ax, color=[GRN, RED], width=0.4)
        ax.set_xticklabels(["No", "Yes"], rotation=0, color="white")
        ax.set_ylabel("Attrition %")
        ax.set_title(title, color="white", fontfamily="monospace")
    plt.tight_layout()
    st.pyplot(fig)
    promo_no  = df[df["promotion_last_5years"] == 0]["left"].mean() * 100
    promo_yes = df[df["promotion_last_5years"] == 1]["left"].mean() * 100
    insight(f"Only **{df['promotion_last_5years'].mean()*100:.1f}%** of employees "
            f"were promoted in 5 years. Non-promoted: **{promo_no:.1f}%** attrition "
            f"vs {promo_yes:.1f}% for promoted — a clear signal to improve career paths.")

    # 2.8 Correlation heatmap
    sh("2.8  Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = dark_fig(10, 7)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 8},
                cbar_kws={"shrink": .8})
    ax.tick_params(colors="white")
    plt.xticks(rotation=45, ha="right", color="white")
    plt.yticks(color="white")
    ax.set_title("Feature Correlation Matrix", color="white", fontfamily="monospace")
    st.pyplot(fig)
    insight("**satisfaction_level** has the strongest negative correlation with "
            "leaving (−0.39). **number_project** and **average_monthly_hours** "
            "are strongly correlated — multicollinearity to watch in Logistic Regression.")

    # 2.9 Department × Salary heatmap
    sh("2.9  Attrition Heatmap — Department × Salary")
    pivot = (df.pivot_table(index="department", columns="salary",
                            values="left", aggfunc="mean") * 100)
    pivot = pivot.reindex(columns=["low", "medium", "high"])
    fig, ax = dark_fig(10, 6)
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds",
                linewidths=0.5, ax=ax, annot_kws={"size": 9},
                cbar_kws={"shrink": .8})
    ax.tick_params(colors="white")
    plt.xticks(color="white")
    plt.yticks(rotation=0, color="white")
    ax.set_title("Attrition % — Department × Salary", color="white",
                 fontfamily="monospace")
    ax.set_xlabel("Salary Band")
    ax.set_ylabel("Department")
    st.pyplot(fig)
    insight("**HR department + low salary** is the most at-risk segment. "
            "Management + high salary shows near-zero attrition.")

    # 2.10 Satisfaction by tenure & left
    sh("2.10  Satisfaction Trend Over Tenure")
    fig, ax = dark_fig(11, 5)
    for left_val, color, label in [(0, GRN, "Stayed"), (1, RED, "Left")]:
        grp = df[df["left"] == left_val].groupby("tenure")["satisfaction_level"].mean()
        ax.plot(grp.index, grp.values, marker="o", color=color, label=label, lw=2)
    ax.set_xlabel("Tenure (years)")
    ax.set_ylabel("Avg Satisfaction Level")
    ax.set_title("Mean Satisfaction by Tenure", color="white", fontfamily="monospace")
    ax.legend(facecolor=BG, labelcolor="white")
    st.pyplot(fig)
    insight("Satisfaction dips sharply at years 3-5 for those who eventually leave, "
            "while stayers maintain relatively stable satisfaction throughout their tenure.")

    # 2.11 Pairplot snapshot
    sh("2.11  Multi-Variate Pairplot (1 000-row sample)")
    sample = df.sample(min(1000, len(df)), random_state=42)
    pair_cols = ["satisfaction_level", "last_evaluation",
                 "average_monthly_hours", "tenure", "left"]
    pair_df = sample[pair_cols].copy()
    pair_df["Status"] = pair_df["left"].map({0: "Stayed", 1: "Left"})
    fig = sns.pairplot(pair_df.drop(columns="left"),
                       hue="Status", palette={"Stayed": GRN, "Left": RED},
                       plot_kws={"alpha": 0.3, "s": 10},
                       diag_kind="kde", corner=True)
    fig.figure.patch.set_facecolor(BG)
    for ax in fig.axes.flatten():
        if ax:
            ax.set_facecolor(BG)
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
    st.pyplot(fig.figure)
    insight("Clear separation in the satisfaction-vs-hours and "
            "satisfaction-vs-evaluation planes confirms these are the most "
            "discriminative features for the model.")

# ═══════════════════════════════════════════════════════════════
# TAB 3 – MODEL RESULTS
# ═══════════════════════════════════════════════════════════════
with tab_model:
    sh("Model Training & Evaluation")

    if not run_btn and "model_results" not in st.session_state:
        st.info("Click **🚀 Train Models** in the sidebar to start training.")
    else:
        if run_btn:
            with st.spinner("Training… (may take 1-2 min)"):
                res = train_models(df, model_choice, cv_method)
            st.session_state["model_results"] = res

        (results_df, best_pipe, feat_names,
         X_test, y_test, all_trained) = st.session_state["model_results"]
        best_name = results_df.iloc[0]["Model"]

        st.success(f"✅  Best model: **{best_name}** · "
                   f"ROC-AUC = {results_df.iloc[0]['ROC-AUC']:.4f}")

        st.dataframe(
            results_df.style
            .highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                color="#14532d")
            .format("{:.4f}",
                    subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]),
            use_container_width=True)

        st.markdown("---")
        inspect_name = st.selectbox("Inspect model", list(all_trained.keys()))
        inspect_pipe = all_trained[inspect_name]
        y_pred  = inspect_pipe.predict(X_test)
        y_proba = inspect_pipe.predict_proba(X_test)[:, 1]

        c_cm, c_roc = st.columns(2)
        with c_cm:
            sh("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = dark_fig(5, 4)
            disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Left"])
            disp.plot(ax=ax, colorbar=False, cmap="Reds")
            ax.tick_params(colors="white")
            for txt in ax.texts:
                txt.set_color("white")
            ax.set_xlabel("Predicted", color="white")
            ax.set_ylabel("Actual",    color="white")
            st.pyplot(fig)

        with c_roc:
            sh("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            fig, ax = dark_fig(5, 4)
            ax.plot(fpr, tpr, color=ACC, lw=2, label=f"AUC = {auc_val:.4f}")
            ax.plot([0, 1], [0, 1], "--", color=MUTED)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(facecolor=BG, labelcolor="white")
            ax.set_title("ROC Curve", color="white", fontfamily="monospace")
            st.pyplot(fig)

        clf = inspect_pipe.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            sh(f"Top-10 Feature Importances — {inspect_name}")
            imp = (pd.Series(clf.feature_importances_, index=feat_names)
                   .nlargest(10).sort_values())
            fig, ax = dark_fig(9, 4)
            imp.plot.barh(ax=ax, color=ACC)
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importances", color="white", fontfamily="monospace")
            st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════
# TAB 4 – EXPLAINABILITY (XAI)
# ═══════════════════════════════════════════════════════════════
with tab_xai:
    sh("Explainable AI — SHAP Analysis")

    if "model_results" not in st.session_state:
        st.warning("Train models first.")
    elif not SHAP_OK:
        st.warning("Install **shap** (`pip install shap`) to enable XAI.")
    else:
        (results_df, best_pipe, feat_names,
         X_test, y_test, all_trained) = st.session_state["model_results"]

        xai_model_name = st.selectbox("Choose model for XAI", list(all_trained.keys()))
        xai_pipe = all_trained[xai_model_name]
        clf_xai  = xai_pipe.named_steps["classifier"]
        prep_xai = xai_pipe.named_steps["preprocessor"]
        X_test_t = pd.DataFrame(prep_xai.transform(X_test), columns=feat_names)

        tree_models = ["Decision Tree", "Random Forest", "XGBoost"]

        if xai_model_name in tree_models:
            with st.spinner("Computing SHAP values…"):
                explainer = shap.TreeExplainer(clf_xai)
                shap_vals  = explainer.shap_values(X_test_t)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]

            sh("SHAP Beeswarm — Global Feature Impact")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            shap.summary_plot(shap_vals, X_test_t, plot_type="dot", show=False)
            plt.gcf().set_facecolor(BG)
            for txt in plt.gcf().findobj(plt.Text):
                txt.set_color("white")
            st.pyplot(plt.gcf())
            plt.clf()

            insight("Each dot = one employee. **Pink = high feature value, blue = low**. "
                    "Dots to the right push the prediction toward 'will leave'. "
                    "**Low satisfaction** is the single biggest driver of attrition.")

            sh("SHAP Waterfall — Single Employee Explanation")
            idx = st.slider("Employee index (test set)", 0, len(X_test_t) - 1, 0)
            base_val = (explainer.expected_value
                        if not isinstance(explainer.expected_value, list)
                        else explainer.expected_value[1])
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            fig2.patch.set_facecolor(BG)
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[idx],
                    base_values=base_val,
                    data=X_test_t.iloc[idx],
                    feature_names=feat_names,
                ), show=False, max_display=12)
            plt.gcf().set_facecolor(BG)
            for txt in plt.gcf().findobj(plt.Text):
                txt.set_color("white")
            st.pyplot(plt.gcf())
            plt.clf()

            # Natural-language SHAP explanation
            sh("Natural Language Explanation")
            row_shap = (pd.Series(shap_vals[idx], index=feat_names)
                        .abs().sort_values(ascending=False))
            top3     = row_shap.head(3)
            actual_pred = xai_pipe.predict(X_test.iloc[[idx]])[0]
            prob        = xai_pipe.predict_proba(X_test.iloc[[idx]])[0, 1]
            direction   = "likely to **leave**" if actual_pred == 1 else "likely to **stay**"
            insight(
                f"For this employee, the model predicts they are {direction} "
                f"(leave probability = **{prob:.1%}**). "
                f"Top 3 drivers: "
                f"**{top3.index[0]}** (|impact|={top3.iloc[0]:.3f}), "
                f"**{top3.index[1]}** (|impact|={top3.iloc[1]:.3f}), "
                f"**{top3.index[2]}** (|impact|={top3.iloc[2]:.3f})."
            )
        else:
            st.info("SHAP TreeExplainer supports Decision Tree, Random Forest, "
                    "and XGBoost. Please select one of those models.")

# ═══════════════════════════════════════════════════════════════
# TAB 5 – PREDICT
# ═══════════════════════════════════════════════════════════════
with tab_predict:
    sh("Single Employee Risk Prediction")

    if "model_results" not in st.session_state:
        st.warning("Train models first.")
    else:
        (results_df, best_pipe, feat_names,
         X_test, y_test, all_trained) = st.session_state["model_results"]
        best_name = results_df.iloc[0]["Model"]
        st.caption(f"Using: **{best_name}**")

        with st.form("predict_form"):
            st.markdown("#### Employee Profile")
            r1c1, r1c2, r1c3 = st.columns(3)
            satisfaction  = r1c1.slider("Satisfaction Level",    0.0, 1.0, 0.5, 0.01)
            last_eval     = r1c2.slider("Last Evaluation Score", 0.0, 1.0, 0.7, 0.01)
            num_projects  = r1c3.number_input("# Projects", 1, 20, 4)

            r2c1, r2c2, r2c3 = st.columns(3)
            avg_hours     = r2c1.number_input("Avg Monthly Hours", 50, 400, 200)
            tenure_val    = r2c2.number_input("Tenure (years)",     1,  40,   3)
            work_accident = r2c3.radio("Work Accident?", [0, 1],
                                       format_func=lambda x: "Yes" if x else "No")

            r3c1, r3c2, r3c3 = st.columns(3)
            promoted   = r3c1.radio("Promoted last 5 yrs?", [0, 1],
                                    format_func=lambda x: "Yes" if x else "No")
            department = r3c2.selectbox("Department",
                         ["sales", "technical", "support", "IT", "product_mng",
                          "marketing", "RandD", "accounting", "hr", "management"])
            salary     = r3c3.selectbox("Salary Band", ["low", "medium", "high"])
            submitted  = st.form_submit_button("Predict 🔮", type="primary")

        if submitted:
            row = pd.DataFrame([{
                "satisfaction_level":    satisfaction,
                "last_evaluation":       last_eval,
                "number_project":        num_projects,
                "average_monthly_hours": avg_hours,
                "tenure":                tenure_val,
                "work_accident":         work_accident,
                "promotion_last_5years": promoted,
                "department":            department,
                "salary":                salary,
            }])
            prob_leave = best_pipe.predict_proba(row)[0, 1]
            prediction = best_pipe.predict(row)[0]

            if prediction == 1:
                st.markdown(
                    f'<div class="pred-leave">'
                    f'<div class="pred-emoji">🚨</div>'
                    f'<div class="pred-title" style="color:{RED}">HIGH ATTRITION RISK</div>'
                    f'<div class="pred-sub">Probability of leaving: <b>{prob_leave:.1%}</b></div>'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="pred-stay">'
                    f'<div class="pred-emoji">✅</div>'
                    f'<div class="pred-title" style="color:{GRN}">LOW ATTRITION RISK</div>'
                    f'<div class="pred-sub">Probability of leaving: <b>{prob_leave:.1%}</b></div>'
                    f'</div>', unsafe_allow_html=True)

            # Probability bar
            fig, ax = dark_fig(7, 1.5)
            ax.barh(["Risk"], [prob_leave], color=RED, height=0.4)
            ax.barh(["Risk"], [1 - prob_leave], left=[prob_leave],
                    color=GRN, height=0.4)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="white", lw=1, ls="--", alpha=0.4)
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="white")
            ax.tick_params(left=False, labelleft=False)
            ax.set_title("Attrition Probability", color="white", fontfamily="monospace")
            st.pyplot(fig)

            # SHAP for this prediction
            if SHAP_OK and best_name in ["Decision Tree", "Random Forest", "XGBoost"]:
                clf_p  = best_pipe.named_steps["classifier"]
                prep_p = best_pipe.named_steps["preprocessor"]
                row_t  = pd.DataFrame(prep_p.transform(row), columns=feat_names)
                exp    = shap.TreeExplainer(clf_p)
                sv     = exp.shap_values(row_t)
                if isinstance(sv, list):
                    sv = sv[1]
                top_feat = (pd.Series(sv[0], index=feat_names)
                            .abs().nlargest(3))

                sh("Why this prediction? (XAI)")
                for feat in top_feat.index:
                    feat_idx  = feat_names.index(feat)
                    shap_val  = sv[0][feat_idx]
                    raw_val   = row_t[feat].values[0]
                    direction_txt = ("pushing toward **leaving**" if shap_val > 0
                                     else "pushing toward **staying**")
                    insight(f"**{feat}** = {raw_val:.3f} — {direction_txt} "
                            f"(SHAP impact = {shap_val:+.3f})")

            # HR Recommendations
            sh("HR Recommendations")
            if prediction == 1:
                st.markdown(f"- 🔴 Schedule a 1:1 check-in — satisfaction score of **{satisfaction:.2f}** is below healthy range.")
                st.markdown(f"- 🔴 Review workload — **{num_projects} projects** & **{avg_hours} hrs/month** may indicate overload.")
                if not promoted:
                    st.markdown(f"- 🔴 Consider promotion eligibility — **{tenure_val} years** tenure with no recent promotion.")
                st.markdown(f"- 🔴 Compensation review recommended — employee is on **{salary}** salary band.")
            else:
                st.markdown(f"- 🟢 Employee appears engaged — satisfaction = {satisfaction:.2f}.")
                st.markdown(f"- 🟢 Workload is balanced ({num_projects} projects, {avg_hours} hrs/month).")
                st.markdown("- 🟢 Continue regular check-ins to sustain engagement.")

# ═══════════════════════════════════════════════════════════════
# TAB 6 – AI CHATBOT
# ═══════════════════════════════════════════════════════════════
with tab_chat:
    sh("💬 HR Analytics AI Assistant")
    st.markdown(
        "Ask anything about employee retention, the dataset, model results, "
        "SHAP explanations, or HR best practices. Powered by **Claude**.")

    # Build live dataset context
    left_pct_ctx = df["left"].mean() * 100
    top_dept = (df.groupby("department")["left"].mean() * 100
                .sort_values(ascending=False).head(3).to_dict())
    model_ctx = ""
    if "model_results" in st.session_state:
        r = st.session_state["model_results"][0]
        model_ctx = (f"Models trained: {list(r['Model'])}. "
                     f"Best: {r.iloc[0]['Model']} "
                     f"(AUC={r.iloc[0]['ROC-AUC']:.4f}, "
                     f"F1={r.iloc[0]['F1-Score']:.4f}).")

    data_context = (
        f"Dataset: {len(df):,} employees, {left_pct_ctx:.1f}% overall attrition. "
        f"Top attrition departments: {top_dept}. "
        f"Avg satisfaction (leavers): {df[df['left']==1]['satisfaction_level'].mean():.2f}, "
        f"(stayers): {df[df['left']==0]['satisfaction_level'].mean():.2f}. "
        f"Avg tenure: {df['tenure'].mean():.1f} yrs. "
        f"{model_ctx}"
    )

    SYSTEM_PROMPT = (
        "You are an expert HR Data Scientist and People Analytics consultant "
        "for Salifort Motors. You have deep knowledge of:\n"
        "- Employee retention, attrition analysis, and HR strategy\n"
        "- Machine learning (Logistic Regression, Decision Tree, Random Forest, XGBoost)\n"
        "- Explainable AI (SHAP values, feature importances)\n"
        "- The Salifort Motors HR dataset\n\n"
        f"Live dataset context:\n{data_context}\n\n"
        "Answer clearly and actionably. Use bullet points where helpful. "
        "Explain model/SHAP concepts in plain English suitable for HR managers. "
        "Ground answers in data where possible."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Suggested questions
    st.markdown("**Quick questions:**")
    sugg_cols = st.columns(3)
    suggestions = [
        "Why do employees leave Salifort?",
        "What does SHAP tell us about satisfaction?",
        "How can we reduce attrition in HR?",
        "Explain precision vs recall in plain English",
        "What salary changes would most reduce attrition?",
        "Which employees should we prioritise for retention?",
    ]
    for i, sugg in enumerate(suggestions):
        if sugg_cols[i % 3].button(sugg, key=f"sugg_{i}", use_container_width=True):
            st.session_state["chat_history"].append({"role": "user", "content": sugg})
            with st.spinner("Thinking…"):
                reply = call_claude(st.session_state["chat_history"], SYSTEM_PROMPT)
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    st.markdown("---")

    # Chat history display
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-label">You</div>'
                        f'<div class="chat-user">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-label">AI Assistant</div>'
                        f'<div class="chat-ai">{msg["content"]}</div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        user_q = st.text_input(
            "Ask a question…",
            placeholder="e.g. What are the main drivers of attrition?")
        send = st.form_submit_button("Send 💬", type="primary")

    if send and user_q.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_q})
        with st.spinner("Claude is thinking…"):
            reply = call_claude(st.session_state["chat_history"], SYSTEM_PROMPT)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("🗑️ Clear chat"):
        st.session_state["chat_history"] = []
        st.rerun()
