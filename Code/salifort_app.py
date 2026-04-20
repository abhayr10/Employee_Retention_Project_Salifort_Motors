"""
Salifort Motors – Employee Retention Predictor
A Streamlit web app wrapping the full EDA + ML pipeline from the capstone notebook.
"""

# ─────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────
import io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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

# ─────────────────────────────────────────────
# 1.  Page config & custom CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Salifort Motors – Employee Retention",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f1117;
    color: #e8e8e8;
}
section[data-testid="stSidebar"] * { color: #e8e8e8 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #aaa !important; }

/* ── main header ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    border-left: 5px solid #e94560;
}
.hero h1 { font-family: 'IBM Plex Mono', monospace;
           color: #fff; font-size: 2rem; margin:0; letter-spacing: -1px; }
.hero p  { color: #a0aec0; margin-top: .5rem; font-size: .95rem; }

/* ── metric cards ── */
.metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
    border-top: 3px solid #e94560;
}
.metric-card .value { font-family:'IBM Plex Mono',monospace;
                      font-size:2rem; font-weight:600; color:#e94560; }
.metric-card .label { color:#a0aec0; font-size:.8rem; margin-top:.3rem; }

/* ── section headers ── */
.section-header {
    font-family:'IBM Plex Mono',monospace;
    font-size:1.1rem; font-weight:600;
    color:#e94560; border-bottom:1px solid #e9456030;
    padding-bottom:.4rem; margin-bottom:1rem;
}

/* ── prediction box ── */
.pred-box-leave {
    background: #2d1b1b; border:2px solid #e94560;
    border-radius:12px; padding:1.5rem; text-align:center;
}
.pred-box-stay {
    background: #1b2d1b; border:2px solid #38a169;
    border-radius:12px; padding:1.5rem; text-align:center;
}
.pred-emoji  { font-size:3rem; }
.pred-result { font-family:'IBM Plex Mono',monospace; font-size:1.4rem;
               font-weight:600; margin:.5rem 0; }
.pred-prob   { color:#a0aec0; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2.  Data loading & caching
# ─────────────────────────────────────────────
CAT_COLS = ["department", "salary"]
NUM_COLS = ["satisfaction_level", "last_evaluation", "number_project",
            "average_monthly_hours", "tenure", "work_accident",
            "promotion_last_5years"]

@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns={
        "Work_accident":         "work_accident",
        "average_montly_hours":  "average_monthly_hours",
        "time_spend_company":    "tenure",
        "Department":            "department",
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
    """Train one or all models and return (results_df, best_pipeline, feature_names)."""
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
                                 "classifier__min_samples_split": [2, 5, 10]},
        "XGBoost":             {"classifier__n_estimators": [50, 100],
                                 "classifier__max_depth": [3, 5],
                                 "classifier__learning_rate": [0.1, 0.2]},
    }

    # Restrict to selected model if specified
    if model_choice != "All Models":
        models = {model_choice: models[model_choice]}
        param_grids = {model_choice: param_grids[model_choice]}

    results = []
    trained = {}

    for name, model in models.items():
        pg = param_grids[name]

        if cv_method == "5-Fold Cross-Validation":
            cv_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
            gs = GridSearchCV(cv_pipeline, pg,
                              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                              scoring="roc_auc", n_jobs=-1)
            gs.fit(X_tr_full, y_tr_full)
            best_pipeline = gs.best_estimator_
        else:  # holdout
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

            best_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", best_mdl),
            ])
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

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_name  = results_df.iloc[0]["Model"]
    best_pipe  = trained[best_name]

    # Feature names
    prep_fitted = best_pipe.named_steps["preprocessor"]
    cat_out     = prep_fitted.named_transformers_["cat"].get_feature_names_out(CAT_COLS)
    feat_names  = NUM_COLS + list(cat_out)

    return results_df, best_pipe, feat_names, X_test, y_test, trained


# ─────────────────────────────────────────────
# 3.  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 Salifort Motors")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload HR dataset (CSV)", type=["csv"],
        help="Use HR_capstone_dataset.csv or any compatible file.")

    st.markdown("---")
    st.markdown("#### ⚙️ Model Settings")

    model_options = ["All Models", "Logistic Regression", "Decision Tree",
                     "Random Forest"] + (["XGBoost"] if XGBOOST_OK else [])
    model_choice = st.selectbox("Model to train", model_options)

    cv_method = st.radio("Validation strategy",
                         ["5-Fold Cross-Validation", "Holdout (60/20 split)"])

    run_btn = st.button("🚀 Train Models", type="primary",
                        disabled=uploaded is None,
                        use_container_width=True)

    st.markdown("---")
    st.caption("Built with Streamlit • Salifort Motors Capstone")

# ─────────────────────────────────────────────
# 4.  Hero header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏭 Salifort Motors — Employee Retention</h1>
  <p>Upload the HR dataset, train machine-learning models, explore the data,
     and predict whether an employee is likely to leave.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5.  Main content
# ─────────────────────────────────────────────
if uploaded is None:
    st.info("👈  Upload **HR_capstone_dataset.csv** in the sidebar to get started.")
    st.stop()

df = load_and_clean(uploaded)

# ── Tabs ──────────────────────────────────────
tab_data, tab_eda, tab_model, tab_predict = st.tabs(
    ["📋 Data Overview", "📊 EDA", "🤖 Model Results", "🔮 Predict"])

# ═══════════════════════════════════════════════
# TAB 1 – DATA OVERVIEW
# ═══════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)

    # KPI row
    left_pct  = df["left"].mean() * 100
    stay_pct  = 100 - left_pct

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [len(df), df["left"].sum(), f"{left_pct:.1f}%", df["tenure"].median()],
        ["Total Employees", "Employees Who Left", "Attrition Rate", "Median Tenure (yrs)"]
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="value">{val}</div>
          <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**First 10 rows**")
        st.dataframe(df.head(10), use_container_width=True)

    with col_right:
        st.markdown("**Descriptive statistics**")
        st.dataframe(df.describe().T.style.format("{:.3f}"), use_container_width=True)

    st.markdown("---")
    st.markdown("**Missing values & dtypes**")
    info_df = pd.DataFrame({
        "dtype":   df.dtypes.astype(str),
        "nulls":   df.isnull().sum(),
        "null_%":  (df.isnull().mean() * 100).round(2),
        "n_unique":df.nunique(),
    })
    st.dataframe(info_df, use_container_width=True)

    # Outlier check on tenure
    st.markdown("---")
    st.markdown('<div class="section-header">Outlier Check – Tenure</div>',
                unsafe_allow_html=True)
    q25, q75 = df["tenure"].quantile([0.25, 0.75])
    iqr       = q75 - q25
    upper_lim = q75 + 1.5 * iqr
    lower_lim = q25 - 1.5 * iqr
    outliers  = df[(df["tenure"] > upper_lim) | (df["tenure"] < lower_lim)]

    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    sns.boxplot(x=df["tenure"], ax=ax, color="#e94560")
    ax.set_xlabel("Tenure (years)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig)
    st.caption(f"Upper limit: **{upper_lim:.1f}**  |  Lower limit: **{lower_lim:.1f}**  "
               f"|  Outlier rows: **{len(outliers)}**")

# ═══════════════════════════════════════════════
# TAB 2 – EDA
# ═══════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    # --- Left vs. Stayed breakdown ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Attrition distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#1e2130")
        ax.set_facecolor("#1e2130")
        counts = df["left"].value_counts()
        ax.pie(counts, labels=["Stayed", "Left"], autopct="%1.1f%%",
               colors=["#38a169", "#e94560"], textprops={"color": "white"})
        st.pyplot(fig)

    with col_b:
        st.markdown("**Attrition by department**")
        dept_left = (df.groupby("department")["left"]
                     .mean()
                     .sort_values(ascending=True) * 100)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#1e2130")
        ax.set_facecolor("#1e2130")
        dept_left.plot.barh(ax=ax, color="#e94560")
        ax.set_xlabel("Attrition rate (%)", color="white")
        ax.tick_params(colors="white")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        st.pyplot(fig)

    st.markdown("---")

    # --- Boxplot: avg monthly hours × tenure × left ---
    st.markdown("**Average Monthly Hours vs Tenure (coloured by attrition)**")
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    palette = {0: "#38a169", 1: "#e94560"}
    sns.boxplot(data=df, x="average_monthly_hours", y="tenure",
                hue="left", orient="h", ax=ax, palette=palette)
    ax.set_xlabel("Average Monthly Hours", color="white")
    ax.set_ylabel("Tenure (years)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor("#1e2130")
        for txt in legend.get_texts():
            txt.set_color("white")
    st.pyplot(fig)

    st.markdown("---")

    # --- Satisfaction & evaluation medians ---
    st.markdown("**Satisfaction & Last Evaluation — Stayed vs Left**")
    left_df   = df[df["left"] == 1]
    stayed_df = df[df["left"] == 0]
    med_df = pd.DataFrame({
        "Group": ["Stayed", "Left"],
        "Median Satisfaction": [stayed_df["satisfaction_level"].median(),
                                left_df["satisfaction_level"].median()],
        "Median Last Eval":    [stayed_df["last_evaluation"].median(),
                                left_df["last_evaluation"].median()],
    }).set_index("Group")
    st.dataframe(med_df.style.format("{:.3f}"), use_container_width=True)

    st.markdown("---")

    # --- Salary breakdown ---
    st.markdown("**Salary distribution among employees who left**")
    salary_pct = left_df["salary"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    salary_pct.sort_values().plot.barh(ax=ax, color="#e94560")
    ax.set_xlabel("% of leavers", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig)

    st.markdown("---")

    # --- Correlation heatmap ---
    st.markdown("**Correlation heatmap (numeric columns)**")
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": .8})
    ax.tick_params(colors="white")
    plt.xticks(rotation=45, ha="right", color="white")
    plt.yticks(color="white")
    st.pyplot(fig)

# ═══════════════════════════════════════════════
# TAB 3 – MODEL RESULTS
# ═══════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="section-header">Model Training & Evaluation</div>',
                unsafe_allow_html=True)

    if not run_btn and "model_results" not in st.session_state:
        st.info("Click **🚀 Train Models** in the sidebar to start training.")
    else:
        if run_btn:
            with st.spinner("Training models… this may take a minute."):
                (results_df, best_pipe, feat_names,
                 X_test, y_test, all_trained) = train_models(
                    df, model_choice, cv_method)
            st.session_state["model_results"] = (
                results_df, best_pipe, feat_names, X_test, y_test, all_trained)

        (results_df, best_pipe, feat_names,
         X_test, y_test, all_trained) = st.session_state["model_results"]

        best_name = results_df.iloc[0]["Model"]
        st.success(f"✅  Best model: **{best_name}**  |  "
                   f"ROC-AUC = {results_df.iloc[0]['ROC-AUC']:.4f}")

        # Performance table
        st.markdown("**Comparison table (sorted by ROC-AUC)**")
        styled = (results_df.style
                  .highlight_max(subset=["Accuracy", "Precision", "Recall",
                                         "F1-Score", "ROC-AUC"],
                                 color="#1b4332")
                  .format("{:.4f}", subset=["Accuracy", "Precision", "Recall",
                                            "F1-Score", "ROC-AUC"]))
        st.dataframe(styled, use_container_width=True)

        st.markdown("---")

        # Choose model to inspect
        inspect_name = st.selectbox("Inspect model details",
                                    list(all_trained.keys()),
                                    index=0)
        inspect_pipe = all_trained[inspect_name]
        y_pred  = inspect_pipe.predict(X_test)
        y_proba = inspect_pipe.predict_proba(X_test)[:, 1]

        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            fig.patch.set_facecolor("#1e2130")
            ax.set_facecolor("#1e2130")
            disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Left"])
            disp.plot(ax=ax, colorbar=False, cmap="Reds")
            ax.tick_params(colors="white")
            ax.set_xlabel("Predicted", color="white")
            ax.set_ylabel("Actual", color="white")
            for txt in ax.texts:
                txt.set_color("white")
            st.pyplot(fig)

        with col_roc:
            st.markdown("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            fig.patch.set_facecolor("#1e2130")
            ax.set_facecolor("#1e2130")
            ax.plot(fpr, tpr, color="#e94560", lw=2,
                    label=f"AUC = {auc_val:.4f}")
            ax.plot([0, 1], [0, 1], "--", color="#555")
            ax.set_xlabel("False Positive Rate", color="white")
            ax.set_ylabel("True Positive Rate",  color="white")
            ax.tick_params(colors="white")
            ax.legend(facecolor="#1e2130", labelcolor="white", framealpha=.6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            st.pyplot(fig)

        # Feature importance (tree-based only)
        clf = inspect_pipe.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            st.markdown("---")
            st.markdown(f"**Top-10 Feature Importances — {inspect_name}**")
            imp_series = (pd.Series(clf.feature_importances_, index=feat_names)
                          .nlargest(10).sort_values())
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#1e2130")
            ax.set_facecolor("#1e2130")
            imp_series.plot.barh(ax=ax, color="#e94560")
            ax.set_xlabel("Importance", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            st.pyplot(fig)

# ═══════════════════════════════════════════════
# TAB 4 – PREDICT
# ═══════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-header">Predict Employee Attrition Risk</div>',
                unsafe_allow_html=True)

    if "model_results" not in st.session_state:
        st.warning("Train the models first (click **🚀 Train Models** in the sidebar).")
    else:
        (_, best_pipe, _, _, _, _) = st.session_state["model_results"]
        best_name = st.session_state["model_results"][0].iloc[0]["Model"]

        st.caption(f"Using best model: **{best_name}**")

        with st.form("predict_form"):
            st.markdown("#### Employee Profile")

            r1c1, r1c2, r1c3 = st.columns(3)
            satisfaction   = r1c1.slider("Satisfaction Level",    0.0, 1.0, 0.5, 0.01)
            last_eval      = r1c2.slider("Last Evaluation Score", 0.0, 1.0, 0.7, 0.01)
            num_projects   = r1c3.number_input("Number of Projects", 1, 20, 4)

            r2c1, r2c2, r2c3 = st.columns(3)
            avg_hours      = r2c1.number_input("Avg Monthly Hours", 50, 400, 200)
            tenure         = r2c2.number_input("Tenure (years)",     1,  40,   3)
            work_accident  = r2c3.radio("Work Accident?", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")

            r3c1, r3c2, r3c3 = st.columns(3)
            promoted       = r3c1.radio("Promoted last 5 yrs?", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")
            department     = r3c2.selectbox("Department",
                                ["sales","technical","support","IT","product_mng",
                                 "marketing","RandD","accounting","hr","management"])
            salary         = r3c3.selectbox("Salary Band", ["low", "medium", "high"])

            submitted = st.form_submit_button("Predict 🔮", type="primary")

        if submitted:
            row = pd.DataFrame([{
                "satisfaction_level":   satisfaction,
                "last_evaluation":      last_eval,
                "number_project":       num_projects,
                "average_monthly_hours":avg_hours,
                "tenure":               tenure,
                "work_accident":        work_accident,
                "promotion_last_5years":promoted,
                "department":           department,
                "salary":               salary,
            }])

            prob_leave = best_pipe.predict_proba(row)[0, 1]
            prediction = best_pipe.predict(row)[0]

            if prediction == 1:
                st.markdown(f"""
                <div class="pred-box-leave">
                  <div class="pred-emoji">🚨</div>
                  <div class="pred-result" style="color:#e94560">HIGH ATTRITION RISK</div>
                  <div class="pred-prob">Probability of leaving: <b>{prob_leave:.1%}</b></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-box-stay">
                  <div class="pred-emoji">✅</div>
                  <div class="pred-result" style="color:#38a169">LOW ATTRITION RISK</div>
                  <div class="pred-prob">Probability of leaving: <b>{prob_leave:.1%}</b></div>
                </div>""", unsafe_allow_html=True)

            # Mini probability gauge
            st.markdown("---")
            fig, ax = plt.subplots(figsize=(6, 1.5))
            fig.patch.set_facecolor("#1e2130")
            ax.set_facecolor("#1e2130")
            ax.barh(["Risk"], [prob_leave], color="#e94560", height=0.4)
            ax.barh(["Risk"], [1 - prob_leave], left=[prob_leave],
                    color="#38a169", height=0.4)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="white")
            ax.tick_params(left=False, labelleft=False, colors="white")
            ax.axvline(0.5, color="white", lw=1, ls="--", alpha=0.4)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            ax.set_title("Attrition probability", color="white", fontsize=10)
            st.pyplot(fig)
