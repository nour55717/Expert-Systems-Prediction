import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CardioSense | Heart Disease Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RAW_RANGES = {
    "age": (29, 77),
    "trestbps": (94, 200),
    "chol": (126, 564),
    "thalach": (71, 202),
    "oldpeak": (0.0, 6.2),
}

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "ca": "Blocked Vessels",
    "cp_1": "Chest Pain Type 1",
    "cp_2": "Chest Pain Type 2",
    "cp_3": "Chest Pain Type 3",
    "restecg_1": "Rest ECG Type 1",
    "restecg_2": "Rest ECG Type 2",
    "slope_1": "Slope Type 1",
    "slope_2": "Slope Type 2",
    "thal_1": "Thal Type 1",
    "thal_2": "Thal Type 2",
    "thal_3": "Thal Type 3",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root{
            --bg1:#f7fbff;
            --bg2:#eef5fb;
            --card:#ffffff;
            --line:#dbe7f3;
            --text:#0f172a;
            --soft:#334155;
            --muted:#5b6b80;
            --primary:#1456d9;
            --accent:#18b7c9;
            --high:#ef4444;
            --medium:#f59e0b;
            --low:#10b981;
            --shadow:0 18px 45px rgba(15,23,42,.07);
        }

        .stApp{
            background:
                radial-gradient(circle at 0% 0%, rgba(24,183,201,.08), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(20,86,217,.08), transparent 30%),
                linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
            color:var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0px !important;
        }
        .stAppToolbar {
            display: none !important;
        }
        div[data-testid="stToolbar"] {
            display: none !important;
        }
        div[data-testid="stDecoration"] {
            display: none !important;
        }
        #MainMenu {
            visibility: hidden !important;
        }
        footer {
            visibility: hidden !important;
        }

        .block-container{
            max-width:1280px;
            padding-top:2.2rem;
            padding-bottom:2rem;
        }

        h1,h2,h3,h4,h5{
            color:var(--text) !important;
            letter-spacing:-0.02em;
        }

        p, label, div{
            color:var(--soft);
        }

        .topbar{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:18px;
            margin-bottom:18px;
            padding:0;
        }

        .brand-wrap{
            display:flex;
            align-items:center;
            gap:14px;
        }

        .brand-logo{
            width:58px;
            height:58px;
            border-radius:18px;
            background:linear-gradient(135deg,#1456d9,#18b7c9);
            color:white !important;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:1.5rem;
            font-weight:800;
            box-shadow:0 12px 28px rgba(20,86,217,.24);
            flex-shrink:0;
        }

        .brand-title{
            font-size:1.18rem;
            font-weight:900;
            color:#0f172a !important;
            margin:0;
            line-height:1.1;
        }

        .brand-sub{
            margin:4px 0 0 0;
            color:#6b7c93 !important;
            font-size:.92rem;
        }

        .top-pill{
            background:rgba(255,255,255,.85);
            border:1px solid rgba(148,163,184,.18);
            border-radius:999px;
            padding:12px 16px;
            font-size:.84rem;
            font-weight:800;
            color:#1e293b !important;
            box-shadow:0 8px 20px rgba(15,23,42,.05);
            white-space:nowrap;
        }

        .hero{
            position:relative;
            overflow:hidden;
            background:linear-gradient(135deg, rgba(255,255,255,.98), rgba(244,249,255,.96));
            border:1px solid rgba(226,232,240,.9);
            border-radius:34px;
            padding:46px 42px;
            box-shadow:0 24px 60px rgba(15,23,42,.07);
            margin-bottom:1.2rem;
        }

        .hero::before{
            content:"";
            position:absolute;
            top:-40px;
            right:-40px;
            width:180px;
            height:180px;
            border-radius:50%;
            background:radial-gradient(circle, rgba(24,183,201,.18), rgba(24,183,201,0));
        }

        .hero::after{
            content:"";
            position:absolute;
            right:-70px;
            bottom:-90px;
            width:280px;
            height:280px;
            border-radius:50%;
            background:radial-gradient(circle, rgba(20,86,217,.12), rgba(20,86,217,.02));
        }

        .eyebrow{
            display:inline-block;
            padding:9px 15px;
            border-radius:999px;
            background:rgba(20,86,217,.08);
            color:#1456d9 !important;
            font-size:.82rem;
            font-weight:900;
            margin-bottom:16px;
        }

        .hero-title{
            font-size:3.25rem;
            line-height:1.04;
            font-weight:900;
            margin:0;
            color:#0f172a !important;
            max-width:760px;
        }

        .hero-subtitle{
            font-size:1.02rem;
            color:#516274 !important;
            line-height:1.85;
            max-width:760px;
            margin:18px 0 22px 0;
        }

        .hero-badges{
            display:flex;
            gap:12px;
            flex-wrap:wrap;
            margin-top:8px;
        }

        .hero-badge{
            background:white;
            border:1px solid rgba(148,163,184,.16);
            border-radius:999px;
            padding:11px 15px;
            font-size:.84rem;
            font-weight:800;
            color:#1e293b !important;
            box-shadow:0 8px 18px rgba(15,23,42,.04);
        }

        .glass-card{
            background:rgba(255,255,255,.95);
            border:1px solid rgba(226,232,240,.9);
            box-shadow:var(--shadow);
            border-radius:28px;
            padding:24px 24px 20px 24px;
            margin-bottom:1rem;
        }

        .section-kicker{
            color:#1456d9 !important;
            font-size:.83rem;
            font-weight:900;
            text-transform:uppercase;
            letter-spacing:.09em;
            margin-bottom:6px;
        }

        .section-title{
            color:#0f172a !important;
            font-size:1.68rem;
            font-weight:900;
            margin-bottom:.45rem;
        }

        .section-copy{
            color:#5b6b80 !important;
            font-size:.97rem;
            line-height:1.7;
        }

        .stat-card{
            background:#ffffff;
            border:1px solid rgba(226,232,240,.95);
            border-radius:22px;
            padding:18px;
            box-shadow:0 12px 28px rgba(15,23,42,.05);
        }

        .stat-label{
            color:#64748b !important;
            font-size:.88rem;
            margin-bottom:7px;
            font-weight:700;
        }

        .stat-value{
            color:#0f172a !important;
            font-size:1.65rem;
            font-weight:900;
            line-height:1.1;
        }

        .input-label{
            color:#0f172a !important;
            font-size:.98rem;
            font-weight:900;
            margin-bottom:2px;
        }

        .input-help{
            color:#6b7c93 !important;
            font-size:.82rem;
            margin-bottom:10px;
        }

        .risk-banner{
            border-radius:28px;
            padding:28px;
            color:white;
            box-shadow:0 18px 35px rgba(15,23,42,.14);
            margin-bottom:1rem;
        }

        .risk-title{
            font-size:2.1rem;
            font-weight:900;
            margin-bottom:4px;
            color:white !important;
        }

        .risk-copy{
            color:rgba(255,255,255,.92) !important;
            font-size:.98rem;
        }

        .pill{
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:8px 13px;
            border-radius:999px;
            background:rgba(255,255,255,.16);
            font-size:.84rem;
            font-weight:800;
            color:white !important;
            margin-top:14px;
        }

        .factor-item{
            display:flex;
            align-items:flex-start;
            gap:12px;
            padding:13px 0;
            border-bottom:1px solid rgba(148,163,184,.14);
        }

        .factor-icon{
            width:34px;
            height:34px;
            border-radius:12px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:1rem;
            font-weight:800;
            flex-shrink:0;
        }

        .factor-text{
            color:#1e293b !important;
            line-height:1.5;
            font-size:.96rem;
            font-weight:500;
        }

        .rec-card{
            padding:18px;
            border-radius:24px;
            background:linear-gradient(180deg, #ffffff, #f8fbff);
            border:1px solid rgba(226,232,240,.95);
            min-height:145px;
            box-shadow:0 12px 28px rgba(15,23,42,.05);
        }

        .recommendation-title{
            font-weight:900;
            margin-top:10px;
            margin-bottom:8px;
            color:#0f172a !important;
        }

        .rec-body{
            color:#55657a !important;
            line-height:1.6;
            font-size:.94rem;
        }

        .footer-note{
            text-align:center;
            color:#5b6b80 !important;
            font-size:.9rem;
            padding:12px 0 8px;
        }

        .stTabs [data-baseweb="tab-list"]{
            gap:10px;
        }

        .stTabs [data-baseweb="tab"]{
            border-radius:999px;
            padding:10px 18px;
            background:rgba(255,255,255,.75);
            color:#1e293b !important;
            font-weight:800;
        }

        .stButton > button{
            width:100%;
            border-radius:18px;
            background:linear-gradient(135deg, #1456d9, #18b7c9);
            color:white !important;
            font-weight:900;
            border:0;
            padding:.88rem 1rem;
            box-shadow:0 14px 30px rgba(20,86,217,.22);
        }

        .stSlider label, .stSelectbox label{
            color:#0f172a !important;
            font-weight:900 !important;
        }

        /* Selected dropdown box */
        div[data-baseweb="select"] > div{
            background-color:#ffffff !important;
            color:#0f172a !important;
            border:1px solid #d9e4ef !important;
            border-radius:16px !important;
        }

        /* Popover container */
        div[data-baseweb="popover"]{
            background:transparent !important;
        }

        /* Dropdown menu panel */
        div[data-baseweb="menu"]{
            background-color:#ffffff !important;
            border-radius:14px !important;
            box-shadow:0 12px 28px rgba(15,23,42,.12) !important;
            overflow:hidden !important;
        }

        /* Menu list */
        ul[role="listbox"]{
            background-color:#ffffff !important;
        }

        /* Dropdown options */
        li[role="option"]{
            background-color:#ffffff !important;
            color:#0f172a !important;
        }

        /* Hover / selected look */
        li[role="option"]:hover,
        li[aria-selected="true"]{
            background-color:#f3f6fb !important;
            color:#0f172a !important;
        }

        .stNumberInput input, .stTextInput input{
            color:#0f172a !important;
        }

        [data-testid="stMarkdownContainer"] p{
            color:inherit;
        }

        .mini-note{
            margin-top:10px;
            background:#f8fbff;
            border:1px solid #e3edf7;
            border-radius:16px;
            padding:12px 14px;
            color:#59708a !important;
            font-size:.88rem;
            line-height:1.6;
        }

        @media (max-width: 900px){
            .topbar{
                flex-direction:column;
                align-items:flex-start;
            }
            .hero{
                padding:30px 24px;
            }
            .hero-title{
                font-size:2.4rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = BASE_DIR / "D:/Semester 6/Intelligent Programming/project 1/cleaned_data.csv"
    df = pd.read_csv(csv_path)
    return df.replace({True: 1, False: 0})


def expert_predict_normalized(row: pd.Series) -> int:
    risk = 0
    if row["chol"] > 0.6:
        risk += 1
    if row["age"] > 0.5:
        risk += 1
    if row["trestbps"] > 0.6:
        risk += 1
    if row["thalach"] < 0.4:
        risk += 1
    if row["exang"] == 1:
        risk += 1
    if row["ca"] >= 2:
        risk += 1
    if row["oldpeak"] > 0.5:
        risk += 1
    return 1 if risk >= 3 else 0


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    tree_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }

    expert_pred = X_test.apply(expert_predict_normalized, axis=1)
    expert_metrics = {
        "Accuracy": accuracy_score(y_test, expert_pred),
        "Precision": precision_score(y_test, expert_pred, zero_division=0),
        "Recall": recall_score(y_test, expert_pred, zero_division=0),
        "F1": f1_score(y_test, expert_pred, zero_division=0),
    }

    comparison = pd.DataFrame(
        [tree_metrics, expert_metrics],
        index=["Decision Tree", "Expert System"],
    )
    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return model, tree_metrics, expert_metrics, comparison, importances, grid.best_params_


def normalize_raw(value: float, key: str) -> float:
    lo, hi = RAW_RANGES[key]
    value = min(max(value, lo), hi)
    if math.isclose(hi, lo):
        return 0.0
    return (value - lo) / (hi - lo)


def map_inputs_to_features(user_inputs: Dict) -> pd.DataFrame:
    sex_val = 1 if user_inputs["sex"] == "Male" else 0

    row = {
        "age": normalize_raw(user_inputs["age"], "age"),
        "sex": sex_val,
        "trestbps": normalize_raw(user_inputs["trestbps"], "trestbps"),
        "chol": normalize_raw(user_inputs["chol"], "chol"),
        "fbs": 1 if user_inputs["fbs"] == "Yes" else 0,
        "thalach": normalize_raw(user_inputs["thalach"], "thalach"),
        "exang": 1 if user_inputs["exang"] == "Yes" else 0,
        "oldpeak": normalize_raw(user_inputs["oldpeak"], "oldpeak"),
        "ca": int(user_inputs["ca"]),
        "cp_1": 1 if user_inputs["cp"] == "Type 1" else 0,
        "cp_2": 1 if user_inputs["cp"] == "Type 2" else 0,
        "cp_3": 1 if user_inputs["cp"] == "Type 3" else 0,
        "restecg_1": 1 if user_inputs["restecg"] == "Type 1" else 0,
        "restecg_2": 1 if user_inputs["restecg"] == "Type 2" else 0,
        "slope_1": 1 if user_inputs["slope"] == "Type 1" else 0,
        "slope_2": 1 if user_inputs["slope"] == "Type 2" else 0,
        "thal_1": 1 if user_inputs["thal"] == "Type 1" else 0,
        "thal_2": 1 if user_inputs["thal"] == "Type 2" else 0,
        "thal_3": 1 if user_inputs["thal"] == "Type 3" else 0,
    }

    return pd.DataFrame([row])


def expert_rule_signals_raw(user_inputs: Dict) -> Tuple[int, int, int, List[Tuple[str, str]]]:
    high = medium = low = 0
    details: List[Tuple[str, str]] = []

    if user_inputs["chol"] > 240:
        high += 1
        details.append(("high", "Cholesterol is above 240, which is flagged as a high-risk signal in the expert system."))
    if user_inputs["age"] > 50 and user_inputs["trestbps"] > 140:
        high += 1
        details.append(("high", "Age above 50 combined with elevated blood pressure triggered a strong warning rule."))
    if user_inputs["thalach"] < 120:
        medium += 1
        details.append(("medium", "Maximum heart rate is relatively low, so the expert system added a medium-risk signal."))
    if user_inputs["oldpeak"] > 2:
        high += 1
        details.append(("high", "ST depression value is above 2, which may reflect notable cardiac stress."))
    if user_inputs["exang"] == "Yes":
        medium += 1
        details.append(("medium", "Exercise-induced angina was selected, which raises the rule-based risk level."))
    if user_inputs["fbs"] == "Yes":
        medium += 1
        details.append(("medium", "Fasting blood sugar is elevated, so the expert system adds a medium-risk flag."))
    if user_inputs["sex"] == "Male" and user_inputs["age"] > 45:
        medium += 1
        details.append(("medium", "Male patient above 45 activates one of the predefined increased-risk rules."))
    if user_inputs["ca"] >= 2:
        high += 1
        details.append(("high", "Two or more colored vessels were selected, which the rule base treats as a high-risk signal."))
    if user_inputs["cp"] == "Type 1":
        low += 1
        details.append(("low", "Chest pain type 1 contributes a lower-risk indicator inside the current rule base."))
    if user_inputs["slope"] == "Type 2":
        medium += 1
        details.append(("medium", "Slope type 2 is treated as an abnormal cardiac slope in the expert system."))

    return high, medium, low, details


def get_recommendations(user_inputs: Dict, risk_level: str):
    recs = []

    if user_inputs["chol"] > 220:
        recs.append(("🥗", "Improve lipid control", "Reduce saturated fat intake and prefer fiber-rich meals to support healthier cholesterol levels."))
    if user_inputs["trestbps"] > 135:
        recs.append(("🩺", "Track blood pressure", "Monitor blood pressure regularly and review salt intake, hydration, stress, and sleep quality."))
    if user_inputs["exang"] == "Yes" or user_inputs["oldpeak"] > 2:
        recs.append(("🏥", "Clinical follow-up", "Exercise-related symptoms or elevated ST depression deserve proper medical follow-up."))
    if user_inputs["thalach"] < 120:
        recs.append(("🚶", "Build safe activity", "Discuss a guided and medically safe cardio routine before increasing exercise intensity."))

    if risk_level == "High Risk":
        recs.append(("📋", "Prioritize evaluation", "This profile shows multiple warning signals, so cardiology evaluation should be prioritized."))
    elif risk_level == "Medium Risk":
        recs.append(("🧠", "Reduce cumulative risk", "Improving blood pressure, glucose control, and activity consistency may lower overall risk."))
    else:
        recs.append(("✨", "Maintain prevention", "The current profile looks more reassuring, but long-term preventive habits still matter."))

    seen = set()
    unique = []
    for item in recs:
        if item[1] not in seen:
            seen.add(item[1])
            unique.append(item)
    return unique[:4]


def risk_palette(risk_label: str):
    if risk_label == "High Risk":
        return "linear-gradient(135deg, #ef4444, #f97316)", "Elevated cardiac warning pattern detected"
    if risk_label == "Medium Risk":
        return "linear-gradient(135deg, #f59e0b, #f97316)", "Moderate concern with mixed clinical indicators"
    return "linear-gradient(135deg, #10b981, #06b6d4)", "Lower immediate concern based on the current profile"


def render_stat_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_factor_list(details):
    color_map = {
        "high": ("rgba(239,68,68,.12)", "#ef4444", "↑"),
        "medium": ("rgba(245,158,11,.14)", "#f59e0b", "•"),
        "low": ("rgba(16,185,129,.14)", "#10b981", "✓"),
    }

    if not details:
        st.markdown(
            '<div class="section-copy">No major rule triggers were activated from the selected profile.</div>',
            unsafe_allow_html=True,
        )
        return

    html = []
    for level, text in details:
        bg, fg, icon = color_map[level]
        html.append(
            f"""
            <div class="factor-item">
                <div class="factor-icon" style="background:{bg}; color:{fg};">{icon}</div>
                <div class="factor-text">{text}</div>
            </div>
            """
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def render_recommendations(cards):
    cols = st.columns(len(cards))
    for col, (icon, title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="rec-card">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="recommendation-title">{title}</div>
                    <div class="rec-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def input_block(title: str, help_text: str):
    st.markdown(f'<div class="input-label">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="input-help">{help_text}</div>', unsafe_allow_html=True)


def render_top_brand():
    st.markdown(
        """
        <div class="topbar">
            <div class="brand-wrap">
                <div class="brand-logo">🫀</div>
                <div>
                    <div class="brand-title">CardioSense</div>
                    <div class="brand-sub">Advanced Heart Disease Detection Platform</div>
                </div>
            </div>
            <div class="top-pill">AI + Expert System Powered Assessment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">SMART CARDIOLOGY SCREENING</div>
            <h1 class="hero-title">Professional heart disease detection in a modern medical interface.</h1>
            <p class="hero-subtitle">
                CardioSense combines a rule-based expert system with a tuned decision tree model to deliver
                a clean, explainable, and presentation-ready clinical experience for heart disease risk assessment.
            </p>
            <div class="hero-badges">
                <div class="hero-badge">Clinical-Style Experience</div>
                <div class="hero-badge">Explainable Risk Logic</div>
                <div class="hero-badge">Machine Learning Prediction</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_analytics(df, comparison_df, importances):
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">Model Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Decision Tree vs Expert System</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-copy">The dashboard compares the trained machine learning model with the human-defined rule engine.</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        comparison_df[["Accuracy", "F1"]].plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">Feature Importance</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What drives the model most</div>', unsafe_allow_html=True)
        top_importances = importances.head(8).sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4.1))
        ax2.barh([FEATURE_LABELS.get(i, i) for i in top_importances.index], top_importances.values)
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    lower_left, lower_right = st.columns(2, gap="large")

    with lower_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">Data Snapshot</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key normalized distributions</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(7, 4.2))
        df[["age", "trestbps", "chol", "thalach"]].plot(kind="box", ax=ax3)
        ax3.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        st.markdown("</div>", unsafe_allow_html=True)

    with lower_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">Class Mix</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Target balance</div>', unsafe_allow_html=True)
        counts = df["target"].value_counts().sort_index()
        labels = ["No Disease", "Disease"]
        fig4, ax4 = plt.subplots(figsize=(7, 4.2))
        ax4.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    inject_css()
    df = load_data()
    model, tree_metrics, expert_metrics, comparison_df, importances, best_params = train_model(df)

    render_top_brand()
    render_hero()

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_stat_card("Dataset Rows", f"{len(df)}")
    with metric_cols[1]:
        render_stat_card("Disease Rate", f"{df['target'].mean() * 100:.1f}%")
    with metric_cols[2]:
        render_stat_card("Features", f"{df.shape[1] - 1}")
    with metric_cols[3]:
        render_stat_card("Best F1", f"{comparison_df['F1'].max() * 100:.1f}%")

    tabs = st.tabs(["Assessment", "Analytics", "Project Story"])

    with tabs[0]:
        left, right = st.columns([0.96, 1.04], gap="large")

        with left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-kicker">Clinical Intake Form</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Patient assessment inputs</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-copy">Fill the patient profile below. Each field is labeled clearly so the interface feels trustworthy, structured, and clinically readable.</div>', unsafe_allow_html=True)

            st.markdown("### Personal & Vital Information")

            col1, col2 = st.columns(2)

            with col1:
                input_block("Age", "Patient age in years")
                age = st.slider("Age", 29, 77, 54, label_visibility="collapsed")

                input_block("Sex", "Biological sex used in the dataset")
                sex = st.selectbox("Sex", ["Male", "Female"], label_visibility="collapsed")

                input_block("Resting Blood Pressure", "Measured resting blood pressure")
                trestbps = st.slider("Resting Blood Pressure", 94, 200, 132, label_visibility="collapsed")

                input_block("Cholesterol", "Serum cholesterol level")
                chol = st.slider("Cholesterol", 126, 564, 246, label_visibility="collapsed")

            with col2:
                input_block("Fasting Blood Sugar", "Whether fasting blood sugar is above 120 mg/dl")
                fbs = st.selectbox("Fasting Blood Sugar", ["No", "Yes"], label_visibility="collapsed")

                input_block("Maximum Heart Rate", "Highest achieved heart rate")
                thalach = st.slider("Maximum Heart Rate", 71, 202, 150, label_visibility="collapsed")

                input_block("Exercise Angina", "Chest pain triggered by exercise")
                exang = st.selectbox("Exercise Angina", ["No", "Yes"], label_visibility="collapsed")

                input_block("ST Depression (Oldpeak)", "ECG depression induced by exercise")
                oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.2, 0.1, label_visibility="collapsed")

            st.markdown("### Diagnostic Categories")

            col3, col4 = st.columns(2)

            with col3:
                input_block("Blocked Vessels", "Number of major vessels observed")
                ca = st.select_slider("Blocked Vessels", options=[0, 1, 2, 3], value=0, label_visibility="collapsed")

                input_block("Chest Pain Type", "Clinical chest pain classification")
                cp = st.selectbox("Chest Pain Type", ["Type 0", "Type 1", "Type 2", "Type 3"], label_visibility="collapsed")

            with col4:
                input_block("Rest ECG", "Resting electrocardiographic result")
                restecg = st.selectbox("Rest ECG", ["Type 0", "Type 1", "Type 2"], label_visibility="collapsed")

                input_block("Slope", "Slope of peak exercise ST segment")
                slope = st.selectbox("Slope", ["Type 0", "Type 1", "Type 2"], label_visibility="collapsed")

                input_block("Thal", "Thalassemia test category")
                thal = st.selectbox("Thal", ["Type 0", "Type 1", "Type 2", "Type 3"], label_visibility="collapsed")

            st.markdown(
                """
                <div class="mini-note">
                    The model uses the cleaned project dataset, while the explanation layer is informed by the expert-system logic.
                </div>
                """,
                unsafe_allow_html=True,
            )

            user_inputs = {
                "age": age,
                "sex": sex,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "ca": ca,
                "cp": cp,
                "restecg": restecg,
                "slope": slope,
                "thal": thal,
            }

            submit = st.button("Generate Clinical Assessment")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            if submit:
                features = map_inputs_to_features(user_inputs)
                ml_pred = int(model.predict(features)[0])
                ml_proba = float(model.predict_proba(features)[0][1])

                high, medium, low, details = expert_rule_signals_raw(user_inputs)

                if ml_pred == 1 or high >= 2:
                    risk_label = "High Risk"
                elif medium >= 2 or high == 1 or ml_proba >= 0.45:
                    risk_label = "Medium Risk"
                else:
                    risk_label = "Low Risk"

                gradient, subtitle = risk_palette(risk_label)
                confidence = ml_proba * 100 if ml_pred == 1 else (1 - ml_proba) * 100

                st.markdown(
                    f"""
                    <div class="risk-banner" style="background:{gradient};">
                        <div class="section-kicker" style="color:rgba(255,255,255,.78) !important;">Risk Assessment Result</div>
                        <div class="risk-title">{risk_label}</div>
                        <div class="risk-copy">{subtitle}</div>
                        <div class="pill">Machine Learning Confidence · {confidence:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                stat_cols = st.columns(4)
                with stat_cols[0]:
                    render_stat_card("High Signals", str(high))
                with stat_cols[1]:
                    render_stat_card("Medium Signals", str(medium))
                with stat_cols[2]:
                    render_stat_card("Low Signals", str(low))
                with stat_cols[3]:
                    render_stat_card("Disease Probability", f"{ml_proba * 100:.1f}%")

                c1, c2 = st.columns([1.04, 0.96], gap="large")

                with c1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-kicker">Rule Breakdown</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Why this profile was flagged</div>', unsafe_allow_html=True)
                    render_factor_list(details)
                    st.markdown("</div>", unsafe_allow_html=True)

                with c2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-kicker">Patient Snapshot</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Selected clinical indicators</div>', unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    with m1:
                        render_stat_card("Age", str(age))
                        render_stat_card("Cholesterol", str(chol))
                        render_stat_card("Oldpeak", f"{oldpeak:.1f}")
                    with m2:
                        render_stat_card("Blood Pressure", str(trestbps))
                        render_stat_card("Max Heart Rate", str(thalach))
                        render_stat_card("Blocked Vessels", str(ca))

                    st.markdown("</div>", unsafe_allow_html=True)

                chart_left, chart_right = st.columns(2, gap="large")

                with chart_left:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-kicker">Signal Distribution</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Expert system signals</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(6.8, 4.1))
                    ax.bar(["High", "Medium", "Low"], [high, medium, low])
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.markdown("</div>", unsafe_allow_html=True)

                with chart_right:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-kicker">Model Lens</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Most influential features</div>', unsafe_allow_html=True)
                    top5 = importances.head(5).sort_values(ascending=True)
                    fig2, ax2 = plt.subplots(figsize=(6.8, 4.1))
                    ax2.barh([FEATURE_LABELS.get(i, i) for i in top5.index], top5.values)
                    ax2.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-kicker">Recommended Actions</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Next clinical and lifestyle steps</div>', unsafe_allow_html=True)
                render_recommendations(get_recommendations(user_inputs, risk_label))
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.markdown(
                    """
                    <div class="glass-card" style="min-height: 620px; display:flex; flex-direction:column; justify-content:center;">
                        <div class="section-kicker">Professional Result Space</div>
                        <div class="section-title">Your assessment report will appear here</div>
                        <div class="section-copy" style="max-width:580px;">
                            The result area is designed to feel like a premium medical dashboard, with a bold risk decision,
                            explainable rule triggers, key indicators, charts, and actionable recommendations.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tabs[1]:
        render_analytics(df, comparison_df, importances)

    with tabs[2]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">Project Story</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">How this product presents your project professionally</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="section-copy">
                This interface combines cleaned data, a rule-based explanation layer, and a tuned decision tree model
                inside one polished experience that feels closer to a medical product than a classroom form.
                <br><br>
                <strong>Best Decision Tree Parameters:</strong> {best_params}<br>
                <strong>Decision Tree Metrics:</strong> Accuracy {tree_metrics['Accuracy']:.3f}, Precision {tree_metrics['Precision']:.3f}, Recall {tree_metrics['Recall']:.3f}, F1 {tree_metrics['F1']:.3f}<br>
                <strong>Expert System Metrics:</strong> Accuracy {expert_metrics['Accuracy']:.3f}, Precision {expert_metrics['Precision']:.3f}, Recall {expert_metrics['Recall']:.3f}, F1 {expert_metrics['F1']:.3f}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(comparison_df.style.format("{:.3f}"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">CardioSense — intelligent heart screening interface designed for academic presentation with a clinical product feel.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()