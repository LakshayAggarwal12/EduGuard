import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="EduGuard | Student Retention Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME + CONSTANTS
# =========================================================
NAVY = "#142D3A"
TEAL = "#168C86"
BLUE = "#527DA8"
AMBER = "#E5A647"
RED = "#C96B6B"
INK = "#26343B"
MUTED = "#71818A"
BG = "#F3F6F7"

OUTCOMES = ["Dropout", "Enrolled", "Graduate"]
OUTCOME_COLORS = {"Dropout": RED, "Enrolled": AMBER, "Graduate": TEAL}
PLOT_CONFIG = {"responsive": True, "displayModeBar": False}

# UCI "Predict students' dropout and academic success" code books
GENDER_LABELS = {0: "Female", 1: "Male"}
YES_NO = {0: "No", 1: "Yes"}
COURSE_NAMES = {
    33: "Biofuel Production Tech.",
    171: "Animation & Multimedia Design",
    8014: "Social Service (evening)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising & Marketing Mgmt.",
    9773: "Journalism & Communication",
    9853: "Basic Education",
    9991: "Management (evening)",
}

# Streamlit >= 1.50 uses width="stretch"; older versions use use_container_width.
try:
    _VERSION = tuple(int(p) for p in st.__version__.split(".")[:2])
except ValueError:
    _VERSION = (1, 0)
STRETCH = {"width": "stretch"} if _VERSION >= (1, 50) else {"use_container_width": True}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    /* Font: set on text elements only so Streamlit's icon font keeps working */
    html, body, .stApp, .stMarkdown, p, label, li, input, textarea, button,
    h1, h2, h3, h4, h5, h6, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        font-family: 'DM Sans', sans-serif;
    }}
    .stApp {{ background: {BG}; color: {INK}; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}

    .block-container {{
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background: {NAVY}; border-right: 0; }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {{ color: #EAF2F3; }}
    [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.14); margin: 1rem 0; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        padding: .45rem .6rem;
        border-radius: 8px;
        width: 100%;
        font-weight: 500;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background: rgba(255,255,255,.08); }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background: rgba(22,140,134,.35);
    }}
    [data-testid="stSidebar"] div.stButton > button {{
        background: transparent;
        color: #EAF2F3;
        border: 1px solid rgba(255,255,255,.35);
    }}
    [data-testid="stSidebar"] div.stButton > button:hover {{
        border-color: #FFFFFF;
        color: #FFFFFF;
    }}

    /* Page header */
    .page-title {{ color: {NAVY}; font-size: 1.85rem; font-weight: 700; margin: 0; letter-spacing: -.01em; }}
    .page-subtitle {{ color: {MUTED}; margin: .2rem 0 1.1rem 0; font-size: .93rem; max-width: 70ch; }}

    /* KPI cards */
    .kpi-card {{
        background: #FFFFFF;
        border: 1px solid #E3EAEC;
        border-left: 4px solid {TEAL};
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
        min-height: 104px;
        overflow-wrap: anywhere;
    }}
    .kpi-label {{ color: {MUTED}; font-size: .8rem; font-weight: 600; margin-bottom: 6px; }}
    .kpi-value {{ color: {NAVY}; font-size: 1.7rem; line-height: 1.15; font-weight: 700; }}
    .kpi-note {{ color: {MUTED}; font-size: .74rem; margin-top: 5px; }}

    /* Insight strip */
    .insight {{
        background: #FFFFFF;
        border: 1px solid #E3EAEC;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: .87rem;
        line-height: 1.45;
        height: 100%;
    }}
    .insight b {{ color: {NAVY}; }}

    /* Section headings */
    .section-heading {{ color: {NAVY}; font-size: 1.02rem; font-weight: 700; margin: .4rem 0 .1rem 0; }}
    .section-caption {{ color: {MUTED}; font-size: .78rem; margin-bottom: .5rem; }}

    /* Risk badge */
    .badge {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: .82rem; font-weight: 700; color: #FFFFFF;
    }}

    /* Containers */
    div[data-testid="stPlotlyChart"] {{
        background: #FFFFFF;
        border: 1px solid #E5EBED;
        border-radius: 10px;
        padding: 6px 6px 0 6px;
    }}
    div[data-testid="stDataFrame"] {{ border: 1px solid #E5EBED; border-radius: 9px; }}
    div[data-testid="stForm"] {{ background: #FFFFFF; border: 1px solid #E3EAEC; border-radius: 10px; }}
    div.stButton > button, div.stDownloadButton > button {{ border-radius: 8px; font-weight: 600; }}
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {{
        background: {TEAL}; border: 1px solid {TEAL}; color: #FFFFFF;
    }}
    button[role="tab"] {{ font-weight: 600; }}

    /* Responsive: let columns wrap on tablets and phones */
    @media (max-width: 1000px) {{
        [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: .85rem !important; }}
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            flex: 1 1 calc(50% - .85rem) !important;
            min-width: calc(50% - .85rem) !important;
        }}
    }}
    @media (max-width: 640px) {{
        .block-container {{ padding: 1rem .75rem 1.5rem .75rem !important; }}
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }}
        .page-title {{ font-size: 1.45rem; }}
        .kpi-value {{ font-size: 1.5rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL AND DATA
# =========================================================
def clean_columns(frame):
    frame.columns = frame.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return frame


@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")


@st.cache_data
def load_data():
    # sep=None auto-detects comma or semicolon; utf-8-sig strips the BOM
    data = clean_columns(pd.read_csv("student_data.csv", sep=None, engine="python", encoding="utf-8-sig"))
    results = clean_columns(pd.read_csv("model_results.csv", encoding="utf-8-sig"))
    return data, results


try:
    bundle = load_model()
    df, results = load_data()
except Exception as exc:
    st.error("EduGuard could not load its model or data files. Check that student_model.pkl, student_data.csv and model_results.csv are next to this script.")
    st.exception(exc)
    st.stop()

model = bundle["model"]
encoder = bundle["label_encoder"]
features = bundle["features"]
model_name = bundle.get("best_model_name", "Trained classifier")

missing_data_columns = set(features + ["Target"]) - set(df.columns)
if missing_data_columns:
    st.error(f"student_data.csv is missing required columns: {sorted(missing_data_columns)}")
    st.stop()

# Human-readable helper columns (the model never sees these)
if "Gender" in df.columns:
    df["Gender label"] = df["Gender"].map(GENDER_LABELS).fillna(df["Gender"].astype(str))
if "Course" in df.columns:
    df["Course name"] = df["Course"].map(COURSE_NAMES).fillna(df["Course"].astype(str))
for _col, _label in [("Scholarship holder", "Scholarship"), ("Tuition fees up to date", "Tuition up to date"), ("Debtor", "Debtor label")]:
    if _col in df.columns:
        df[_label] = df[_col].map(YES_NO).fillna(df[_col].astype(str))


# =========================================================
# HELPERS
# =========================================================
def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def heading(title, subtitle=None):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-caption">{subtitle}</div>', unsafe_allow_html=True)


def kpi(label, value, note="", accent=TEAL):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color:{accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig, height=300, show_legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=36, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color=INK, size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text="", font=dict(size=10)),
        showlegend=show_legend,
        hoverlabel=dict(bgcolor=NAVY, font_color="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E8EDEF")
    fig.update_yaxes(gridcolor="#EDF1F2", zeroline=False)
    return fig


def show(fig, height=300, legend=True):
    st.plotly_chart(style_fig(fig, height, legend), config=PLOT_CONFIG, **STRETCH)


def pct(numerator, denominator):
    return (numerator / denominator * 100) if denominator else 0.0


def options_with_all(series):
    values = series.dropna().unique().tolist()
    try:
        values = sorted(values)
    except TypeError:
        values = sorted(values, key=str)
    return ["All"] + values


def filter_widgets(container_cols, prefix):
    """Gender + course selectboxes that show readable labels but filter on raw codes."""
    gender_choice = course_choice = "All"
    if "Gender" in df.columns:
        gender_choice = container_cols[0].selectbox(
            "Gender", options_with_all(df["Gender"]),
            format_func=lambda v: v if v == "All" else GENDER_LABELS.get(v, str(v)),
            key=f"{prefix}_gender",
        )
    if "Course" in df.columns:
        course_choice = container_cols[1].selectbox(
            "Course", options_with_all(df["Course"]),
            format_func=lambda v: v if v == "All" else COURSE_NAMES.get(v, str(v)),
            key=f"{prefix}_course",
        )
    return gender_choice, course_choice


def apply_filters(data, gender_choice, course_choice):
    out = data
    if gender_choice != "All":
        out = out[out["Gender"] == gender_choice]
    if course_choice != "All":
        out = out[out["Course"] == course_choice]
    return out


def require_rows(view):
    if view.empty:
        st.warning("No students match these filters. Choose different filter values.")
        st.stop()


def risk_level(prob_pct):
    """Illustrative bands for communicating a dropout probability, not calibrated thresholds."""
    if prob_pct >= 66:
        return "High", RED
    if prob_pct >= 33:
        return "Medium", AMBER
    return "Low", TEAL


def badge(text, color):
    return f'<span class="badge" style="background:{color}">{text}</span>'


def share_chart(view, column, title, caption, x_title):
    """100% stacked outcome share per category: fair when group sizes differ."""
    heading(title, caption)
    if column not in view.columns:
        st.info("This field is not available in the dataset.")
        return
    fig = px.histogram(
        view, x=column, color="Target", barnorm="percent",
        category_orders={"Target": OUTCOMES}, color_discrete_map=OUTCOME_COLORS,
    )
    fig.update_traces(texttemplate="%{y:.0f}%", textposition="inside")
    fig.update_layout(xaxis_title=x_title, yaxis_title="Share of students (%)", bargap=0.35)
    show(fig, 315)


def get_importances(m):
    est = m.steps[-1][1] if hasattr(m, "steps") else m
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(np.asarray(est.coef_)).mean(axis=0)
    else:
        return None
    if len(values) != len(features):
        return None
    return pd.Series(values, index=features).sort_values(ascending=False)


def read_upload(file):
    frame = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
    return clean_columns(frame)


def default_value(col):
    s = df[col].dropna()
    return s.mode().iloc[0] if s.nunique() <= 60 else s.median()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:10px 4px 6px 4px;">
            <div style="font-size:1.6rem;font-weight:700;letter-spacing:-.03em;color:#FFFFFF;">EduGuard<span style="color:{AMBER}">.</span></div>
            <div style="font-size:.78rem;color:#B8C9CE;">Student retention intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Workspace",
        ["Overview", "Student Prediction", "Risk Analytics", "Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if st.button("Refresh data", **STRETCH):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption(f"Model: {model_name}")
    st.caption(f"{len(df):,} historical records loaded")


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":
    page_header("Education performance overview", "Historical student outcomes and academic indicators in the dataset.")

    f1, f2, _ = st.columns([1, 1.4, 2])
    gender_choice, course_choice = filter_widgets([f1, f2], "overview")
    view = apply_filters(df, gender_choice, course_choice)
    require_rows(view)

    total = len(view)
    dropout = int((view["Target"] == "Dropout").sum())
    graduate = int((view["Target"] == "Graduate").sum())
    enrolled = int((view["Target"] == "Enrolled").sum())
    grade_col = "Curricular units 1st sem (grade)"
    avg_grade = view[grade_col].mean() if grade_col in view.columns else np.nan

    cards = st.columns(5)
    with cards[0]:
        kpi("Total students", f"{total:,}", "Records in current view", NAVY)
    with cards[1]:
        kpi("Dropout rate", f"{pct(dropout, total):.1f}%", f"{dropout:,} students", RED)
    with cards[2]:
        kpi("Graduate rate", f"{pct(graduate, total):.1f}%", f"{graduate:,} students", TEAL)
    with cards[3]:
        kpi("Still enrolled", f"{pct(enrolled, total):.1f}%", f"{enrolled:,} students", BLUE)
    with cards[4]:
        kpi("Avg. 1st-sem grade", f"{avg_grade:.1f}" if pd.notna(avg_grade) else "N/A", "Historical average", AMBER)

    # ---- Auto-generated insights (computed from the current view) ----
    insights = []
    if "Course name" in view.columns and view["Course"].nunique() > 1:
        cs = view.groupby("Course name").agg(n=("Target", "size"), d=("Target", lambda x: (x == "Dropout").sum()))
        cs = cs[cs["n"] >= 30]
        if not cs.empty:
            cs["rate"] = cs["d"] / cs["n"] * 100
            top = cs["rate"].idxmax()
            insights.append(f"<b>{top}</b> has the highest dropout rate at <b>{cs.loc[top, 'rate']:.0f}%</b> (courses with 30+ students).")
    if "Tuition fees up to date" in view.columns and view["Tuition fees up to date"].nunique() > 1:
        r = view.groupby("Tuition fees up to date")["Target"].apply(lambda x: (x == "Dropout").mean() * 100)
        if 0 in r.index and 1 in r.index:
            insights.append(f"Students with unpaid tuition dropped out at <b>{r[0]:.0f}%</b> vs <b>{r[1]:.0f}%</b> for those up to date.")
    if grade_col in view.columns:
        med = view.groupby("Target")[grade_col].median()
        if "Dropout" in med.index and "Graduate" in med.index:
            insights.append(f"Median 1st-semester grade: <b>{med['Dropout']:.1f}</b> for dropouts vs <b>{med['Graduate']:.1f}</b> for graduates.")
    if insights:
        st.write("")
        icols = st.columns(len(insights))
        for col, text in zip(icols, insights):
            col.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

    st.write("")
    a, b, c = st.columns([1, 1, 1.35])
    with a:
        heading("Students by gender", "Share of records in the current view")
        if "Gender label" in view.columns:
            g = view["Gender label"].value_counts().rename_axis("Gender").reset_index(name="Students")
            fig = px.pie(g, names="Gender", values="Students", hole=0.68, color_discrete_sequence=[TEAL, BLUE, AMBER])
            fig.update_traces(textposition="outside", textinfo="percent+label")
            fig.add_annotation(text=f"<b>{total:,}</b>", x=.5, y=.5, showarrow=False, font=dict(size=18, color=NAVY))
            show(fig, 285, False)
    with b:
        heading("Outcome distribution", "Dropout, enrolled and graduate")
        o = view["Target"].value_counts().reindex(OUTCOMES, fill_value=0).rename_axis("Outcome").reset_index(name="Students")
        fig = px.pie(o, names="Outcome", values="Students", hole=0.68, color="Outcome", color_discrete_map=OUTCOME_COLORS)
        fig.update_traces(textposition="outside", textinfo="percent")
        fig.add_annotation(text=f"<b>{total:,}</b>", x=.5, y=.5, showarrow=False, font=dict(size=18, color=NAVY))
        show(fig, 285, True)
    with c:
        heading("Dropout rate by course", "Courses with at least 20 students in the view")
        if "Course name" in view.columns:
            cs = view.groupby("Course name").agg(Students=("Target", "size"), Dropouts=("Target", lambda x: (x == "Dropout").sum())).reset_index()
            cs["Dropout rate"] = cs["Dropouts"] / cs["Students"] * 100
            cs = cs[cs["Students"] >= 20] if (cs["Students"] >= 20).any() else cs
            cs = cs.sort_values("Dropout rate").tail(12)
            fig = px.bar(cs, x="Dropout rate", y="Course name", orientation="h",
                         color="Dropout rate", color_continuous_scale=["#A9D8D3", TEAL, NAVY],
                         hover_data={"Students": True, "Dropouts": True, "Dropout rate": ":.1f"})
            fig.update_traces(texttemplate="%{x:.0f}%", textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Dropout rate (%)")
            show(fig, 285, False)

    d, e = st.columns([1.2, 1])
    with d:
        heading("First-semester grades", "Grade distribution by observed outcome")
        if grade_col in view.columns:
            fig = px.histogram(view, x=grade_col, color="Target", barmode="overlay", opacity=.75, nbins=24,
                               category_orders={"Target": OUTCOMES}, color_discrete_map=OUTCOME_COLORS)
            fig.update_layout(xaxis_title="First-semester grade", yaxis_title="Students")
            show(fig, 315, True)
    with e:
        heading("Age at enrolment", "Spread of age within each outcome")
        if "Age at enrollment" in view.columns:
            fig = px.box(view, x="Target", y="Age at enrollment", color="Target", points=False,
                         category_orders={"Target": OUTCOMES}, color_discrete_map=OUTCOME_COLORS)
            fig.update_layout(xaxis_title="", yaxis_title="Age at enrolment")
            show(fig, 315, False)

    st.caption("Source: UCI student dataset. These are historical associations, not causal effects or live institutional statistics.")


# =========================================================
# STUDENT PREDICTION
# =========================================================
elif page == "Student Prediction":
    page_header("Student outcome prediction", "Score one student or a whole file with the trained end-of-first-semester model.")

    dropout_idx = list(encoder.classes_).index("Dropout")
    tab_single, tab_batch = st.tabs(["Single student", "Batch upload"])

    # ---------------- Single student ----------------
    with tab_single:
        key_fields = [
            "Age at enrollment", "Gender", "Tuition fees up to date", "Scholarship holder", "Debtor",
            "Admission grade", "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
        ]
        key_cols = [c for c in key_fields if c in features] or features[:8]
        st.caption("Fields not shown below use the most common value in the training data.")

        with st.form("single_student"):
            values = {c: default_value(c) for c in features}
            grid = st.columns(3)
            for i, col in enumerate(key_cols):
                s = df[col].dropna()
                box = grid[i % 3]
                if s.nunique() <= 2:
                    opts = sorted(s.unique().tolist())
                    labels = GENDER_LABELS if col == "Gender" else YES_NO
                    default_idx = opts.index(values[col]) if values[col] in opts else 0
                    values[col] = box.selectbox(col, opts, index=default_idx, format_func=lambda v, L=labels: L.get(v, str(v)))
                elif pd.api.types.is_integer_dtype(s):
                    lo, hi = int(s.min()), int(s.max())
                    values[col] = box.number_input(col, min_value=lo, max_value=hi, value=int(np.clip(s.median(), lo, hi)), step=1)
                else:
                    lo, hi = float(s.min()), float(s.max())
                    values[col] = box.number_input(col, min_value=lo, max_value=hi, value=float(np.clip(s.median(), lo, hi)), step=0.5)
            submitted = st.form_submit_button("Predict outcome", type="primary")

        if submitted:
            row = pd.DataFrame([values])[features]
            try:
                row = row.astype({c: df[c].dtype for c in features})
            except (ValueError, TypeError):
                pass
            try:
                probs = model.predict_proba(row)[0]
                label = encoder.inverse_transform([int(np.argmax(probs))])[0]
                p_drop = float(probs[dropout_idx] * 100)
                level, color = risk_level(p_drop)

                left, right = st.columns([1, 1.2])
                with left:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=p_drop,
                        number={"suffix": "%", "font": {"size": 38, "color": NAVY}},
                        title={"text": "Dropout probability", "font": {"size": 14, "color": MUTED}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": NAVY},
                            "steps": [
                                {"range": [0, 33], "color": "#CFE8E5"},
                                {"range": [33, 66], "color": "#F6E3BC"},
                                {"range": [66, 100], "color": "#EBC5C5"},
                            ],
                        },
                    ))
                    show(gauge, 270, False)
                with right:
                    st.markdown(f"**Predicted outcome:** {label} &nbsp; {badge(level + ' risk', color)}", unsafe_allow_html=True)
                    prob_df = pd.DataFrame({"Outcome": encoder.classes_, "Probability": probs * 100})
                    fig = px.bar(prob_df, x="Probability", y="Outcome", orientation="h", color="Outcome",
                                 color_discrete_map=OUTCOME_COLORS)
                    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False)
                    fig.update_layout(xaxis=dict(range=[0, 110], title="Probability (%)"), yaxis_title="")
                    show(fig, 200, False)
                st.caption("Risk bands (under 33%, 33–66%, over 66%) are for readability only. The model's probabilities are not guaranteed to be calibrated.")
            except Exception as exc:
                st.error("The model could not score this record.")
                st.exception(exc)

    # ---------------- Batch upload ----------------
    with tab_batch:
        st.info("Upload the same features used in training. Second-semester fields and the Target column are not needed.")
        st.download_button(
            "Download CSV template",
            pd.DataFrame(columns=features).to_csv(index=False),
            "eduguard_student_template.csv",
            "text/csv",
        )
        uploaded = st.file_uploader("Student records (CSV)", type=["csv"])
        if uploaded is not None:
            try:
                input_df = read_upload(uploaded)
                missing = sorted(set(features) - set(input_df.columns))
                if missing:
                    st.error(f"Missing required columns: {missing}")
                elif input_df.empty:
                    st.warning("The uploaded CSV has no student records.")
                else:
                    input_df = input_df[features].copy()
                    for col in features:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
                    if input_df.isna().any().any():
                        bad = input_df.columns[input_df.isna().any()].tolist()
                        st.error(f"Missing or non-numeric values found in: {bad}. Fix the file and upload it again.")
                    else:
                        preds = encoder.inverse_transform(model.predict(input_df))
                        probs = model.predict_proba(input_df)
                        out = input_df.copy()
                        out.insert(0, "Predicted outcome", preds)
                        out.insert(1, "Dropout probability (%)", (probs[:, dropout_idx] * 100).round(1))
                        out.insert(2, "Risk level", [risk_level(p)[0] for p in out["Dropout probability (%)"]])
                        out = out.sort_values("Dropout probability (%)", ascending=False)

                        m = st.columns(4)
                        m[0].metric("Records scored", f"{len(out):,}")
                        m[1].metric("Predicted dropout", int((preds == "Dropout").sum()))
                        m[2].metric("Predicted enrolled", int((preds == "Enrolled").sum()))
                        m[3].metric("Predicted graduate", int((preds == "Graduate").sum()))

                        left, right = st.columns([1, 2])
                        with left:
                            heading("Risk levels")
                            counts = out["Risk level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0).reset_index()
                            counts.columns = ["Risk level", "Students"]
                            fig = px.bar(counts, x="Risk level", y="Students", color="Risk level",
                                         color_discrete_map={"High": RED, "Medium": AMBER, "Low": TEAL})
                            show(fig, 300, False)
                        with right:
                            heading("Highest-risk students first")
                            st.dataframe(out, hide_index=True, height=300, **STRETCH)
                        st.download_button("Download prediction report", out.to_csv(index=False),
                                           "eduguard_predictions.csv", "text/csv", type="primary")
            except Exception as exc:
                st.error("The uploaded file could not be processed.")
                st.exception(exc)


# =========================================================
# RISK ANALYTICS
# =========================================================
elif page == "Risk Analytics":
    page_header("Student risk analytics", "How student characteristics vary across observed outcomes.")

    f1, f2, _ = st.columns([1, 1.4, 2])
    gender_choice, course_choice = filter_widgets([f1, f2], "risk")
    view = apply_filters(df, gender_choice, course_choice)
    require_rows(view)
    grade_col = "Curricular units 1st sem (grade)"

    r1, r2 = st.columns(2)
    with r1:
        heading("Age at enrolment by outcome")
        fig = px.histogram(view, x="Age at enrollment", color="Target", barmode="overlay", opacity=.75, nbins=25,
                           category_orders={"Target": OUTCOMES}, color_discrete_map=OUTCOME_COLORS)
        fig.update_layout(xaxis_title="Age at enrolment", yaxis_title="Students")
        show(fig, 330)
    with r2:
        heading("First-semester grades by outcome")
        fig = px.box(view, x="Target", y=grade_col, color="Target", points=False,
                     category_orders={"Target": OUTCOMES}, color_discrete_map=OUTCOME_COLORS)
        fig.update_layout(xaxis_title="", yaxis_title="1st-semester grade")
        show(fig, 330, False)

    r3, r4 = st.columns(2)
    with r3:
        share_chart(view, "Scholarship", "Scholarship status and outcome", "Outcome share within each group", "Scholarship holder")
    with r4:
        share_chart(view, "Tuition up to date", "Tuition status and outcome", "Outcome share within each group", "Tuition fees up to date")

    r5, r6 = st.columns(2)
    with r5:
        heading("Features most associated with dropout", "Correlation with dropping out (1 = dropout, 0 = otherwise)")
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(view[c])]
        corr = view[numeric].corrwith((view["Target"] == "Dropout").astype(int)).dropna()
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(10).sort_values()
        if corr.empty:
            st.info("Not enough variation in this view to compute correlations.")
        else:
            fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                                   marker_color=[RED if v > 0 else TEAL for v in corr.values],
                                   hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
            fig.update_layout(xaxis_title="Correlation with dropout", yaxis_title="")
            show(fig, 360, False)
    with r6:
        approved = "Curricular units 1st sem (approved)"
        heading("Units approved in 1st semester", "Passed course units by outcome")
        if approved in view.columns:
            fig = px.box(view, x="Target", y=approved, color="Target", points=False,
                         category_orders={"Target": OUTCOMES}, color_discrete_map=OUTCOME_COLORS)
            fig.update_layout(xaxis_title="", yaxis_title="Units approved")
            show(fig, 360, False)
        else:
            st.info("This field is not available in the dataset.")

    st.caption("These charts describe patterns in the historical dataset. They do not show that a characteristic causes a student to drop out.")


# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":
    page_header("Model performance", "Candidate classifiers evaluated during model development.")

    if results.empty:
        st.warning("model_results.csv is empty.")
        st.stop()

    lower = {c.lower(): c for c in results.columns}
    model_col = lower.get("model")
    f1_col = lower.get("f1 score") or lower.get("f1")
    acc_col = lower.get("accuracy")

    best = results[results[model_col] == model_name] if model_col else results.iloc[0:0]
    m = st.columns(3)
    with m[0]:
        kpi("Selected model", model_name, "Used for all predictions", NAVY)
    with m[1]:
        kpi("Accuracy", f"{best.iloc[0][acc_col]:.3f}" if (not best.empty and acc_col) else "See table", "Held-out test set", TEAL)
    with m[2]:
        kpi("Macro F1", f"{best.iloc[0][f1_col]:.3f}" if (not best.empty and f1_col) else "See table", "Held-out test set", BLUE)

    st.write("")
    heading("Algorithm comparison", "Best value in each metric is highlighted")
    numeric_cols = results.select_dtypes("number").columns.tolist()
    try:
        styled = results.style.format({c: "{:.3f}" for c in numeric_cols}).highlight_max(subset=numeric_cols, color="#DDF0EE")
        st.dataframe(styled, hide_index=True, **STRETCH)
    except Exception:
        st.dataframe(results, hide_index=True, **STRETCH)

    if model_col and f1_col and acc_col:
        x, y = st.columns(2)
        with x:
            heading("Macro F1 score")
            fig = px.bar(results.sort_values(f1_col), x=f1_col, y=model_col, orientation="h",
                         color=f1_col, color_continuous_scale=["#B7DCD8", TEAL, NAVY])
            fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Macro F1", yaxis_title="")
            show(fig, 320, False)
        with y:
            heading("Accuracy")
            fig = px.bar(results.sort_values(acc_col), x=acc_col, y=model_col, orientation="h",
                         color=acc_col, color_continuous_scale=["#D6E1EA", BLUE, NAVY])
            fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Accuracy", yaxis_title="")
            show(fig, 320, False)

    importances = get_importances(model)
    if importances is not None:
        heading("What the selected model relies on", "Top 10 features by importance")
        top = importances.head(10).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h", color_discrete_sequence=[TEAL])
        fig.update_layout(xaxis_title="Importance", yaxis_title="")
        show(fig, 360, False)

    st.caption("Metrics come from the held-out test split used during training. They are not live metrics.")