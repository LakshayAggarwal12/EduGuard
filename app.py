import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib


# -------------------------
# PAGE CONFIGURATION
# -------------------------

st.set_page_config(
    page_title="EduGuard",
    page_icon="🎓",
    layout="wide"
)


# -------------------------
# LOAD MODEL AND DATA
# -------------------------

@st.cache_resource
def load_model():

    return joblib.load("student_model.pkl")


@st.cache_data
def load_data():

    df = pd.read_csv("student_data.csv")

    results = pd.read_csv("model_results.csv")

    return df, results


bundle = load_model()

df, results = load_data()

model = bundle["model"]

encoder = bundle["label_encoder"]

features = bundle["features"]


# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.title("EduGuard")

st.sidebar.caption(
    "Student Retention Intelligence"
)

page = st.sidebar.radio(

    "Navigation",

    [
        "Overview",
        "Student Prediction",
        "Risk Analytics",
        "Model Performance"
    ]

)


# -------------------------
# OVERVIEW
# -------------------------

if page == "Overview":

    st.title("Student Retention Dashboard")

    st.write(
        "Monitor student outcomes and identify "
        "patterns associated with academic success."
    )

    total = len(df)

    dropout = (df["Target"] == "Dropout").sum()

    graduate = (df["Target"] == "Graduate").sum()

    enrolled = (df["Target"] == "Enrolled").sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", total)

    c2.metric(
        "Dropout Students",
        dropout,
        f"{dropout / total * 100:.1f}%"
    )

    c3.metric("Graduated", graduate)

    c4.metric("Enrolled", enrolled)

    st.divider()

    fig = px.pie(

        df,

        names="Target",

        title="Student Outcome Distribution",

        hole=0.45

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# -------------------------
# STUDENT PREDICTION
# -------------------------

elif page == "Student Prediction":

    st.title("Student Outcome Prediction")

    st.write(
        "Upload student records to predict their "
        "academic outcomes."
    )

    sample = pd.DataFrame(
        [df.drop(columns=["Target"]).iloc[0][features].to_dict()]
    )

    st.download_button(

        "Download Input Template",

        sample.to_csv(index=False),

        "student_template.csv",

        "text/csv"

    )

    uploaded_file = st.file_uploader(

        "Upload Student CSV",

        type=["csv"]

    )

    if uploaded_file:

        input_df = pd.read_csv(uploaded_file)

        missing = set(features) - set(input_df.columns)

        if missing:

            st.error(
                f"Missing columns: {list(missing)}"
            )

        else:

            input_df = input_df[features]

            predictions = model.predict(input_df)

            probabilities = model.predict_proba(input_df)

            predicted_labels = encoder.inverse_transform(
                predictions
            )

            dropout_index = list(
                encoder.classes_
            ).index("Dropout")

            output = input_df.copy()

            output["Predicted Outcome"] = predicted_labels

            output["Dropout Probability"] = (
                probabilities[:, dropout_index] * 100
            ).round(2)

            st.subheader("Prediction Results")

            st.dataframe(
                output,
                use_container_width=True
            )

            st.download_button(

                "Download Prediction Report",

                output.to_csv(index=False),

                "student_predictions.csv",

                "text/csv"

            )


# -------------------------
# RISK ANALYTICS
# -------------------------

elif page == "Risk Analytics":

    st.title("Student Risk Analytics")

    c1, c2 = st.columns(2)

    with c1:

        fig = px.histogram(

            df,

            x="Age at enrollment",

            color="Target",

            title="Age vs Student Outcome"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        fig = px.box(

            df,

            x="Target",

            y="Curricular units 1st sem (grade)",

            color="Target",

            title="Academic Performance"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    fig = px.histogram(

        df,

        x="Scholarship holder",

        color="Target",

        title="Scholarship Status Analysis"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.histogram(

        df,

        x="Tuition fees up to date",

        color="Target",

        title="Tuition Fee Analysis"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# -------------------------
# MODEL PERFORMANCE
# -------------------------

elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "Selected Model:",
        bundle["best_model_name"]
    )

    st.subheader("Algorithm Comparison")

    st.dataframe(
        results,
        use_container_width=True
    )

    fig = px.bar(

        results,

        x="Model",

        y="F1 Score",

        color="Model",

        title="F1 Score Comparison"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.bar(

        results,

        x="Model",

        y="Accuracy",

        color="Model",

        title="Accuracy Comparison"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )