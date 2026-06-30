
import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


model = joblib.load("/content/best_model.pkl")
feature_columns = joblib.load("/content/features.pkl")

df = pd.read_csv("/content/train_u6lujuX_CVtuZ9i.csv")



df['TotalIncome'] = (
    df['ApplicantIncome']
    +
    df['CoapplicantIncome']
)

df['TotalIncome_log'] = np.log(df['TotalIncome'])

df['LoanAmount_log'] = np.log(df['LoanAmount'])

df['EMI'] = (
    df['LoanAmount']
    /
    df['Loan_Amount_Term']
)

df['Income_Loan_Ratio'] = (
    df['TotalIncome']
    /
    df['LoanAmount']
)



st.sidebar.title("🏦 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Dataset",
        "EDA Dashboard",
        "Prediction"
    ]
)


# ==========================
# HOME PAGE
# ==========================

if page == "Home":

    st.title("🏦 Loan Approval Prediction System")
    st.markdown("---")

    st.markdown("""
    ### 📌 Project Overview

    This Machine Learning application predicts whether a loan application
    is likely to be approved or rejected based on applicant details such as
    income, loan amount, credit history, education, and property area.

    The model has been trained using multiple machine learning algorithms,
    with the best-performing model selected for deployment.
    """)

    st.markdown("")

    total_records = len(df)
    total_features = df.shape[1]

    approved_loans = df[df["Loan_Status"] == 1].shape[0]
    rejected_loans = total_records - approved_loans

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Total Records",
            value=f"{total_records:,}"
        )

    with col2:
        st.metric(
            label="📂 Features",
            value=total_features
        )

    with col3:
        st.metric(
            label="✅ Approved Loans",
            value=approved_loans
        )

    with col4:
        st.metric(
            label="❌ Rejected Loans",
            value=rejected_loans
        )

    st.markdown("---")

    try:
        st.image(
            "/content/bank.jpeg",
            caption="AI Powered Loan Approval Prediction",
            use_container_width=True
        )
    except:
        st.warning("Banner image not found.")

    st.markdown("---")


    st.subheader("🚀 Key Features")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:
        st.info("""
        **📈 Data Analysis**

        Explore applicant demographics,
        income patterns and loan trends.
        """)

    with feature2:
        st.success("""
        **🤖 ML Prediction**

        Predict loan approval using
        trained machine learning models.
        """)

    with feature3:
        st.warning("""
        **📊 Interactive Dashboard**

        Visualize insights through
        interactive charts and metrics.
        """)

    st.markdown("---")

    st.subheader("🛠 Technologies Used")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-Learn
    - XGBoost
    - Plotly
    - Streamlit
    - Machine Learning
    """)

    st.markdown("---")

    st.success("🎯 Navigate using the sidebar to explore the dashboard and make predictions.")

elif page == "Dataset":

    st.title("📊 Dataset Overview")

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("First 10 Rows")

    st.dataframe(df.head(10))

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())


elif page == "EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")

    col1,col2 = st.columns(2)

    with col1:

        loan_status = px.histogram(
            df,
            x="Loan_Status",
            title="Loan Status Distribution"
        )

        st.plotly_chart(
            loan_status,
            use_container_width=True
        )

    with col2:

        gender = px.histogram(
            df,
            x="Gender",
            color="Loan_Status",
            title="Gender vs Loan Status"
        )

        st.plotly_chart(
            gender,
            use_container_width=True
        )

    st.divider()

    income = px.histogram(
        df,
        x="ApplicantIncome",
        title="Applicant Income Distribution"
    )

    st.plotly_chart(
        income,
        use_container_width=True
    )

    area = px.histogram(
        df,
        x="Property_Area",
        color="Loan_Status",
        title="Property Area Analysis"
    )

    st.plotly_chart(
        area,
        use_container_width=True
    )

elif page == "Prediction":

    st.title("🤖 Loan Approval Prediction")

    col1,col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male","Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes","No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0","1","2","3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate","Not Graduate"]
        )

        self_emp = st.selectbox(
            "Self Employed",
            ["Yes","No"]
        )

    with col2:

        applicant_income = st.number_input(
            "Applicant Income",
            value=5000
        )

        co_income = st.number_input(
            "Coapplicant Income",
            value=0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            value=120
        )

        loan_term = st.number_input(
            "Loan Term",
            value=360
        )

        credit_history = st.selectbox(
            "Credit History",
            [1,0]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban","Semiurban","Rural"]
        )

    if st.button("Predict Loan Status"):

        total_income = (
            applicant_income
            +
            co_income
        )

        total_income_log = np.log(
            total_income
        )

        loan_amount_log = np.log(
            loan_amount
        )

        emi = (
            loan_amount
            /
            loan_term
        )

        income_loan_ratio = (
            total_income
            /
            loan_amount
        )

        input_df = pd.DataFrame({

            'Gender':[1 if gender=="Male" else 0],

            'Married':[1 if married=="Yes" else 0],

            'Dependents':[0],

            'Education':[1 if education=="Graduate" else 0],

            'Self_Employed':[1 if self_emp=="Yes" else 0],

            'ApplicantIncome':[applicant_income],

            'CoapplicantIncome':[co_income],

            'LoanAmount':[loan_amount],

            'Loan_Amount_Term':[loan_term],

            'Credit_History':[credit_history],

            'Property_Area':[2],

            'TotalIncome':[total_income],

            'TotalIncome_log':[total_income_log],

            'LoanAmount_log':[loan_amount_log],

            'EMI':[emi],

            'Income_Loan_Ratio':[income_loan_ratio]
        })

        prediction = model.predict(
            input_df
        )[0]

        probability = (
            model.predict_proba(
                input_df
            )[0].max()*100
        )

        if prediction == 1:

            st.success(
                f"✅ Loan Approved\n\nConfidence : {probability:.2f}%"
            )

        else:

            st.error(
                f"❌ Loan Rejected\n\nConfidence : {probability:.2f}%"
            )
