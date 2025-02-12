# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import shap

st.header("An artificial intelligence platform to predict the early suicide risk after discharge for self-poisoning patients")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Sedentarytime = st.sidebar.selectbox("Sedentary time (hours)", ("﹤1", "≧1 and ﹤3", "≧3 and ﹤6", "≧6"))
Suicidetime = st.sidebar.selectbox("Suicide frequency", ("1", "2", "3", "≧4"))
Previousdepression= st.sidebar.selectbox("Previous diagnosis of depression", ("Yes", "No"))
Previouspsychiologicaldisorder= st.sidebar.selectbox("Previous diagnosis of psychiological disorders", ("Yes", "No"))
GAD7 = st.sidebar.slider("GAD-7 score", 0, 21)
BECK20 = st.sidebar.slider("BECK score", 0, 19)
Antithyroglobulin = st.sidebar.slider("Anti-thyroglobulin (IU/ml)", 1.0, 40.0)

if st.button("Submit"):
    rf_clf = jl.load("ensemble_clf_final_roundweb.pkl")
    x = pd.DataFrame([[Sedentarytime, Suicidetime, Previousdepression, Previouspsychiologicaldisorder, GAD7, BECK20, Antithyroglobulin]],
                     columns=["Sedentarytime",  "Suicidetime", "Previousdepression", "Previouspsychiologicaldisorder", "GAD7", "BECK20", "Antithyroglobulin"])

    x = x.replace(["﹤1", "≧1 and ﹤3", "≧3 and ﹤6", "≧6"], [1, 2, 3, 4])
    x = x.replace(["1", "2", "3", "≧4"], [1, 2, 3, 4])
    x = x.replace(["Yes", "No"], [1, 2])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Early suicide risk after discharge: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.500:
        st.success(f"Risk group: low-risk group")
    else:
        st.error(f"Risk group: High-risk group")

st.subheader('About the model')
st.markdown('The complimentary online calculator utilizes an ensemble machine learning algorithm and has demonstrated exceptional performance during validation. The risk of patient suicide within three months post-discharge can be predicted through this platform, providing important reasons for the high-risk status based on the risk assessment report. Based on this, it can serve as a reference for suicide management in clinical patients after discharge.')