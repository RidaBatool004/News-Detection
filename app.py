import streamlit as st
import joblib
vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('Logistic_Regression_Model.jb')
st.title("Fake News Detector")
st.write("Enter the News Article below to check whether it is Fake or Real")
news_input = st.text_area("News Article","")
if st.button("Check Article"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0]==1:
            st.success("The News is Real")
        else:
            st.error("The News is Fake")
    else:
        st.warning("Please enter any text to analyze. ")