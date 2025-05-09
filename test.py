import streamlit as st
import pandas as pd
 
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
 
if uploaded_file:
    df = pd.read_csv(uploaded_file)
 
    st.dataframe(df)
    st.table(df)