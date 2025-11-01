import streamlit as st
import pandas as pd
import numpy as np

## Title of application 
st.title("Hello Streamlit")

## Display  a Simple Text
st.write("This is simple text")

## Create dataFrame 
dp = pd.DataFrame({
    "first column":[1,2,3,4],
    "second column":[10,20,30,40]
})

## Display the DataFrame
st.write("Here is DataFrame")
st.write(dp)

## create Line chat

chart_data=pd.DataFrame(
    np.random.randn(20,3),columns=['a','b','c']
)
st.line_chart(chart_data)