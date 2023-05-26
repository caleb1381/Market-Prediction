import streamlit as st
import pandas as pd
from datetime import datetime
st.set_page_config(page_title= "market price prediction", page_icon= ':shark:')

#show the date and time on my app
mydate = datetime.now()
st.sidebar.write("**Date & Time** :", mydate)

@st.cache_data
def load_data():
    st.text(''' About Dataset
This dataset contains the
daily recorded stock prices (and more) for the company,
from day one that they went public. Twitter, 
originally founded in 2006, 
has been publicly-traded since November 2013, when it held an initial public
offering that raised **$1.8 billion.
Twitter,Inc. 
operates as a platform for public self-expression and conversation in real-time. 
The company offers Twitter, 
a platform that allows users to consume, create, distribute, 
and discover content.
The company is now in the midst of a battle of take over
with Elon Musk for a by out of $46+ billion and since stock prices have soared.''')
    st.subheader('Dataset of Twitter Stock Market Price')
    df = pd.read_csv('TWITTER.csv')
    return df

# Display data in Streamlit app
def main():
    # Display the title of the app
    st.title("Twitter Stock Market Price Prediction")
    
    df = load_data()
    st.write(df)
     # Display basic information about the dataset
    st.subheader("Dataset Information")
    st.write("**Total Numbers of Rows and Columns :** ", df.shape)
    st.write("**Number of Features:**", len(df.columns))
    st.write("**Data Types of Each Column:**")
    st.write(df.dtypes)
    
     # Display EDA
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("First Few Rows of the Data")
    st.write(df.head())
    
    #discribe the data
    st.subheader("Summary of the Data")
    st.write(df.describe())
    
     # Add some visualizations
    st.subheader("Visualization")
    st.line_chart(df["Close"])
    st.area_chart(df[["Open", "High", "Low", "Close"]])
    st.bar_chart(df["Volume"])
    
     # Add Insights
    st.subheader("Insights")
    st.write("From the visualization, we can observe that the stock prices of Twitter have been increasing steadily over the years. We can also see a spike in volume towards the end of 2021, which coincides with the news of Elon Musk's buyout offer.")


if __name__ == '__main__':
    main()
