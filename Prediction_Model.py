import streamlit as st
import warnings
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import glob
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA,ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from scipy.stats.distributions import chi2
from pmdarima.arima import auto_arima

st.set_page_config(page_title= "market price prediction", page_icon= ':shark:')


#show the date and time on my app
mydate = datetime.now()
formated_date_time = f"**Date and Time:** {mydate.strftime('%Y-%m-%d %H:%M:%S')}"
st.sidebar.write(formated_date_time)


matplotlib.rcParams['figure.figsize'] = (14, 10)
sns.set_style('darkgrid')
#ignore warnings on console
warnings.filterwarnings("ignore")


@st.cache_data
def load_data():
    st.text(''' About Dataset
This dataset contains the
daily recorded stock prices (and more) for the company,
from day one that they went public. Twitter, 
originally founded in 2006, 
has been publicly-traded since November 2013, when it held an initial public
offering that raised $1.8 billion.
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

def preprocessing(df):
#converting normal date column to date pandas date and time
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def plotting(df):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(df[['Open','High','Low','Close']])
    axes[0].set_title('Twitter Stock Price', size=20)

    axes[1].plot(df['Volume'])
    axes[1].set_title('Twitter Volume', size=20)
    axes[1].get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# display the plot using st.pyplot()
    st.pyplot(fig)
    
    
def resampling(df):
    df = df.set_index("Date")
    fig, axes = plt.subplots(2,2,figsize=[15, 7])

# Resampling to daily frequency
    axes[0,0].plot(df[['Open','High','Low','Close']])
    axes[0,0].set_title("Daily Price",size=16)

# Resampling to monthly frequency
    df_month = df.resample('M').mean()
    axes[0,1].plot(df_month[['Open','High','Low','Close']])
    axes[0,1].set_title("Monthly mean Price",size=16)

# Resampling to annual frequency
    df_year = df.resample('A-DEC').mean()
    axes[1,0].plot(df_year[['Open','High','Low','Close']])
    axes[1,0].set_title("Annually mean Price",size=16)

# Resampling to quarterly frequency
    df_Q = df.resample('Q-DEC').mean()
    axes[1,1].plot(df_Q[['Open','High','Low','Close']])
    axes[1,1].set_title("Quarterly mean Price",size=16)
    st.pyplot(fig)

# Display data in Streamlit app
def main():
    
    # Display the title of the app
    st.title("Twitter Stock Market Price Prediction")
    
    df = load_data()
    df = preprocessing(df)
    st.write(df)
    
    # Display basic information about the dataset
    st.subheader("Dataset Information")
    st.write("**Total Numbers of Rows and Columns :** ", df.shape)
    st.write("**Number of Features:**", len(df.columns))
    st.subheader("List all the columns")
    st.write(df.columns)
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
    st.subheader("Visualization of Twitter Stock Price")
    st.line_chart(df["Close"])
    st.area_chart(df[["Open", "High", "Low", "Close"]])
    st.bar_chart(df["Volume"])
    
    # pass my chart and graph into the app
    st.subheader("Creating Figure and Axes for my plot")    
    plottingres = plotting(df)
    st.write(plottingres)
    
    #data sampling
    st.subheader("Data Resampling")
    samp = resampling(df)
    st.write(samp)
    
    # Add Insights
    st.subheader("Insights")
    st.write("From the visualization, we can observe that the stock prices of Twitter have been increasing steadily over the years. We can also see a spike in volume towards the end of 2021, which coincides with the news of Elon Musk's buyout offer.")


if __name__ == '__main__':
    main()