import streamlit as st
import warnings
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import glob
import numpy as np
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
    
def seasonality(df):
    # Set frequency for date range
    start_date_str = '2014-02-25 00:00:00'
    end_date_str = '2022-06-22 00:00:00'
    date_rng = pd.date_range(start=pd.to_datetime(start_date_str), end=pd.to_datetime(end_date_str), freq='B')
    
    df['Date'] = date_rng
    df = df.set_index('Date')
     
    # Perform seasonal decomposition
    decompose_result_mult = seasonal_decompose(df['Close'], model="additive", period=5)
    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid
    
    # Plot results
    st.line_chart(decompose_result_mult.observed)
    st.line_chart(trend)
    st.line_chart(seasonal)
    st.line_chart(residual)
    st.subheader("Training Set 80%")
    train, test = df.iloc[:int(0.8*len(df))], df.iloc[int(0.8*len(df)):]
    st.write(train)
    st.subheader("Test Set 20%")
    st.write(test)
    

def stationary_check(series, window=5):
    # Plot rolling statistics
    fig = plt.figure(figsize=(10,5))
    orig = plt.plot(series, color='blue', label='Original')
    mean = plt.plot(series.rolling(window).mean(), color='red', label='Rolling Mean')
    std = plt.plot(series.rolling(window).std(), color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation', size=20)
    plt.show(block=False)
    
    # Augmented Dickey-Fuller test for stationarity
    result = adfuller(series) #Regular differentiation
    dfoutput = pd.Series(result[0:4], index=['Test Statistic','p-value',
                                             '#Lags Used','Number of Observations Used'])
    for key,value in result[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    st.subheader("Augmented Dickey-Fuller test for stationarity")
    st.write(dfoutput)
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
    
    #Seasonality trends
    st.subheader("Seasonality Trends")
    season_trends = seasonality(df)
    st.write(season_trends)
    
    #Stationary check on the data
    output_checks = stationary_check(df['Adj Close'])
    st.write(output_checks)
    
    # Add Insights
    st.subheader("Insights")
    st.write("From the visualization, we can observe that the stock prices of Twitter have been increasing steadily over the years. We can also see a spike in volume towards the end of 2021, which coincides with the news of Elon Musk's buyout offer.")


if __name__ == '__main__':
    main()