import streamlit as st
import warnings
import pandas as pd
from datetime import datetime
import sklearn.preprocessing 
import sklearn.metrics 
import math 
import glob
import numpy as np
import itertools 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import boxcox
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

# Set frequency for date range
start_date_str ='2014-02-25 00:00:00'
end_date_str = '2022-06-22 00:00:00'

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
    # Return the train and test subsets
    return  df

def preprocessing(df):
#converting normal date column to date pandas date and time
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
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
    
@st.cache_resource   
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
    
@st.cache_resource    
def seasonality(df):
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
    train =df.iloc[:int(0.8*len(df))]
    test = df.iloc[int(0.8*len(df)):]
    st.write(train)
    st.subheader("Test Set 20%")
    st.write(test)
    
@st.cache_data
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
    st.header("First Test")
    st.write(dfoutput)
    st.pyplot(fig)
    
@st.cache_data    
def fuller_test(series, window=5):
    # Apply Box-Cox transformation
    series_boxcox, _ = boxcox(series)

# Convert series_boxcox to Pandas Series object
    series_boxcox = pd.Series(series_boxcox)

# Plot rolling statistics of transformed data
    fig = plt.figure(figsize=(10,5))
    orig = plt.plot(series_boxcox, color='blue', label='Original')
    mean = plt.plot(series_boxcox.rolling(window).mean(), color='red', label='Rolling Mean')
    std = plt.plot(series_boxcox.rolling(window).std(), color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation', size=20)
    plt.show(block=False)

# Perform ADF test on transformed data
    result = adfuller(series_boxcox)
    doutput = pd.Series(result[0:4], index=['Test Statistic','p-value',
                                         '#Lags Used','Number of Observations Used'])
    for key,value in result[4].items():
        doutput['Critical Value (%s)'%key] = value
    st.write(doutput)
    st.pyplot(fig)
      
def regular_trans(series, window=5):
    # Apply regular transformation
    series_diff = series.diff(1)

    # Plot rolling statistics of transformed data
    def plot_rolling_stats(series, window):
        fig = plt.figure(figsize=(10,5))
        orig = plt.plot(series, color='blue', label='Original')
        mean = plt.plot(series.rolling(window).mean(), color='red', label='Rolling Mean')
        std = plt.plot(series.rolling(window).std(), color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation', size=20)
        plt.show(block=False)
        st.pyplot(fig)
    plot_rolling_stats(series_diff[1:], window)
    # Perform ADF test on transformed data
    res = adfuller(series_diff[1:])
    doutputs = pd.Series(res[0:4], index=['Test Statistic','p-value',
                                             '#Lags Used','Number of Observations Used'])
    for key,value in res[4].items():
        doutputs['Critical Value (%s)'%key] = value
    st.subheader('Regular Data')
    st.write(doutputs)
    
@st.cache_data   
def autocorrelation(series, window=5):
    fig = plt.figure(figsize=(14,7))
    ax_1 = fig.add_subplot(221)
    orig = ax_1.plot(series, color='blue', label='Original')
    mean = ax_1.plot(series.rolling(window).mean(), color='red', label='Rolling Mean')
    std = ax_1.plot(series.rolling(window).std(), color='black', label='Rolling Std')
    ax_1.legend(loc='best')
    ax_1.set_title('Rolling Mean & Standard Deviation', size=20)

    ax_2 = fig.add_subplot(222)
    plot_acf(series,ax=ax_2,lags=40,zero=False)
    ax_2.set_ylim(-0.5, 0.5)
    ax_2.set_title("Autocorrelation Function (ACF)",size=20)

    ax_3 = fig.add_subplot(223)
    plot_pacf(series,ax=ax_3,lags=40,zero=False,method='ols')
    ax_3.set_ylim(-0.5, 0.5)
    ax_3.set_title("Partial Autocorrelation Function (PACF)",size=20)

    ax_4 = fig.add_subplot(224)
    series_diff = series.diff(1) # regular Transformations
    orig = ax_4.plot(series_diff, color='blue', label='Original')
    mean = ax_4.plot(series_diff.rolling(window).mean(), color='red', label='Rolling Mean')
    std = ax_4.plot(series_diff.rolling(window).std(), color='black', label='Rolling Std')
    ax_4.legend(loc='best')
    ax_4.set_title('Rolling Mean & Standard Deviation (Transformed)', size=20)

    plt.tight_layout()
    plt.show(block=False)
    st.write(fig)

@st.cache_data
def fit_arima(df, exog_cols=['Open','Close'], m=5, alpha=0.05, crit='oob', maxiter=20, oos_size=0.1):
    # Split the time series data into train and test sets, and extract exogenous variables
    train = df.iloc[:int(0.8*len(df))]
    test = df.iloc[int(0.8*len(df)):]
    exogenous = train[exog_cols][5:]

    # Automatically select the best ARIMA model using pmdarima.auto_arima()
    try:
        model_auto = auto_arima(train.Close[1:], exogenous=exogenous, m=m, alpha=alpha, 
                                information_criterion=crit, out_of_sample_size=int(oos_size*len(train)), 
                                maxiter=maxiter, trend='ct', n_jobs=-1)
    except ValueError as e:
        st.write(f"ARIMA model fitting failed with error: {e}")
        return None

    return model_auto

def predict_arima(model, df):
    # Split the data into train and test sets
    train = df.iloc[:int(0.8*len(df))]
    test = df.iloc[int(0.8*len(df)):]

    # Extract the exogenous variables for the test set
    exogenous = test[['Open', 'Close']][5:]

    # Make predictions using the ARIMA model
    predictions = model.predict(n_periods=len(test), exogenous=exogenous)

    return predictions

def generate_predictions(train, test, exogenous):
    # Fit SARIMAX model to training data
    model = SARIMAX(train['Open'], order=(1, 1, 0), seasonal_order=(1, 0, 0, 12))
    model_fit = model.fit()

    # Generate predictions for test data using fitted model
    preds = model_fit.predict(steps=len(test), exog=test[exogenous])

    # Convert predictions and actual test data to Pandas Series and return as a tuple
    preds_series = pd.Series(preds, index=test.index)
    actual_series = test['Open']
    return (actual_series, preds_series)

#residual Anaylsis of trianed data
def plot_residuals(df):
    # Retrieve model residuals
    residuals = df.resid()

    # Create figure
    fig = plt.figure(figsize=(14, 5))

    # Plot residuals
    ax_1 = fig.add_subplot(121)
    ax_1.plot(residuals)
    ax_1.set_title("Residuals of Returns", size=24)

    # Plot ACF for residuals
    ax_2 = fig.add_subplot(122)
    plot_acf(residuals, lags=40, zero=False, ax=ax_2)
    ax_2.set_title("ACF for Residuals of Returns", size=24)
    ax_2.set_ylim(-0.5, 0.5)

    # Display plots
    st.pyplot(fig)


# Display data in Streamlit app
@st.cache_resource(experimental_allow_widgets=True)
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
    output_checks = stationary_check(df['Close'])
    st.write(output_checks)
    st.subheader("Second Test")
    fuller_output = fuller_test(df['Close'])
    st.write(fuller_output)
    
    #regular transform
    st.subheader("Regular Transform ")
    regular =regular_trans(df['Close'])
    st.write(regular)
    
    st.subheader('Auto Correlation')
    auto = autocorrelation(df['Close'])
    st.write(auto)
    
    st.subheader('Show the model output using ARIMA')
    df = pd.read_csv('TWITTER.csv', index_col='Date')
    # Fit an ARIMA model to the data using pmdarima
    model = fit_arima(df, exog_cols=['Open', 'Close'], m=5, alpha=0.01, crit='aic', maxiter=50)

    if model is not None:
        st.write(model.summary())
    else:
        st.write("ARIMA model fitting failed; please check your data and parameters.")
        
    #my Residual Anaylsis
    st.subheader("Residual Analysis")
    models = plot_residuals(model)
    st.write(models)
    # Add Insights
    #using the test data to make predictions
    st.header("Predicted Data")
    predicted_data = predict_arima(model, df)
    st.write(predicted_data)
    p_conversion = pd.DataFrame(predicted_data)
    st.subheader("Data Histogram")
    st.bar_chart(p_conversion)
    #predicted data
    def run():
        
        # Load stock price data
        stock_data = pd.read_csv('TWITTER.csv', parse_dates=['Date'])
        train = stock_data.loc[stock_data['Date'] < '2020-01-01 00:00:00']
        test = stock_data.loc[(stock_data['Date'] >= '2020-01-01 00:00:00') & (stock_data['Date'] <= '2022-06-22 00:00:00')]

        # Create sidebar with options for user input
        st.sidebar.header('User Input Parameters')
        exogenous_var = st.sidebar.selectbox('Exogenous Variable', ['Volume', 'High', 'Low'])
        train_start_date = pd.Timestamp(st.sidebar.date_input('Training Start Date', value=train['Date'].iloc[0]))
        train_end_date = pd.Timestamp(st.sidebar.date_input('Training End Date', value=train['Date'].iloc[-1]))
        test_start_date = pd.Timestamp(st.sidebar.date_input('Test Start Date', value=test['Date'].iloc[0]))
        test_end_date = pd.Timestamp(st.sidebar.date_input('Test End Date', value=test['Date'].iloc[-1]))
        
        # Subset data based on user input
        train = train.loc[(train['Date'] >= train_start_date) & (train['Date'] <= train_end_date)]
        test = test.loc[(test['Date'] >= test_start_date) & (test['Date'] <= test_end_date)]

        # Generate and display predictions
        actual, preds = generate_predictions(train, test, exogenous_var)
        st.header('Stock Price Data')
        st.write(actual)
   
        st.header('**Actual vs Predicted Values**')
        fig, ax = plt.subplots()
        ax.plot(actual, label='Actual')
        ax.plot(preds, label='Predicted')
        ax.legend() 
        st.pyplot(fig)
    run()
    
    st.subheader("Insights")
    st.write("From the visualization, we can observe that the stock prices of Twitter have been increasing steadily over the years. We can also see a spike in volumn towards its take over")

if __name__ == '__main__':
    main()