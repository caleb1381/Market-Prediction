import streamlit as st
import pandas as pd
from datetime import datetime

#show the date and time on my app
mydate = datetime.now()
st.write(" Date & Time : ", mydate)
st.title("MARKET PRICE PREDICTION")
st.write("MARKET **STOCK** PRICE PREDICTION ")


# Load dataset from CSV file
@st.cache_data
def load_data():
    df = pd.read_csv('wfp_market_food_prices.csv', chunksize= 2000)
    for chunksize in df:
        return chunksize
    
# Display data in Streamlit app
def main():
    df = load_data()
    st.write(df)
  

if __name__ == '__main__':
    main()
