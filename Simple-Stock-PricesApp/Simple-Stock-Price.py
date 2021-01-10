import streamlit as st
import yfinance as yf
from datetime import datetime
from PIL import Image

# title and image
st.title('Simple Stock Price App')
st.sidebar.header('User input')
#image = Image.open("C:/Users/danil/PycharmProjects/Finance datasets/finance-image.png")
#st.image(image, use_column_width=True)

# Create a select box with the stocks
get_data = st.sidebar.selectbox('Select your stock', ['Amazon', 'Google', 'Apple', 'Tesla'])


def get_symbol(Symbol):

    # Get the start date to search
    start_date = st.sidebar.text_input('Start Date', '2019-01-02')

    # Get the end date to search
    end_date = st.sidebar.text_input('End Date', '2021-01-02')

    # Get the stock price and search on yahoo finance api
    tickerData = yf.Ticker(Symbol)
    # today = datetime.today().strftime('%Y-%m-%d')
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

    # first chart text
    st.write(f"""
        ## Closing Price of {get_data}
        """)

    # second chart text
    st.line_chart(tickerDf.Close)
    st.write(f"""
        ## Volume Price of {get_data}
        """)
    st.line_chart(tickerDf.Volume)

# convert a company name into stock ticker
if get_data == 'Google':
    get_symbol('GOOGL')
elif get_data == 'Amazon':
    get_symbol('AMZN')
elif get_data == 'Apple':
    get_symbol('AAPL')
elif get_data == 'Tesla':
    get_symbol('TSLA')
