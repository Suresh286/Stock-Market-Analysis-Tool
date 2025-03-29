import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup

# App title
st.set_page_config(page_title="Stock Market Analysis Tool", layout="wide")
st.title("📈 Stock Market Analysis Tool")

# Sidebar Navigation
st.sidebar.header("📌 Select Feature")
menu = st.sidebar.radio("Choose an Option:", [
    "Stock Price Viewer",
    "Moving Averages",
    "RSI Analysis",
    "Stock Price Prediction",
    "Latest Stock News",
    "Candlestick Chart Analyzer",
    "Portfolio Tracking",
    "Stock Screener",
    "Options Data"
])

# Sidebar for stock selection
st.sidebar.header("📌 Stock Selection")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()

def get_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

data = get_stock_data(stock_symbol, period="20y")

if menu == "Stock Price Viewer":
    st.header("💰 Current Stock Price")
    stock = yf.Ticker(stock_symbol)
    current_price = stock.history(period='1d')['Close'].iloc[-1]
    st.write(f"**{stock_symbol} Latest Closing Price:** ${current_price:.2f}")

elif menu == "Moving Averages":
    st.header("📊 Stock Price & Moving Averages")
    data['Short_MA'] = data['Close'].rolling(window=20).mean()
    data['Long_MA'] = data['Close'].rolling(window=100).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data["Short_MA"], mode='lines', name='20-Day MA'))
    fig.add_trace(go.Scatter(x=data.index, y=data["Long_MA"], mode='lines', name='100-Day MA'))
    st.plotly_chart(fig)

elif menu == "RSI Analysis":
    st.header("📉 Relative Strength Index (RSI)")
    def calculate_rsi(data, window=14):
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    data['RSI'] = calculate_rsi(data)
    st.line_chart(data[['RSI']])

elif menu == "Stock Price Prediction":
    st.header("🔮 Stock Price Prediction")
    data = get_stock_data(stock_symbol, period="5y")
    if not data.empty:
        data['Short_MA'] = data['Close'].rolling(window=20).mean()
        data['Long_MA'] = data['Close'].rolling(window=100).mean()
        data.dropna(inplace=True)
        X = data[['Close', 'Short_MA', 'Long_MA']]
        y = data['Close'].shift(-1).dropna()
        X, y = X[:-1], y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        future_days = st.slider("Days to Predict", 1, 30, 10)
        last_known_data = X.iloc[-future_days:].values
        predicted_prices = model.predict(last_known_data)
        future_dates = pd.date_range(start=data.index[-1], periods=future_days+1, freq='B')[1:]
        predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})
        st.write(predictions_df)
        st.line_chart(predictions_df.set_index("Date"))
    else:
        st.error("⚠️ No stock data found.")

elif menu == "Latest Stock News":
    st.header("📰 Latest Stock News")
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey=b51abc126a3f46c8af66df29cd268416"
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json().get('articles', [])[:5]
        for article in news_data:
            st.write(f"**{article.get('title', 'No Title')}**")
            st.write(article.get('description', 'No Description'))
            st.write(f"[Read more]({article.get('url', '#')})")
            st.write("---")

elif menu == "Candlestick Chart Analyzer":
    st.header("📈 Candlestick Chart Analyzer")
    data = get_stock_data(stock_symbol, period="1y")
    fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
    fig.update_layout(title=f"{stock_symbol} Stock Price", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

elif menu == "Portfolio Tracking":
    st.sidebar.header("Portfolio Tracking")
    st.header("🚀📈 Portfolio Tracking")
    stocks = st.sidebar.text_area("Enter Stocks (comma-separated)")
    if stocks:
        tickers = [ticker.strip().upper() for ticker in stocks.split(",")]
        portfolio = {ticker: get_stock_data(ticker)['Close'].iloc[-1] for ticker in tickers if not get_stock_data(ticker).empty}
        st.table(pd.DataFrame(portfolio.items(), columns=["Stock", "Latest Price"]))

elif menu == "Stock Screener":
    st.sidebar.header("📌 Stock Screener")
    screener_criteria = st.sidebar.selectbox("Select Criteria", ["High Volume", "Low P/E Ratio", "Top Gainers"])
    url = "https://finance.yahoo.com/screener/predefined/most_actives"
    if screener_criteria == "Top Gainers":
        url = "https://finance.yahoo.com/screener/predefined/day_gainers"
    elif screener_criteria == "Low P/E Ratio":
        url = "https://finance.yahoo.com/screener/predefined/undervalued_growth_stocks"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    st.write("**Top Stocks Based on Criteria:**")
    st.write([tag.text for tag in soup.find_all("td", class_="Va(m)")[:10]])

elif menu == "Options Data":
    options_ticker = st.sidebar.text_input("Enter Stock Ticker for Options Data")
    if options_ticker:
        stock = yf.Ticker(options_ticker)
        exp_dates = stock.options
        selected_date = st.selectbox("Select Expiration Date", exp_dates)
        options_data = stock.option_chain(selected_date)
        st.subheader("Calls")
        st.write(options_data.calls)
        st.subheader("Puts")
        st.write(options_data.puts)

st.success("✅ Stock Market Analysis Tool Ready!")
