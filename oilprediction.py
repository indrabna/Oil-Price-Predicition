import streamlit as st
import pandas as pd

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


st.header('Oil Price Prediction')
n_years= st.slider("Predction for month", 1,12)
period = n_years*30

# file = "C:\Users\INDRA BUANA\OneDrive\Desktop\Kalla\oilprice.csv"
df2 = pd.read_csv('https://raw.githubusercontent.com/indrabna/dataset/main/oilprice.csv')
# read dataset
df = pd.read_csv('https://raw.githubusercontent.com/indrabna/Data_Analytics_Zenius/main/Copy%20of%20brentcrudeoil%20-%20dailybrentoil.csv')
# read dataset



st.subheader('Raw Data')
st.write(df.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df2['Date'], y=df2['Price'], name="safafsf"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting
df_train = df2[['Date',"Price"]]
df_train = df_train.rename(columns={"Date" : "ds", "Price" : "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write("Forecast Data Visualization")
fig1= plot_plotly(m, forecast)
st.plotly_chart(fig1)


# st.write("Forecast Data Components")
# fig2 = m.plot_components(forecast)
# st.plotly_chart(fig2)
