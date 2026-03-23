import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet

st.set_page_config(page_title="AI Financial Intelligence Dashboard", layout="wide")

st.title("AI-Driven Financial Intelligence and Decision Support System")
st.write("Upload financial data to forecast revenue, detect anomalies, assess risks, and simulate scenarios.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # REMOVE DUPLICATE COLUMNS
    # -----------------------------
    df = df.loc[:, ~df.columns.duplicated()]

    # -----------------------------
    # AUTO COLUMN DETECTION
    # -----------------------------
    cols = df.columns

    date_col = None
    sales_col = None
    profit_col = None

    for col in cols:
        col_lower = col.lower()
        if 'date' in col_lower:
            date_col = col
        elif 'sales' in col_lower or 'revenue' in col_lower:
            sales_col = col
        elif 'profit' in col_lower:
            profit_col = col

    # Manual fallback
    if date_col is None or sales_col is None:
        st.warning("Auto detection failed. Please select columns manually.")

        date_col = st.selectbox("Select Date Column", cols)
        sales_col = st.selectbox("Select Sales/Revenue Column", cols)
        profit_col = st.selectbox("Select Profit Column (optional)", ["None"] + list(cols))

        if profit_col == "None":
            profit_col = None

    # Rename columns
    df.rename(columns={
        date_col: 'Order Date',
        sales_col: 'Sales'
    }, inplace=True)

    if profit_col:
        df.rename(columns={profit_col: 'Profit'}, inplace=True)
    else:
        df['Profit'] = df['Sales'] * 0.3  # fallback

    # -----------------------------
    # SAFE DATA CLEANING
    # -----------------------------
    # Ensure single column
    if isinstance(df['Order Date'], pd.DataFrame):
        df['Order Date'] = df['Order Date'].iloc[:, 0]

    # Convert types safely
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')

    # Drop invalid rows
    df = df.dropna(subset=['Order Date', 'Sales'])

    # Aggregate
    df = df.groupby('Order Date').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()

    df = df.sort_values('Order Date')

    # Fill missing dates
    df = df.set_index('Order Date').asfreq('D').fillna(0).reset_index()

    df['revenue'] = df['Sales']
    df['expenses'] = df['Sales'] - df['Profit']

    st.subheader("Processed Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # REVENUE GRAPH
    # -----------------------------
    st.subheader("Revenue Trend")

    fig1, ax1 = plt.subplots()
    ax1.plot(df['Order Date'], df['revenue'])
    ax1.set_title("Revenue Over Time")
    st.pyplot(fig1)

    # -----------------------------
    # FORECASTING
    # -----------------------------
    st.subheader("Revenue Forecast for Next 30 Days")

    forecast_df = df[['Order Date', 'revenue']].rename(columns={'Order Date': 'ds', 'revenue': 'y'})

    if len(forecast_df) < 10:
        st.warning("Not enough data for forecasting.")
    else:
        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig2 = model.plot(forecast)
        st.pyplot(fig2)

        st.subheader("Forecast Output (Next 30 Days)")
        st.write("This shows expected future revenue based on past trends.")

        forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
        forecast_table.columns = [
            "Date",
            "Predicted Revenue",
            "Minimum Expected Revenue",
            "Maximum Expected Revenue"
        ]

        st.dataframe(forecast_table.round(2))

    # -----------------------------
    # ANOMALY DETECTION
    # -----------------------------
    st.subheader("Anomaly Detection")

    if df['revenue'].std() == 0:
        st.warning("No variation in data to detect anomalies.")
    else:
        df['z_score'] = (df['revenue'] - df['revenue'].mean()) / df['revenue'].std()
        df['anomaly'] = df['z_score'].apply(lambda x: 'Yes' if abs(x) > 3 else 'No')
        anomalies = df[df['anomaly'] == 'Yes']

        fig3, ax3 = plt.subplots()
        ax3.plot(df['Order Date'], df['revenue'], label='Revenue')
        ax3.scatter(anomalies['Order Date'], anomalies['revenue'], label='Anomalies')
        ax3.legend()
        st.pyplot(fig3)

        st.dataframe(anomalies[['Order Date', 'revenue']])

    # -----------------------------
    # RISK DETECTION
    # -----------------------------
    st.subheader("Early Risk Detection")

    if len(df) > 7:
        recent_avg = df['revenue'].tail(7).mean()
        overall_avg = df['revenue'].mean()

        recent_expense = df['expenses'].tail(7).mean()
        overall_expense = df['expenses'].mean()

        if recent_avg < overall_avg * 0.8:
            st.error("High Risk: Revenue is declining significantly.")
        elif recent_avg < overall_avg:
            st.warning("Medium Risk: Revenue is slightly decreasing.")
        else:
            st.success("Low Risk: Revenue trend is stable.")

        if recent_expense > overall_expense * 1.2:
            st.warning("Expenses are rising faster than usual.")
    else:
        st.info("Not enough data for risk analysis.")

    # -----------------------------
    # SCENARIO SIMULATION
    # -----------------------------
    st.subheader("Scenario Simulation")

    increase_pct = st.slider("Increase expenses by (%)", -50, 50, 10)

    df['simulated_expenses'] = df['expenses'] * (1 + increase_pct / 100)
    df['simulated_profit'] = df['revenue'] - df['simulated_expenses']

    st.write("Average Profit Before:", round((df['revenue'] - df['expenses']).mean(), 2))
    st.write("Average Profit After:", round(df['simulated_profit'].mean(), 2))

    # -----------------------------
    # INSIGHTS
    # -----------------------------
    st.subheader("Insights and Recommendations")

    if len(df) > 7:
        if recent_avg < overall_avg:
            st.info("Sales have dropped recently. Consider improving marketing or promotions.")

    if 'anomaly' in df.columns and len(df[df['anomaly'] == 'Yes']) > 0:
        st.info("Unusual spikes detected. Check specific dates.")

    if len(df) > 7 and recent_expense > overall_expense:
        st.info("Expenses are increasing. Consider cost optimization.")

else:
    st.info("Please upload a CSV file to start analysis.")
