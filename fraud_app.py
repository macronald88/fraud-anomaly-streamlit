import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.title("Fraud & Anomaly Detection Dashboard")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
    data['TransactionHour'] = data['TransactionDate'].dt.hour
    data['TransactionDayOfWeek'] = data['TransactionDate'].dt.dayofweek
    data['TimeSinceLastTransaction'] = (
        data.groupby('AccountID')['TransactionDate'].diff().dt.total_seconds().fillna(0)
    )

    categorical_cols = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    X = data[numerical_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_scaled)
    scores = model.decision_function(X_scaled)
    threshold = np.percentile(scores, 5)
    data['Anomaly'] = (scores <= threshold).astype(int)

    st.subheader("Anomaly Score Distribution")
    fig = px.histogram(x=scores, nbins=100, title="Anomaly Scores")
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    st.plotly_chart(fig)

    st.subheader("Detected Anomalies")
    st.dataframe(data[data['Anomaly'] == 1])

    def analyze_login_attempts(df, login_threshold=4):
        df['SuspiciousLogin'] = (df['LoginAttempts'] > login_threshold).astype(int)
        return df

    data = analyze_login_attempts(data)
    st.subheader("Suspicious Login Attempts")
    st.dataframe(data[data['SuspiciousLogin'] == 1])

    def analyze_transaction_amounts(df, multiplier=2):
        flags = []
        for acc_id, group in df.groupby('AccountID'):
            mean = group['TransactionAmount'].mean()
            std = group['TransactionAmount'].std()
            if std > 0:
                threshold = mean + multiplier * std
                flags.extend(group[group['TransactionAmount'] > threshold].index.tolist())
        df['UnusualAmount'] = 0
        df.loc[df.index.isin(flags), 'UnusualAmount'] = 1
        return df

    data = analyze_transaction_amounts(data)
    st.subheader("Unusual Transaction Amounts")
    st.dataframe(data[data['UnusualAmount'] == 1])

    def analyze_device_anomalies(df, threshold=8):
        anomalies = []
        for acc_id, group in df.groupby('AccountID'):
            if group['DeviceID'].nunique() > threshold:
                anomalies.extend(group.index.tolist())
        df['DeviceAnomaly'] = 0
        df.loc[df.index.isin(anomalies), 'DeviceAnomaly'] = 1
        return df

    data = analyze_device_anomalies(data)
    st.subheader("Device Anomalies")
    st.dataframe(data[data['DeviceAnomaly'] == 1])

    def analyze_ip_anomalies(df, threshold=9):
        anomalies = []
        for acc_id, group in df.groupby('AccountID'):
            if group['IP Address'].nunique() > threshold:
                anomalies.extend(group.index.tolist())
        df['IPAnomaly'] = 0
        df.loc[df.index.isin(anomalies), 'IPAnomaly'] = 1
        return df

    data = analyze_ip_anomalies(data)
    st.subheader("IP Address Anomalies")
    st.dataframe(data[data['IPAnomaly'] == 1])

    def analyze_account_balance(df, change_threshold=0.7):
        anomalies = []
        for acc_id, group in df.groupby('AccountID'):
            changes = group['AccountBalance'].pct_change().abs()
            anomalies.extend(changes[changes > change_threshold].index.tolist())
        df['BalanceAnomaly'] = 0
        df.loc[df.index.isin(anomalies), 'BalanceAnomaly'] = 1
        return df

    data = analyze_account_balance(data)
    st.subheader("Account Balance Anomalies")
    st.dataframe(data[data['BalanceAnomaly'] == 1])
