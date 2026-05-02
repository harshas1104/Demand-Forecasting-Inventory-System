import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("📊 Demand Forecasting & Inventory Decision Dashboard")


# LOAD DATA

df = pd.read_csv("train.csv")
df['date'] = pd.to_datetime(df['date'])

# Filter (same as your model)
df = df[(df['store'] == 1) & (df['item'] == 1)]


# FEATURE ENGINEERING

df['day'] = (df['date'] - df['date'].min()).dt.days
df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)
df = df.dropna()

df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# -------------------------
# MODEL (Random Forest)
# -------------------------
from sklearn.ensemble import RandomForestRegressor

X = df[['day', 'month', 'day_of_week', 'lag_1', 'lag_7']]
y = df['sales']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------
# FUTURE PREDICTION (7 DAYS)
# -------------------------
future_rows = []

last_day = df['day'].max()
last_date = df['date'].max()

lag_1 = df.iloc[-1]['sales']
lag_7 = df.iloc[-7]['sales']

for i in range(1, 8):
    new_date = last_date + pd.Timedelta(days=i)
    future_rows.append([
        last_day + i,
        new_date.month,
        new_date.dayofweek,
        lag_1,
        lag_7
    ])

future_df = pd.DataFrame(future_rows, columns=X.columns)

predictions = model.predict(future_df)

# -------------------------
# LAYOUT (2 COLUMNS)
# -------------------------
col1, col2 = st.columns(2)


# GRAPH 1: SALES TREND

with col1:
    st.subheader("📈 Sales Trend")

    fig1, ax1 = plt.subplots(figsize=(6,3))
    ax1.plot(df['date'], df['sales'])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sales")

    st.pyplot(fig1)


# GRAPH 2: FUTURE DEMAND

with col2:
    st.subheader("🔮 Future Demand (Next 7 Days)")

    fig2, ax2 = plt.subplots(figsize=(6,3))
    ax2.plot(range(1, 8), predictions, marker='o')
    ax2.set_xlabel("Days Ahead")
    ax2.set_ylabel("Predicted Sales")

    st.pyplot(fig2)


# INVENTORY DECISION SECTION

st.subheader(" Inventory Decision System")

current_stock = st.slider("Select Current Stock", 0, 300, 50)

lead_time = 5
lead_demand = sum(predictions[:lead_time])

safety_stock = 10
reorder_point = lead_demand + safety_stock

# -------------------------
# SHOW METRICS
# -------------------------
st.write(" Expected Demand (Next 5 Days):", round(lead_demand, 2))
st.write(" Reorder Point:", round(reorder_point, 2))

# -------------------------
# DECISION LOGIC
# -------------------------
daily_avg = predictions.mean()
days_left = current_stock / daily_avg

if current_stock <= reorder_point:
    st.error(" Reorder Now!")

    order_qty = reorder_point - current_stock
    st.write(" Suggested Order Quantity:", int(order_qty))
    st.write(" Stock will last approx:", round(days_left, 1), "days")

else:
    st.success("✅ Stock is sufficient")
    st.write("⏳ Stock will last approx:", round(days_left, 1), "days")

# -------------------------
# INVENTORY VISUALIZATION
# -------------------------
st.subheader("📉 Inventory Depletion Visualization")

cumulative_demand = np.cumsum(predictions)
stock_remaining = current_stock - cumulative_demand

fig3, ax3 = plt.subplots(figsize=(6,4))

ax3.plot(range(1, 8), stock_remaining, label="Stock Remaining", marker='o')
ax3.plot(range(1, 8), cumulative_demand, label="Cumulative Demand", marker='o')

ax3.axhline(y=0, linestyle='--', label="Stock Out Level")
ax3.axhline(y=reorder_point, linestyle='--', label="Reorder Point")

if current_stock <= reorder_point:
    ax3.axvline(x=1, linestyle='--', label="Reorder Today")

ax3.set_xlabel("Days Ahead")
ax3.set_ylabel("Units")
ax3.legend()
st.pyplot(fig3, use_container_width=False)
#st.pyplot(fig3)