print("NEW CODE RUNNING")
print("Project started")
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("train.csv")

# show first 5 rows
print(df.head())
print(df.info())
df['date'] = pd.to_datetime(df['date'])
df = df[(df['store'] == 1) & (df['item'] == 1)]
print(df.head())
print(df.shape)

#feature engineering
df['day'] = (df['date'] - df['date'].min()).dt.days
df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)

df = df.dropna()
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
print(df.head())
print(df.shape)



#define input(x) and output(y)
X = df[['day', 'month','day_of_week','lag_1','lag_7']]
y = df['sales']
print("Training columns:", X.columns) 
print(X.head())
#import from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
# split data (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
# train model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("\n====================")
print("MODEL EVALUATION")
print("MAE:", mae)
print("====================\n")

#Upgrade the model to Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
print("Random Forest MAE:", rf_mae)

last_row = df.iloc[-1]


#predict future demand(7 days)

# predict future demand (7 days)

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

print("Future DF columns:", future_df.columns)   

#predictions = model.predict(future_df)
predictions = rf_model.predict(future_df)  # FOR RF
print("\nFUTURE PREDICTIONS")
print(predictions)
#check
print(df['day'].max())


# Reorder logic
print("\n====================")
print("REORDER LOGIC")
print("====================")

current_stock = 50   # assume
lead_time = 5        # days

# demand during lead time
lead_demand = sum(predictions[:lead_time])

print("Current Stock:", current_stock)
print("Expected demand in next", lead_time, "days:", round(lead_demand,2))

if current_stock <= lead_demand:
    print("⚠️ REORDER NOW!")
else:
    print("✅ Stock is sufficient")

# Safety stock
safety_stock = 10
reorder_point = sum(predictions[:lead_time]) + safety_stock

print("Reorder Point:", round(reorder_point,2))

if current_stock <= reorder_point:
    print("⚠️ REORDER NOW (with safety stock)")
else:
    print("✅ Safe")

#visualising reorder graphs
import numpy as np

plt.figure(figsize=(10,6))

days = np.arange(1, len(predictions)+1)

# cumulative demand (realistic)
cumulative_demand = np.cumsum(predictions)

# stock level decreasing
stock_remaining = current_stock - cumulative_demand

# plot
plt.plot(days, stock_remaining, marker='o', label="Stock Remaining")
plt.plot(days, cumulative_demand, marker='o', label="Cumulative Demand")

# reorder line
plt.axhline(y=0, linestyle='--', label="Stock Out Level")
plt.axhline(y=reorder_point, linestyle='--', label="Reorder Point")
if current_stock < reorder_point:
    reorder_day = 1
    plt.axvline(x=reorder_day, linestyle='--', label="Reorder Today")
plt.title("Inventory Depletion & Reorder Decision")
plt.xlabel("Days Ahead")
plt.ylabel("Units")

plt.legend()
plt.grid()
plt.savefig("dashboard1.png")
plt.show()

# dashboard graphs
print("\nGenerating dashboard graphs...")

plt.figure(figsize=(14,10))

# GRAPH 1 
plt.subplot(3,2,1)
plt.plot(df['date'], df['sales'])
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")

# GRAPH 2
plt.subplot(3,2,2)
df.groupby('day_of_week')['sales'].mean().plot(kind='bar')
plt.title("Avg Sales by Day of Week")
plt.xlabel("Day (0=Mon)")
plt.ylabel("Sales")

# GRAPH 3
plt.subplot(3,2,3)
df.groupby('month')['sales'].mean().plot(kind='line')
plt.title("Avg Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")

#GRAPH 4 
plt.subplot(3,2,4)
plt.scatter(df['lag_1'], df['sales'])
plt.title("Lag1 vs Sales")
plt.xlabel("Previous Day Sales")
plt.ylabel("Today Sales")

# GRAPH 5 
plt.subplot(3,2,5)
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Model Performance")
plt.xlabel("Time")
plt.ylabel("Sales")

# GRAPH 6 
plt.subplot(3,2,6)
future_days = range(1,8)
plt.plot(future_days, predictions, marker='o')
plt.title("Future Demand (Next 7 Days)")
plt.xlabel("Days Ahead")
plt.ylabel("Predicted Sales")

plt.tight_layout()
plt.savefig("dashboard.png")
plt.show()
