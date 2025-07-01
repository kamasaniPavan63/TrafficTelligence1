import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Step 1: Load Dataset (use local path!)
df = pd.read_csv("traffic volume.csv")

# Step 2: Clean column names
df.columns = df.columns.str.strip()

# Step 3: Combine 'date' and 'Time' into 'date_time'
if 'date' in df.columns and 'Time' in df.columns:
    df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['Time'], errors='coerce')
    df.drop(['date', 'Time'], axis=1, inplace=True)
else:
    raise KeyError("Expected 'date' and 'Time' columns not found in the data.")

# Step 4: Handle missing values
df.dropna(inplace=True)

# Step 5: Feature Engineering
df['hour'] = df['date_time'].dt.hour
df['dayofweek'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
df.drop(['date_time'], axis=1, inplace=True)

# Step 6: One-hot encoding
df = pd.get_dummies(df, columns=['holiday', 'weather'], drop_first=True)

# Step 7: Features and target
X = df.drop(['traffic_volume'], axis=1)
y = df['traffic_volume']

# Step 8: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 9: Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 10: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 11: Evaluate
# Step 11: Evaluate
y_pred = model.predict(X_test)  # <--- This line is essential
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)



# Step 12: Save model and scaler
os.makedirs("Flask", exist_ok=True)
joblib.dump(model, "Flask/model.pkl")
joblib.dump(scaler, "Flask/encoder.pkl")
print("✅ Model and scaler saved to 'Flask/' folder.")
