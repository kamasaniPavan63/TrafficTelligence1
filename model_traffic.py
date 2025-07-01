import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Step 1: Load your dataset
df = pd.read_csv("traffic volume.csv")

# Step 2: Extract necessary features
df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['Time'], errors='coerce')
df.dropna(inplace=True)

df['holiday'] = df['holiday'].apply(lambda x: 0 if x == 'None' else 1)
df['weather'] = df['weather'].astype('category').cat.codes  # convert to numbers

df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hours'] = df['date_time'].dt.hour
df['minutes'] = df['date_time'].dt.minute
df['seconds'] = df['date_time'].dt.second

# Step 3: Rename for consistency with app.py/index.html
df.rename(columns={'rain_1h': 'rain', 'snow_1h': 'snow'}, inplace=True)

# Step 4: Final feature selection
features = ['holiday', 'temp', 'rain', 'snow', 'weather',
            'year', 'month', 'day', 'hours', 'minutes', 'seconds']

X = df[features]
y = df['traffic_volume']

# Step 5: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Model Trained")
print("RMSE:", rmse)
print("R2 Score:", r2)

# Step 9: Save model and scaler
os.makedirs("Flask", exist_ok=True)
joblib.dump(model, "Flask/model.pkl")
joblib.dump(scaler, "Flask/encoder.pkl")
print("✅ model.pkl and encoder.pkl saved to Flask folder.")
