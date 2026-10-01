import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


df = pd.read_csv("insurance.csv")


df = pd.get_dummies(df, columns=["region", "sex", "smoker"], drop_first=True)


X = df.drop("charges", axis=1)
y = df["charges"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = GradientBoostingRegressor(n_estimators=100)
model.fit(X_train_scaled, y_train)


joblib.dump(model, "gradient_boost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
