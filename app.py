import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


st.title("🌍 Life Expectancy Prediction using Linear Regression")

st.write(
"""
📊 This app predicts Life Expectancy using socio-economic and health factors.

Features used:
Adult Mortality, Alcohol, GDP, Schooling, HIV/AIDS
"""
)

# -----------------------
# 1. LOAD DATA
# -----------------------

df = pd.read_csv("Life Expectancy Data.csv")

df.columns = df.columns.str.strip()

st.subheader("📂 Dataset Preview")
st.write(df.head())


# -----------------------
# 2. EXPLORE DATA
# -----------------------

st.subheader("🔍 Missing Values")
st.write(df.isnull().sum())

st.subheader("📈 Statistics")
st.write(df.describe())


# -----------------------
# 3. SELECT REQUIRED COLUMNS
# -----------------------

st.subheader("⚙️ Select Required Columns")

cols = [
    "Adult Mortality",
    "Alcohol",
    "GDP",
    "Schooling",
    "HIV/AIDS",
    "Life expectancy"
]

df = df[cols]

for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna()


# -----------------------
# Scatter plot
# -----------------------

st.subheader("📉 Scatter Plot")

fig = plt.figure()
sns.scatterplot(
    x=df["Schooling"],
    y=df["Life expectancy"]
)
st.pyplot(fig)


# -----------------------
# 4. PREPARE DATA
# -----------------------

st.subheader("🧠 Prepare Data")

X = df[[
    "Adult Mortality",
    "Alcohol",
    "GDP",
    "Schooling",
    "HIV/AIDS"
]]

y = df["Life expectancy"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------
# 5. BUILD MODEL
# -----------------------

st.subheader("🤖 Build Model")

model = LinearRegression()
model.fit(X_train, y_train)


# -----------------------
# 6. EVALUATE MODEL
# -----------------------

st.subheader("📊 Model Evaluation")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("MSE:", mse)
st.write("R2 Score:", r2)


# -----------------------
# 7. COEFFICIENTS
# -----------------------

st.subheader("📌 Model Coefficients")

coeff = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

st.write(coeff)


# -----------------------
# 8. ACTUAL VS PREDICTED
# -----------------------

st.subheader("📉 Actual vs Predicted")

fig2 = plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
st.pyplot(fig2)


# -----------------------
# 9. USER INPUT
# -----------------------

st.subheader("🧮 Predict Life Expectancy")

adult = st.number_input("Adult Mortality", 0.0, 1000.0, 200.0)
alcohol = st.number_input("Alcohol", 0.0, 20.0, 5.0)
gdp = st.number_input("GDP", 0.0, 100000.0, 5000.0)
school = st.number_input("Schooling", 0.0, 20.0, 10.0)
hiv = st.number_input("HIV/AIDS", 0.0, 50.0, 1.0)


if st.button("🔮 Predict"):

    input_data = pd.DataFrame(
        [[adult, alcohol, gdp, school, hiv]],
        columns=[
            "Adult Mortality",
            "Alcohol",
            "GDP",
            "Schooling",
            "HIV/AIDS"
        ]
    )

    pred = model.predict(input_data)

    st.success(
        f"✅ Predicted Life Expectancy: {pred[0]:.2f}"
    )
