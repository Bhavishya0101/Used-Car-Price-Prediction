import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Used Car Price Prediction",
    layout="centered"
)

st.title("🚘 Used Car Price Prediction")
st.write(
    "Predict used car prices using Machine Learning to help buyers, sellers, "
    "and dealers make informed decisions."
)

# ======================================
# LOAD & CLEAN DATA
# ======================================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("usedCars.csv")

    # -------- Convert Price (Lakhs → Numeric) --------
    def convert_price(val):
        if isinstance(val, str):
            val = val.lower().replace(",", "").strip()
            if "lakh" in val:
                return float(val.replace("lakhs", "").replace("lakh", "")) * 100000
            if "crore" in val:
                return float(val.replace("crore", "")) * 10000000
        return float(val)

    df["Price"] = df["Price"].apply(convert_price)

    # -------- Feature Engineering --------
    df["car_age"] = 2025 - df["ModelYear"]

    return df

df = load_and_clean_data()

# ======================================
# FEATURES & TARGET (MATCH DATASET)
# ======================================
TARGET = "Price"

FEATURES = [
    "Company",
    "FuelType",
    "TransmissionType",
    "Owner",
    "BodyStyle",
    "ModelYear",
    "Kilometer",
    "car_age"
]

X = df[FEATURES]
y = df[TARGET]

# ======================================
# COLUMN GROUPS
# ======================================
NUM_COLS = ["ModelYear", "Kilometer", "car_age"]
CAT_COLS = ["Company", "FuelType", "TransmissionType", "Owner", "BodyStyle"]

# ======================================
# TRAIN MODEL (SAFE)
# ======================================
@st.cache_resource
def train_model(X, y):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUM_COLS),
        ("cat", categorical_pipeline, CAT_COLS)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model

model = train_model(X, y)

# ======================================
# USER INPUT SECTION
# ======================================
st.header("🔮 Predict Used Car Price")

company = st.selectbox("Company", sorted(df["Company"].unique()))
fuel = st.selectbox("Fuel Type", df["FuelType"].unique())
transmission = st.selectbox("Transmission", df["TransmissionType"].unique())
owner = st.selectbox("Owner", df["Owner"].unique())
body = st.selectbox("Body Style", df["BodyStyle"].unique())

year = st.slider("Manufacturing Year", 2000, 2025, 2018)
km_driven = st.number_input("Kilometers Driven", 0, 500000, 50000)

car_age = 2025 - year

# ======================================
# CAR IMAGE
# ======================================
image_path = f"car_images/{company.lower()}.png"
if not os.path.exists(image_path):
    image_path = "car_images/default.png"

st.image(image_path, width=280)

# ======================================
# PREDICTION
# ======================================
input_df = pd.DataFrame([{
    "Company": company,
    "FuelType": fuel,
    "TransmissionType": transmission,
    "Owner": owner,
    "BodyStyle": body,
    "ModelYear": year,
    "Kilometer": km_driven,
    "car_age": car_age
}])

if st.button("Predict Price"):
    price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Used Car Price: ₹ {int(price):,}")
