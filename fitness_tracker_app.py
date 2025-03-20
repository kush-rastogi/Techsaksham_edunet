import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 22)
    duration = st.sidebar.slider("Exercise Duration (min)", 0, 60, 20)
    heart_rate = st.sidebar.slider("Heart Rate", 50, 150, 85)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35, 42, 37)
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    activity_level = st.sidebar.selectbox("Activity Level", ["Low", "Medium", "High"])
    
    gender_encoded = 1 if gender == "Female" else 0
    activity_factor = {"Low": 1, "Medium": 2, "High": 3}[activity_level]
    
    bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    
    data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_Female": gender_encoded,
        "Activity_Level": activity_factor
    }
    
    return pd.DataFrame(data, index=[0]), bmi_category

df, bmi_category = user_input_features()

st.title("Personal Fitness Tracker")
st.write("### Your Input Parameters")
st.write(df)
st.write(f"#### BMI Category: {bmi_category}")

# Simulated progress bar
progress_bar = st.progress(0)
for percent in range(101):
    time.sleep(0.005)
    progress_bar.progress(percent)

# Load & Process Data
df_calories = pd.read_csv("calories.csv")
df_exercise = pd.read_csv("exercise.csv")
data = df_exercise.merge(df_calories, on="User_ID").drop(columns="User_ID")

data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)
data = data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
data = pd.get_dummies(data, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(data.drop("Calories", axis=1), data["Calories"], test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=500, max_depth=5)
model.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = model.predict(df)

st.write("### Estimated Calories Burned: ")
st.subheader(f"{round(prediction[0], 2)} kcal")

# Similar results
st.write("### Similar Results")
similar_range = [prediction[0] - 15, prediction[0] + 15]
similar_data = data[(data["Calories"] >= similar_range[0]) & (data["Calories"] <= similar_range[1])]
st.write(similar_data.sample(min(5, len(similar_data))))

st.write("### Additional Insights")
st.write(f"You are older than {round((data['Age'] < df['Age'].values[0]).mean() * 100, 2)}% of users.")
st.write(f"Your heart rate is higher than {round((data['Heart_Rate'] < df['Heart_Rate'].values[0]).mean() * 100, 2)}% of users.")
