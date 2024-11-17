import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pandas as pd

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Read the CSV file using the base directory
df = pd.read_csv(os.path.join(BASE_DIR, 'gym_members_exercise_tracking.csv'))


# Streamlit App Title
st.title("Calories Burned Prediction App")
st.write("From the Historical Data of the gym we have the following important visualisations.")

# Display Histograms
st.header("Histograms")

# Calories burned Distribution
fig1, ax1 = plt.subplots()
df['Calories_Burned'].plot(kind='hist', bins=30, ax=ax1, color='skyblue', edgecolor='black')
ax1.set_title('Distribution of Calories Burned')
ax1.set_xlabel('Calories Burned')
st.pyplot(fig1)

# Fat percentage Distribution
fig2, ax2 = plt.subplots()
df['Fat_Percentage'].plot(kind='hist', bins=30, ax=ax2, color='lightgreen', edgecolor='black')
ax2.set_title('Distribution of Fat Percentage')
ax2.set_xlabel('Fat Percentage (%)')
st.pyplot(fig2)

# BMI Distribution
fig3, ax3 = plt.subplots()
df['BMI'].plot(kind='hist', bins=30, ax=ax3, color='salmon', edgecolor='black')
ax3.set_title('Distribution of BMI')
ax3.set_xlabel('BMI')
st.pyplot(fig3)

# Correlation Analysis
st.header("Correlation Analysis")
correlation_features = ['Calories_Burned', 'Session_Duration (hours)', 'Workout_Frequency (days/week)', 'Experience_Level']
correlation_matrix = df[correlation_features].corr()

fig4, ax4 = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax4)
ax4.set_title('Correlation between Selected Features')
st.pyplot(fig4)

# Boxplots
st.header("Boxplots")

# Avg_BPM across Workout Types
fig5, ax5 = plt.subplots()
sns.boxplot(x='Workout_Type', y='Avg_BPM', data=df, ax=ax5)
ax5.set_title('Average BPM across Workout Types')
st.pyplot(fig5)

# BMI across Workout Types
fig6, ax6 = plt.subplots()
sns.boxplot(x='Workout_Type', y='BMI', data=df, ax=ax6)
ax6.set_title('BMI across Workout Types')
st.pyplot(fig6)

# Fat_Percentage across Workout Types
fig7, ax7 = plt.subplots()
sns.boxplot(x='Workout_Type', y='Fat_Percentage', data=df, ax=ax7)
ax7.set_title('Fat Percentage across Workout Types')
st.pyplot(fig7)

st.write("Visualizations help us better understand the data and the relationships between different features.")

# Load the trained linear regression model
model_filename = r"C:\Users\Satyajit\Documents\Masai_ML_Project\linear_regression_model.pkl"
with open(model_filename, 'rb') as file:
    model = pickle.load(file)
    print("Model loaded successfully!")


st.title("Enter features manually to predict Calories Burned.")

# Input Boxes for all features
age = st.number_input("Age", min_value=18, max_value=100, value=25)
weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=70)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.75)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
gender = st.selectbox("Gender", ["Female", "Male"])
workout_type = st.selectbox("Workout Type", ["HIIT", "Strength", "Yoga"])
session_duration = st.slider("Session Duration (hours)", min_value=0.5, max_value=3.0, value=1.0)
max_bpm = st.number_input("Max BPM", min_value=60, max_value=220, value=120)
resting_bpm = st.number_input("Resting BPM", min_value=40, max_value=100, value=70)
avg_bpm = st.number_input("Average BPM", min_value=40, max_value=200, value=80)
fat_percentage = st.number_input("Fat Percentage (%)", min_value=0.0, max_value=50.0, value=20.0)
experience_level = st.slider("Experience Level (1=Beginner, 3=Advanced)", min_value=1, max_value=3, value=2)
water_intake = st.number_input("Water Intake (liters)", min_value=0.5, max_value=10.0, value=2.0)
workout_frequency = st.number_input("Workout Frequency (days/week)", min_value=1, max_value=7, value=3)

# Display the Inputs
st.write("Here are the inputs you provided:")
st.write(f"Age: {age}, Weight: {weight}, Height: {height}, BMI: {bmi}")
st.write(f"Gender: {gender}, Workout Type: {workout_type}")
st.write(f"Session Duration: {session_duration} hours, Max BPM: {max_bpm}, Resting BPM: {resting_bpm}")
st.write(f"Average BPM: {avg_bpm}, Fat Percentage: {fat_percentage}%, Experience Level: {experience_level}")
st.write(f"Water Intake: {water_intake} liters, Workout Frequency: {workout_frequency} days/week")

input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [1 if gender == "Male" else 0],  # Encoding Gender: 1 for Male, 0 for Female
    "Weight (kg)": [weight],
    "Height (m)": [height],
    "Max_BPM": [max_bpm],
    "Avg_BPM": [avg_bpm],
    "Resting_BPM": [resting_bpm],
    "Session_Duration (hours)": [session_duration],
    "Fat_Percentage": [fat_percentage],
    "Water_Intake (liters)": [water_intake],
    "Workout_Frequency (days/week)": [workout_frequency],
    "Experience_Level": [experience_level],
    "BMI": [bmi],
    "Workout_Type_HIIT": [1 if workout_type == "HIIT" else 0],
    "Workout_Type_Strength": [1 if workout_type == "Strength" else 0],
    "Workout_Type_Yoga": [1 if workout_type == "Yoga" else 0]
})

# Predict Calories Burned
if st.button("Predict Calories Burned"):
    if model is not None:
        # Make a prediction
        prediction = model.predict(input_data)
        st.success(f"Predicted Calories Burned: {prediction[0]:.2f}")
    else:
        st.error("Model is not loaded. Please check if the model file exists.")











