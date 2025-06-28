import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# loading, preprocessing data
@st.cache_data
def preprocess_data():
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")          # loading CSV
    X = df.drop("DEATH_EVENT", axis=1)                                      # dropping "DEATH_EVENT" column
    y = df["DEATH_EVENT"]
    scaler = StandardScaler()                                               # initialize standard scaler
    X_scaled = scaler.fit_transform(X)                                      # scaled features
    return df, X_scaled, y, scaler

# training both models
@st.cache_resource
def train_models(X_train, y_train):
    log_reg = LogisticRegression(random_state=42)                           # Initializing the logistic regression model
    log_reg.fit(X_train, y_train)                                           # Training logistic regression model on training data

    rf = RandomForestClassifier(random_state=42)                            # Initializing random forest classifier
    rf.fit(X_train, y_train)                                                # Training random forest model on training data
    return log_reg, rf

# tryinbg to optimize Random Forest using GridSearchCV for hyperparameter tuning
@st.cache_resource
def optimize_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)           # Initializing GridSearchCV
    grid_search.fit(X_train, y_train)                                                               # Performing grid search on training data
    return grid_search.best_estimator_                                                              # Returns the best model

# load and preprocess data
df, X_scaled, y, scaler = preprocess_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training models and optimizing Random Forest
log_reg, rf = train_models(X_train, y_train)
best_rf = optimize_random_forest(X_train, y_train)

# Evaluating models
y_pred_log = log_reg.predict(X_test)                                    # prediction from logistic regression model
y_pred_rf = rf.predict(X_test)                                          # prediction from the baseline random forest model
y_pred_best_rf = best_rf.predict(X_test)                                # prediction from optimized random forest model

# print("Logistic Regression Performance:")
# print(classification_report(y_test, y_pred_log))

# print("Random Forest Performance:")
# print(classification_report(y_test, y_pred_rf))

# print("Optimized Random Forest Performance:")
# print(classification_report(y_test, y_pred_best_rf))

# app title nad description
st.title("Heart Failure Mortality Prediction Dashboard")
st.write("""
This dashboard predicts the risk of mortality due to heart failure using machine learning models.
Adjust the patient features in the sidebar to see the prediction results.
""")

# sidebar input sliders for patient features
with st.form("user_input_form"):
    age = st.slider("Age", int(df["age"].min()), int(df["age"].max()))
    serum_creatinine = st.slider("Serum Creatinine", float(df["serum_creatinine"].min()), float(df["serum_creatinine"].max()))
    ejection_fraction = st.slider("Ejection Fraction", int(df["ejection_fraction"].min()), int(df["ejection_fraction"].max()))
    
    # submit button for the form
    submitted = st.form_submit_button("Submit")

if submitted:
    patient_data = pd.DataFrame([{
        "age": age,
        "anaemia": 0,
        "creatinine_phosphokinase": df["creatinine_phosphokinase"].mean(),
        "diabetes": 0,
        "ejection_fraction": ejection_fraction,
        "high_blood_pressure": 0,
        "platelets": df["platelets"].mean(),
        "serum_creatinine": serum_creatinine,
        "serum_sodium": df["serum_sodium"].mean(),
        "sex": 1,
        "smoking": 0,
        "time": df["time"].mean()
    }])

    patient_data_scaled = scaler.transform(patient_data)

    prediction = best_rf.predict(patient_data_scaled)                                                   # make prediction using optimized Random Forest model
    probability = best_rf.predict_proba(patient_data_scaled)[:, 1]

    st.write(f"Predicted Risk of Mortality: {'High' if prediction[0] == 1 else 'Low'}")                 # display prediction results on the dashboard
    st.write(f"Probability of Mortality: {probability[0]:.2f}")

# visualization: correlation heatmap
if "correlation_heatmap" not in st.session_state:
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Correlation Heatmap")
    st.session_state["correlation_heatmap"] = fig_corr

st.pyplot(st.session_state["correlation_heatmap"])

# visualization: pairplot of selected features
if "pairplot" not in st.session_state:
    sns.set_theme(style="ticks")
    pairplot_fig = sns.pairplot(df, vars=["age", "serum_creatinine", "ejection_fraction"], hue="DEATH_EVENT")
    pairplot_fig.fig.suptitle("Pairplot of Selected Features", y=1.02)          # add title to pairplot
    st.session_state["pairplot"] = pairplot_fig

st.pyplot(st.session_state["pairplot"])

