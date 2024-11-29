# Import necessary libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import streamlit as st

# Cache for loading and preprocessing the dataset
@st.cache_data
def load_data():
    file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    column_names = ["ID", "Diagnosis"] + [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
        'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
        'fractal_dimension_worst'
    ]
    data = pd.read_csv(file_path, header=None, names=column_names)
    data.drop('ID', axis=1, inplace=True)
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})
    return data

data = load_data()

# Cache for preprocessing and splitting the data
@st.cache_data
def preprocess_data(data):
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, X_test, y_train_smote, y_test, scaler

X_train_smote, X_test, y_train_smote, y_test, scaler = preprocess_data(data)

# Cache for training the best AdaBoost model
@st.cache_resource
def train_model(X_train_smote, y_train_smote):
    ada_classifier = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), 
        n_estimators=50,
        learning_rate=1.0,
        algorithm="SAMME",
        random_state=42
    )
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.5, 1.0, 1.5],
        'estimator__max_depth': [1, 2, 3]
    }
    grid_search = GridSearchCV(estimator=ada_classifier, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train_smote, y_train_smote)
    return grid_search.best_estimator_

best_model = train_model(X_train_smote, y_train_smote)

# Default values for malignant diagnosis
default_values = {
    'radius_mean': 20.0,
    'texture_mean': 25.0,
    'perimeter_mean': 120.0,
    'area_mean': 1000.0,
    'smoothness_mean': 0.15,
    'compactness_mean': 0.30,
    'concavity_mean': 0.40,
    'concave_points_mean': 0.20,
    'symmetry_mean': 0.25,
    'fractal_dimension_mean': 0.08,
    'radius_se': 1.2,
    'texture_se': 1.5,
    'perimeter_se': 8.0,
    'area_se': 40.0,
    'smoothness_se': 0.01,
    'compactness_se': 0.03,
    'concavity_se': 0.05,
    'concave_points_se': 0.02,
    'symmetry_se': 0.03,
    'fractal_dimension_se': 0.01,
    'radius_worst': 25.0,
    'texture_worst': 30.01,
    'perimeter_worst': 140.0,
    'area_worst': 1500.0,
    'smoothness_worst': 0.20,
    'compactness_worst': 0.40,
    'concavity_worst': 0.50,
    'concave_points_worst': 0.25,
    'symmetry_worst': 0.30
}

# Streamlit Layout
st.title('Breast Cancer Diagnosis Prediction')
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Make Prediction"])

# Home Page: Overview and Data Visualizations
if page == "Home":
    st.subheader("Basic Overview of the Dataset")
    st.write(data.head())
    st.write(f"Shape of the dataset: {data.shape}")
    
    # Pie chart for the distribution of diagnosis
    diagnosis_counts = data['Diagnosis'].value_counts()
    st.subheader("Distribution of Breast Cancer Diagnosis")
    fig, ax = plt.subplots()
    ax.pie(diagnosis_counts, labels=['Benign (0)', 'Malignant (1)'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
    ax.axis('equal')
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(data.corr(), annot=True, fmt='.0%', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Confusion Matrix for Best AdaBoost Model
    st.subheader("Confusion Matrix for Best AdaBoost Classifier")
    y_pred_best = best_model.predict(X_test)
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Best Model)')
    st.pyplot(fig)

# Make Prediction Page: User Input for Prediction
# Make Prediction Page: User Input for Prediction
elif page == "Make Prediction":
    st.subheader("Make Predictions")
    st.write("Enter the values for the features to predict whether the tumor is MALIGNANT or BENIGN.")
    
    # Create input fields with default malignant values
    user_inputs = {}
    for feature in data.columns[:-1]:
        user_inputs[feature] = st.number_input(
            f"Enter value for {feature}",
            value=default_values.get(feature, 0.0)  # Use default malignant value
        )
    
    if st.button("Make Prediction"):
        user_input_array = np.array(list(user_inputs.values())).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input_array)
        prediction = best_model.predict(user_input_scaled)
        
        # Set background color based on prediction
        if prediction == 1:
            st.write("The tumor is *MALIGNANT*.")
            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #8B0000;
                }
                </style>
                """, unsafe_allow_html=True
            )
        else:
            st.write("The tumor is *BENIGN*.")
            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #013220;
                }
                </style>
                """, unsafe_allow_html=True
            )
