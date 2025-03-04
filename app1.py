import streamlit as st

st.title("Machine Learning Workflow")

# Code snippets
def get_code(section):
    codes = {
        "Load Libraries": """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score""",
        "Load Dataset": """df = pd.read_csv('data.csv')
st.write(df.head())""",
        "Train-Test Split": """X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)""",
        "Train Model": """model = RandomForestClassifier()
model.fit(X_train, y_train)""",
        "Make Predictions": """y_pred = model.predict(X_test)""",
        "Calculate Accuracy": """accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')""",
        "Visualize Results": """sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()"""
    }
    return codes.get(section, "No code available")

# Buttons to show code
sections = [
    "Load Libraries", "Load Dataset", "Train-Test Split", "Train Model", "Make Predictions", "Calculate Accuracy", "Visualize Results"
]

for section in sections:
    if st.button(f"Show {section} Code"):
        st.code(get_code(section), language='python')