import os
import pickle
import warnings
import streamlit as st
import random
import streamlit_lottie
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.express as px
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Initialize session state for animations
if 'animation_placeholder' not in st.session_state:
    st.session_state.animation_placeholder = None
if 'current_animation' not in st.session_state:
    st.session_state.current_animation = None
if 'tips_key' not in st.session_state:
    st.session_state.tips_key = 0

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

working_dir = os.path.dirname(os.path.abspath(__file__))

# Cache Lottie files loading
@st.cache_resource
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load animations once
bg_file = load_lottiefile('video/bg.json')
healthy_heart = load_lottiefile("video/healthy.json")
diseased_heart = load_lottiefile("video/diseased.json")

BRAND_COLORS = {
    'primary': '#E65050',
    'secondary': '#E67878',
    'complementary': '#f8f9fa',  # Light background color
    'text': '#2C3E50'  # Dark text color
}

# Update the custom colormap for heatmaps
custom_cmap = sns.light_palette(BRAND_COLORS['primary'], as_cmap=True)

# Load model and dataset
@st.cache_resource
def load_model():
    return pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

heart_disease_model = load_model()

@st.cache_data
def load_data():
    data = pd.read_csv(f'{working_dir}/heart.csv')  # Adjust path as needed
    return data

# Load data and prepare model metrics
data = load_data()
X = data.drop('target', axis=1)
y = data['target']

# Split the data and calculate metrics (cached)
@st.cache_data
def calculate_metrics():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = heart_disease_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, conf_matrix, class_report, X_test, y_test

accuracy, conf_matrix, class_report, X_test, y_test = calculate_metrics()

# Cached function for feature importance calculation
@st.cache_data
def get_feature_importance():
    if hasattr(heart_disease_model, 'feature_importances_'):
        importance = heart_disease_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance
    # For models without feature_importances_ (like logistic regression)
    elif hasattr(heart_disease_model, 'coef_'):
        importance = np.abs(heart_disease_model.coef_[0])
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance
    return None

def plot_confusion_matrix():
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=custom_cmap,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title('Confusion Matrix', color=BRAND_COLORS['text'])
    plt.ylabel('True Label', color=BRAND_COLORS['text'])
    plt.xlabel('Predicted Label', color=BRAND_COLORS['text'])
    # Set the figure background color
    fig.patch.set_facecolor(BRAND_COLORS['complementary'])
    ax.set_facecolor(BRAND_COLORS['complementary'])
    return fig

def create_feature_importance_plot(feature_importance_df):
    fig = px.bar(feature_importance_df, x='feature', y='importance',
                 title='Feature Importance',
                 labels={'importance': 'Importance Score', 'feature': 'Features'})
    
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor=BRAND_COLORS['complementary'],
        paper_bgcolor=BRAND_COLORS['complementary'],
        title_font_color=BRAND_COLORS['text'],
        font_color=BRAND_COLORS['text']
    )
    
    fig.update_traces(marker_color=BRAND_COLORS['primary'])
    return fig

def create_histogram(data, selected_feature):
    fig = px.histogram(data, x=selected_feature, color='target',
                      title=f'Distribution of {selected_feature} by Heart Disease',
                      labels={'target': 'Heart Disease'},
                      color_discrete_map={0: BRAND_COLORS['secondary'], 
                                        1: BRAND_COLORS['primary']})
    
    fig.update_layout(
        plot_bgcolor=BRAND_COLORS['complementary'],
        paper_bgcolor=BRAND_COLORS['complementary'],
        title_font_color=BRAND_COLORS['text'],
        font_color=BRAND_COLORS['text']
    )
    return fig

def create_boxplot(data, selected_feature):
    fig = px.box(data, x='target', y=selected_feature,
                 title=f'Box Plot of {selected_feature} by Heart Disease',
                 labels={'target': 'Heart Disease'},
                 color='target',
                 color_discrete_map={0: BRAND_COLORS['secondary'], 
                                   1: BRAND_COLORS['primary']})
    
    fig.update_layout(
        plot_bgcolor=BRAND_COLORS['complementary'],
        paper_bgcolor=BRAND_COLORS['complementary'],
        title_font_color=BRAND_COLORS['text'],
        font_color=BRAND_COLORS['text']
    )
    return fig

def create_correlation_heatmap(data):
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap=custom_cmap, fmt='.2f')
    plt.title('Correlation Heatmap', color=BRAND_COLORS['text'])
    # Set the figure background color
    fig.patch.set_facecolor(BRAND_COLORS['complementary'])
    plt.gca().set_facecolor(BRAND_COLORS['complementary'])
    return fig



healthy_heart_tips = [
    "Maintain a Healthy Diet: Eat a variety of nutrient-rich foods, including fruits, vegetables, whole grains, and lean proteins.",
    "Stay Physically Active: Aim for at least 30 minutes of moderate exercise most days of the week.",
    "Avoid Smoking: Smoking is a major risk factor for heart disease. Seek help to quit if necessary.",
    "Limit Alcohol Intake: Drink alcohol in moderation, if at all. Excessive drinking can lead to heart problems.",
    "Monitor Blood Pressure: Keep track of your blood pressure and seek medical advice if it's consistently high.",
    "Maintain a Healthy Weight: Being overweight can increase your risk of heart disease. Aim for a healthy weight through diet and exercise.",
    "Manage Stress: Practice relaxation techniques like yoga, meditation, or deep breathing to manage stress.",
    "Get Regular Health Screenings: Regular check-ups can help detect risk factors early.",
    "Stay Hydrated: Drink plenty of water throughout the day to support heart health.",
    "Get Enough Sleep: Aim for 7-9 hours of quality sleep per night to support heart health."
]

def choice_tip():
    st.write(healthy_heart_tips[st.session_state.tips_key])

def new_tip():
    st.session_state.tips_key = (st.session_state.tips_key + 1) % len(healthy_heart_tips)

# Sidebar content
with st.sidebar:
    st.header(":heart: Heart Disease Prediction App")
    st.write("───────── ⋆⋅:heart:⋅⋆ ───────────")
    
    # Display background animation
    streamlit_lottie.st_lottie(
        bg_file,
        speed=5,
        loop=True,
        quality='medium',
        reverse=False,
        key="bg_animation"
    )
    
    st.header('Tip of the Day', divider='rainbow')
    with st.container(border=True):
        choice_tip()
        st.button("New Tip", on_click=new_tip)
    
    # Model metrics in sidebar
    st.header('Model Performance', divider='rainbow')
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    st.subheader("Class-wise Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precision (Disease)", f"{class_report['1']['precision']:.2%}")
        st.metric("Recall (Disease)", f"{class_report['1']['recall']:.2%}")
    with col2:
        st.metric("Precision (No Disease)", f"{class_report['0']['precision']:.2%}")
        st.metric("Recall (No Disease)", f"{class_report['0']['recall']:.2%}")

# Main content
st.header('Heart Disease Prediction using ML', divider='rainbow')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Analysis", "Data Insights"])

with tab1:
    with st.form(key="prediction_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        
        # Form inputs (keep your existing input fields here)
        with col1:
            age = st.slider('Age', 1, 100, 30, help='Select your age')
        with col2:
            sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female', help='Select your sex')
        with col3:
            choice = st.selectbox('Chest Pain Types (CP)', options=["Asymptomatic", "Atypical Angina", "Typical Angina", "Non-Angina"], help='0: Asymptomatic, 1: Atypical Angina, 2: Typical Angina, 3: Non-Angina')
            cp = {"Asymptomatic": 0, "Atypical Angina": 1, "Typical Angina": 2, "Non-Angina": 3}[choice]

        with col1:
            trestbps = st.number_input('Resting Blood Pressure (TRTBPS)', min_value=0, help='Enter your resting blood pressure in mmHg')
        with col2:
            chol = st.number_input('Serum Cholesterol in mg/dl (Chol)', min_value=0, help='Enter your serum cholesterol level')
        with col3:
            choice = st.selectbox('Fasting Blood Sugar (FBS)', options=["FBS > 120 mg/dl", "FBS <= 120 mg/dl"], help='Select Your BP')
            fbs = 1 if choice == "FBS > 120 mg/dl" else 0

        with col1:
            choice = st.selectbox('Resting Electrocardiographic Results (Rest ECG)', options=["Normal", "ST Elevation", "Others"], help='0: Normal, 1: ST Elevation, 2: Others')
            restecg = {"Normal": 0, "ST Elevation": 1, "Others": 2}[choice]
        with col2:
            thalach = st.number_input('Maximum Heart Rate Achieved (Thalachh)', min_value=0, help='Enter your maximum heart rate achieved')
        with col3:
            choice = st.selectbox('Exercise Induced Angina (Exng)', options=["Yes", "No"], help='1: Yes, 0: No')
            exang = 1 if choice == "Yes" else 0

        with col1:
            oldpeak = st.number_input('ST Depression Induced by Exercise (Oldpeak)', min_value=0.0, format="%.2f", help='Enter the ST depression value')
        with col2:
            choice = st.selectbox('Slope of the Peak Exercise ST Segment (Slope)', options=["Flat", "Up Sloping", "Down Sloping"], help='0: Flat, 1: Up Sloping, 2: Down Sloping')
            slope = {"Flat": 0, "Up Sloping": 1, "Down Sloping": 2}[choice]
        with col3:
            ca = st.selectbox('Number of Major Vessels (CA)', options=[0, 1, 2, 3], help='Number of major vessels colored by fluoroscopy')

        with col1:
            choice = st.selectbox('Thalassemia (thal)', options=["Normal", "Fixed Defect", "Reversible Defect"], help='1: Normal, 2: Fixed Defect, 3: Reversible Defect')
            thal = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[choice]

        submitted = st.form_submit_button("Results", type='primary')

    # Create a placeholder for the prediction result
    result_placeholder = st.empty()
    animation_placeholder = st.empty()

    if submitted:
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        if any(x is None or x == '' for x in user_input):
            result_placeholder.error("Please fill in all the details....", icon="🥲")
        else:
            user_input = [float(x) for x in user_input]
            heart_prediction = heart_disease_model.predict([user_input])
            prediction_proba = heart_disease_model.predict_proba([user_input])[0]
            
            if heart_prediction[0] == 1:
                result_placeholder.error('The person is having heart disease')
                with animation_placeholder:
                    streamlit_lottie.st_lottie(
                        diseased_heart,
                        speed=1,
                        loop=True,
                        quality='medium',
                        reverse=False,
                        height=300,
                        key="diseased_animation"
                    )
            else:
                result_placeholder.success('The person does not have any heart disease')
                with animation_placeholder:
                    streamlit_lottie.st_lottie(
                        healthy_heart,
                        speed=1,
                        loop=True,
                        quality='medium',
                        reverse=False,
                        height=300,
                        key="healthy_animation"
                    )
            
            st.write(f"Confidence: {max(prediction_proba):.2%}")

with tab2:
    st.subheader("Confusion Matrix")
    st.pyplot(plot_confusion_matrix())
    
    st.subheader("Feature Importance")
    feature_importance_df = get_feature_importance()
    if feature_importance_df is not None:
        st.plotly_chart(create_feature_importance_plot(feature_importance_df))
    else:
        st.info("Feature importance is not available for this model type.")

# Update the Data Insights tab
with tab3:
    st.subheader("Data Distribution")
    selected_feature = st.selectbox("Select Feature to Visualize", X.columns)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_histogram(data, selected_feature))
    
    with col2:
        st.plotly_chart(create_boxplot(data, selected_feature))
    
    st.subheader("Feature Correlations")
    st.pyplot(create_correlation_heatmap(data))