import streamlit as st 
import pandas as pd
import joblib
# import ploty.graph_objects as go
import pickle 
import os
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Churn Prediction",layout="wide", page_icon="👨‍🦰🤶")

st.markdown("""
            <style>
            .stButton>button {
                width: 100%;
                background-color: #FF4B4B;
                color: white;
                font-weight: bold;
                padding: 0.5rem;
                border-radius: 10px;
            }
            prediction-box {
                padding: 1.5rem;
                border-radius: 10px;
                text-align: center;
                font-size: 1.3rem;
                font-weight: bold;
            }
            .churn { background-color: #ffebee; color: #c62828; }
            .no-churn{ background-color: #e8f5e9; color: #2e7d32; }
            </style>
        """,unsafe_allow_html=True)
            
st.cache_resource
def load_model():
    try:
        model=joblib.load('Churn_Prediction.pkl')
        return model
    except FileNotFoundError:
        st.error('Model file not found. Run the notebook first.')
        return None, None
    
model = load_model()

st.title('Customer Churn Predition')
st.markdown('Predict customer churn with Machine Learning')
st.markdown('---')
                
working_dir = os.path.dirname(os.path.abspath(__file__))

# model = pickle.load(open(f'{working_dir}Churn_prediction.pkl','rb'))
# model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Churn_Prediction.pkl")
# model = joblib.load(model_path)
joblib.dump(model, 'Churn_Prediction.pkl')
print("Model saved!")

NewBMI_Overweight=0
NewBMI_Underweight=0
NewBMI_Obesity_1=0
NewBMI_Obesity_2=0 
NewBMI_Obesity_3=0
NewInsulinScore_Normal=0 
NewGlucose_Low=0
NewGlucose_Normal=0 
NewGlucose_Overweight=0
NewGlucose_Secret=0

# with st.sidebar:
#     selected = option_menu("Churn Prediction",
#                  menu_icon='hospital-fill',
#                  icons=['person'],
#                  default_index=0)
with st.sidebar:
    selected = option_menu(
        menu_title="Churn Prediction",
        options=["Churn Prediction"],
        icons=["person"],
        menu_icon="bank",
        default_index=0
    )
    
if selected == 'Churn Prediction':
    st.title("Churn Prediction Using Machine Learning")

    col1, col2, col3 = st.columns(3)

    with col1:
        Year = st.text_input("Year")
    with col2:
        Credit_Score = st.text_input("Credit_score")
    with col3:
        Age = st.text_input("Age")
    with col1:
        Tenure = st.text_input("Tenure")
    with col2:
        Balance = st.text_input("Balance")
    with col3:
        NumOfProducts = st.text_input("NumOfProducts")
    with col1:
        HasCrCard = st.text_input("HasCrCard")
    with col2:
        IsActiveMember = st.text_input("IsActiveMember")
    with col3:
        EstimatedSalary = st.text_input("EstimatedSalary")
    with col1:
        Exited = st.text_input("Exited")
    with col2:
        Geography_Germany = st.text_input("Geography_Germany")
    with col3:
        Geography_Spain = st.text_input("Geography_Spain")
    with col1:
        Gender_Male = st.text_input("Gender_Male")  
    
    churn_result = ""
    if st.button("Churn Prediction Result"):

        user_input=['Year', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
       'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited',
       'Geography_Germany', 'Geography_Spain', 'Gender_Male']
        
        user_input = [float(x) for x in user_input]
        prediction = model.predict([user_input])
        if prediction[0]==1:
            churn_result = "The person has exited"
        else:
            churn_result = "The person has no exited"
    st.success(churn_result)

