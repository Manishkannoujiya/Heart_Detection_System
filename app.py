import streamlit as st 
import pandas as pd
import numpy as np 
import pickle 
import base64
import plotly.express as px
import joblib
from sklearn.impute import SimpleImputer




# Function to download prediction results
def get_binary_files_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Prediction CSV</a>'
    return href

st.title("Heart Disease Predictor")

tab1, tab2, tab3 = st.tabs(['Predict','Bulk Predict', 'Model Information'])

# ------------------------- TAB 1 -------------------------
with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non_Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    
    # Convert Categorical inputs to numeric
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non_Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 0 if fasting_bs == "<= 120 mg/dl" else 1
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)
    
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = ['DecisionTreeC.joblib', 'LogisticR.joblib', 'RandomForestC.joblib', 'SVM.joblib']  
 

    # Prediction function with safe loading
    def predict_heart_disease(data):
        predictions = []
        for modelname in modelnames:
            try:
                with open(modelname, 'rb') as f:
                    model = joblib.load(f)
                # Ensure columns match training
                model_features = model.feature_names_in_
                data_to_predict = data[model_features]
                prediction = model.predict(data_to_predict)
                predictions.append(prediction)
            except FileNotFoundError:
                st.error(f"Model file {modelname} not found!")
                predictions.append([None])
            except EOFError:
                st.error(f"Model file {modelname} is empty or corrupted!")
                predictions.append([None])
        return predictions

    # Submit button
    if st.button("Submit"):
        st.subheader('Results....')
        st.markdown('---------------')

        result = predict_heart_disease(input_data)

        for i in range(len(result)):
            st.subheader(algonames[i])
            if result[i][0] is None:
                st.write("Prediction could not be made due to model error.")
            elif result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
            st.markdown('--------------------')


# ------------------------- TAB 2 -------------------------
with tab2:
    st.title("Upload CSV File")
    st.subheader("Instruction: Ensure CSV has correct columns and categorical values match training.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        
        expected_column = ['Age' , 'Sex', 'CheastPainType','RestingBP' , 'Cholesterol' , 'FasatingBS' , 'RestingECG' , 'MaxHR' , 'ExerciseAngina' , 'Oldpeak', 'ST_Slope']
        if set(expected_column).issubset(input_data.columns):input_data['Prediction LR'] = ''
        
        try:
            # Load the model
            model = joblib.load('SVM.joblib')

            # ---------- Encode categorical columns ----------
            if 'Sex' in input_data.columns:
                input_data['Sex'] = input_data['Sex'].map({'M': 0, 'F': 1})
            if 'ExerciseAngina' in input_data.columns:
                input_data['ExerciseAngina'] = input_data['ExerciseAngina'].map({'Yes': 1, 'No': 0})
            if 'ChestPainType' in input_data.columns:
                mapping = {"Typical Angina":0, "Atypical Angina":1, "Non_Anginal Pain":2, "Asymptomatic":3}
                input_data['ChestPainType'] = input_data['ChestPainType'].map(mapping)
            if 'RestingECG' in input_data.columns:
                mapping = {"Normal":0, "ST-T Wave Abnormality":1, "Left Ventricular Hypertrophy":2}
                input_data['RestingECG'] = input_data['RestingECG'].map(mapping)
            if 'ST_Slope' in input_data.columns:
                mapping = {"Upsloping":0, "Flat":1, "Downsloping":2}
                input_data['ST_Slope'] = input_data['ST_Slope'].map(mapping)

            # ---------- Select model features ----------
            model_features = model.feature_names_in_
            input_data = input_data[model_features]

            # ---------- Impute missing values ----------
            imputer = SimpleImputer(strategy='median')
            input_data_imputed = pd.DataFrame(imputer.fit_transform(input_data), columns=model_features)

            # ---------- Run predictions ----------
            preds = model.predict(input_data_imputed)
            input_data['Prediction SVM'] = preds

            # ---------- Display and download ----------
            st.subheader("Predictions:")
            st.write(input_data)
            st.markdown(get_binary_files_downloader_html(input_data), unsafe_allow_html=True)

        except FileNotFoundError:
            st.error("Model file SVM.joblib not found!")
        except EOFError:
            st.error("Model file SVM.joblib is empty or corrupted!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    else:
        st.info("Upload a CSV file to get prediction.")

# ------------------------- TAB 3 -------------------------
with tab3:
    data = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 80.97,
        'Support Vector Machine': 84.22,

    }

    Models = list(data.keys())
    Accuracies = list(data.values())

    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])

    fig = px.bar(df, y='Accuracies', x='Models', text='Accuracies', title="Model Accuracies")
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    st.plotly_chart(fig)
