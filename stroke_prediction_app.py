import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Aplikasi Prediksi Risiko Stroke",
    page_icon="🧠",
    layout="wide"
)

# Title of the application
st.title("🧠 Aplikasi Prediksi Risiko Stroke")
st.markdown("""
Aplikasi ini memprediksi kemungkinan seseorang mengalami stroke berdasarkan profil pribadi dan kesehatan mereka.
""")

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        # Load the complete model data (contains model, scaler, and encoders)
        with open('model_terbaik.pkl', 'rb') as file:
            model_data = pickle.load(file)
        return model_data
    except FileNotFoundError:
        st.error("File model 'model_terbaik.pkl' tidak ditemukan. Pastikan model telah dilatih dan disimpan.")
        return None

model_data = load_model()

if model_data is not None:
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    numerical_columns = model_data['numerical_columns']
    categorical_columns = model_data['categorical_columns']

    # Remove 'id' from numerical_columns if it exists, as it shouldn't be used for prediction
    if 'id' in numerical_columns:
        numerical_columns = [col for col in numerical_columns if col != 'id']

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Fitur Input")
        
        # Numerical inputs (excluding 'id')
        age = st.slider("Usia", min_value=1, max_value=100, value=40, 
                        help="Usia dalam tahun")
        avg_glucose_level = st.slider("Rata-rata Kadar Glukosa", min_value=50, max_value=300, value=100,
                                     help="Rata-rata kadar glukosa dalam darah")
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0,
                             help="Indeks massa tubuh")
        
        # Categorical inputs
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        hypertension = st.selectbox("Hipertensi", ["0", "1"], format_func=lambda x: "Tidak" if x == "0" else "Ya")
        heart_disease = st.selectbox("Penyakit Jantung", ["0", "1"], format_func=lambda x: "Tidak" if x == "0" else "Ya")
        ever_married = st.selectbox("Pernah Menikah", ["No", "Yes"])

    with col2:
        st.header("Fitur Tambahan")
        
        # More categorical inputs
        work_type = st.selectbox("Jenis Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Jenis Tempat Tinggal", ["Urban", "Rural"])
        smoking_status = st.selectbox("Status Merokok", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    # Create a button to make prediction
    if st.button("🔮 Prediksi Risiko Stroke", type="primary"):
        # Create a dataframe with the input values (excluding 'id')
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [int(hypertension)],
            'heart_disease': [int(heart_disease)],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status]
        })

        try:
            # Encode categorical variables using the saved label encoders
            input_encoded = input_data.copy()
            
            for col in categorical_columns:
                if col in input_encoded.columns:
                    # Handle unseen labels by using the first label in the encoder
                    le = label_encoders[col]
                    input_values = input_encoded[col].astype(str)
                    
                    # Transform values, handling unseen labels
                    transformed_values = []
                    for val in input_values:
                        try:
                            transformed_val = le.transform([val])[0]
                        except ValueError:
                            # If value not seen during training, use the first label
                            transformed_val = le.transform([le.classes_[0]])[0]
                        transformed_values.append(transformed_val)
                    
                    input_encoded[col] = transformed_values

            # Scale numerical features using the saved scaler
            input_scaled = input_encoded.copy()
            # Only scale the numerical columns that were used during training (excluding 'id')
            cols_to_scale = [col for col in numerical_columns if col in input_encoded.columns]
            input_scaled[cols_to_scale] = scaler.transform(input_encoded[cols_to_scale])
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Display results
            st.success("Prediksi selesai!")
            
            # Create two columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.subheader("Hasil Prediksi")
                if prediction[0] == 1:
                    st.error("🚨 RISIKO TINGGI: Pasien ini berisiko mengalami stroke!")
                    st.image("https://cdn-icons-png.flaticon.com/512/2972/2972585.png", width=10)
                else:
                    st.success("✅ RISIKO RENDAH: Pasien ini berisiko rendah mengalami stroke!")
                    st.image("https://cdn-icons-png.flaticon.com/512/2972/2972590.png", width=100)
            
            with result_col2:
                st.subheader("Tingkat Kepastian Prediksi")
                stroke_prob = prediction_proba[0][1]  # Probability of stroke
                no_stroke_prob = prediction_proba[0][0]   # Probability of no stroke
                
                st.metric(label="Probabilitas Stroke", value=f"{stroke_prob:.2%}")
                st.metric(label="Probabilitas Tidak Stroke", value=f"{no_stroke_prob:.2%}")
                
                # Visual representation of probability
                st.progress(stroke_prob)
                st.caption(f"Tingkat Risiko: {'Tinggi' if stroke_prob > 0.7 else 'Sedang' if stroke_prob > 0.3 else 'Rendah'}")
            
            # Detailed explanation
            st.subheader("Analisis Fitur")
            st.write("Berdasarkan profil pasien:")
            features_info = []
            features_info.append(f"- Usia: {age} tahun {'(usia tua, risiko lebih tinggi)' if age > 50 else '(usia muda, risiko lebih rendah)'}")
            features_info.append(f"- Kadar Glukosa: {avg_glucose_level} mg/dL {'(tinggi, risiko lebih tinggi)' if avg_glucose_level > 120 else '(normal, risiko lebih rendah)'}")
            features_info.append(f"- BMI: {bmi} {'(tinggi, risiko lebih tinggi)' if bmi > 30 else '(normal, risiko lebih rendah)'}")
            features_info.append(f"- Hipertensi: {'Ya' if int(hypertension) == 1 else 'Tidak'} {'(faktor risiko penting)' if hypertension == '1' else ''}")
            features_info.append(f"- Penyakit Jantung: {'Ya' if int(heart_disease) == 1 else 'Tidak'} {'(faktor risiko penting)' if heart_disease == '1' else ''}")
            
            for info in features_info:
                st.write(info)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
            st.info("Pastikan semua input valid dan coba lagi.")

    # Add some information about the model
    with st.expander("ℹ️ Tentang Model Ini"):
        st.markdown("""
        Model ini dilatih menggunakan beberapa algoritma machine learning untuk memprediksi risiko stroke:
        
        - **Algoritma yang Digunakan**: Regresi Logistik, Random Forest, K-Nearest Neighbors, dan SVM
        - **Fitur**: Informasi demografi, kesehatan, dan gaya hidup
        - **Metrik Evaluasi**: F1-Score, yang menyeimbangkan presisi dan recall
        
        Model mempertimbangkan berbagai faktor seperti usia, tekanan darah, penyakit jantung, dan faktor gaya hidup lainnya untuk memprediksi risiko stroke.
        """)
        
else:
    st.error("Tidak dapat memuat model. Pastikan 'model_terbaik.pkl' ada di direktori yang sama dengan aplikasi ini.")