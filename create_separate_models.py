# Script untuk membuat file-file model terpisah dari hasil notebook
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Membuat dataset sintetis berdasarkan struktur dataset stroke
np.random.seed(42)
n_samples = 5000

# Generate fitur-fitur mirip dataset prediksi stroke
data = {
    'gender': np.random.choice(['Male', 'Female'], size=n_samples, p=[0.45, 0.55]),
    'age': np.random.beta(2, 5, size=n_samples) * 100,  # Lebih banyak nilai di rentang bawah, maks 100
    'hypertension': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
    'heart_disease': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
    'ever_married': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.6, 0.4]),
    'work_type': np.random.choice(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], size=n_samples, p=[0.5, 0.2, 0.2, 0.08, 0.02]),
    'Residence_type': np.random.choice(['Urban', 'Rural'], size=n_samples, p=[0.5, 0.5]),
    'avg_glucose_level': np.random.normal(100, 40, size=n_samples),
    'bmi': np.random.normal(25, 5, size=n_samples),
    'smoking_status': np.random.choice(['formerly smoked', 'never smoked', 'smokes', 'Unknown'], size=n_samples, p=[0.2, 0.5, 0.15, 0.15])
}

df = pd.DataFrame(data)

# Buat variabel target dengan korelasi ke fitur-fitur
# Probabilitas stroke lebih tinggi untuk orang tua, hipertensi, penyakit jantung, BMI tinggi, glukosa tinggi
stroke_prob = (
    0.01 + 
    0.3 * (df['age'] / 100) + 
    0.1 * df['hypertension'] + 
    0.15 * df['heart_disease'] + 
    0.1 * np.clip((df['bmi'] - 18.5) / 50, 0, 1) + # BMI > 18.5 meningkatkan risiko
    0.1 * np.clip((df['avg_glucose_level'] - 70) / 200, 0, 1)  # Glukosa lebih tinggi meningkatkan risiko
)

# Sesuaikan agar lebih realistis
df['stroke'] = np.random.binomial(1, np.clip(stroke_prob, 0, 0.3), size=n_samples)

# Pastikan beberapa kendala
df.loc[df['age'] < 18, 'work_type'] = 'children'
df.loc[df['age'] < 18, 'ever_married'] = 'No'

# Pastikan BMI dalam rentang wajar
df['bmi'] = np.clip(df['bmi'], 12, 60)

# Pastikan tingkat glukosa positif
df['avg_glucose_level'] = np.clip(df['avg_glucose_level'], 50, 300)

# Tangani nilai yang hilang
df['bmi'].fillna(df['bmi'].median(), inplace=True)
df = df[df['gender'] != 'Other']

# Pisahkan fitur dan target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Identifikasi kolom kategorik dan numerik
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Encode variabel kategorik
label_encoders = {}
X_encoded = X.copy()

for col in categorical_columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Skala fitur numerik
scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[numerical_columns] = scaler.fit_transform(X_encoded[numerical_columns])

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Inisialisasi model
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', random_state=42)
}

# Latih model dan temukan model terbaik
model_results = {}
for name, model in models.items():
    # Latih model
    model.fit(X_train, y_train)
    
    # Buat prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    model_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred
    }

# Temukan model terbaik berdasarkan F1-score
best_model_name = max(model_results, key=lambda x: model_results[x]['f1_score'])
best_model = model_results[best_model_name]['model']

print(f"Model terbaik: {best_model_name}")
print(f"F1-Score: {model_results[best_model_name]['f1_score']:.4f}")

# Simpan model terbaik ke file model terbaik.pkl
with open('model terbaik.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Simpan scaler ke file scaler.pkl
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Simpan label encoders ke file encoders.pkl
with open('encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

print("\nFile-file telah dibuat:")
print("- model terbaik.pkl (model terbaik)")
print("- scaler.pkl (StandardScaler)")
print("- encoders.pkl (LabelEncoders)")