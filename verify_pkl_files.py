# Script untuk memverifikasi file-file .pkl yang telah dibuat
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("Memverifikasi file-file .pkl...")

# Memuat model terbaik
try:
    with open('model terbaik.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✓ model terbaik.pkl berhasil dimuat")
    print(f"  Tipe model: {type(model).__name__}")
except Exception as e:
    print(f"✗ Gagal memuat model terbaik.pkl: {e}")

# Memuat scaler
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    print("✓ scaler.pkl berhasil dimuat")
    print(f"  Tipe scaler: {type(scaler).__name__}")
except Exception as e:
    print(f"✗ Gagal memuat scaler.pkl: {e}")

# Memuat encoders
try:
    with open('encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    print("✓ encoders.pkl berhasil dimuat")
    print(f"  Jumlah encoder: {len(encoders)}")
    print(f"  Tipe encoder: {type(list(encoders.values())[0]).__name__ if encoders else 'N/A'}")
except Exception as e:
    print(f"✗ Gagal memuat encoders.pkl: {e}")

print("\nVerifikasi selesai!")