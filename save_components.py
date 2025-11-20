# Script untuk menyimpan komponen model secara terpisah jika diperlukan
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Ini adalah contoh script untuk menyimpan komponen secara terpisah
# Dalam praktiknya, kita akan melatih model dulu, lalu menyimpannya

# Contoh: menyimpan model terbaik, scaler, dan label_encoders secara terpisah
# Dalam konteks proyek ini, kita sebenarnya sudah menyimpan semuanya dalam satu file
# Tapi jika ingin menyimpan terpisah:

# Misalnya kita sudah punya model, scaler, dan label_encoders dari proses pelatihan
# best_model = model_terbaik_yang_sudah_dilatih
# scaler = scaler_yang_sudah_difit
# label_encoders = dictionary_dari_label_encoders

# Contoh menyimpan masing-masing komponen:
# with open('best_model.pkl', 'wb') as f:
#     pickle.dump(best_model, f)
# 
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)
# 
# with open('label_encoders.pkl', 'wb') as f:
#     pickle.dump(label_encoders, f)

# Tapi untuk proyek ini, kita menggunakan pendekatan menyimpan semuanya dalam satu file
# seperti yang sudah dilakukan di notebook, karena lebih praktis untuk deployment