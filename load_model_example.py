# Contoh script untuk memuat model dari file .pkl
import pickle

def load_stroke_model():
    """
    Fungsi untuk memuat model stroke yang telah disimpan
    """
    try:
        # Muat file model
        with open('best_stroke_model.pkl', 'rb') as file:
            model_data = pickle.load(file)
        
        print("Model berhasil dimuat!")
        print("Komponen yang dimuat:")
        print("- Model: {}".format(type(model_data['model']).__name__))
        print("- Scaler: {}".format(type(model_data['scaler']).__name__))
        print("- Label Encoders: {} buah".format(len(model_data['label_encoders'])))
        print("- Kolom Numerik: {}".format(model_data['numerical_columns']))
        print("- Kolom Kategorik: {}".format(model_data['categorical_columns']))
        
        return model_data
        
    except FileNotFoundError:
        print("File 'best_stroke_model.pkl' tidak ditemukan!")
        return None
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat model: {str(e)}")
        return None

# Contoh penggunaan
if __name__ == "__main__":
    model_components = load_stroke_model()
    
    if model_components:
        model = model_components['model']
        scaler = model_components['scaler']
        label_encoders = model_components['label_encoders']
        
        print("\nModel siap digunakan untuk prediksi!")