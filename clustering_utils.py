from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# Fungsi untuk melatih dan menyimpan model K-Means
def train_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(data)
    return kmeans

# Fungsi untuk menyimpan model
def save_model(model, file_path="models/kmeans_model.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

# Fungsi untuk normalisasi data
def normalize_data(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

# Fungsi untuk melakukan decoding pada hasil clustering
def decode_clustered_data(encoded_data, encoders):
    decoded_data = encoded_data.copy()
    for column, encoder in encoders.items():
        decoded_data[column] = encoder.inverse_transform(decoded_data[column])
    return decoded_data

#Fungsi untuk memuat encoder
def load_encoders(file_path="models/label_encoders.pkl"):
    try:
        with open(file_path, "rb") as f:
            encoders = pickle.load(f)
        return encoders
    except FileNotFoundError:
        return None
    
#Fungsi untuk decoding data
def label_decode_data(encoded_df, encoders):
    decoded_df = encoded_df.copy()
    for column, le in encoders.items():
        decoded_df[column] = le.inverse_transform(encoded_df[column])
    return decoded_df

#Fungsi untuk meload model
import pickle

def load_model(file_path="kmeans_model.pkl"):
    """
    Memuat model K-Means dari file pickle.

    Parameters:
    - file_path (str): Path ke file model K-Means yang telah disimpan.

    Returns:
    - model: Objek model K-Means yang telah dimuat.
    """
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"File model {file_path} tidak ditemukan.")
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model dari {file_path}: {e}")
