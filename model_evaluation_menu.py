import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score
from clustering_utils import normalize_data, load_encoders, label_decode_data

folder_path = "Hasil Cluster"
file_name = "hasil_clustering_dataUji.csv"
file_path = f"{folder_path}/{file_name}"

# Halaman Evaluasi Model
def model_evaluation():
    # Jika test_df tidak ada, tampilkan file CSV clustering data uji
    if 'test_df' not in st.session_state:
        # Cek apakah file hasil clustering data uji sudah ada
        if os.path.exists(file_path):
            st.warning("Untuk melakukan evaluasi model, silahkan lakukan clustering untuk membuat model K-Means nya.")
            st.info("Hasil clustering data uji sebelumnya:")
            clustering_result = pd.read_csv(file_path)
            st.dataframe(clustering_result)
        else:
            st.warning("Silahkan lakukan clustering untuk membuat model K-Means.")
        return

    # Periksa keberadaan file model K-Means
    model_path = 'models/kmeans_model.pkl'
    if not os.path.exists(model_path):
        st.warning("Model K-Means tidak ditemukan. Silahkan lakukan clustering terlebih dahulu!")
        return

    # Inisialisasi state evaluasi jika belum ada
    if 'evaluation_done' not in st.session_state:
        st.session_state['evaluation_done'] = False
    if 'evaluation_results' not in st.session_state:
        st.session_state['evaluation_results'] = {}

    # Fungsi untuk menampilkan hasil evaluasi
    def display_evaluation_results(results):
        st.write("### Hasil Evaluasi")
        st.write(f"- **Silhouette Score**: {results['silhouette_avg']:.3f}")
        st.write(f"- **Davies-Bouldin Index (DBI)**: {results['dbi_score']:.3f}")
        st.write("Dataset setelah Decoding:")
        st.info("Hasil clustering data uji telah disimpan ke dalam folder sistem")
        st.dataframe(results['decoded_test_df'], use_container_width=True)

    # Periksa apakah evaluasi sudah dilakukan
    if st.session_state['evaluation_done']:
        st.info("Evaluasi model sudah dilakukan.")
        # Tampilkan hasil evaluasi yang disimpan
        display_evaluation_results(st.session_state['evaluation_results'])
    else:
        # Tombol evaluasi model
        if st.button("Evaluasi Model"):
            try:
                # Muat model K-Means dari file
                kmeans_model = joblib.load(model_path)
                test_df = st.session_state['test_df']

                st.write("Data Uji:")
                st.dataframe(test_df)

                # Lakukan normalisasi data uji jika belum
                st.write("Normalisasi Data Uji...")
                normalized_test_data, _ = normalize_data(test_df)

                # Prediksi cluster untuk data uji
                st.write("Melakukan prediksi cluster...")
                cluster_labels = kmeans_model.predict(normalized_test_data)
                test_df['Predicted Cluster'] = cluster_labels + 1

                # Hitung metrik evaluasi
                st.write("Menghitung metrik evaluasi model...")
                silhouette_avg = silhouette_score(normalized_test_data, cluster_labels)
                dbi_score = davies_bouldin_score(normalized_test_data, cluster_labels)

                # Simpan hasil evaluasi di session state
                encoders = load_encoders()
                decoded_test_df = label_decode_data(test_df, encoders)

                st.session_state['evaluation_results'] = {
                    'silhouette_avg': silhouette_avg,
                    'dbi_score': dbi_score,
                    'decoded_test_df': decoded_test_df
                }

                # Tandai bahwa evaluasi telah selesai
                st.session_state['evaluation_done'] = True

                # Tampilkan hasil evaluasi
                display_evaluation_results(st.session_state['evaluation_results'])

                # Cek folder
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                # Simpan dataset ke dalam file .csv
                decoded_test_df.to_csv(file_path, index=False)

            except Exception as e:
                st.error(f"Terjadi kesalahan selama evaluasi: {e}")
        else:
            st.info("Tekan tombol 'Evaluasi Model' untuk memulai proses evaluasi.")

# Eksekusi menu evaluasi
if __name__ == "__main__":
    model_evaluation()