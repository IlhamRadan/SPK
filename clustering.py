import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from clustering_utils import train_kmeans, save_model, normalize_data, decode_clustered_data, load_encoders, label_decode_data
import os

# Nama file dan folder penyimpanan
folder_path = "Hasil Cluster"
file_name = "hasil_clustering_dataLatih.csv"
file_path = f"{folder_path}/{file_name}"

# Halaman Visualisasi Cluster
def clustering_menu():
    # Jika train_df tidak ada, tampilkan file CSV clustering
    if 'train_df' not in st.session_state:
        # Cek apakah file hasil clustering sudah ada
        if os.path.exists(file_path):
            st.warning("Untuk melakukan clustering, silahkan lakukan pengolahan data terlebih dahulu di menu Unggah Dataset")
            st.info("Hasil clustering data latih sebelumnya:")
            clustering_result = pd.read_csv(file_path)
            st.dataframe(clustering_result)
        else:
            st.warning("Silahkan lakukan pengolahan data terlebih dahulu di menu Unggah Dataset.")
        return
        
    # Tampilkan data latih
    train_data = st.session_state['train_df']
    st.write("Data Latih:")
    st.dataframe(train_data)

    # Periksa apakah data sudah dinormalisasi sebelumnya
    if 'normalized_data' not in st.session_state:
        st.write("Melakukan normalisasi data...")
        try:
            normalized_data, scaler = normalize_data(train_data)
            st.session_state['normalized_data'] = normalized_data
            st.session_state['scaler'] = scaler
        except Exception as e:
            st.error(f"Terjadi kesalahan saat normalisasi data: {e}")
            return

    # Setelah proses normalisasi selesai
    if 'scaler' in st.session_state:
        scaler = st.session_state['scaler']
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(train_data)
        joblib.dump(scaler, 'models/scaler.pkl')  # Simpan scaler ke file

    normalized_data = st.session_state['normalized_data']
    st.write("Data setelah normalisasi:")
    st.dataframe(pd.DataFrame(normalized_data, columns=train_data.columns))

    # Periksa apakah clustering sudah dilakukan
    if 'clustered_data' in st.session_state:
        st.success("Clustering telah selesai sebelumnya.")
        clustered_data = st.session_state['clustered_data']
        st.write("Hasil Clustering:")
        st.dataframe(clustered_data)

        # Visualisasi Scatter Plot
        st.write("Visualisasi Scatter Plot:")

        if normalized_data.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(normalized_data)
        else:
            reduced_data = normalized_data

        plt.figure(figsize=(8, 6))
        for cluster_id in range(clustered_data['Cluster'].nunique()):
            cluster_points = reduced_data[clustered_data['Cluster'] == cluster_id + 1]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id + 1}")

        plt.title("Visualisasi Cluster")
        plt.xlabel("Komponen 1")
        plt.ylabel("Komponen 2")
        plt.legend()
        st.pyplot(plt)

        encoders = load_encoders()
        decoded_df = label_decode_data(clustered_data, encoders)
        st.write("Dataset setelah Decoding:")
        st.dataframe(decoded_df)

    else:
        # Input jumlah cluster
        n_clusters = st.slider("Pilih jumlah cluster:", min_value=2, max_value=10, value=4)

        if st.button("Lakukan Clustering"):
            try:
                # Latih model K-Means
                kmeans = train_kmeans(normalized_data, n_clusters)

                # Prediksi cluster
                cluster_labels = kmeans.labels_
                clustered_data = train_data.copy()
                clustered_data['Cluster'] = cluster_labels + 1
                st.session_state['kmeans_model'] = kmeans
                st.session_state['clustered_data'] = clustered_data

                # Tampilkan hasil clustering dalam bentuk tabel
                st.write("Hasil Clustering:")
                st.dataframe(clustered_data)

                # Simpan model
                save_model(kmeans)
                st.success("Clustering selesai. Model K-Means telah disimpan.")

                # Visualisasi Scatter Plot
                st.write("Visualisasi Scatter Plot:")

                if normalized_data.shape[1] > 2:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(normalized_data)
                else:
                    reduced_data = normalized_data

                plt.figure(figsize=(8, 6))
                for cluster_id in range(n_clusters):
                    cluster_points = reduced_data[clustered_data['Cluster'] == cluster_id + 1]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id + 1}")

                plt.title("Visualisasi Cluster")
                plt.xlabel("Komponen 1")
                plt.ylabel("Komponen 2")
                plt.legend()
                st.pyplot(plt)

                encoders = load_encoders()
                decoded_df = label_decode_data(clustered_data, encoders)
                st.success("Decoding selesai.")
                st.write("Dataset setelah Decoding:")
                st.info("Hasil clustering data latih telah disimpan ke dalam folder sistem")
                st.dataframe(decoded_df, use_container_width=True)

                # Pastikan folder penyimpanan ada, jika tidak buat folder
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # Simpan dataset ke dalam file .csv
                decoded_df.to_csv(file_path, index=False)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat clustering: {e}")
          
            # Setelah data pelatihan selesai diproses
            if 'train_df' in st.session_state:
                data_columns = [
                    col for col in st.session_state['train_df'].columns 
                    if col not in ['Cluster']  # Hilangkan kolom Cluster
                ]
                joblib.dump(data_columns, 'models/data_columns.pkl')  # Simpan kolom data ke file