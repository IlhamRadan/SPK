import streamlit as st
import pandas as pd
import joblib
from clustering_utils import load_model, normalize_data, load_encoders, label_decode_data
import os

# Halaman Input Data Baru
def input_data():
    # Muat model dan scaler dari file
    try:
        kmeans_model = load_model('models/kmeans_model.pkl')  # Muat model K-Means
        scaler = joblib.load('models/scaler.pkl')             # Muat scaler
        encoders = load_encoders()                    # Muat encoder
    except Exception as e:
        st.error(f"Gagal memuat model atau file pendukung. Silahkan lakukan clustering & model training terlebih dahulu!")
        return

    # Dapatkan kolom yang akan digunakan untuk input
    try:
        data_columns = load_model('models/data_columns.pkl')  # Muat daftar kolom input
    except Exception as e:
        st.error(f"Gagal memuat daftar kolom input: {e}")
        return

    # Inisialisasi session state untuk form input dan hasil prediksi
    if 'input_data' not in st.session_state:
        st.session_state['input_data'] = {col: f"Pilih {col}..." for col in data_columns}
    if 'predicted_data' not in st.session_state:
        st.session_state['predicted_data'] = pd.DataFrame(columns=data_columns + ['Cluster'])

    # Input data baru menggunakan selection box
    st.write("Pilih data surat untuk diprediksi clusternya:")
    
    # Variabel untuk melacak kelengkapan input
    inputs_complete = True
    
    for col in data_columns:
        # Dapatkan unique values dari encoder
        unique_values = list(encoders[col].classes_)
        
        # Tambahkan placeholder
        placeholder = f"Pilih {col}..."
        full_options = [placeholder] + unique_values

        # Tentukan index awal
        current_index = 0
        if st.session_state['input_data'][col] in full_options:
            current_index = full_options.index(st.session_state['input_data'][col])

        # Gunakan selectbox dengan placeholder
        selected_value = st.selectbox(
            f"Pilih {col}:",
            options=full_options,
            index=current_index
        )

        # Perbarui session state dan periksa kelengkapan
        if selected_value == placeholder:
            inputs_complete = False
        
        st.session_state['input_data'][col] = selected_value

    # Tombol Prediksi dengan validasi input
    if st.button("Prediksi Cluster"):
        # Cek apakah semua input sudah terisi
        if not inputs_complete:
            st.error("Harap lengkapi semua input sebelum melakukan prediksi!")
            return

        try:
            # Buat salinan input data tanpa placeholder
            clean_input_data = {
                col: val for col, val in st.session_state['input_data'].items() 
                if not val.startswith("Pilih ")
            }

            # Convert input ke DataFrame
            input_df = pd.DataFrame([clean_input_data])

            # Lakukan Label Encoding menggunakan encoder yang ada
            for col, encoder in encoders.items():
                if col in input_df:
                    # Encode nilai input
                    input_df[col] = encoder.transform(input_df[col])

            # Normalisasi data
            try:
                normalized_input = scaler.transform(input_df)
            except Exception as e:
                st.error(f"Error saat normalisasi: {e}")
                return

            # Prediksi cluster
            try:
                cluster_label = kmeans_model.predict(normalized_input)[0] + 1
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")
                return

            # Tambahkan cluster ke DataFrame
            input_df['Cluster'] = cluster_label

            # Decode data untuk ditampilkan
            decoded_df = label_decode_data(input_df, encoders)

            # Tambahkan hasil ke session state
            st.session_state['predicted_data'] = pd.concat(
                [st.session_state['predicted_data'], decoded_df], ignore_index=True
            )

            # Reset form input ke placeholder
            st.session_state['input_data'] = {col: f"Pilih {col}..." for col in data_columns}

            # Tampilkan hasil prediksi
            st.success(f"Data baru termasuk ke dalam Cluster {cluster_label}.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

    # Tampilkan data hasil prediksi
    st.subheader("Hasil Prediksi Cluster:")
    st.warning("Jangan lupa untuk selalu mengunduh file hasil prediksi!")
    predicted_data = st.session_state['predicted_data'].copy()
    predicted_data.index = range(1, len(predicted_data) + 1)

    st.dataframe(predicted_data, use_container_width=True)

    # Tambahkan tombol download
    if not predicted_data.empty:
        # Buat direktori untuk menyimpan hasil
        output_dir = "clustering_results"
        os.makedirs(output_dir, exist_ok=True)

        # Generate nama file unik dengan timestamp
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"hasil_prediksi_cluster_{current_time}.csv")

        # Simpan DataFrame ke CSV
        predicted_data.to_csv(file_path, index=False)

        # Tambahkan tombol download
        with open(file_path, "rb") as file:
            st.download_button(
                label="📥 Download Hasil Prediksi",
                data=file,
                file_name=f"hasil_prediksi_cluster_{current_time}.csv",
                mime="text/csv",
                key="download_prediksi_btn"
            )

        # Tambahan: Informasi jumlah data dan cluster
        st.write(f"Total data prediksi: {len(predicted_data)}")
        cluster_counts = predicted_data['Cluster'].value_counts()
        st.write("Distribusi Cluster:")
        st.dataframe(cluster_counts)