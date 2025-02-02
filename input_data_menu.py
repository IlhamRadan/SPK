import streamlit as st
import pandas as pd
import joblib
from clustering_utils import load_model, normalize_data, load_encoders, label_decode_data
from label_utils import save_encoders
import os
import numpy as np

# Fungsi untuk memuat entri baru dari file CSV
def load_new_entries(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

# Fungsi untuk menyimpan entri baru ke file CSV
def save_new_entry(file_path, new_entry):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=new_entry.keys())
    
    new_entry_df = pd.DataFrame([new_entry])
    df = pd.concat([df, new_entry_df], ignore_index=True)
    df.to_csv(file_path, index=False)

# Fungsi untuk memuat hasil prediksi sebelumnya
def load_previous_predictions(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

# Halaman Input Data Baru
def input_data():
    try:
        kmeans_model = load_model('models/kmeans_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = load_encoders()
        data_columns = load_model('models/data_columns.pkl')
    except Exception as e:
        st.error(f"Gagal memuat model atau file pendukung. Silahkan lakukan clustering & model training terlebih dahulu!")
        return

    if 'input_data' not in st.session_state:
        st.session_state['input_data'] = {col: f"Pilih {col}..." for col in data_columns}
    if 'show_new_entry_form' not in st.session_state:
        st.session_state['show_new_entry_form'] = False

    new_entries_file = 'new_entries.csv'
    prediction_file = 'clustering_results/hasil_prediksi_cluster.csv'
    new_entries_df = load_new_entries(new_entries_file)
    previous_predictions = load_previous_predictions(prediction_file)

    inputs_complete = True

    st.write("Klik untuk menambah entri baru")    
    if st.button("Tambah Entri Baru"):
        st.session_state['show_new_entry_form'] = True

    if st.session_state['show_new_entry_form']:
        st.subheader("Tambah Entri Baru:")
        new_entry = {}
        for col in data_columns:
            new_value = st.text_input(f"Masukkan {col}:")
            if new_value:
                new_entry[col] = new_value

        if st.button("Simpan Entri Baru"):
            if new_entry:
                save_new_entry(new_entries_file, new_entry)
                st.success("Entri baru berhasil disimpan!")
                st.session_state['show_new_entry_form'] = False
            else:
                st.error("Harap masukkan nilai untuk entri baru.") 

    st.write("Pilih data surat untuk diprediksi clusternya:")
    for col in data_columns:
        unique_values = list(encoders[col].classes_)
        if col in new_entries_df.columns:
            unique_values += new_entries_df[col].dropna().unique().tolist()
        
        placeholder = f"Pilih {col}..."
        full_options = [placeholder] + list(set(unique_values))
        
        current_index = 0
        if st.session_state['input_data'][col] in full_options:
            current_index = full_options.index(st.session_state['input_data'][col])

        selected_value = st.selectbox(
            f"Pilih {col}:",
            options=full_options,
            index=current_index
        )

        if selected_value == placeholder:
            inputs_complete = False
        
        st.session_state['input_data'][col] = selected_value

    if st.button("Prediksi Cluster"):
        if not inputs_complete:
            st.error("Harap lengkapi semua input sebelum melakukan prediksi!")
            return

        try:
            clean_input_data = {
                col: val for col, val in st.session_state['input_data'].items() 
                if not val.startswith("Pilih ")
            }

            input_df = pd.DataFrame([clean_input_data])

            for col, encoder in encoders.items():
                if col in input_df:
                    try:    
                        input_df[col] = encoder.transform(input_df[col])
                    except ValueError:
                        new_classes = list(encoder.classes_) + [input_df[col].values[0]]
                        encoder.classes_ = np.array(new_classes)
                        encoders[col] = encoder
                        save_encoders(encoders)
                        input_df[col] = encoder.transform(input_df[col])

            normalized_input = scaler.transform(input_df)
            cluster_label = kmeans_model.predict(normalized_input)[0] + 1

            input_df['Cluster'] = cluster_label
            decoded_df = label_decode_data(input_df, encoders)
            previous_predictions = pd.concat(
                [previous_predictions, decoded_df], ignore_index=True
            )

            st.session_state['input_data'] = {col: f"Pilih {col}..." for col in data_columns}
            st.success(f"Data baru termasuk ke dalam Cluster {cluster_label}.")

            # Simpan hasil prediksi secara otomatis
            os.makedirs("clustering_results", exist_ok=True)
            previous_predictions.to_csv(prediction_file, index=False)
            """st.success(f"Hasil prediksi berhasil disimpan ke {prediction_file}")"""

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

    st.subheader("Hasil Prediksi Cluster:")
    """st.warning("Jangan lupa untuk selalu mengunduh file hasil prediksi!")"""
    previous_predictions.index = range(1, len(previous_predictions) + 1)
    st.dataframe(previous_predictions, use_container_width=True)

    if not previous_predictions.empty:
        with open(prediction_file, "rb") as file:
            st.download_button(
                label="📥 Download Hasil Prediksi",
                data=file,
                file_name="hasil_prediksi_cluster.csv",
                mime="text/csv",
                key="download_prediksi_btn"
            )
        st.write(f"Total data prediksi: {len(previous_predictions)}")
        st.write("Distribusi Cluster:")
        st.dataframe(previous_predictions['Cluster'].value_counts())
