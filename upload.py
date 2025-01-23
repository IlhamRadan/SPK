import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from label_utils import label_encode_data, save_encoders
from preprocessing_utils import preprocess_text
import nltk
from nltk.corpus import stopwords
import re
import os

# Unduh stopwords jika belum tersedia
def load_stopwords(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            stop_words = set(file.read().splitlines())
        return stop_words
    except Exception as e:
        st.error(f"Gagal memuat stopwords: {e}")
        return set()

# Coba muat stopwords dari file lokal
stopwords_path = 'stopwords/indonesian'  # Sesuaikan path
if os.path.exists(stopwords_path):
    stop_words = load_stopwords(stopwords_path)
else:
    # Fallback jika file tidak ditemukan
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('indonesian'))

# Fungsi utama untuk menu "Unggah Dataset"
def upload_menu():
    st.title("Unggah Dataset")

    # Periksa apakah dataset sudah diunggah sebelumnya
    if 'uploaded_df' in st.session_state:
        st.write("Dataset yang telah diunggah sebelumnya:")
        st.dataframe(st.session_state['uploaded_df'].set_index(st.session_state['uploaded_df'].columns[0]))
    else:
        uploaded_file = st.file_uploader("Unggah file dataset dalam format CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_df'] = df
            st.session_state['preprocessed'] = False
            st.write("Dataset yang diunggah:")
            st.dataframe(df.set_index(df.columns[0]))

    # Data Selection (Menghapus Kolom) - Opsional, hanya jika belum preprocessing
    if 'uploaded_df' in st.session_state and not st.session_state.get('preprocessed', False):
        st.write("### Data Selection")
        columns = list(st.session_state['uploaded_df'].columns)
        selected_columns = st.multiselect("Pilih kolom untuk dihapus (opsional):", columns)

        if st.button("Hapus Kolom (opsional)"):
            try:
                st.session_state['uploaded_df'] = st.session_state['uploaded_df'].drop(columns=selected_columns)
                st.success("Kolom berhasil dihapus.")
                st.write("Dataset setelah penghapusan kolom:")
                st.dataframe(st.session_state['uploaded_df'].set_index(st.session_state['uploaded_df'].columns[0]))
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghapus kolom: {e}")

    # Preprocessing Data
    if 'uploaded_df' in st.session_state:
        st.write("### Preprocessing Data")
        if st.session_state.get('preprocessed', False):
            st.success("Preprocessing sudah dilakukan.")
        else:
            if st.button("Lakukan Preprocessing"):
                try:
                    # Preprocessing semua kolom teks secara otomatis
                    df = st.session_state['uploaded_df']
                    text_columns = df.select_dtypes(include=['object']).columns
                    for col in text_columns:
                        df[col] = df[col].apply(preprocess_text)
                    st.session_state['uploaded_df'] = df
                    st.session_state['preprocessed'] = True
                    st.success("Preprocessing selesai.")
                    st.write("Dataset setelah Preprocessing:")
                    st.dataframe(df.set_index(df.columns[0]))
                except Exception as e:
                    st.error(f"Terjadi kesalahan selama Preprocessing: {e}")

    # Periksa apakah Label Encoding sudah dilakukan
    if 'encoded_df' in st.session_state:
        st.write("Dataset setelah Label Encoding:")
        st.dataframe(st.session_state['encoded_df'].set_index(st.session_state['encoded_df'].columns[0]))
        st.success("Label Encoding sudah dilakukan.")
    else:
        if st.button("Lakukan Label Encoding"):
            if 'uploaded_df' not in st.session_state:
                st.error("Silakan unggah dataset terlebih dahulu sebelum melakukan Label Encoding.")
            else:
                try:
                    encoded_df, encoders = label_encode_data(st.session_state['uploaded_df'])
                    st.session_state['encoded_df'] = encoded_df
                    save_encoders(encoders)
                    st.success("Label Encoding selesai. Model encoder disimpan sebagai `label_encoders.pkl`.")
                    st.write("Dataset setelah Label Encoding:")
                    st.dataframe(encoded_df.set_index(encoded_df.columns[0]))
                except Exception as e:
                    st.error(f"Terjadi kesalahan selama Label Encoding: {e}")

    # Periksa apakah Split Data sudah dilakukan
    if 'train_df' in st.session_state and 'test_df' in st.session_state:
        st.success("Data telah di-split sebelumnya.")
        st.write(f"Jumlah data latih: {len(st.session_state['train_df'])}")
        st.write(f"Jumlah data uji: {len(st.session_state['test_df'])}")
        st.write("Data Latih:")
        st.dataframe(st.session_state['train_df'].set_index(st.session_state['train_df'].columns[0]))
        st.write("Data Uji:")
        st.dataframe(st.session_state['test_df'].set_index(st.session_state['test_df'].columns[0]))
    else:
        if st.button("Lakukan Split Data"):
            if 'encoded_df' in st.session_state:
                try:
                    train_df, test_df = train_test_split(
                        st.session_state['encoded_df'], test_size=0.2, random_state=42
                    )
                    st.session_state['train_df'] = train_df
                    st.session_state['test_df'] = test_df
                    st.success("Data berhasil di-split.")
                    st.write(f"Jumlah data latih: {len(train_df)}")
                    st.write(f"Jumlah data uji: {len(test_df)}")
                    st.write("Data Latih:")
                    st.dataframe(train_df.set_index(train_df.columns[0]))
                    st.write("Data Uji:")
                    st.dataframe(test_df.set_index(test_df.columns[0]))
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
            else:
                st.error("Silakan lakukan Label Encoding terlebih dahulu sebelum Split Data.")

# Eksekusi menu upload
if __name__ == "__main__":
    upload_menu()
