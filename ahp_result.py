import streamlit as st
import numpy as np
import pandas as pd
import os

def ahp_finalResult():
    st.subheader("3. Hasil Akhir")
    
    # Pastikan data tersedia di session state
    if ("kriteria" not in st.session_state or 
        "bobot_kriteria" not in st.session_state or 
        len(st.session_state.get("kriteria", [])) == 0 or 
        len(st.session_state.get("bobot_kriteria", [])) == 0):
        st.error("Data kriteria dan alternatif tidak tersedia, lengkapi data terlebih dahulu.")
        return

    if ("alternatif" not in st.session_state or 
        "bobot_alternatif_per_kriteria" not in st.session_state or 
        len(st.session_state.get("alternatif", [])) == 0 or 
        len(st.session_state.get("bobot_alternatif_per_kriteria", {})) == 0):
        st.error("Data alternatif tidak tersedia, lengkapi data terlebih dahulu.")
        return

    kriteria = st.session_state["kriteria"]
    bobot_kriteria = np.array(st.session_state["bobot_kriteria"])
    alternatif = st.session_state["alternatif"]
    bobot_alternatif_per_kriteria = st.session_state["bobot_alternatif_per_kriteria"]

    # Pastikan semua data bobot alternatif tersedia
    if len(bobot_alternatif_per_kriteria) != len(kriteria):
        st.error("Data bobot alternatif belum lengkap. Harap kembali ke halaman sebelumnya untuk melengkapi data.")
        return

    # Konversi bobot alternatif ke array
    bobot_alternatif_array = np.array([bobot_alternatif_per_kriteria[k] for k in kriteria])

    # Hitung skor akhir
    skor_akhir = np.dot(bobot_alternatif_array.T, bobot_kriteria)

    # Buat DataFrame hasil
    hasil_df = pd.DataFrame({
        "Alternatif": alternatif, 
        "Skor Akhir": skor_akhir,
        "Ranking": range(1, len(alternatif) + 1)
    })
    hasil_df = hasil_df.sort_values(by="Skor Akhir", ascending=False).reset_index(drop=True)
    hasil_df["Ranking"] = range(1, len(hasil_df) + 1)

    # Tambahkan informasi bobot kriteria
    bobot_kriteria_df = pd.DataFrame({
        "Kriteria": kriteria,
        "Bobot Kriteria": bobot_kriteria
    })

    # Tambahkan informasi bobot alternatif per kriteria
    bobot_alternatif_detail_df = pd.DataFrame()
    for k in kriteria:
        temp_df = pd.DataFrame({
            "Kriteria": [k] * len(alternatif),
            "Alternatif": alternatif,
            "Bobot Alternatif": bobot_alternatif_per_kriteria[k]
        })
        bobot_alternatif_detail_df = pd.concat([bobot_alternatif_detail_df, temp_df], ignore_index=True)

    # Buat direktori untuk menyimpan hasil
    output_dir = "ahp_results"
    os.makedirs(output_dir, exist_ok=True)

    # Simpan hasil ke CSV
    current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Simpan file-file CSV
    hasil_ranking_path = os.path.join(output_dir, f"hasil_ranking_{current_time}.csv")
    bobot_kriteria_path = os.path.join(output_dir, f"bobot_kriteria_{current_time}.csv")
    bobot_alternatif_path = os.path.join(output_dir, f"bobot_alternatif_{current_time}.csv")

    hasil_df.to_csv(hasil_ranking_path, index=False)
    bobot_kriteria_df.to_csv(bobot_kriteria_path, index=False)
    bobot_alternatif_detail_df.to_csv(bobot_alternatif_path, index=False)

    # Tampilkan hasil
    if not st.session_state.get("edit_mode", False):
        st.write("Peringkat Alternatif:")
        st.dataframe(hasil_df, use_container_width=True)

        st.write("Prioritas Alternatif berdasarkan AHP adalah:")
        st.success(hasil_df.iloc[0]["Alternatif"])

        # Tambahkan tombol download
        col1, col2, col3 = st.columns(3)
        with col1:
            with open(hasil_ranking_path, "rb") as file:
                st.download_button(
                    label="Download Hasil Ranking",
                    data=file,
                    file_name=f"hasil_ranking_{current_time}.csv",
                    mime="text/csv"
                )
        with col2:
            with open(bobot_kriteria_path, "rb") as file:
                st.download_button(
                    label="Download Bobot Kriteria",
                    data=file,
                    file_name=f"bobot_kriteria_{current_time}.csv",
                    mime="text/csv"
                )
        with col3:
            with open(bobot_alternatif_path, "rb") as file:
                st.download_button(
                    label="Download Bobot Alternatif",
                    data=file,
                    file_name=f"bobot_alternatif_{current_time}.csv",
                    mime="text/csv"
                )

    else:
        st.error("Silakan kembali ke halaman sebelumnya untuk memperbarui data kriteria atau alternatif.")