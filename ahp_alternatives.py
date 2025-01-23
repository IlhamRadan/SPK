import streamlit as st
import numpy as np
import pandas as pd

def alternatives_form():
    st.subheader("2. Input Alternatif dan Bobotnya Untuk Setiap Kriteria")
    # Pastikan data kriteria tersedia di session state
    if ("kriteria" not in st.session_state or 
        "bobot_kriteria" not in st.session_state or 
        len(st.session_state.get("kriteria", [])) == 0 or 
        len(st.session_state.get("bobot_kriteria", [])) == 0):
        st.error("Data kriteria tidak tersedia. Harap kembali ke halaman sebelumnya untuk mengisi data kriteria.")
        return

    kriteria = st.session_state["kriteria"]

    # Initialize session state untuk alternatif
    if "alternatif" not in st.session_state:
        st.session_state["alternatif"] = []
    if "matriks_alternatif" not in st.session_state:
        st.session_state["matriks_alternatif"] = {}
    if "bobot_alternatif_per_kriteria" not in st.session_state:
        st.session_state["bobot_alternatif_per_kriteria"] = {}
    if "normalisasi_alternatif" not in st.session_state:
        st.session_state["normalisasi_alternatif"] = {}
    if "eigen_values_alternatif" not in st.session_state:
        st.session_state["eigen_values_alternatif"] = {}
    if "total_per_kolom_alternatif" not in st.session_state:
        st.session_state["total_per_kolom_alternatif"] = {}
    if "edit_mode" not in st.session_state:
        st.session_state["edit_mode"] = True
    
    # Tombol untuk toggle mode edit
    if not st.session_state["edit_mode"]:
        if st.button("Ubah Alternatif dan Bobot"):
            st.session_state["edit_mode"] = True

    if st.session_state["edit_mode"]:

        # Input daftar alternatif
        alternatif_input = st.text_area(
            "Masukkan daftar alternatif (pisahkan dengan koma):",
            value=", ".join(st.session_state["alternatif"]) if st.session_state["alternatif"] else "Cluster 1, Cluster 2, Cluster 3, Cluster 4"
        )

        alternatif = [a.strip() for a in alternatif_input.split(",") if a.strip()]
        st.session_state["alternatif"] = alternatif

        if not alternatif:
            st.warning("Masukkan setidaknya satu alternatif.")
            return
#########################################
        # Input nilai/bobot alternatif untuk setiap kriteria
        for idx_k, k in enumerate(kriteria):
            st.subheader(f"Perbandingan Alternatif untuk Kriteria: {k}")

            # Tambahkan inisialisasi untuk menyimpan CI dan CR per kriteria
            if f"ci_{k}" not in st.session_state:
                st.session_state[f"ci_{k}"] = None
            if f"cr_{k}" not in st.session_state:
                st.session_state[f"cr_{k}"] = None

            # Update matriks alternatif sesuai jumlah alternatif
            alternatif_len = len(alternatif)
            if k not in st.session_state["matriks_alternatif"]:
                st.session_state["matriks_alternatif"][k] = np.ones((alternatif_len, alternatif_len))
            else:
                # Perbarui ukuran matriks jika jumlah alternatif berubah
                matriks_alternatif = st.session_state["matriks_alternatif"][k]
                if matriks_alternatif.shape[0] != alternatif_len:
                    st.session_state["matriks_alternatif"][k] = np.ones((alternatif_len, alternatif_len))

            matriks_alternatif = st.session_state["matriks_alternatif"][k]

            # Set diagonal ke 1 untuk konsistensi
            for i in range(alternatif_len):
                matriks_alternatif[i, i] = 1.0

            for i in range(alternatif_len):
                for j in range(i + 1, alternatif_len):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        pilihan = st.radio(
                            f"Mana yang lebih penting? ({alternatif[i]} atau {alternatif[j]})",
                            options=[alternatif[i], alternatif[j]],
                            key=f"radio_{k}_{i}_{j}"
                        )

                    with col2:
                        nilai = st.number_input(
                            f"Nilai perbandingan ({alternatif[i]} dan {alternatif[j]}):",
                            min_value=0.1, max_value=9.0, step=0.1, value=1.0,
                            key=f"nilai_{k}_{i}_{j}"
                        )

                    # Perbarui nilai matriks
                    if pilihan == alternatif[i]:
                        matriks_alternatif[i, j] = nilai
                        matriks_alternatif[j, i] = 1 / nilai
                    else:
                        matriks_alternatif[i, j] = 1 / nilai
                        matriks_alternatif[j, i] = nilai
            
            st.divider()
            st.session_state["matriks_alternatif"][k] = matriks_alternatif
##################################################

            # Hitung bobot alternatif
            total_per_kolom_alt = matriks_alternatif.sum(axis=0)
            normalisasi_alternatif = matriks_alternatif / total_per_kolom_alt
            bobot_alternatif = normalisasi_alternatif.mean(axis=1)

            # Hitung eigen values
            eigen_values = normalisasi_alternatif.mean(axis=1) * total_per_kolom_alt

            # Perhitungan Consistency Index (CI) dan Consistency Ratio (CR)
            lambda_max = (matriks_alternatif @ bobot_alternatif).sum() / bobot_alternatif.sum()
            ci = (lambda_max - len(alternatif)) / (len(alternatif) - 1)
            ri_values = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
            ri = ri_values.get(len(alternatif), 1.45)  # Default to max RI for sizes > 9
            cr = ci / ri if ri > 0 else 0.0

            # Simpan hasil ke session state
            st.session_state["bobot_alternatif_per_kriteria"][k] = bobot_alternatif
            st.session_state["normalisasi_alternatif"][k] = normalisasi_alternatif
            st.session_state["total_per_kolom_alternatif"][k] = total_per_kolom_alt
            st.session_state["eigen_values_alternatif"][k] = eigen_values
            st.session_state[f"ci_{k}"] = ci
            st.session_state[f"cr_{k}"] = cr

        # Tombol untuk menyimpan data dan keluar dari mode edit
        if st.button("Proses"):
            st.session_state["edit_mode"] = False

    # Menampilkan hasil jika tidak dalam mode edit
    if not st.session_state["edit_mode"]:
        st.subheader("Hasil Akhir")

        for k in kriteria:
            st.subheader(f"Matriks Perbandingan Alternatif untuk Kriteria: {k}")
            matriks_alt_df = pd.DataFrame(
                st.session_state["matriks_alternatif"][k],
                index=st.session_state["alternatif"],
                columns=st.session_state["alternatif"]
            )
            
            # Tambahkan baris total untuk matriks perbandingan
            total_kolom = matriks_alt_df.sum()
            total_kolom['Alternatif'] = 'Total'
            matriks_alt_df.loc['Total'] = total_kolom
            
            st.dataframe(matriks_alt_df)

            # Matriks Normalisasi
            normalisasi_df = pd.DataFrame(
                st.session_state["normalisasi_alternatif"][k],
                index=st.session_state["alternatif"],
                columns=st.session_state["alternatif"]
            )
            
            # Tambahkan kolom jumlah baris
            normalisasi_df['Jumlah Baris'] = normalisasi_df.sum(axis=1)
            
            # Tambahkan kolom bobot
            normalisasi_df['Bobot'] = st.session_state["bobot_alternatif_per_kriteria"][k]
            
            # Tambahkan kolom eigen values
            normalisasi_df['Eigen Value'] = st.session_state["eigen_values_alternatif"][k]

            # Tambahkan baris total untuk jumlah kolom
            total_kolom_normalisasi = normalisasi_df.sum()
            total_kolom_normalisasi['Alternatif'] = 'Total'
            normalisasi_df.loc['Total'] = total_kolom_normalisasi

            st.write(f"Matriks Normalisasi dan Bobot Alternatif untuk Kriteria: {k}")
            st.dataframe(normalisasi_df)

            st.write(f"Rekomendasi alternatif dan bobotnya untuk Kriteria: {k}")
            bobot_df = pd.DataFrame({
                "Alternatif": st.session_state["alternatif"],
                "Bobot": st.session_state["bobot_alternatif_per_kriteria"][k]
            })
            bobot_df = bobot_df.sort_values("Bobot", ascending=False).reset_index(drop=True)
            bobot_df.index = range(1, len(bobot_df) + 1)
            st.dataframe(bobot_df)

            # Tampilkan hasil CI dan CR
            if st.session_state[f"ci_{k}"] is not None and st.session_state[f"cr_{k}"] is not None:
                st.write(f"Consistency Index (CI): {st.session_state[f'ci_{k}']:.4f}")
                st.write(f"Consistency Ratio (CR): {st.session_state[f'cr_{k}']:.4f}")

                if st.session_state[f"cr_{k}"] <= 0.1:
                    st.success("CR <= 0.1, matriks konsisten!")
                else:
                    st.warning("CR > 0.1, matriks tidak konsisten!")
                
                st.write(f"Alternatif terbaik untuk kriteria {k}:")
                st.success(bobot_df.iloc[0]["Alternatif"])
            st.divider()
