import streamlit as st
import numpy as np
import pandas as pd

def criteria_form():
    st.title("AHP - Analytical Hierarchy Process")

    # Initialize session state
    if "matriks_kriteria" not in st.session_state:
        st.session_state["matriks_kriteria"] = None
    if "bobot_kriteria" not in st.session_state:
        st.session_state["bobot_kriteria"] = None
    if "kriteria" not in st.session_state:
        st.session_state["kriteria"] = []
    if "show_form" not in st.session_state:
        st.session_state["show_form"] = True
    if "CI" not in st.session_state:
        st.session_state["CI"] = None
    if "CR" not in st.session_state:
        st.session_state["CR"] = None
    if "normalisasi_kriteria" not in st.session_state:
        st.session_state["normalisasi_kriteria"] = None

    st.subheader("1. Input Kriteria dan Bobotnya")

    # Button to toggle form visibility
    if not st.session_state["show_form"]:
        if st.button("Ubah Kriteria dan Bobot"):
            st.session_state["show_form"] = True

    # Form input and processing logic
    if st.session_state["show_form"]:
        kriteria = st.text_area(
            "Masukkan daftar kriteria (pisahkan dengan koma):",
            value=", ".join(st.session_state["kriteria"]) if st.session_state["kriteria"] else "Urgensi surat, Jenis Kegiatan, Jenis Program"
        )
        kriteria = [k.strip() for k in kriteria.split(",")]

        matriks_kriteria = np.zeros((len(kriteria), len(kriteria)))
        for i in range(len(kriteria)):
            for j in range(i + 1, len(kriteria)):
                col1, col2 = st.columns([2, 1])

                with col1:
                    pilihan = st.radio(
                        f"Mana yang lebih penting?",
                        options=[kriteria[i], kriteria[j]],
                        key=f"radio_kriteria_{i}_{j}"
                    )

                with col2:
                    nilai = st.number_input(
                        f"Masukkan nilai perbandingan ({pilihan}):",
                        min_value=0.1, max_value=9.0, step=0.1, value=1.0,
                        key=f"nilai_kriteria_{i}_{j}"
                    )

                    if pilihan == kriteria[i]:
                        matriks_kriteria[i][j] = nilai
                        matriks_kriteria[j][i] = 1 / nilai
                    else:
                        matriks_kriteria[i][j] = 1 / nilai
                        matriks_kriteria[j][i] = nilai
                st.divider()

        for i in range(len(kriteria)):
            matriks_kriteria[i][i] = 1

        if st.button("Proses"):
            # Save to session state
            st.session_state["kriteria"] = kriteria
            st.session_state["matriks_kriteria"] = matriks_kriteria

            # Normalisasi matriks kriteria dan hitung bobot
            total_per_kolom = matriks_kriteria.sum(axis=0)
            normalisasi_kriteria = matriks_kriteria / total_per_kolom
            bobot_kriteria = normalisasi_kriteria.mean(axis=1)

            # Hitung eigen values
            eigen_values = normalisasi_kriteria.mean(axis=1) * total_per_kolom

            # Consistency Index (CI) and Consistency Ratio (CR)
            lambda_max = eigen_values.sum()
            CI = (lambda_max - len(kriteria)) / (len(kriteria) - 1)
            
            # Random Index (RI) values for matrices of size 1 to 10
            RI_values = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
                         6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
            RI = RI_values.get(len(kriteria), 1.49)  # Default to 1.49 for sizes > 10
            CR = CI / RI if RI != 0 else 0

            # Save results to session state
            st.session_state["bobot_kriteria"] = bobot_kriteria
            st.session_state["normalisasi_kriteria"] = normalisasi_kriteria
            st.session_state["eigen_values"] = eigen_values
            st.session_state["total_per_kolom"] = total_per_kolom
            st.session_state["CI"] = CI
            st.session_state["CR"] = CR
            st.session_state["show_form"] = False

    # Display results if available in session state
    if st.session_state["matriks_kriteria"] is not None and st.session_state["kriteria"]:
        # Buat DataFrame untuk matriks kriteria
        matriks_df = pd.DataFrame(
            st.session_state["matriks_kriteria"],
            index=st.session_state["kriteria"],
            columns=st.session_state["kriteria"]
        )
        
        # Tambahkan baris total untuk jumlah kolom
        total_kolom = matriks_df.sum()
        total_kolom['Kriteria'] = 'Total'
        matriks_df.loc['Total'] = total_kolom

        st.write("Matriks Perbandingan Kriteria:")
        st.dataframe(matriks_df)

    if st.session_state["normalisasi_kriteria"] is not None and st.session_state["kriteria"]:
        # Persiapkan data untuk tabel normalisasi
        normalisasi_df = pd.DataFrame(
            st.session_state["normalisasi_kriteria"],
            index=st.session_state["kriteria"],
            columns=st.session_state["kriteria"]
        )
        
        # Tambahkan kolom jumlah pada baris normalisasi
        normalisasi_df['Jumlah Baris'] = normalisasi_df.sum(axis=1)
        
        # Tambahkan kolom bobot
        normalisasi_df['Bobot'] = st.session_state["bobot_kriteria"]

        # Tambahkan kolom eigen values
        normalisasi_df['Eigen Value'] = st.session_state["eigen_values"]

        # Tambahkan baris total untuk jumlah kolom
        total_kolom = normalisasi_df.sum()
        total_kolom['Kriteria'] = 'Total'
        normalisasi_df.loc['Total'] = total_kolom

        st.write("Matriks Normalisasi dan Bobot Kriteria:")
        st.dataframe(normalisasi_df)

        # Tampilkan informasi konsistensi
        st.write("Consistency Index (CI):", st.session_state["CI"])
        st.write("Consistency Ratio (CR):", st.session_state["CR"])

        if st.session_state["CR"] <= 0.1:
            st.success("CR <= 0.1, matriks konsisten!")
        else:
            st.warning("CR > 0.1, matriks tidak konsisten!")

# Contoh menjalankan aplikasi
if __name__ == "__main__":
    criteria_form()