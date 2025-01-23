import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu
from upload import upload_menu
from clustering import clustering_menu
from model_evaluation_menu import model_evaluation
from input_data_menu import input_data
from ahp_criteria import criteria_form
from ahp_alternatives import alternatives_form
from ahp_result import ahp_finalResult

def main():
    with st.sidebar:
        selected = option_menu(
            menu_title = "Menu",
            options = ["Beranda", "Unggah Data", "Clustering", "Evaluasi Model K-Means", "Prediksi Cluster", "AHP - Kriteria", "AHP - Alternatif", "AHP - Hasil"],
            icons = ["house", "upload", "grid", "check-square", "plus-circle", "1-square", "2-square", "3-square"],
            menu_icon = "",
            default_index = 0
        )

    if selected == "Beranda":
        st.title("Sistem Penunjang Keputusan Metode K-Means, PCA, dan AHP")
        st.info(
        """
        ### Selamat datang
        Sistem ini dirancang untuk melakukan clustering data surat, serta memberikan rekomendasi prioritas cluster untuk ditindaklanjuti.
        
        Silakan pilih menu dari sidebar untuk memulai!
        """
        )
        

    elif selected == "Unggah Data":
        upload_menu()

    elif selected == "Clustering":
        st.title("Clustering")
        st.write("Halaman untuk melakukan clustering dan pelatihan model K-Means.")
        clustering_menu()

    elif selected == "Evaluasi Model K-Means":
        st.title("Evaluasi Model K-Means")
        st.write("Halaman untuk evaluasi model K-Means.")
        model_evaluation()

    elif selected == "Prediksi Cluster":
        st.title("Prediksi Cluster")
        input_data()
    
    elif selected == "AHP - Kriteria":
        criteria_form()

    elif selected == "AHP - Alternatif":
        alternatives_form()

    elif selected == "AHP - Hasil":
        ahp_finalResult()

if __name__ == "__main__":
    main()
