import streamlit as st
import pandas as pd

def main():
    st.title("OPTIMASI PORTOFOLIO")

    menu = ["Home"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to our Homepage. This is a Portofolio Optimization app for who seek a best your Portofolio.")
        # --- Bagian Unggah File CSV ---
    st.header("Unggah Data (CSV)")
    uploaded_file = st.file_uploader(
        "Seret & lepas file CSV di sini, atau klik untuk menelusuri",
        type=["csv"],
        help="Pastikan file CSV Anda memiliki kolom tanggal dan kolom harga untuk setiap aset."
    )

    # Memproses file yang diunggah
    if uploaded_file is not None:
        try:
            # Membaca file CSV ke dalam DataFrame pandas
            # Asumsi: Kolom pertama adalah tanggal/indeks, sisa kolom adalah harga aset
            df = pd.read_csv(uploaded_file)

            # Menampilkan pratinjau data yang diunggah
            st.subheader("Pratinjau Data yang Diunggah:")
            st.dataframe(df.head()) # Menampilkan 5 baris pertama DataFrame

            st.success("File CSV berhasil diunggah dan dibaca!")

            # Di sini Anda bisa menambahkan logika lebih lanjut untuk optimasi portofolio
            # Misalnya, tombol untuk memulai analisis atau opsi konfigurasi.
            st.markdown(
                """
                Data Anda telah berhasil dimuat.
                Selanjutnya, Anda dapat menambahkan fitur untuk:
                - Memilih kolom tanggal dan harga.
                - Menghitung return dan risiko.
                - Menerapkan model optimasi portofolio (misalnya, Markowitz).
                - Menampilkan hasil optimasi (alokasi aset, Frontier Efisien).
                """
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
            st.warning("Pastikan file CSV Anda diformat dengan benar.")
    else:
        st.info("Menunggu Anda mengunggah file CSV...")


if __name__ == '__main__':
    main()
