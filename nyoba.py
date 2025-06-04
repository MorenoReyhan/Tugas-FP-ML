import streamlit as st
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import matplotlib.pyplot as plt
import io # Untuk menyimpan plot ke buffer

def plot_pie_chart(weights, title="Alokasi Portofolio Optimal"):
    """Fungsi untuk membuat dan menampilkan pie chart alokasi portofolio."""
    if not weights or all(value == 0 for value in weights.values()):
        st.warning("Tidak ada alokasi untuk ditampilkan dalam pie chart (semua bobot nol).")
        return

    # Filter aset dengan bobot lebih besar dari 0 untuk pie chart yang lebih bersih
    filtered_weights = {k: v for k, v in weights.items() if v > 0.001} # ambang batas kecil
    if not filtered_weights:
        st.warning("Bobot portofolio terlalu kecil untuk divisualisasikan dalam pie chart.")
        return

    labels = filtered_weights.keys()
    sizes = [filtered_weights[key] * 100 for key in labels] # persentase
    explode = [0.05] * len(labels) # sedikit memisahkan irisan

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal') # Memastikan pie chart berbentuk lingkaran.
    plt.title(title)

    # Simpan plot ke buffer BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption=title)
    plt.close(fig) # Tutup plot agar tidak ditampilkan ganda jika menggunakan plt.show() di backend

def main():
    st.set_page_config(layout="wide") # Menggunakan layout lebar
    st.title("📊 OPTIMASI PORTOFOLIO")

    menu = ["Home"]
    choice = st.sidebar.selectbox("Menu", menu)

    if "data_df" not in st.session_state:
        st.session_state.data_df = None
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "optimal_weights" not in st.session_state:
        st.session_state.optimal_weights = None

    if choice == "Home":
        st.write("""
        Selamat datang di Aplikasi Optimasi Portofolio! 📈
        Aplikasi ini membantu Anda menemukan alokasi aset yang optimal berdasarkan data historis harga saham.
        """)

        # --- Bagian Unggah File CSV ---
        st.header("Unggah Data Harga Saham (CSV)")
        uploaded_file = st.file_uploader(
            "Seret & lepas file CSV di sini, atau klik untuk menelusuri",
            type=["csv"],
            help="Pastikan file CSV Anda memiliki kolom 'Date' (atau tanggal sebagai kolom pertama) dan kolom harga penutupan untuk setiap aset."
        )

        # Memproses file yang diunggah
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Pratinjau Data yang Diunggah:")
                st.dataframe(df.head())

                # Asumsi kolom pertama adalah tanggal, atau ada kolom bernama 'Date'/'Tanggal'
                date_col = None
                if df.columns[0].lower() in ['date', 'tanggal']:
                    date_col = df.columns[0]
                elif pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                     # Jika tidak ada nama kolom spesifik, tapi kolom pertama adalah datetime
                    df = df.rename(columns={df.columns[0]: 'Date'})
                    date_col = 'Date'

                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.set_index(date_col)
                    st.session_state.data_df = df # Simpan df ke session state
                    st.success("File CSV berhasil diunggah dan kolom tanggal ditetapkan sebagai indeks!")
                    st.session_state.analysis_done = False # Reset status analisis
                    st.session_state.optimal_weights = None
                else:
                    st.warning("Kolom tanggal tidak terdeteksi secara otomatis. Pastikan kolom pertama adalah tanggal atau beri nama 'Date'/'Tanggal'. Optimasi mungkin tidak akurat.")
                    # Tetap simpan df, tapi pengguna harus sadar akan potensi masalah
                    st.session_state.data_df = df.copy() # Simpan salinan
                    # Coba konversi kolom pertama ke datetime jika memungkinkan sebagai fallback
                    try:
                        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                        df = df.set_index(df.columns[0])
                        st.session_state.data_df = df
                        st.info(f"Mencoba menggunakan kolom '{df.index.name}' sebagai tanggal.")
                    except Exception as e:
                        st.error(f"Tidak dapat mengkonversi kolom pertama ke tanggal: {e}")
                        st.session_state.data_df = None


            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
                st.warning("Pastikan file CSV Anda diformat dengan benar.")
                st.session_state.data_df = None
                st.session_state.analysis_done = False
                st.session_state.optimal_weights = None
        elif st.session_state.data_df is None: # Hanya tampilkan jika belum ada file yang diunggah
             st.info("Menunggu Anda mengunggah file CSV...")


        # Tombol Analisis dan Tampilan Hasil
        if st.session_state.data_df is not None:
            st.markdown("---")
            st.header("🚀 Analisis Portofolio")

            # Penjelasan Metode ARO (Sederhana)
            st.subheader("Tentang Metode ARO (Conceptual)")
            st.markdown("""
            **Absolute Robust Optimization (ARO)** bertujuan untuk menghasilkan portofolio yang kinerjanya tetap baik meskipun ada ketidakpastian dalam estimasi parameter input (seperti return yang diharapkan).
            Metode ARO yang sebenarnya melibatkan formulasi matematis yang kompleks untuk mendefinisikan 'ketidakpastian' dan mengoptimalkan terhadap skenario terburuk dalam batas ketidakpastian tersebut.

            Untuk **demonstrasi ini**, kita akan menggunakan pendekatan **Markowitz standar (Mean-Variance Optimization)** untuk memaksimalkan Sharpe Ratio. Ini memberikan dasar yang baik.
            Implementasi ARO yang lebih canggih akan:
            1.  Mendefinisikan *uncertainty sets* untuk *expected returns* dan *covariance matrix*.
            2.  Menggunakan *solver* khusus untuk menemukan portofolio yang optimal di bawah asumsi ketidakpastian tersebut.
            """)

            if st.button("Lakukan Analisis Optimasi Portofolio", key="analyze_button"):
                with st.spinner("Melakukan analisis optimasi... Harap tunggu."):
                    try:
                        df_prices = st.session_state.data_df.copy()

                        # Pastikan semua kolom harga adalah numerik, coba konversi jika tidak
                        for col in df_prices.columns:
                            if df_prices[col].dtype == 'object':
                                try:
                                    df_prices[col] = pd.to_numeric(df_prices[col].str.replace(',', ''))
                                except ValueError:
                                    st.error(f"Kolom '{col}' mengandung nilai non-numerik yang tidak dapat dikonversi. Harap periksa data Anda.")
                                    st.stop()

                        # 1. Hitung expected returns dan sample covariance
                        # Untuk ARO sejati, mu akan disesuaikan berdasarkan uncertainty set
                        mu = expected_returns.mean_historical_return(df_prices)
                        # Untuk ARO sejati, S (matriks kovarians) juga bisa memiliki komponen ketidakpastian
                        S = risk_models.sample_cov(df_prices)

                        # 2. Optimasi untuk Sharpe Ratio Maksimum (sebagai contoh)
                        ef = EfficientFrontier(mu, S)
                        # Anda bisa menambahkan batasan di sini, misalnya:
                        # ef.add_constraint(lambda w: w[0] + w[1] >= 0.1) # Contoh batasan
                        # ef.add_objective(objective_functions.L2_reg, gamma=0.1) # Contoh regularisasi (mirip ARO sederhana)

                        try:
                            weights = ef.max_sharpe() # Atau min_volatility(), etc.
                        except Exception as opt_error:
                            st.error(f"Optimasi gagal: {opt_error}")
                            st.warning("Ini bisa terjadi jika aset sangat berkorelasi atau data tidak cukup beragam. Coba dengan aset yang berbeda atau periode yang lebih panjang.")
                            st.session_state.analysis_done = False
                            st.session_state.optimal_weights = None
                            st.stop()


                        cleaned_weights = ef.clean_weights() # Membersihkan bobot kecil dan membulatkan
                        st.session_state.optimal_weights = cleaned_weights
                        st.session_state.analysis_done = True

                    except Exception as e:
                        st.error(f"Terjadi kesalahan selama analisis: {e}")
                        st.session_state.analysis_done = False
                        st.session_state.optimal_weights = None

            if st.session_state.analysis_done and st.session_state.optimal_weights:
                st.success("Analisis optimasi portofolio berhasil diselesaikan!")
                st.subheader("Hasil Optimasi Portofolio:")

                # Tampilkan Alokasi Aset dalam bentuk tabel
                weights_df = pd.DataFrame.from_dict(st.session_state.optimal_weights, orient='index', columns=['Bobot Optimal'])
                weights_df['Bobot Optimal (%)'] = (weights_df['Bobot Optimal'] * 100).round(2)
                st.dataframe(weights_df)

                # Tampilkan Pie Chart
                st.subheader("Visualisasi Alokasi Aset (Pie Chart):")
                plot_pie_chart(st.session_state.optimal_weights)

                # Penjelasan Persentase Hasil
                st.subheader("Penjelasan Persentase Hasil Optimasi:")
                explanation = "Berikut adalah alokasi aset yang disarankan untuk portofolio Anda, berdasarkan tujuan memaksimalkan Sharpe Ratio (return per unit risiko) menggunakan data historis yang diberikan:\n\n"
                for asset, weight in st.session_state.optimal_weights.items():
                    if weight > 0.0001: # Hanya tampilkan aset dengan bobot signifikan
                        explanation += f"- **{asset}:** {weight*100:.2f}%\n"
                explanation += "\nAlokasi ini dihitung menggunakan model Markowitz. Dalam konteks ARO, hasil ini akan menjadi titik awal, yang kemudian akan diuji ketahanannya terhadap berbagai skenario ketidakpastian pasar."
                st.markdown(explanation)

                # Kinerja Portofolio (opsional, bisa ditambahkan)
                try:
                    perf = ef.portfolio_performance(verbose=False, risk_free_rate=0.02) # asumsikan risk_free_rate 2%
                    st.subheader("Perkiraan Kinerja Portofolio (Berdasarkan Data Historis):")
                    st.markdown(f"- **Expected Annual Return:** {perf[0]*100:.2f}%")
                    st.markdown(f"- **Annual Volatility (Risk):** {perf[1]*100:.2f}%")
                    st.markdown(f"- **Sharpe Ratio:** {perf[2]:.2f}")
                except Exception as e:
                    st.warning(f"Tidak dapat menghitung kinerja portofolio: {e}")

if __name__ == '__main__':
    main()