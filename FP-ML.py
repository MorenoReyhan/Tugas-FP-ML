import streamlit as st
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns
import matplotlib.pyplot as plt
import io

# --- Functions from Code 2 (adapted for Streamlit) ---
def calculate_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Calculates the Sharpe ratio for a given set of weights."""
    port_return = np.sum(mean_returns * weights)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if port_volatility == 0:
        return -np.inf
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    return sharpe_ratio

def normalize_weights(weights):
    """Normalizes weights to sum to 1 and ensures they are non-negative."""
    weights = np.maximum(weights, 0)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return np.ones(len(weights)) / len(weights) if len(weights) > 0 else np.array([])
    return weights / total_weight

def aro_portfolio_optimization(df_prices, n_rabbits, n_iterations, alpha, risk_free_rate):
    """
    Performs portfolio optimization using the custom ARO-like algorithm.
    Returns optimal weights, best Sharpe ratio, and history of Sharpe ratios.
    """
    if df_prices.empty or len(df_prices.columns) == 0:
        st.warning("Data harga kosong atau tidak ada aset yang terdeteksi.")
        return None, -np.inf, []

    tickers = df_prices.columns.tolist()
    n_assets = len(tickers)

    if n_assets == 0:
        st.warning("Tidak ada aset yang terdeteksi dalam data.")
        return None, -np.inf, []

    df_prices_sorted = df_prices.sort_index()
    daily_returns = df_prices_sorted.pct_change().dropna()

    if daily_returns.empty:
        st.warning("Return harian tidak dapat dihitung (mungkin data terlalu sedikit atau konstan).")
        return None, -np.inf, []
    if daily_returns.shape[0] < 2:
        st.warning("Data return harian tidak cukup untuk menghitung matriks kovarians.")
        return None, -np.inf, []

    mean_returns_calc = daily_returns.mean() * 252
    cov_matrix_calc = daily_returns.cov() * 252

    rabbits = [normalize_weights(np.random.rand(n_assets)) for _ in range(n_rabbits)]
    best_score = -np.inf
    best_weights_arr = np.array([])
    history = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for iteration in range(n_iterations):
        for i in range(n_rabbits):
            current_weights = rabbits[i]
            if current_weights.ndim > 1:
                current_weights = current_weights.flatten()

            random_vector = np.random.uniform(-1, 1, n_assets)
            adjustment = alpha * random_vector
            new_weights_raw = current_weights + adjustment
            new_weights = normalize_weights(new_weights_raw)

            if len(mean_returns_calc) != n_assets or cov_matrix_calc.shape[0] != n_assets:
                st.error("Dimensi mean_returns atau cov_matrix tidak sesuai dengan jumlah aset.")
                return None, -np.inf, history

            old_fitness = calculate_sharpe_ratio(current_weights, mean_returns_calc.values, cov_matrix_calc.values, risk_free_rate)
            new_fitness = calculate_sharpe_ratio(new_weights, mean_returns_calc.values, cov_matrix_calc.values, risk_free_rate)

            if new_fitness > old_fitness:
                rabbits[i] = new_weights
                if new_fitness > best_score:
                    best_score = new_fitness
                    best_weights_arr = new_weights
        
        history.append(best_score if best_score != -np.inf else None)
        progress = (iteration + 1) / n_iterations
        progress_bar.progress(progress)
        status_text.text(f"Iterasi {iteration + 1}/{n_iterations} | Sharpe Ratio Terbaik Sementara: {best_score:.4f}")

    status_text.text("Optimasi ARO Selesai!")
    progress_bar.empty()

    if not best_weights_arr.any():
        st.warning("Tidak ditemukan bobot optimal. Ini bisa terjadi jika semua iterasi gagal menghasilkan peningkatan.")
        return None, -np.inf, history

    optimal_weights_dict = {ticker: weight for ticker, weight in zip(tickers, best_weights_arr)}
    
    return optimal_weights_dict, best_score, history

# --- Functions from Code 1 (plot_pie_chart and main structure) ---
def plot_pie_chart(weights, title="Alokasi Portofolio Optimal (ARO)"):
    """Fungsi untuk membuat dan menampilkan pie chart alokasi portofolio."""
    if not weights or all(value == 0 for value in weights.values()):
        st.warning("Tidak ada alokasi untuk ditampilkan dalam pie chart (semua bobot nol).")
        return

    filtered_weights = {k: v for k, v in weights.items() if v > 0.001}
    if not filtered_weights:
        st.warning("Bobot portofolio terlalu kecil untuk divisualisasikan dalam pie chart.")
        return

    labels = filtered_weights.keys()
    sizes = [filtered_weights[key] * 100 for key in labels]
    explode = [0.05] * len(labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption=title)
    plt.close(fig)

def main():
    st.set_page_config(layout="wide")
    st.title("📊 OPTIMASI PORTOFOLIO dengan ARO (Metaheuristic)")

    menu = ["Home"]
    choice = st.sidebar.selectbox("Menu", menu)

    if "data_df" not in st.session_state:
        st.session_state.data_df = None
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "optimal_weights" not in st.session_state:
        st.session_state.optimal_weights = None
    if "best_sharpe_aro" not in st.session_state:
        st.session_state.best_sharpe_aro = None
    if "sharpe_history" not in st.session_state:
        st.session_state.sharpe_history = []

    # Default parameters based on N (number of assets)
    default_n_rabbits = 20
    default_n_iterations = 100
    default_alpha = 0.2

    if choice == "Home":
        st.write("""
        Selamat datang di Aplikasi Optimasi Portofolio! 📈
        Aplikasi ini membantu Anda menemukan alokasi aset yang optimal menggunakan pendekatan heuristik yang terinspirasi dari Absolute Robust Optimization (ARO) berdasarkan data historis harga saham.
        """)

        st.header("Unggah Data Harga Saham (CSV)")
        uploaded_file = st.file_uploader(
            "Seret & lepas file CSV di sini, atau klik untuk menelusuri",
            type=["csv"],
            help="Pastikan file CSV Anda memiliki kolom 'Date' (atau tanggal sebagai kolom pertama) dan kolom harga penutupan untuk setiap aset."
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Pratinjau Data yang Diunggah:")
                st.dataframe(df.head())

                date_col_name = None
                potential_date_cols = [col for col in df.columns if col.lower() in ['date', 'tanggal', 'time', 'timestamp']]
                if potential_date_cols:
                    date_col_name = potential_date_cols[0]
                elif df.columns[0].lower() in ['unnamed: 0'] and pd.api.types.is_datetime64_any_dtype(df.iloc[:, 1]):
                    df = df.rename(columns={df.columns[1]: 'Date_derived'})
                    date_col_name = 'Date_derived'
                elif pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                    df = df.rename(columns={df.columns[0]: 'Date_derived'})
                    date_col_name = 'Date_derived'

                if date_col_name:
                    try:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])
                        df = df.set_index(date_col_name)
                        price_cols = [col for col in df.columns if col != date_col_name]
                        for col in price_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df = df.dropna(axis=1, how='any')
                        
                        st.session_state.data_df = df
                        st.success(f"File CSV berhasil diunggah dan kolom '{date_col_name}' ditetapkan sebagai indeks!")
                        st.session_state.analysis_done = False
                        st.session_state.optimal_weights = None
                        st.session_state.best_sharpe_aro = None
                        st.session_state.sharpe_history = []
                    except Exception as e_date:
                        st.error(f"Gagal memproses kolom tanggal '{date_col_name}': {e_date}")
                        st.session_state.data_df = None
                else:
                    try:
                        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                        df = df.set_index(df.columns[0])
                        price_cols = df.columns
                        for col in price_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df = df.dropna(axis=1, how='any')

                        st.session_state.data_df = df
                        st.info(f"Mencoba menggunakan kolom '{df.index.name}' sebagai tanggal.")
                        st.session_state.analysis_done = False
                        st.session_state.optimal_weights = None
                        st.session_state.best_sharpe_aro = None
                        st.session_state.sharpe_history = []
                    except Exception as e_fallback:
                        st.warning("Kolom tanggal tidak terdeteksi secara otomatis atau gagal dikonversi. Pastikan kolom pertama adalah tanggal atau beri nama 'Date'/'Tanggal'. Optimasi mungkin tidak akurat.")
                        st.error(f"Detail kesalahan konversi: {e_fallback}")
                        numeric_df = df.select_dtypes(include=np.number)
                        if not numeric_df.empty:
                            st.warning("Menggunakan kolom numerik yang terdeteksi. Pastikan ini adalah harga aset yang benar.")
                            st.session_state.data_df = numeric_df.copy()
                        else:
                            st.error("Tidak ada kolom numerik yang valid ditemukan untuk harga aset.")
                            st.session_state.data_df = None

            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca atau memproses file CSV: {e}")
                st.session_state.data_df = None
        elif st.session_state.data_df is None:
            st.info("Menunggu Anda mengunggah file CSV...")

        if st.session_state.data_df is not None:
            # Determine N and set default parameters based on code 2 rules
            N = len(st.session_state.data_df.columns)
            if N < 10:
                default_n_rabbits = 20
                default_n_iterations = 100
                default_alpha = 0.2
            elif 10 <= N <= 19:
                default_n_rabbits = 20
                default_n_iterations = 500
                default_alpha = 0.1
            else: # N > 20
                default_n_rabbits = 100
                default_n_iterations = 1000
                default_alpha = 0.05

            st.markdown("---")
            st.header("🚀 Analisis Portofolio (ARO Heuristik)")

            st.subheader(f"Parameter Optimasi ARO (Default berdasarkan {N} aset)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                n_rabbits = st.number_input("Jumlah Agen (Rabbits)", min_value=5, max_value=200, value=default_n_rabbits, step=1)
            with col2:
                n_iterations = st.number_input("Jumlah Iterasi", min_value=10, max_value=2000, value=default_n_iterations, step=10)
            with col3:
                alpha = st.slider("Faktor Pergerakan (Alpha)", min_value=0.01, max_value=1.0, value=default_alpha, step=0.01, format="%.2f")
            with col4:
                risk_free_rate = st.number_input("Risk-Free Rate (Tahunan)", min_value=0.0, max_value=0.2, value=0.0575, step=0.0025, format="%.4f")

            st.subheader("Tentang Metode ARO (Heuristik)")
            st.markdown("""
            **Pendekatan Absolute Robust Optimization (ARO) Heuristik yang Digunakan:**

            Metode optimasi portofolio yang digunakan di sini adalah pendekatan heuristik yang terinspirasi oleh algoritma optimasi berbasis populasi (mirip dengan algoritma evolusioner sederhana atau optimasi swarm). Ini bertujuan untuk menemukan alokasi bobot aset yang memaksimalkan **Sharpe Ratio**.

            **Cara Kerja (Sederhana):**
            1.  **Inisialisasi:** Sejumlah portofolio acak (disebut 'agen' atau 'kelinci') dibuat. Bobot aset dalam setiap portofolio dinormalisasi (dijumlahkan menjadi 1 dan non-negatif).
            2.  **Evaluasi:** Sharpe Ratio dihitung untuk setiap portofolio berdasarkan return historis rata-rata, matriks kovarians, dan tingkat bebas risiko yang diberikan.
            3.  **Iterasi & Eksplorasi:**
                * Dalam setiap iterasi, setiap portofolio ('agen') mencoba bergerak ke posisi baru di ruang solusi. Pergerakan ini melibatkan penambahan vektor acak (dikendalikan oleh parameter 'alpha') ke bobot saat ini.
                * Bobot baru kemudian dinormalisasi lagi.
            4.  **Seleksi:** Jika portofolio baru memiliki Sharpe Ratio yang lebih baik daripada yang lama, portofolio tersebut diperbarui.
            5.  **Pelacakan Terbaik:** Bobot portofolio dengan Sharpe Ratio tertinggi yang ditemukan sejauh ini disimpan.
            6.  **Pengulangan:** Proses ini diulang untuk jumlah iterasi yang ditentukan.

            Meskipun metode ini tidak secara formal mengimplementasikan *uncertainty sets* dari teori ARO klasik, sifat eksploratif dan iteratifnya bertujuan untuk menemukan solusi yang cukup kuat terhadap variasi kecil dalam data historis dengan mencari di berbagai kombinasi bobot.
            """)

            if st.button("Lakukan Analisis Optimasi Portofolio (ARO)", key="analyze_aro_button"):
                with st.spinner("Melakukan analisis optimasi ARO... Harap tunggu."):
                    try:
                        df_prices = st.session_state.data_df.copy()

                        for col in df_prices.columns:
                            if not pd.api.types.is_numeric_dtype(df_prices[col]):
                                try:
                                    df_prices[col] = pd.to_numeric(df_prices[col].astype(str).str.replace(',', ''), errors='raise')
                                except Exception as e_conv:
                                    st.error(f"Kolom '{col}' tidak dapat dikonversi menjadi numerik: {e_conv}. Harap periksa data Anda.")
                                    st.stop()
                        
                        df_prices = df_prices.dropna()

                        if df_prices.empty or df_prices.shape[0] < 2:
                            st.error("Data harga tidak cukup setelah pembersihan NaN atau konversi. Perlu minimal 2 baris data.")
                            st.stop()

                        optimal_weights, best_sharpe, sharpe_hist = aro_portfolio_optimization(
                            df_prices, n_rabbits, n_iterations, alpha, risk_free_rate
                        )

                        if optimal_weights:
                            st.session_state.optimal_weights = optimal_weights
                            st.session_state.best_sharpe_aro = best_sharpe
                            st.session_state.sharpe_history = sharpe_hist
                            st.session_state.analysis_done = True
                        else:
                            st.error("Optimasi ARO tidak menghasilkan bobot yang valid.")
                            st.session_state.analysis_done = False
                            st.session_state.optimal_weights = None

                    except Exception as e:
                        st.error(f"Terjadi kesalahan selama analisis ARO: {e}")
                        st.session_state.analysis_done = False
                        st.session_state.optimal_weights = None

            if st.session_state.analysis_done and st.session_state.optimal_weights:
                st.success("Analisis optimasi portofolio (ARO) berhasil diselesaikan!")
                st.subheader("Hasil Optimasi Portofolio (ARO):")

                weights_df = pd.DataFrame.from_dict(st.session_state.optimal_weights, orient='index', columns=['Bobot Optimal (ARO)'])
                weights_df['Bobot Optimal (%)'] = (weights_df['Bobot Optimal (ARO)'] * 100).round(2)
                st.dataframe(weights_df)

                st.subheader("Visualisasi Alokasi Aset (Pie Chart - ARO):")
                plot_pie_chart(st.session_state.optimal_weights, title="Alokasi Portofolio Optimal (ARO)")

                st.subheader("Perkembangan Sharpe Ratio Terbaik (ARO):")
                if st.session_state.sharpe_history:
                    fig_history, ax_history = plt.subplots(figsize=(10, 5))
                    valid_history = [s for s in st.session_state.sharpe_history if s is not None]
                    if valid_history:
                        ax_history.plot(range(len(valid_history)), valid_history)
                        ax_history.set_title("Perkembangan Sharpe Ratio Terbaik per Iterasi (ARO)")
                        ax_history.set_xlabel("Iterasi")
                        ax_history.set_ylabel("Sharpe Ratio")
                        ax_history.grid(True)
                        st.pyplot(fig_history)
                        plt.close(fig_history)
                    else:
                        st.warning("Tidak ada data riwayat Sharpe Ratio yang valid untuk ditampilkan.")
                else:
                    st.info("Tidak ada data riwayat Sharpe Ratio yang dihasilkan.")

                st.subheader("Penjelasan Persentase Hasil Optimasi (ARO):")
                explanation = "Berikut adalah alokasi aset yang disarankan berdasarkan optimasi ARO heuristik, yang bertujuan memaksimalkan Sharpe Ratio:\n\n"
                for asset, weight in st.session_state.optimal_weights.items():
                    if weight > 0.0001:
                        explanation += f"- **{asset}:** {weight*100:.2f}%\n"
                st.markdown(explanation)

                st.subheader("Perkiraan Kinerja Portofolio (ARO - Berdasarkan Data Historis):")
                try:
                    df_prices_perf = st.session_state.data_df.copy()
                    df_prices_perf = df_prices_perf.sort_index()
                    daily_returns_perf = df_prices_perf.pct_change().dropna()
                    
                    if not daily_returns_perf.empty and st.session_state.optimal_weights:
                        mean_returns_perf = daily_returns_perf.mean() * 252
                        cov_matrix_perf = daily_returns_perf.cov() * 252
                        
                        ordered_weights = np.array([st.session_state.optimal_weights[ticker] for ticker in mean_returns_perf.index])

                        expected_annual_return = np.sum(mean_returns_perf.values * ordered_weights)
                        annual_volatility = np.sqrt(np.dot(ordered_weights.T, np.dot(cov_matrix_perf.values, ordered_weights)))
                        sharpe_ratio_perf = st.session_state.best_sharpe_aro

                        st.markdown(f"- **Expected Annual Return:** {expected_annual_return*100:.2f}%")
                        st.markdown(f"- **Annual Volatility (Risk):** {annual_volatility*100:.2f}%")
                        st.markdown(f"- **Sharpe Ratio (dari Optimasi):** {sharpe_ratio_perf:.2f}")
                    else:
                        st.warning("Tidak dapat menghitung kinerja portofolio, data return kosong atau bobot tidak ada.")

                except Exception as e:
                    st.warning(f"Tidak dapat menghitung kinerja portofolio secara detail: {e}")
                    if st.session_state.best_sharpe_aro is not None:
                           st.markdown(f"- **Sharpe Ratio (dari Optimasi):** {st.session_state.best_sharpe_aro:.2f}")


if __name__ == '__main__':
    main()