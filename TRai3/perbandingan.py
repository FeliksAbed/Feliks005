import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load Data dari File Log ---
try:
    # Load data untuk sistem Adaptif (CSP)
    df_adaptive = pd.read_csv('queue_length.txt')
    # Rename columns for clarity and consistency if needed for plotting
    df_adaptive.rename(columns={
        'total_halting_vehicles': 'halting_vehicles',
        'ns_avg_wait_current': 'ns_avg_waiting_time',
        'ew_avg_wait_current': 'ew_avg_waiting_time'
    }, inplace=True)

    # Load data untuk sistem Statis
    df_static = pd.read_csv('static_queue_length.txt')
    # Rename columns for clarity and consistency if needed for plotting
    df_static.rename(columns={
        'queue_length': 'halting_vehicles', # Use a common name for plotting
    }, inplace=True)

except FileNotFoundError:
    print("Error: Pastikan file 'queue_length.txt' dan 'static_queue_length.txt' berada di direktori yang sama.")
    exit()

# --- 2. Siapkan Data Ringkasan (dari output konsol sebelumnya) ---
# Data Ringkasan untuk CSP Adaptif
csp_summary = {
    "Total vehicles departed": 1158,
    "Average waiting time per vehicle": 533.90,
    "Average travel time per vehicle": 50.71,
    "Throughput": 1.1580,
}

# Data Ringkasan untuk Statis
static_summary = {
    "Total vehicles departed": 1132,
    "Average waiting time per vehicle": 544.92,
    "Average travel time per vehicle": 53.56,
    "Throughput": 1.1320,
}

# --- 3. Buat Grafik Perbandingan Time Series ---

plt.figure(figsize=(15, 10))

# Plot Kepadatan Antrean (Halting Vehicles)
plt.subplot(2, 1, 1) # 2 baris, 1 kolom, plot ke-1
plt.plot(df_adaptive['step'], df_adaptive['halting_vehicles'], label='CSP Adaptif', color='blue', alpha=0.7)
plt.plot(df_static['step'], df_static['halting_vehicles'], label='Statis', color='red', alpha=0.7, linestyle='--')
plt.title('Perbandingan Kepadatan Antrean (Halting Vehicles) Seiring Waktu')
plt.xlabel('Langkah Simulasi')
plt.ylabel('Jumlah Kendaraan Berhenti')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(bottom=0) # Pastikan Y-axis mulai dari 0

# Plot Waktu Tunggu Kumulatif (per langkah)
# Note: total_waiting_time_step adalah waktu tunggu kumulatif dari semua kendaraan yang berhenti pada langkah itu.
# Bukan waktu tunggu kumulatif sepanjang simulasi.
# Jika ingin total kumulatif sepanjang simulasi, perlu dihitung ulang dari data per kendaraan.
# Namun, untuk perbandingan dinamika per langkah, ini sudah cukup.
plt.subplot(2, 1, 2) # 2 baris, 1 kolom, plot ke-2
plt.plot(df_adaptive['step'], df_adaptive['total_waiting_time_step'], label='CSP Adaptif', color='blue', alpha=0.7)
plt.plot(df_static['step'], df_static['waiting_time'], label='Statis', color='red', alpha=0.7, linestyle='--') # static log uses 'waiting_time'
plt.title('Perbandingan Total Waktu Tunggu per Langkah')
plt.xlabel('Langkah Simulasi')
plt.ylabel('Total Waktu Tunggu (detik) per Langkah')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# --- 4. Buat Grafik Perbandingan Bar Chart untuk Metrik Kunci ---

metrics = {
    'Total Kendaraan Berangkat': (csp_summary['Total vehicles departed'], static_summary['Total vehicles departed']),
    'Waktu Tunggu Rata-Rata per Kendaraan': (csp_summary['Average waiting time per vehicle'], static_summary['Average waiting time per vehicle']),
    'Waktu Perjalanan Rata-Rata per Kendaraan': (csp_summary['Average travel time per vehicle'], static_summary['Average travel time per vehicle']),
    'Throughput (kendaraan/langkah)': (csp_summary['Throughput'], static_summary['Throughput']),
}

metric_names = list(metrics.keys())
csp_values = [metrics[name][0] for name in metric_names]
static_values = [metrics[name][1] for name in metric_names]

x = np.arange(len(metric_names)) # Posisi label di x-axis
width = 0.35 # Lebar bar

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, csp_values, width, label='CSP Adaptif', color='skyblue')
rects2 = ax.bar(x + width/2, static_values, width, label='Statis', color='lightcoral')

# Fungsi untuk menambahkan nilai di atas bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

ax.set_ylabel('Nilai Metrik')
ax.set_title('Perbandingan Metrik Performa Utama')
ax.set_xticks(x)
ax.set_xticklabels(metric_names, rotation=20, ha='right')
ax.legend()
ax.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()


# --- 5. Buat Grafik Perbandingan Waktu Tunggu Spesifik Jalur (Time Series) ---

plt.figure(figsize=(15, 10))

# Waktu Tunggu NS
plt.subplot(2, 1, 1)
plt.plot(df_adaptive['step'], df_adaptive['ns_avg_waiting_time'], label='CSP Adaptif NS', color='blue', alpha=0.7)
plt.plot(df_static['step'], df_static['ns_avg_waiting_time'], label='Statis NS', color='skyblue', linestyle='--')
plt.title('Perbandingan Rata-rata Waktu Tunggu NS per Langkah')
plt.xlabel('Langkah Simulasi')
plt.ylabel('Waktu Tunggu Rata-rata (detik)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(bottom=0)

# Waktu Tunggu EW
plt.subplot(2, 1, 2)
plt.plot(df_adaptive['step'], df_adaptive['ew_avg_waiting_time'], label='CSP Adaptif EW', color='red', alpha=0.7)
plt.plot(df_static['step'], df_static['ew_avg_waiting_time'], label='Statis EW', color='lightcoral', linestyle='--')
plt.title('Perbandingan Rata-rata Waktu Tunggu EW per Langkah')
plt.xlabel('Langkah Simulasi')
plt.ylabel('Waktu Tunggu Rata-rata (detik)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

print("\nGrafik perbandingan telah dibuat dan ditampilkan.")
print("Periksa jendela pop-up grafik.")