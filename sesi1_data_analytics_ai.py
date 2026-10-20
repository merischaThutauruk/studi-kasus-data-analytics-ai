"""
Studi Kasus Sesi 1: Make Sense of Data with Analysis and AI
Pemateri: Bapak Andria Arisal (BRIN)
Topik: Implementasi CRISP-DM & 4 Level Analytics pada Kasus Piutang PT Maju Bersama
Catatan: Pure Python Implementation (Bebas Error ModuleNotFoundError)
"""

from datetime import datetime, timedelta
import math
import random

def generate_messy_dataset(n_samples=300):
  random.seed(42)
  base_date = datetime(2024, 7, 1)
  records = []

  for i in range(1, n_samples + 1):
    inv_date = base_date + timedelta(days=i)
    if random.random() < 0.08:
      ret_date = inv_date - timedelta(days=random.randint(1, 5))
    else:
      ret_date = None

    amount = (
        None if random.random() < 0.05 else random.uniform(1_000_000, 50_000_000)
    )
    discount = random.choices(
        [0.0, 0.05, 0.10, 0.15, 0.45, 0.60],
        weights=[40, 25, 20, 10, 3, 2],
    )[0]
    hist_late = random.choices([0, 1, 2, 3, 4], weights=[50, 25, 15, 7, 3])[0]
    credit_limit = random.choice([20_000_000, 50_000_000, 100_000_000])
    pay_method = random.choice(["Credit", "credit ", "CREDIT", "Bank Transfer"])

    eff_amount = amount if amount is not None else 15_000_000
    risk_score = (
        (eff_amount / credit_limit) * 0.4
        + (hist_late * 0.18)
        + (0.25 if discount > 0.3 else 0.0)
    )
    is_late = 1 if (risk_score + random.uniform(-0.1, 0.1)) > 0.55 else 0

    records.append({
        "invoice_id": f"INV-{i:04d}",
        "customer_id": f"CUST-{random.randint(1, 80):03d}",
        "invoice_date": inv_date,
        "return_date": ret_date,
        "payment_method": pay_method,
        "invoice_amount": amount,
        "discount_rate": discount,
        "historical_late_count": hist_late,
        "credit_limit": credit_limit,
        "actual_days_overdue": random.randint(31, 90) if is_late else 0,  # Leakage!
        "is_late_payment": is_late,
    })

  records.extend(records[:10])
  return records

def clean_and_prepare_data(records):
  print("=== [1] TAHAP DATA PREPARATION (MITIGASI DATA TRAPS) ===")
  print(f"Jumlah baris awal (kotor): {len(records)}")

  unique_map = {}
  for r in records:
    unique_map[r["invoice_id"]] = dict(r)
  dedup_records = list(unique_map.values())

  valid_amounts = sorted(
      [r["invoice_amount"] for r in dedup_records if r["invoice_amount"] is not None]
  )
  median_amount = valid_amounts[len(valid_amounts) // 2]

  clean_records = []
  impossible_count = 0
  anomaly_discount_count = 0

  for r in dedup_records:
    if r["return_date"] is not None and r["return_date"] < r["invoice_date"]:
      impossible_count += 1
      continue

    if r["invoice_amount"] is None:
      r["invoice_amount"] = median_amount
    r["payment_method"] = r["payment_method"].strip().title()

    r["flag_discount_anomaly"] = 1 if r["discount_rate"] > 0.30 else 0
    if r["flag_discount_anomaly"] == 1:
      anomaly_discount_count += 1

    r.pop("actual_days_overdue", None)
    clean_records.append(r)

  print(f"- Menghapus {len(records) - len(dedup_records)} baris duplikat.")
  print(
      f"- Menghapus {impossible_count} baris Impossible Order (Retur < Tanggal Beli)."
  )
  print(
      f"- Mendeteksi {anomaly_discount_count} transaksi Anomali Diskon Sales (>30%)."
  )
  print("- Menghapus kolom 'actual_days_overdue' untuk mencegah Data Leakage.")
  print(f"Jumlah baris bersih siap analisis: {len(clean_records)}\n")
  return clean_records

def run_four_level_analytics(data):
  print("=== [2] DESCRIPTIVE ANALYTICS (What Happened?) ===")
  total_rev = sum(r["invoice_amount"] for r in data)
  late_rate = (sum(r["is_late_payment"] for r in data) / len(data)) * 100
  print(f"Total Gross Revenue                : Rp {total_rev:,.2f}")
  print(f"Persentase Piutang Telat (>30 Hari): {late_rate:.2f}%\n")

  print("=== [3] DIAGNOSTIC ANALYTICS (Why Did It Happen?) ===")
  for flag in [0, 1]:
    sub = [r for r in data if r["flag_discount_anomaly"] == flag]
    late_pct = (
        (sum(r["is_late_payment"] for r in sub) / len(sub) * 100) if sub else 0
    )
    label = "Diskon Normal (<=30%)" if flag == 0 else "Anomali Diskon (>30%)"
    print(f"- {label:<24}: {len(sub):3d} transaksi | Rasio Telat Bayar: {late_pct:.1f}%")
  print()

  print("=== [4] PREDICTIVE ANALYTICS (What Will Happen?) ===")
  split_idx = int(len(data) * 0.75)
  train_data, test_data = data[:split_idx], data[split_idx:]

  dt_correct = 0
  lr_correct = 0

  for r in test_data:
    dt_pred = (
        1
        if (
            r["historical_late_count"] >= 2
            or (r["invoice_amount"] / r["credit_limit"]) > 0.8
        )
        else 0
    )
    if dt_pred == r["is_late_payment"]:
      dt_correct += 1

    z = (
        -2.2
        + 1.8 * (r["invoice_amount"] / r["credit_limit"])
        + 0.95 * r["historical_late_count"]
        + 1.2 * r["flag_discount_anomaly"]
    )
    prob = 1.0 / (1.0 + math.exp(-z))
    r["risk_probability"] = prob
    lr_pred = 1 if prob >= 0.5 else 0
    if lr_pred == r["is_late_payment"]:
      lr_correct += 1

  print(
      f"Akurasi Model Decision Tree       : {(dt_correct / len(test_data)) * 100:.2f}%"
  )
  print(
      f"Akurasi Model Logistic Regression : {(lr_correct / len(test_data)) * 100:.2f}%\n"
  )

  print("=== [5] PRESCRIPTIVE ANALYTICS (Rekomendasi Keputusan Manajemen) ===")
  print(
      f"{'INVOICE':<10} | {'CUSTOMER':<10} | {'PROBABILITAS RISIKO':<20} | {'REKOMENDASI TINDAKAN (HUMAN DECISION)'}"
  )
  print("-" * 95)
  for r in test_data[:8]:
    p = r["risk_probability"]
    if p > 0.75:
      action = f"TINGGI: Pangkas limit kredit ke Rp {r['credit_limit']*0.5:,.0f} & Penagihan Khusus"
    elif p > 0.45:
      action = "SEDANG: Kirim peringatan otomatis H-7 jatuh tempo"
    else:
      action = "RENDAH: Pertahankan limit kredit normal"
    print(f"{r['invoice_id']:<10} | {r['customer_id']:<10} | {p:<20.2%} | {action}")


if __name__ == "__main__":
  raw_data = generate_messy_dataset(300)
  clean_data = clean_and_prepare_data(raw_data)
  run_four_level_analytics(clean_data)