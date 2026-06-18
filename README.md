# Automated Essay Scoring (AES) — IndoBERT + Cosine Similarity

Sistem penilaian esai otomatis berbahasa Indonesia menggunakan model **IndoBERT** (`indobenchmark/indobert-base-p1`) dan **Cosine Similarity**. Sistem ini menilai kemiripan semantik antara jawaban siswa dengan kunci jawaban, lalu mengubahnya menjadi skor 0–100.

Dataset yang digunakan adalah esai sejarah siswa kelas XI pada dua topik: **Kolonialisme & Pergerakan Nasional** dan **Pendudukan Jepang**.

## Cara Kerja

1. Teks jawaban siswa dan kunci jawaban di-*preprocess* (lowercase, hapus karakter khusus)
2. Keduanya diubah menjadi vektor embedding menggunakan IndoBERT (strategi *mean pooling*)
3. Cosine similarity dihitung antar pasangan vektor, lalu dikalikan 100 sebagai skor akhir
4. Skor dikategorikan menjadi empat tingkat:

| Skor | Kategori |
|:---:|:---|
| 80–100 | Sangat Sesuai |
| 60–79 | Cukup Sesuai |
| 40–59 | Kurang Sesuai |
| 0–39 | Tidak Sesuai |

## Evaluasi

Sistem dievaluasi dari empat sisi: kualitas embedding, konsistensi skor (Cronbach's Alpha), perbandingan dengan model berbasis Euclidean distance, dan validasi terhadap skor manual guru (Pearson r, MAE, Cohen's Kappa).

## Instalasi

```bash
pip install transformers torch openpyxl pandas numpy scikit-learn matplotlib seaborn tqdm accelerate pingouin
```

## Output

Hasil penilaian disimpan ke `Hasil_AES_IndoBERT_Cosine.xlsx`, ringkasan statistik ke `ringkasan_aes.json`, dan seluruh pipeline ke `aes_pipeline.pkl`.

## Link 
https://penilaian-esai-otomatis-sejarah-indonesia.streamlit.app/
