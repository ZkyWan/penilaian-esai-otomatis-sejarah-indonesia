import re
import io
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity as cos_sim_sklearn

warnings.filterwarnings('ignore')

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Penilaian Esai Otomatis | IndoBERT",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS KUSTOM
st.markdown("""
<style>
    /* ── Kartu metrik ── */
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-card.green  { border-left-color: #27ae60; background:#f0fdf4; }
    .metric-card.yellow { border-left-color: #f39c12; background:#fffbeb; }
    .metric-card.orange { border-left-color: #e67e22; background:#fff7ed; }
    .metric-card.red    { border-left-color: #e74c3c; background:#fef2f2; }
    .metric-card.blue   { border-left-color: #3498db; background:#eff6ff; }

    /* ── Skor besar ── */
    .score-big {
        font-size: 3.4rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-label {
        font-size: 0.82rem;
        color: #666;
        margin-top: 6px;
        font-weight: 500;
    }

    /* ── Badge kategori ── */
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .badge-green  { background:#dcfce7; color:#166534; }
    .badge-yellow { background:#fef9c3; color:#854d0e; }
    .badge-orange { background:#ffedd5; color:#9a3412; }
    .badge-red    { background:#fee2e2; color:#991b1b; }

    /* ── Kotak pertanyaan ── */
    .box-pertanyaan {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 13px 16px;
        margin: 8px 0 14px 0;
        font-size: 0.97rem;
        line-height: 1.55;
    }

    /* ── Kotak jawaban ── */
    .box-jawaban {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #1e293b;
    }
    .box-kunci {
        background: #f0fdf4;
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #1e293b;
    }

    /* ── Kotak rekomendasi ── */
    .box-rekomendasi {
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 8px;
        font-size: 0.93rem;
        line-height: 1.6;
    }

    /* ── Sidebar lebih terang ── */
    div[data-testid="stSidebar"] {
        background: #f0f4f8;
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stSidebar"] * { color: #1e293b !important; }
    div[data-testid="stSidebar"] .stCaption { color: #64748b !important; }

    /* ── Judul section ── */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin: 6px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* ── Nama siswa di hasil ── */
    .nama-siswa-tag {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# TextPreprocessor — WAJIB didefinisikan di sini agar joblib.load() berhasil
# Nama dan isi class HARUS identik dengan yang ada di notebook
class TextPreprocessor:
    """
    Preprocessor teks ringan untuk input IndoBERT.
    Stopword TIDAK dihapus karena IndoBERT membutuhkan konteks kalimat utuh.
    """
    def __init__(self):
        pass

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ''
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,!?:;\-]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess_batch(self, texts: list) -> list:
        return [self.clean(t) for t in texts]


# HELPER FUNCTIONS
def badge_html(kategori: str) -> str:
    mapping = {
        "Sangat Sesuai": ("badge-green",  "✅ Sangat Sesuai"),
        "Cukup Sesuai" : ("badge-yellow", "🟡 Cukup Sesuai"),
        "Kurang Sesuai": ("badge-orange", "🟠 Kurang Sesuai"),
        "Tidak Sesuai" : ("badge-red",    "🔴 Tidak Sesuai"),
    }
    cls, label = mapping.get(kategori, ("badge-green", kategori))
    return f'<span class="badge {cls}">{label}</span>'


def score_color(skor: float) -> str:
    if skor >= 75: return "#16a34a"
    if skor >= 50: return "#d97706"
    if skor >= 25: return "#ea580c"
    return "#dc2626"


def kategorikan(skor: float, thr: dict) -> str:
    if skor >= thr['sangat_sesuai']: return "Sangat Sesuai"
    if skor >= thr['cukup_sesuai']:  return "Cukup Sesuai"
    if skor >= thr['kurang_sesuai']: return "Kurang Sesuai"
    return "Tidak Sesuai"


def clean_text(text: str) -> str:
    return TextPreprocessor().clean(text)


def rekomendasi_guru(kategori: str, skor: float) -> tuple[str, str]:
    """Kembalikan (teks_rekomendasi, warna_bg) berdasarkan kategori."""
    if kategori == "Sangat Sesuai":
        return (
            "💡 <b>Rekomendasi:</b> Jawaban siswa sangat sesuai dengan kunci jawaban. "
            "Guru dapat memberikan nilai penuh atau sesuai rubrik penilaian yang berlaku.",
            "#f0fdf4"
        )
    elif kategori == "Cukup Sesuai":
        return (
            "💡 <b>Rekomendasi:</b> Jawaban siswa cukup sesuai namun kurang lengkap. "
            "Pertimbangkan untuk memberikan nilai sebagian dan memberikan umpan balik "
            "agar siswa melengkapi penjelasannya.",
            "#fffbeb"
        )
    elif kategori == "Kurang Sesuai":
        return (
            "💡 <b>Rekomendasi:</b> Jawaban siswa kurang sesuai dengan materi yang diharapkan. "
            "Disarankan guru meninjau ulang jawaban secara manual dan memberikan "
            "bimbingan remedial kepada siswa.",
            "#fff7ed"
        )
    else:
        return (
            "💡 <b>Rekomendasi:</b> Jawaban siswa tidak sesuai dengan kunci jawaban. "
            "Guru perlu meninjau ulang secara manual. Siswa disarankan untuk mengulang "
            "materi dan mengikuti program remedial.",
            "#fef2f2"
        )


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()

# LOAD PIPELINE (.pkl)
@st.cache_resource(show_spinner="Memuat pipeline AES...")
def load_pipeline(path: str = "aes_pipeline_final.pkl"):
    import sys
    import types

    _candidates = ['__main__', 'app', 'aes_app', '__mp_main__']
    for _mod_name in _candidates:
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = types.ModuleType(_mod_name)
        if not hasattr(sys.modules[_mod_name], 'TextPreprocessor'):
            sys.modules[_mod_name].TextPreprocessor = TextPreprocessor

    try:
        return joblib.load(path), None
    except FileNotFoundError:
        return None, f"File tidak ditemukan: `{path}`"
    except Exception as e:
        return None, str(e)

# LOAD INDOBERT
@st.cache_resource(show_spinner="Memuat IndoBERT dari HuggingFace (~440MB, sekali saja)...")
def load_indobert(model_name: str):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device

# ENCODE + SCORING
def encode_text(text, tokenizer, model, device, max_length, strategy):
    text_clean = clean_text(text)
    if not text_clean:
        return None
    with torch.no_grad():
        enc = tokenizer(
            [text_clean], padding=True, truncation=True,
            max_length=max_length, return_tensors='pt'
        )
        input_ids      = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        token_type_ids = enc.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        if strategy == 'cls':
            emb = out.last_hidden_state[:, 0, :]
        else:
            tok = out.last_hidden_state
            msk = attention_mask.unsqueeze(-1).expand(tok.size()).float()
            emb = torch.sum(tok * msk, 1) / torch.clamp(msk.sum(1), min=1e-9)
    return emb.cpu().numpy()


def score_jawaban(jawaban, topik, qid, pipeline, tokenizer, model, device):
    kunci_df = pipeline['kunci_unik']
    row = kunci_df[
        (kunci_df['topik'] == topik) &
        (kunci_df['id_pertanyaan'] == qid)
    ]
    if len(row) == 0:
        return None
    row       = row.iloc[0]
    emb_siswa = encode_text(jawaban, tokenizer, model, device,
                             pipeline['max_length'], pipeline['embedding_strategy'])
    if emb_siswa is None:
        return None
    emb_kunci = np.array(row['embedding_kunci']).reshape(1, -1)
    sim       = float(np.clip(cos_sim_sklearn(emb_siswa, emb_kunci)[0][0], 0, 1))
    skor      = round(sim * 100, 2)
    return {
        'topik'            : topik,
        'id_pertanyaan'    : qid,
        'pertanyaan'       : row['pertanyaan'],
        'jawaban_input'    : jawaban,
        'kunci_jawaban'    : row['kunci_jawaban'],
        'cosine_similarity': round(sim, 6),
        'skor_aes'         : skor,
        'kategori'         : kategorikan(skor, pipeline['thresholds']),
    }

# SIDEBAR
def render_sidebar(pipeline):
    with st.sidebar:
        st.markdown(
            "<div style='font-size:1.3rem;font-weight:800;color:#1e293b;"
            "margin-bottom:4px'>📝 Penilaian Esai Otomatis</div>"
            "<div style='font-size:0.8rem;color:#64748b;margin-bottom:16px'>"
            "IndoBERT + Cosine Similarity</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        page = st.radio(
            "Halaman",
            ["🔍 Penilaian Jawaban", "📊 Dashboard Hasil"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown(
            "<div class='section-title'>Info Model</div>",
            unsafe_allow_html=True
        )
        st.caption(f"🤖 Model: **{pipeline['model_name'].split('/')[-1]}**")
        st.caption(f"📐 Embedding: `{pipeline['embedding_strategy']}`")
        st.caption(f"📏 Max token: `{pipeline['max_length']}`")

        st.markdown("---")
        df = pipeline['df_hasil']
        st.markdown(
            "<div class='section-title'>Statistik Dataset</div>",
            unsafe_allow_html=True
        )
        col_a, col_b = st.columns(2)
        col_a.metric("Siswa",   df['nama'].nunique())
        col_b.metric("Kelas",   df['kelas'].nunique())
        col_a.metric("Topik",   df['topik'].nunique())
        col_b.metric("Jawaban", f"{len(df):,}")

        st.markdown("---")
        st.caption("Universitas Telkom Purwokerto · 2025")

    return page

# GAUGE CHART — SVG modern, compact, tanpa matplotlib
def build_gauge_svg(skor: float, thr: dict, kategori: str) -> str:
    
    # Menentukan warna teks berdasarkan kategori
    if kategori == "Sangat Sesuai":
        warna = "#16a34a" # Hijau
    elif kategori == "Cukup Sesuai":
        warna = "#d97706" # Kuning/Emas
    elif kategori == "Kurang Sesuai":
        warna = "#ea580c" # Oranye
    else:
        warna = "#dc2626" # Merah

    # Tampilan HTML bersih pengganti Gauge Chart
    html_bersih = f"""
    <div style="text-align: center; padding: 20px; background-color: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px;">
        <p style="font-size: 15px; color: #64748b; margin-bottom: 5px; font-weight: 600; font-family: sans-serif;">Skor Keselarasan Akhir</p>
        <h1 style="font-size: 56px; color: {warna}; margin: 0; font-weight: 800; font-family: sans-serif;">
            {skor:.1f} <span style="font-size: 24px; color: #cbd5e1;">/ 100</span>
        </h1>
        <div style="margin-top: 10px; display: inline-block; background-color: {warna}15; padding: 6px 16px; border-radius: 20px;">
            <p style="font-size: 16px; color: {warna}; margin: 0; font-weight: 700; font-family: sans-serif;">{kategori}</p>
        </div>
    </div>
    """
    return html_bersih

# PAGE 1 — PENILAIAN JAWABAN
def page_inference(pipeline, tokenizer, model, device):
    st.title("🔍 Penilaian Jawaban Siswa")
    st.markdown(
        "Masukkan jawaban siswa untuk mendapatkan skor penilaian otomatis "
        "menggunakan **IndoBERT + Cosine Similarity**."
    )

    kunci_df   = pipeline['kunci_unik']
    topik_list = sorted(kunci_df['topik'].unique().tolist())
    thr        = pipeline['thresholds']

    # ── Tab: Satu Jawaban | Batch ─────────────────────────────────────────────
    tab_satu, tab_batch = st.tabs(["✏️ Satu Jawaban", "📋 Penilaian Batch"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB SATU JAWABAN
    # ══════════════════════════════════════════════════════════════════════════
    with tab_satu:
        col_form, col_info = st.columns([3, 1])

        with col_form:
            st.markdown("<div class='section-title'>Input Jawaban</div>",
                        unsafe_allow_html=True)

            # Nama siswa — bisa dikosongkan, opsional
            nama_siswa = st.text_input(
                "Nama Siswa",
                placeholder="Contoh: Budi Santoso",
                key="inp_nama"
            )

            # Selectbox di LUAR form → update real-time
            topik_sel = st.selectbox("Pilih Topik", topik_list, key="sel_topik")

            soal_options = sorted(
                kunci_df[kunci_df['topik'] == topik_sel]['id_pertanyaan']
                .unique().tolist()
            )
            qid_sel = st.selectbox(
                "Pilih Nomor Soal", soal_options,
                format_func=lambda x: f"Soal {x}",
                key="sel_soal"
            )

            # Pertanyaan update real-time
            pertanyaan_row = kunci_df[
                (kunci_df['topik'] == topik_sel) &
                (kunci_df['id_pertanyaan'] == qid_sel)
            ]
            if len(pertanyaan_row):
                teks_pertanyaan = pertanyaan_row.iloc[0]['pertanyaan']
                st.markdown(
                    f"<div class='box-pertanyaan'>"
                    f"❓ <b>Pertanyaan Soal {qid_sel}:</b><br>{teks_pertanyaan}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Form hanya text_area + submit
            with st.form("form_satu"):
                jawaban_input = st.text_area(
                    "Jawaban Siswa",
                    placeholder="Ketik atau tempel jawaban siswa di sini...",
                    height=190
                )
                submitted = st.form_submit_button(
                    "🎯 Nilai Jawaban Ini",
                    use_container_width=True,
                    type="primary"
                )

        # ── Kolom kanan: panduan + kunci ──────────────────────────────────────
        with col_info:
            st.markdown("<div class='section-title'>Panduan Skor</div>",
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card green" style="margin-bottom:8px">
                <b>✅ Sangat Sesuai</b><br>
                <span style="font-size:0.85rem">Skor ≥ {thr['sangat_sesuai']}</span>
            </div>
            <div class="metric-card yellow" style="margin-bottom:8px">
                <b>🟡 Cukup Sesuai</b><br>
                <span style="font-size:0.85rem">Skor ≥ {thr['cukup_sesuai']}</span>
            </div>
            <div class="metric-card orange" style="margin-bottom:8px">
                <b>🟠 Kurang Sesuai</b><br>
                <span style="font-size:0.85rem">Skor ≥ {thr['kurang_sesuai']}</span>
            </div>
            <div class="metric-card red" style="margin-bottom:8px">
                <b>🔴 Tidak Sesuai</b><br>
                <span style="font-size:0.85rem">Skor &lt; {thr['kurang_sesuai']}</span>
            </div>
            """, unsafe_allow_html=True)

            if len(pertanyaan_row):
                with st.expander("🔑 Lihat Kunci Jawaban"):
                    st.markdown(
                        f"<div class='box-kunci'>"
                        f"{pertanyaan_row.iloc[0]['kunci_jawaban']}</div>",
                        unsafe_allow_html=True
                    )

        # ── Hasil penilaian ───────────────────────────────────────────────────
        if submitted:
            if not jawaban_input.strip():
                st.warning("⚠️ Jawaban tidak boleh kosong.")
                st.stop()

            with st.spinner("⏳ Menghitung skor dengan IndoBERT..."):
                hasil = score_jawaban(
                    jawaban_input, topik_sel, qid_sel,
                    pipeline, tokenizer, model, device
                )

            if hasil is None:
                st.error("❌ Gagal memproses jawaban. Periksa kombinasi topik dan soal.")
                st.stop()

            st.markdown("---")

            label_nama    = nama_siswa.strip() if nama_siswa.strip() else "Siswa"
            warna         = score_color(hasil['skor_aes'])
            persen_sesuai = round(hasil['cosine_similarity'] * 100, 1)
            teks_rek, warna_rek = rekomendasi_guru(hasil['kategori'], hasil['skor_aes'])
            gauge_svg     = build_gauge_svg(hasil['skor_aes'], thr, hasil['kategori'])

            badge_map_css = {
                "Sangat Sesuai": ("badge-green",  "&#x2705; Sangat Sesuai"),
                "Cukup Sesuai" : ("badge-yellow", "&#x1F7E1; Cukup Sesuai"),
                "Kurang Sesuai": ("badge-orange", "&#x1F7E0; Kurang Sesuai"),
                "Tidak Sesuai" : ("badge-red",    "&#x1F534; Tidak Sesuai"),
            }
            bcls, blbl = badge_map_css.get(hasil['kategori'], ("badge-green", hasil['kategori']))

            st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
            padding:16px 20px 14px;margin-bottom:12px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap">
    <span style="background:#dbeafe;color:#1e40af;border-radius:6px;
                 padding:3px 10px;font-size:0.83rem;font-weight:600">
      &#x1F464; {label_nama}
    </span>
    <span style="font-size:0.9rem;color:#475569;font-weight:500">
      Hasil Penilaian &mdash; <em>{hasil['topik']}</em> &middot; Soal&nbsp;#{hasil['id_pertanyaan']}
    </span>
  </div>
  <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap">
    <div style="flex:0 0 220px;min-width:180px">{gauge_svg}</div>
    <div style="flex:1;min-width:200px;display:flex;flex-direction:column;gap:12px">
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
        <div>
          <div style="font-size:2.8rem;font-weight:800;line-height:1;color:{warna}">{hasil['skor_aes']:.1f}</div>
          <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px">Skor Akhir (0&ndash;100)</div>
        </div>
        <span class="badge {bcls}">{blbl}</span>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#64748b;margin-bottom:5px">
          <span>Kesesuaian Semantik</span>
          <span style="font-weight:700;color:{warna}">{persen_sesuai}%</span>
        </div>
        <div style="background:#e2e8f0;border-radius:999px;height:9px;overflow:hidden">
          <div style="width:{persen_sesuai}%;height:100%;background:{warna};border-radius:999px"></div>
        </div>
      </div>
      <div style="background:{warna_rek};border-radius:8px;padding:10px 13px;
                  font-size:0.84rem;line-height:1.55;border:1px solid #e2e8f0">
        {teks_rek}
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Perbandingan jawaban vs kunci ─────────────────────────────────
            st.markdown("---")
            with st.expander("📖 Perbandingan Jawaban Siswa vs Kunci Jawaban",
                             expanded=True):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(
                        f"<div class='section-title'>Jawaban Siswa</div>"
                        f"<div class='box-jawaban'>{hasil['jawaban_input']}</div>",
                        unsafe_allow_html=True
                    )
                with cb:
                    st.markdown(
                        f"<div class='section-title'>Kunci Jawaban Guru</div>"
                        f"<div class='box-kunci'>{hasil['kunci_jawaban']}</div>",
                        unsafe_allow_html=True
                    )

            # ── Catatan teknis (collapsed by default) ─────────────────────────
            with st.expander("🔧 Detail Teknis (untuk keperluan penelitian)"):
                st.markdown(f"""
                | Parameter | Nilai |
                |:---|:---|
                | Model | `{pipeline['model_name']}` |
                | Embedding Strategy | `{pipeline['embedding_strategy']}` |
                | Cosine Similarity (raw) | `{hasil['cosine_similarity']:.6f}` |
                | Skor (× 100) | `{hasil['skor_aes']}` |
                | Topik | {hasil['topik']} |
                | Soal | #{hasil['id_pertanyaan']} |
                """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB BATCH
    # ══════════════════════════════════════════════════════════════════════════
    with tab_batch:
        st.markdown("### 📋 Penilaian Batch")
        st.markdown(
            "Upload file Excel atau CSV berisi daftar jawaban siswa. "
            "Sistem akan menilai semua jawaban secara otomatis."
        )

        # Template download
        template_df = pd.DataFrame({
            'nama'          : ['Budi Santoso', 'Siti Rahayu'],
            'kelas'         : ['XI TKRO', 'XI TMI 1'],
            'topik'         : ['Kolonialisme & Pergerakan Nasional', 'Pendudukan Jepang'],
            'id_pertanyaan' : [1, 2],
            'pertanyaan'    : ['Pertanyaan esai di sini...', 'Pertanyaan esai di sini...'],
            'jawaban_siswa' : ['Jawaban siswa di sini...', 'Jawaban siswa di sini...'],
            'kunci_jawaban' : ['Kunci jawaban di sini...', 'Kunci jawaban di sini...']
        })
        buf_tmpl = io.BytesIO()
        template_df.to_excel(buf_tmpl, index=False, engine='openpyxl')
        st.download_button(
            "⬇️ Download Template Excel",
            data=buf_tmpl.getvalue(),
            file_name="template_batch_aes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_template"
        )

        st.markdown("**Kolom yang dibutuhkan sudah sesuai template.**")

        uploaded = st.file_uploader(
            "Upload file jawaban siswa",
            type=['xlsx', 'csv'],
            key="batch_upload"
        )

        if uploaded:
            try:
                df_batch = (pd.read_csv(uploaded)
                            if uploaded.name.endswith('.csv')
                            else pd.read_excel(uploaded))

                required = {'nama', 'topik', 'id_pertanyaan', 'jawaban_siswa'}
                missing  = required - set(df_batch.columns)
                if missing:
                    st.error(f"❌ Kolom tidak ditemukan: {missing}")
                else:
                    st.info(f"📄 {len(df_batch)} jawaban ditemukan. Memulai penilaian...")
                    results = []
                    prog    = st.progress(0, "Memproses...")

                    for idx, row_b in df_batch.iterrows():
                        h = score_jawaban(
                            str(row_b['jawaban_siswa']),
                            str(row_b['topik']),
                            int(row_b['id_pertanyaan']),
                            pipeline, tokenizer, model, device
                        )
                        if h:
                            results.append({
                                'Nama Siswa'        : row_b['nama'],
                                'Topik'             : row_b['topik'],
                                'No. Soal'          : row_b['id_pertanyaan'],
                                'Kesesuaian (%)'    : round(h['cosine_similarity']*100, 1),
                                'Skor AES (0-100)'  : h['skor_aes'],
                                'Kategori'          : h['kategori'],
                            })
                        prog.progress((idx + 1) / len(df_batch))

                    df_res = pd.DataFrame(results)
                    n_gagal = len(df_batch) - len(df_res)

                    if df_res.empty:
                        st.warning(
                            f"⚠️ Semua {len(df_batch)} jawaban gagal dinilai. "
                            "Pastikan kolom **topik** dan **id_pertanyaan** di file "
                            "sesuai dengan data yang ada di pipeline."
                        )
                        # Tampilkan topik & soal yang tersedia di pipeline untuk debug
                        kunci_df_ref = pipeline['kunci_unik']
                        with st.expander("🔍 Topik & Soal yang tersedia di pipeline"):
                            st.dataframe(
                                kunci_df_ref[['topik','id_pertanyaan','pertanyaan']]
                                .drop_duplicates()
                                .sort_values(['topik','id_pertanyaan'])
                                .reset_index(drop=True),
                                use_container_width=True, hide_index=True
                            )
                        with st.expander("🔍 Preview data yang diupload"):
                            st.dataframe(df_batch.head(10), use_container_width=True)
                    else:
                        st.success(f"✅ {len(df_res)} jawaban berhasil dinilai."
                                   + (f" ({n_gagal} baris dilewati karena topik/soal tidak cocok.)" if n_gagal else ""))

                        # Ringkasan cepat
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Sangat Sesuai",
                                  (df_res['Kategori']=='Sangat Sesuai').sum())
                        r2.metric("Cukup Sesuai",
                                  (df_res['Kategori']=='Cukup Sesuai').sum())
                        r3.metric("Kurang Sesuai",
                                  (df_res['Kategori']=='Kurang Sesuai').sum())
                        r4.metric("Tidak Sesuai",
                                  (df_res['Kategori']=='Tidak Sesuai').sum())

                        # Tabel dengan warna kategori
                        def highlight_kat(val):
                            m = {
                                'Sangat Sesuai': 'background-color:#dcfce7;color:#166534',
                                'Cukup Sesuai' : 'background-color:#fef9c3;color:#854d0e',
                                'Kurang Sesuai': 'background-color:#ffedd5;color:#9a3412',
                                'Tidak Sesuai' : 'background-color:#fee2e2;color:#991b1b',
                            }
                            return m.get(val, '')

                        st.dataframe(
                            df_res.style.map(highlight_kat, subset=['Kategori']),
                            use_container_width=True
                        )

                        buf_b = io.BytesIO()
                        df_res.to_excel(buf_b, index=False, engine='openpyxl')
                        st.download_button(
                            "⬇️ Download Hasil Penilaian Batch (.xlsx)",
                            data=buf_b.getvalue(),
                            file_name="hasil_batch_aes.xlsx",
                            mime="application/vnd.openxmlformats-officedocument"
                                 ".spreadsheetml.sheet",
                            key="dl_batch_result"
                        )
            except Exception as e:
                st.error(f"❌ Error membaca file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DASHBOARD HASIL SCORING
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard(pipeline):
    st.title("📊 Dashboard Hasil Scoring")

    df          = pipeline['df_hasil'].copy()
    rekap_siswa = pipeline['rekap_siswa'].copy()
    thr         = pipeline['thresholds']
    evaluasi    = pipeline['hasil_evaluasi']

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distribusi Skor",
        "🏫 Per Kelas & Topik",
        "📝 Per Soal",
        "🔬 Metrik Evaluasi",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — DISTRIBUSI SKOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        total    = len(df)
        n_sangat = (df['kategori'] == 'Sangat Sesuai').sum()
        n_cukup  = (df['kategori'] == 'Cukup Sesuai').sum()
        n_kurang = (df['kategori'] == 'Kurang Sesuai').sum()
        n_tidak  = (df['kategori'] == 'Tidak Sesuai').sum()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Rata-Rata Skor",   f"{df['skor_aes'].mean():.1f}")
        k2.metric("✅ Sangat Sesuai", f"{n_sangat} ({n_sangat/total*100:.0f}%)")
        k3.metric("🟡 Cukup Sesuai",  f"{n_cukup} ({n_cukup/total*100:.0f}%)")
        k4.metric("🟠 Kurang Sesuai", f"{n_kurang} ({n_kurang/total*100:.0f}%)")
        k5.metric("🔴 Tidak Sesuai",  f"{n_tidak} ({n_tidak/total*100:.0f}%)")

        st.markdown("---")
        col_h, col_p = st.columns(2)

        with col_h:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df['skor_aes'], bins=40, color='#3b82f6',
                    edgecolor='white', alpha=0.85)
            ax.axvline(thr['sangat_sesuai'], color='#16a34a', linestyle='--',
                       lw=1.8, label=f"Sangat Sesuai (≥{thr['sangat_sesuai']})")
            ax.axvline(thr['cukup_sesuai'],  color='#d97706', linestyle='--',
                       lw=1.8, label=f"Cukup Sesuai (≥{thr['cukup_sesuai']})")
            ax.axvline(df['skor_aes'].mean(), color='#dc2626', linestyle='-',
                       lw=2, label=f"Rata-rata: {df['skor_aes'].mean():.1f}")
            ax.set_title('Distribusi Skor AES', fontweight='bold', fontsize=12)
            ax.set_xlabel('Skor AES (0–100)')
            ax.set_ylabel('Jumlah Jawaban')
            ax.legend(fontsize=8)
            sns.despine(ax=ax)
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇️ Unduh Grafik", fig_to_bytes(fig),
                               "distribusi_skor.png", "image/png", key="dl_hist")
            plt.close(fig)

        with col_p:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            kat_order = ['Sangat Sesuai','Cukup Sesuai','Kurang Sesuai','Tidak Sesuai']
            vals_pie  = [df['kategori'].value_counts().get(k,0) for k in kat_order]
            warna_pie = ['#86efac','#fde68a','#fdba74','#fca5a5']
            wedges, texts, autotexts = ax2.pie(
                vals_pie, labels=kat_order, colors=warna_pie,
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor':'white','linewidth':2}
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            ax2.set_title('Proporsi Kategori', fontweight='bold', fontsize=12)
            st.pyplot(fig2, use_container_width=True)
            st.download_button("⬇️ Unduh Grafik", fig_to_bytes(fig2),
                               "proporsi_kategori.png", "image/png", key="dl_pie")
            plt.close(fig2)

        st.markdown("#### Statistik Deskriptif")
        stats_df = pd.DataFrame({
            'Metrik': ['Rata-rata','Median','Std Deviasi','Nilai Minimum',
                       'Nilai Maksimum','Kuartil 1 (25%)','Kuartil 3 (75%)'],
            'Nilai' : [
                f"{df['skor_aes'].mean():.2f}",
                f"{df['skor_aes'].median():.2f}",
                f"{df['skor_aes'].std():.2f}",
                f"{df['skor_aes'].min():.2f}",
                f"{df['skor_aes'].max():.2f}",
                f"{df['skor_aes'].quantile(0.25):.2f}",
                f"{df['skor_aes'].quantile(0.75):.2f}",
            ]
        })
        st.dataframe(stats_df, use_container_width=False, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — PER KELAS & TOPIK
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        col_bar, col_heat = st.columns(2)

        with col_bar:
            skor_kelas = (df.groupby('kelas')['skor_aes']
                          .agg(['mean','std','count']).reset_index()
                          .sort_values('mean', ascending=False))
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            warna_k = ['#3b82f6','#22c55e','#ef4444','#f59e0b','#8b5cf6','#06b6d4']
            bars3 = ax3.bar(skor_kelas['kelas'], skor_kelas['mean'],
                            yerr=skor_kelas['std'], capsize=4,
                            color=warna_k[:len(skor_kelas)],
                            edgecolor='white', alpha=0.9)
            for bar_, (_, r_) in zip(bars3, skor_kelas.iterrows()):
                ax3.text(bar_.get_x()+bar_.get_width()/2,
                         bar_.get_height()+r_['std']+1.5,
                         f'{r_["mean"]:.1f}\n(n={int(r_["count"])})',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax3.axhline(df['skor_aes'].mean(), color='#dc2626', linestyle='--',
                        lw=1.5, label=f'Rata-rata: {df["skor_aes"].mean():.1f}')
            ax3.set_title('Rata-Rata Skor per Kelas', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Skor AES (0–100)')
            ax3.set_ylim(0, 118)
            ax3.legend(fontsize=8)
            sns.despine(ax=ax3)
            st.pyplot(fig3, use_container_width=True)
            st.download_button("⬇️ Unduh Grafik", fig_to_bytes(fig3),
                               "skor_per_kelas.png", "image/png", key="dl_kelas")
            plt.close(fig3)

        with col_heat:
            pivot = df.pivot_table(values='skor_aes', index='kelas',
                                   columns='topik', aggfunc='mean')
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                        linewidths=0.5, linecolor='white',
                        cbar_kws={'label':'Skor Rata-Rata'}, ax=ax4)
            ax4.set_title('Heatmap Skor (Kelas × Topik)',
                          fontweight='bold', fontsize=12)
            ax4.set_xlabel('Topik')
            ax4.set_ylabel('Kelas')
            plt.xticks(rotation=20, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            st.download_button("⬇️ Unduh Grafik", fig_to_bytes(fig4),
                               "heatmap_skor.png", "image/png", key="dl_heat")
            plt.close(fig4)

        st.markdown("---")
        st.markdown("#### Rekap Skor per Siswa")
        kelas_filter = st.multiselect(
            "Filter berdasarkan Kelas",
            df['kelas'].unique().tolist(),
            default=df['kelas'].unique().tolist()
        )
        df_rekap_f = (rekap_siswa[rekap_siswa['kelas'].isin(kelas_filter)]
                      .sort_values('rata_skor', ascending=False)
                      .reset_index(drop=True))

        # Rename kolom agar lebih ramah guru
        df_rekap_display = df_rekap_f.rename(columns={
            'nama'      : 'Nama Siswa',
            'kelas'     : 'Kelas',
            'jumlah_soal': 'Jumlah Soal',
            'rata_skor' : 'Rata-Rata Skor',
            'total_skor': 'Total Skor',
            'skor_min'  : 'Skor Terendah',
            'skor_max'  : 'Skor Tertinggi',
        })
        st.dataframe(df_rekap_display, use_container_width=True)

        buf_r = io.BytesIO()
        df_rekap_display.to_excel(buf_r, index=False, engine='openpyxl')
        st.download_button(
            "⬇️ Download Rekap Siswa (.xlsx)",
            data=buf_r.getvalue(),
            file_name="rekap_siswa.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_rekap_xl"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — PER SOAL
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        topik_sel_d = st.selectbox(
            "Pilih Topik", sorted(df['topik'].unique().tolist()), key="topik_tab3"
        )
        df_topik = df[df['topik'] == topik_sel_d]
        soal_ids = sorted(df_topik['id_pertanyaan'].unique())

        col_box, col_bar2 = st.columns(2)

        with col_box:
            fig5, ax5 = plt.subplots(figsize=(7, 4))
            data_box = [df_topik[df_topik['id_pertanyaan']==q]['skor_aes'].values
                        for q in soal_ids]
            bp = ax5.boxplot(data_box, patch_artist=True,
                             medianprops={'color':'#dc2626','linewidth':2.5})
            for patch in bp['boxes']:
                patch.set_facecolor('#bfdbfe')
                patch.set_alpha(0.85)
            ax5.set_xticklabels([f'Soal {q}' for q in soal_ids])
            ax5.set_title(f'Distribusi Skor per Soal\n{topik_sel_d[:40]}',
                          fontweight='bold', fontsize=11)
            ax5.set_ylabel('Skor AES (0–100)')
            ax5.set_ylim(-5, 110)
            sns.despine(ax=ax5)
            st.pyplot(fig5, use_container_width=True)
            st.download_button("⬇️ Unduh Grafik", fig_to_bytes(fig5),
                               "boxplot_soal.png", "image/png", key="dl_box")
            plt.close(fig5)

        with col_bar2:
            mean_soal = df_topik.groupby('id_pertanyaan')['skor_aes'].mean().reset_index()
            fig6, ax6 = plt.subplots(figsize=(7, 4))
            ax6.bar([f'Soal {q}' for q in mean_soal['id_pertanyaan']],
                    mean_soal['skor_aes'],
                    color='#8b5cf6', edgecolor='white', alpha=0.9)
            for i, (_, r_) in enumerate(mean_soal.iterrows()):
                ax6.text(i, r_['skor_aes']+1.5,
                         f'{r_["skor_aes"]:.1f}',
                         ha='center', fontweight='bold', fontsize=11)
            ax6.set_title(f'Rata-Rata Skor per Soal\n{topik_sel_d[:40]}',
                          fontweight='bold', fontsize=11)
            ax6.set_ylabel('Skor AES (0–100)')
            ax6.set_ylim(0, 118)
            sns.despine(ax=ax6)
            st.pyplot(fig6, use_container_width=True)
            st.download_button("⬇️ Unduh Grafik", fig_to_bytes(fig6),
                               "mean_soal.png", "image/png", key="dl_bar2")
            plt.close(fig6)

        st.markdown("---")
        soal_det = st.selectbox("Lihat detail jawaban untuk soal:",
                                soal_ids, format_func=lambda x: f"Soal {x}",
                                key="soal_det")
        df_det = (df_topik[df_topik['id_pertanyaan'] == soal_det]
                  [['nama','kelas','jawaban_siswa','skor_aes','kategori']]
                  .sort_values('skor_aes', ascending=False)
                  .reset_index(drop=True))

        df_det = df_det.rename(columns={
            'nama'          : 'Nama Siswa',
            'kelas'         : 'Kelas',
            'jawaban_siswa' : 'Jawaban Siswa',
            'skor_aes'      : 'Skor AES',
            'kategori'      : 'Kategori',
        })

        def highlight_kat(val):
            m = {
                'Sangat Sesuai': 'background-color:#dcfce7;color:#166534',
                'Cukup Sesuai' : 'background-color:#fef9c3;color:#854d0e',
                'Kurang Sesuai': 'background-color:#ffedd5;color:#9a3412',
                'Tidak Sesuai' : 'background-color:#fee2e2;color:#991b1b',
            }
            return m.get(val, '')

        st.dataframe(
            df_det.style.map(highlight_kat, subset=['Kategori']),
            use_container_width=True
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — METRIK EVALUASI
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        emb_q    = evaluasi.get('embedding_quality', {})
        cronbach = evaluasi.get('cronbach_alpha', {})
        df_shap  = evaluasi.get('df_shapiro', pd.DataFrame())
        df_redun = evaluasi.get('df_redundancy', pd.DataFrame())
        r_panj   = evaluasi.get('r_skor_panjang', 0)
        r_jac    = evaluasi.get('r_jaccard_cosine', 0)

        st.markdown("#### Ringkasan Metrik Evaluasi")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Discriminative Gap",
                  f"{emb_q.get('discriminative_gap', 0):.4f}",
                  help="Intra − Inter similarity. Positif = IndoBERT berhasil membedakan topik.")
        e2.metric("Mean Cohesion",
                  f"{emb_q.get('mean_cohesion', 0):.4f}",
                  help="Kekompakan cluster embedding per soal.")
        best_alpha = max(cronbach.values()) if cronbach else 0
        e3.metric("Cronbach α terbaik", f"{best_alpha:.4f}",
                  delta="Acceptable ✅" if best_alpha >= 0.70 else "Low ⚠️")
        e4.metric("Korelasi Skor–Panjang", f"r = {r_panj:.4f}",
                  delta="Tidak Bias ✅" if abs(r_panj) < 0.40 else "Bias ⚠️")

        st.markdown("---")

        with st.expander("📐 Kelompok 1 — Embedding Quality", expanded=True):
            c1a, c1b = st.columns(2)
            with c1a:
                intra_s    = evaluasi.get('intra_scores', {})
                inter_s    = evaluasi.get('inter_scores', {})
                mean_intra = np.mean([v['mean'] for v in intra_s.values()]) if intra_s else 0
                mean_inter = np.mean([v['mean'] for v in inter_s.values()]) if inter_s else 0

                fig_eq, ax_eq = plt.subplots(figsize=(5, 3))
                bars_eq = ax_eq.bar(['Intra-class','Inter-class'],
                                    [mean_intra, mean_inter],
                                    color=['#86efac','#fca5a5'],
                                    alpha=0.9, edgecolor='white', linewidth=1.5)
                for b_, v_ in zip(bars_eq, [mean_intra, mean_inter]):
                    ax_eq.text(b_.get_x()+b_.get_width()/2,
                               b_.get_height()+0.005,
                               f'{v_:.4f}', ha='center',
                               fontweight='bold', fontsize=11)
                ax_eq.set_title('Intra vs Inter-class Similarity',
                                fontweight='bold', fontsize=10)
                ax_eq.set_ylim(0, 1.1)
                ax_eq.set_ylabel('Cosine Similarity')
                sns.despine(ax=ax_eq)
                st.pyplot(fig_eq, use_container_width=True)
                plt.close(fig_eq)

                gap = mean_intra - mean_inter
                if gap > 0:
                    st.success(f"✅ Gap = {gap:.4f} — IndoBERT berhasil membedakan topik")
                else:
                    st.warning(f"⚠️ Gap = {gap:.4f}")

            with c1b:
                df_coh = evaluasi.get('df_cohesion', pd.DataFrame())
                if not df_coh.empty:
                    fig_coh, ax_coh = plt.subplots(figsize=(5, 3))
                    topik_u = df_coh['topik'].unique()
                    warna_c = ['#3b82f6','#ef4444']
                    soal_c  = sorted(df_coh['qid'].unique())
                    x_c     = np.arange(len(soal_c))
                    for i, (t_, col_) in enumerate(zip(topik_u, warna_c)):
                        sub_c = df_coh[df_coh['topik']==t_].sort_values('qid')
                        ax_coh.bar(x_c + i*0.35, sub_c['cohesion_mean'], 0.35,
                                   label=t_[:22], color=col_,
                                   alpha=0.85, edgecolor='white')
                    ax_coh.set_xticks(x_c + 0.175)
                    ax_coh.set_xticklabels([f'S{q}' for q in soal_c])
                    ax_coh.set_title('Cohesion per Soal',
                                     fontweight='bold', fontsize=10)
                    ax_coh.set_ylim(0, 1.1)
                    ax_coh.legend(fontsize=7)
                    sns.despine(ax=ax_coh)
                    st.pyplot(fig_coh, use_container_width=True)
                    plt.close(fig_coh)

        with st.expander("📏 Kelompok 2 — Scoring Consistency"):
            c2a, c2b = st.columns(2)
            with c2a:
                st.markdown("**Cronbach's Alpha per Topik**")
                for t_k, a_v in cronbach.items():
                    lbl = ("✅ Good" if a_v >= 0.80 else
                           "✅ Acceptable" if a_v >= 0.70 else "⚠️ Low")
                    st.metric(t_k[:30], f"α = {a_v:.4f}", delta=lbl)

            with c2b:
                st.markdown("**Uji Normalitas Shapiro-Wilk**")
                if not df_shap.empty:
                    st.dataframe(
                        df_shap[['topik','soal','W_stat','p_value','keterangan']]
                        .rename(columns={'topik':'Topik','soal':'Soal',
                                         'W_stat':'W','p_value':'p',
                                         'keterangan':'Hasil'}),
                        use_container_width=True, hide_index=True
                    )

            df_sens = evaluasi.get('df_sensitivity', pd.DataFrame())
            if not df_sens.empty:
                st.markdown("**Threshold Sensitivity Analysis**")
                st.dataframe(df_sens, use_container_width=True, hide_index=True)

        with st.expander("🔎 Kelompok 3 — Semantic Validity"):
            c3a, c3b = st.columns(2)
            with c3a:
                fig_sv, ax_sv = plt.subplots(figsize=(5, 3.5))
                ax_sv.scatter(df['panjang_jawaban_siswa'], df['skor_aes'],
                              alpha=0.3, s=10, color='#3b82f6')
                m_sv, b_sv = np.polyfit(df['panjang_jawaban_siswa'],
                                        df['skor_aes'], 1)
                x_sv = np.linspace(df['panjang_jawaban_siswa'].min(),
                                   df['panjang_jawaban_siswa'].max(), 100)
                ax_sv.plot(x_sv, m_sv*x_sv+b_sv, color='#dc2626', lw=2)
                ax_sv.set_title(f'Skor vs Panjang Teks (r={r_panj:.4f})',
                                fontweight='bold', fontsize=10)
                ax_sv.set_xlabel('Panjang Jawaban (kata)')
                ax_sv.set_ylabel('Skor AES')
                sns.despine(ax=ax_sv)
                st.pyplot(fig_sv, use_container_width=True)
                plt.close(fig_sv)
                st.info("✅ Tidak bias panjang teks" if abs(r_panj) < 0.40
                        else "⚠️ Potensi bias panjang teks")

            with c3b:
                if 'jaccard_similarity' in df.columns:
                    fig_jac, ax_jac = plt.subplots(figsize=(5, 3.5))
                    ax_jac.scatter(df['jaccard_similarity'],
                                   df['cosine_similarity'],
                                   alpha=0.3, s=10, color='#f59e0b')
                    m_j, b_j = np.polyfit(df['jaccard_similarity'],
                                          df['cosine_similarity'], 1)
                    x_j = np.linspace(df['jaccard_similarity'].min(),
                                      df['jaccard_similarity'].max(), 100)
                    ax_jac.plot(x_j, m_j*x_j+b_j, color='#dc2626', lw=2)
                    ax_jac.set_title(f'Cosine vs Jaccard (r={r_jac:.4f})',
                                     fontweight='bold', fontsize=10)
                    ax_jac.set_xlabel('Jaccard Similarity')
                    ax_jac.set_ylabel('Cosine Similarity')
                    sns.despine(ax=ax_jac)
                    st.pyplot(fig_jac, use_container_width=True)
                    plt.close(fig_jac)

                st.markdown("**Deteksi Jawaban Redundan (Plagiarisme)**")
                if len(df_redun) == 0:
                    st.success("✅ Tidak ada jawaban redundan terdeteksi.")
                else:
                    st.warning(f"⚠️ {len(df_redun)} pasangan jawaban sangat mirip.")
                    st.dataframe(df_redun, use_container_width=True, hide_index=True)

        st.markdown("---")
        ringkasan_eval = {
            'discriminative_gap': emb_q.get('discriminative_gap', 0),
            'mean_cohesion'     : emb_q.get('mean_cohesion', 0),
            'cronbach_alpha'    : cronbach,
            'r_skor_panjang'    : r_panj,
            'r_jaccard_cosine'  : r_jac,
            'n_redundan'        : len(df_redun),
        }
        st.download_button(
            "⬇️ Download Ringkasan Evaluasi (.json)",
            data=json.dumps(ringkasan_eval, ensure_ascii=False, indent=2),
            file_name="ringkasan_evaluasi.json",
            mime="application/json",
            key="dl_eval_json"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pipeline, err = load_pipeline("aes_pipeline_final.pkl")

    if err:
        st.error(f"❌ Gagal memuat pipeline: {err}")
        st.markdown("""
        **Pastikan:**
        1. File `aes_pipeline.pkl` ada di folder yang sama dengan `app.py`
        2. Dihasilkan dari notebook `TA_IndoBERT_Cosine_v4.ipynb` Section 15
        3. Jalankan: `streamlit run app.py`
        """)
        st.stop()

    tokenizer, model, device = load_indobert(pipeline['model_name'])
    page = render_sidebar(pipeline)

    if page == "🔍 Penilaian Jawaban":
        page_inference(pipeline, tokenizer, model, device)
    else:
        page_dashboard(pipeline)


if __name__ == "__main__":
    main()