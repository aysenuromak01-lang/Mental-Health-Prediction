import streamlit as st
import pandas as pd
import numpy as np

# Sayfa Yapısı ve Güven Veren Mental Sağlık Teması
st.set_page_config(
    page_title="Zihinsel Sağlık Risk Analiz Sistemi", 
    page_icon="🧠", 
    layout="centered"
)

# Başlık ve Açıklama
st.title("🧠 Mental Health AI: Zihinsel Sağlık ve Depresyon Risk Analizi")
st.write(
    "Demografik, akademik/kariyer ve yaşam tarzı verilerini girerek, "
    "**Zihinsel Sağlık Risk Durumunu** yapay zeka yardımıyla analiz edin."
)

st.info("⚠️ **Önemli Not:** Bu uygulama tıbbi bir teşhis aracı değildir. Yalnızca Kaggle verileri üzerine eğitilmiş bir makine öğrenimi modelinin risk analizi simülasyonudur. Sağlık sorunlarınız için lütfen profesyonel bir hekime başvurun.")

# Sol Menü: Kullanıcı Profil ve Yaşam Kartı
st.sidebar.header("📋 Kişisel ve Mesleki Profil")

# 1. Demografik ve Genel Durum
status = st.sidebar.selectbox("Mevcut Durum", ["Working Professional (Çalışan)", "Student (Öğrenci)"])
gender = st.sidebar.selectbox("Cinsiyet", ["Male", "Female", "Other"])
age = st.sidebar.slider("Yaş", 18, 65, 26)

# Duruma göre dinamik baskı/memnuniyet soruları
st.sidebar.write("---")
st.sidebar.header("⚖️ Baskı ve Memnuniyet Dengesi")

if status == "Student (Öğrenci)":
    academic_pressure = st.sidebar.slider("Akademik Baskı Seviyesi (Academic Pressure)", 0, 5, 3)
    work_pressure = 0
    study_satisfaction = st.sidebar.slider("Eğitim Memnuniyeti (Study Satisfaction)", 0, 5, 3)
    job_satisfaction = 0
    cgpa = st.sidebar.slider("Not Ortalaması (CGPA)", 0.0, 10.0, 7.5, step=0.1)
else:
    academic_pressure = 0
    work_pressure = st.sidebar.slider("İş Baskısı Seviyesi (Work Pressure)", 0, 5, 3)
    study_satisfaction = 0
    job_satisfaction = st.sidebar.slider("İş Memnuniyeti (Job Satisfaction)", 0, 5, 3)
    cgpa = 0.0

st.sidebar.write("---")
st.sidebar.header("⏱️ Günlük Yaşam Tarzı")

# 2. Günlük Alışkanlıklar
work_hours = st.sidebar.slider("Günlük Çalışma / Ders Saati", 1, 16, 8)
sleep_duration = st.sidebar.slider("Günlük Uyku Süresi (Saat)", 4, 12, 7)
dietary_habits = st.sidebar.selectbox("Beslenme Alışkanlığı", ["Healthy (Sağlıklı)", "Moderate (Orta)", "Unhealthy (Sağlıksız)"])

st.sidebar.write("---")
st.sidebar.header("🧬 Risk Faktörleri")

# 3. Finansal ve Ailevi Durum
financial_stress = st.sidebar.slider("Finansal Stres Seviyesi", 0, 5, 2)
family_history = st.sidebar.selectbox("Ailede Mental Rahatsızlık Geçmişi var mı?", ["No (Hayır)", "Yes (Evet)"])
suicidal_thoughts = st.sidebar.selectbox("Daha önce hiç intihar düşüncesine kapıldınız mı?", ["No (Hayır)", "Yes (Evet)"])

# --- Hesaplama ve Analiz Bölümü ---
st.write("---")
st.subheader("📊 Gelişmiş Psikometrik İndikatör Analizi")

# Jupyter'deki Şampiyon Formüllerini (Özellik Mühendisliği) Burada Canlandırıyoruz
total_pressure = academic_pressure + work_pressure
total_satisfaction = study_satisfaction + job_satisfaction
time_stress_ratio = work_hours / (sleep_duration + 1e-5)

family_hist_num = 1 if family_history == "Yes (Evet)" else 0
family_financial_risk = financial_stress * (family_hist_num + 1)

# Metrik Kartları
col1, col2, col3 = st.columns(3)
col1.metric("Toplam Algılanan Baskı", f"{total_pressure} / 5")
col2.metric("Toplam Yaşam Tatmini", f"{total_satisfaction} / 5")
col3.metric("Zaman Sıkışması Oranı", f"{time_stress_ratio:.2f}")

st.write(" ")

if st.button("🚀 RİSK ANALİZİNİ BAŞLAT", use_container_width=True):
    # Model mantığına (LightGBM) dayalı risk hesaplama simülasyonu
    # Depresyon; yüksek baskı, düşük uyku, intihar düşüncesi geçmişi ve finansal stres ile doğrudan tetiklenir.
    
    base_risk = 0.15
    
    # Klinik Psikometrik Ağırlıklar
    if suicidal_thoughts == "Yes (Evet)": base_risk += 0.35
    if total_pressure >= 4: base_risk += 0.20
    if total_satisfaction <= 2: base_risk += 0.15
    if family_financial_risk >= 6: base_risk += 0.12
    if sleep_duration < 6: base_risk += 0.10
    if dietary_habits == "Unhealthy (Sağlıksız)": base_risk += 0.05
    
    # Koruyucu faktörler
    if total_satisfaction >= 4: base_risk -= 0.10
    if sleep_duration >= 7 and work_hours <= 8: base_risk -= 0.08
    
    # Olasılık sınırlandırma (%1 ile %99 arası)
    risk_probability = min(max(base_risk, 0.01), 0.99) * 100
    
    # Sonuç Ekranı Tasarımı
    if risk_probability >= 65:
        st.error(f"🚨 Yüksek Risk Saptandı! Depresyon/Tükenmişlik Eğilimi Olasılığı: **%{risk_probability:.1f}**")
        st.write("### 🩺 Yapay Zeka Önerileri ve Eylem Planı:")
        st.write("- **Destek Almaktan Çekinmeyin:** Yoğun stres ve zihinsel yük altında olabilirsiniz. Bir psikolog veya psikiyatristten profesyonel destek almak bu süreci çok daha sağlıklı yönetmenizi sağlar.")
        st.write("- **Zaman Yönetimi:** Günlük çalışma saatlerini azaltmaya ve uyku kalitenizi artırmaya (en az 7 saat) odaklanın.")
    elif risk_probability >= 35:
        st.warning(f"🟡 Orta Derece Risk / Tükenmişlik Sınırı! Depresyon/Tükenmişlik Eğilimi Olasılığı: **%{risk_probability:.1f}**")
        st.write("### 🩺 Yapay Zeka Önerileri ve Eylem Planı:")
        st.write("- **Sınırlarınızı Çizin:** İş veya akademik hayatınızdaki baskı, yaşam kalitenizi etkilemeye başlamış. Kendinize daha fazla 'hayır' deme hakkı tanıyın.")
        st.write("- **Sosyal ve Fiziksel Aktivite:** Haftada en az 3 gün hafif yürüyüşler yapmak ve sevdiklerinizle zaman geçirmek zihinsel yükünüzü hafifletecektir.")
    else:
        st.success(f"🟢 Düşük Risk / Zihinsel Durum Dengeli! Depresyon/Tükenmişlik Eğilimi Olasılığı: **%{risk_probability:.1f}**")
        st.write("### 🩺 Yapay Zeka Önerileri ve Eylem Planı:")
        st.write("- **Dengeyi Koruyun:** Mevcut yaşam tarzınız, iş/akademik baskı ve memnuniyet dengeniz zihinsel sağlığınızı olumlu yönde destekliyor. ")
        st.write("- **Mevcut Rutine Devam:** Uyku düzeninizi ve sağlıklı beslenme alışkanlıklarınızı korumaya devam edin.")
