import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# Sayfa Ayarları
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide"
)

# Yolların Dinamik Çözümlenmesi (app/ klasörü içinde çalıştığını varsayarak)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_random_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Model ve Scaler Yükleme
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu. Yollar kontrol edilmeli.\n{e}")
        return None, None

model, scaler = load_models()

# Servis Sütunları Listesi (Ortalamalar için)
SERVICE_COLS = [
    'inflight_wifi_service', 'departure_arrival_time_convenient', 'ease_of_online_booking',
    'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort', 'inflight_entertainment',
    'onboard_service', 'leg_room_service', 'baggage_handling', 'checkin_service', 'inflight_service', 'cleanliness'
]

def preprocess_input(user_data, model, scaler):
    # 1. DataFrame'e Çevir
    df = pd.DataFrame([user_data])
    
    # 2. Log1p Dönüşümleri
    df["departure_delay_in_minutes_log"] = np.log1p(df["departure_delay_in_minutes"])
    df["arrival_delay_in_minutes_log"] = np.log1p(df["arrival_delay_in_minutes"])
    df["flight_distance_log"] = np.log1p(df["flight_distance"])
    
    # 3. Kategori Sütunlarını Ayarlama & Encoding Hazırlığı
    # One-hot encoding için pandas dummylerini oluşturuyoruz
    cat_features = ["Gender", "customer_type", "type_of_travel", "customer_class"]
    df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=False)
    
    # 4. Feature Engineering
    df_encoded["total_service_score"] = df_encoded[SERVICE_COLS].sum(axis=1)
    df_encoded["avg_service_score"] = df_encoded[SERVICE_COLS].mean(axis=1)
    df_encoded["digital_score"] = df_encoded[["inflight_wifi_service", "ease_of_online_booking", "online_boarding"]].mean(axis=1)
    df_encoded["comfort_score"] = df_encoded[["seat_comfort", "leg_room_service", "cleanliness"]].mean(axis=1)
    df_encoded["total_delay"] = df_encoded["departure_delay_in_minutes"] + df_encoded["arrival_delay_in_minutes"]
    df_encoded["is_delayed"] = (df_encoded["total_delay"] > 0).astype(int)
    
    # Orijinal sütunları düşür
    drop_cols = ["departure_delay_in_minutes", "arrival_delay_in_minutes", "flight_distance"]
    df_encoded = df_encoded.drop(columns=drop_cols, errors="ignore")
    
    # 5. Modelin Beklediği Sütunlarla Eşitleme (Eksik Dummy'leri 0 yap, Fazlaları at)
    expected_cols = model.feature_names_in_
    # Reindex ile hizala, var olmayan sütunlara 0 bas
    df_aligned = df_encoded.reindex(columns=expected_cols, fill_value=0)
    
    # 6. StandardScaler Uygula
    num_scale = scaler.feature_names_in_
    df_aligned[num_scale] = scaler.transform(df_aligned[num_scale])
    
    return df_aligned

# ================= ARAYÜZ (UI) =================
st.title("✈️ Havayolu Yolcu Memnuniyeti Tahmin Sistemi")
st.markdown("""
Bu test arayüzü, arka planda eğitilmiş olan **Random Forest** modeli ile entegre çalışmaktadır. 
Müşteri özelliklerini girerek, seyahat sonunda müşterinin **Memnun (Satisfied)** mu yoksa **Memnuniyetsiz/Nötr (Neutral/Dissatisfied)** mi olacağını tahmin edebilirsiniz.
""")

# --- Sol Menü (Sidebar) Tasarımı ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3125/3125713.png", width=120) 
st.sidebar.title("✈️ Kontrol Paneli")
st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Proje Bilgileri")
st.sidebar.markdown("""
Bu arayüz, CRISP-DM metodolojisinin **Ürünleştirme (Deployment)** adımı kapsamında geliştirilmiştir. 
Sistemde eğitilmiş olan **Random Forest Classifier** modeli aktif olarak tahmin yapmaktadır.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performans Metrikleri")
col_m1, col_m2 = st.sidebar.columns(2)
col_m1.metric("Accuracy", "%96.2", "+0.5%")
col_m2.metric("F1-Score", "%95.8", "+0.3%")

st.sidebar.markdown("---")
st.sidebar.info("💡 **İpucu:** Sol taraftaki menü barından uygulamayı yönetebilir, ana ekranda ise müşteri özelliklerini girerek canlı tahmin yapabilirsiniz.")

# Form 
with st.form("prediction_form"):
    st.subheader("👤 1. Yolcu Demografisi ve Uçuş Bilgileri")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gender = st.selectbox("Cinsiyet (Gender)", ["Male", "Female"])
        age = st.number_input("Yaş (Age)", min_value=1, max_value=120, value=35)
    with col2:
        customer_type = st.selectbox("Müşteri Tipi", ["Loyal Customer", "disloyal Customer"])
        flight_distance = st.number_input("Uçuş Mesafesi (Mesafe)", min_value=10, max_value=15000, value=800)
    with col3:
        type_of_travel = st.selectbox("Seyahat Sebebi", ["Personal Travel", "Business travel"])
    with col4:
        customer_class = st.selectbox("Sınıf (Class)", ["Eco", "Eco Plus", "Business"])

    st.markdown("---")
    st.subheader("⭐ 2. Hizmet Puanları (0: Geçerli Değil, 1: Çok Kötü - 5: Çok İyi)")
    
    # 14 Servis Puanı
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        inflight_wifi_service = st.slider("Wifi Hizmeti", 0, 5, 3)
        departure_arrival_time_convenient = st.slider("Kalkış/Varış Saat Uygunluğu", 0, 5, 3)
        ease_of_online_booking = st.slider("Online Rezervasyon Kolaylığı", 0, 5, 3)
        gate_location = st.slider("Kapı Konumu", 0, 5, 3)
    with sc2:
        food_and_drink = st.slider("Yiyecek & İçecek", 0, 5, 3)
        online_boarding = st.slider("Online Boarding", 0, 5, 3)
        seat_comfort = st.slider("Koltuk Rahatlığı", 0, 5, 3)
        inflight_entertainment = st.slider("Uçak İçi Eğlence", 0, 5, 3)
    with sc3:
        onboard_service = st.slider("Kabin Hizmeti", 0, 5, 3)
        leg_room_service = st.slider("Diz Mesafesi", 0, 5, 3)
        baggage_handling = st.slider("Bagaj İşlemleri", 0, 5, 3)
    with sc4:
        checkin_service = st.slider("Check-in Hizmeti", 0, 5, 3)
        inflight_service = st.slider("Uçuş Hizmeti", 0, 5, 3)
        cleanliness = st.slider("Temizlik", 0, 5, 3)

    st.markdown("---")
    st.subheader("⏱️ 3. Gecikme Süreleri (Dakika)")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        departure_delay = st.number_input("Kalkış Gecikmesi (Departure Delay)", 0, 2000, 0)
    with col_d2:
        arrival_delay = st.number_input("Varış Gecikmesi (Arrival Delay)", 0, 2000, 0)
        
    submit_button = st.form_submit_button(label="🚀 Tahmin Et (Predict Satisfaction)")

# Tahmin Süreci
if submit_button:
    if model is None or scaler is None:
        st.error("Model dosyaları yüklenemedi. Lütfen 'models' klasöründe scaler.pkl ve best_model_random_forest.pkl olduğundan emin olun.")
    else:
        # Kullanıcı girdilerini sözlük formatında toplama
        user_input = {
            "Gender": gender,
            "customer_type": customer_type,
            "age": age,
            "type_of_travel": type_of_travel,
            "customer_class": customer_class,
            "flight_distance": flight_distance,
            "inflight_wifi_service": inflight_wifi_service,
            "departure_arrival_time_convenient": departure_arrival_time_convenient,
            "ease_of_online_booking": ease_of_online_booking,
            "gate_location": gate_location,
            "food_and_drink": food_and_drink,
            "online_boarding": online_boarding,
            "seat_comfort": seat_comfort,
            "inflight_entertainment": inflight_entertainment,
            "onboard_service": onboard_service,
            "leg_room_service": leg_room_service,
            "baggage_handling": baggage_handling,
            "checkin_service": checkin_service,
            "inflight_service": inflight_service,
            "cleanliness": cleanliness,
            "departure_delay_in_minutes": departure_delay,
            "arrival_delay_in_minutes": arrival_delay
        }

        # Veri önişleme
        try:
            X_test_record = preprocess_input(user_input, model, scaler)
            
            # Tahmin
            prediction = model.predict(X_test_record)[0]
            probability = model.predict_proba(X_test_record)[0]
            
            st.markdown("---")
            st.header("🎯 Analiz Sonucu")
            
            # 1 = Satisfied, 0 = Neutral or Dissatisfied (Projende encoding bu şekilde yapıldı)
            if prediction == 1:
                st.success(f"**Sonuç:** Yolcu uçuş sonrasında **MEMNUN (Satisfied)** olacaktır! 🥳")
                st.info(f"Yolcunun memnun olma olasılığı: **%{probability[1]*100:.1f}**")
                st.balloons()
            else:
                st.error(f"**Sonuç:** Yolcu uçuş sonrasında **MEMNUN OLMAYACAK (Neutral or Dissatisfied)**. 😔")
                st.warning(f"Yolcunun memnuniyetsiz olma ihtimali (Risk): **%{probability[0]*100:.1f}**")
                
                # Proaktif Aksiyon Önerisi (Business Insight)
                st.markdown("### ⚠️ Operasyonel Aksiyon Önerisi")
                st.markdown("* Bu yolcu risk grubundadır. Uçuş öncesi ücretsiz WiFi veya Lounge erişimi gibi küçük bir jest planlanabilir.")
                
        except Exception as e:
            st.error(f"Veri önişleme veya tahmin sırasında hata oluştu: {str(e)}")