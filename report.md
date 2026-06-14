# 📘 Analitik Rapor: Airline Passenger Satisfaction

## 1. Problem Tanımı
Havacılık sektöründe müşteri sadakati, işletmelerin temel gelir kaynağıdır. Memnun olmayan bir yolcunun rakip firmaya geçme olasılığının oldukça yüksek olması, müşteri memnuniyetsizliğinin proaktif olarak tespit edilmesini zorunlu kılmaktadır. 

## 2. CRISP-DM Akışı
Projemizde Endüstri Standardı olan CRISP-DM (Sektörler Arası Standart Veri Madenciliği Süreci) uygulanmıştır:
1. **İşin Anlaşılması**: Yanlış alarmden ziyade kaçırılan fırsatların (False Negative) maliyeti daha yüksek olarak belirlendi ve Recall metriği optimize edildi.
2. **Veriyi Anlama (EDA)**: Kategorik değişkenlerin dengesi çıkarıldı. `arrival_delay` sütunudaki çok küçük eksiklikler tespit edildi.
3. **Veri Hazırlama (DataPrep)**: 'Leakage'i engelleyecek şekilde eksikler, aykırılıklar log transformasyonu ile temizlenip OneHot ve Scale işlemleri yapıldı.
4. **Modelleme**: LogReg'den XGBoost'a 12 farklı ML ve DL algoritması denenmiştir.
5. **Değerlendirme**: Random Forest modeline karar verilmiştir.

## 3. Veri Analizi ve İçgörüler
* Sınıf dağılımında (Imbalance) 1.30x oranında fark vardır. SMOTE kullanımına gerek kalmadan class_weight stratejisi ile bu durum aşılmıştır.
* `Online Boarding`, `Inflight Entertainment` ve `Seat Comfort` metriklerinin genel memnuniyet puanı üzerinde çok güçlü etkileri olduğu saptanmıştır. Özellikle gecikmeler (delay) ile birleşince memnuniyetsizlik dramatik şekilde artmaktadır.

## 4. Model Sonuçları
Tüm algoritmalar çapraz doğrulama mantığı ve test setinde kıyaslandığında **Random Forest**:
- F1 Score: 0.9536
- AUC Score: 0.9932 
performansı yakalayarak belirlenen tüm metrik parametrelerini beklentilerin üzerinde karşılamıştır.

## 5. Riskler ve Sınırlılıklar
- Model geçmişe dönük uçuş sonu anket verilerine dayanmaktadır. Gerçek zamanlı uyarı sistemlerinde (in-flight), sensörlerden / uçak içi panellerden gelecek dataya adapte edilmesi gerekebilir. 
- Aşırı büyük rötarlardaki (outlier durumlar) yolcu psikolojisi düzlemsel bir seyir izlemediği için model bu grupta küçük tolerans hatalarına sahiptir.

## 6. Karar Önerileri ve Aksiyon Planı
1. **Erken Uyarı Sistemi ve Müdahale:** Gecikme yaşayan veya skor profili düşük seyreden First / Business Class yolcuları için, uçuş esnasında proaktif davranılarak kabin içi internet (wifi) ücretsiz olarak tahsis edilmelidir.
2. **Kişiselleştirilmiş İyileştirme:** Model feature analizinde en hassas ağırlığın "Online Boarding" üzerine olduğu tespit edilmiştir. Şirketin dijital check-in / online boarding deneyimi acilen modernize edilmeli, varsa buglar temizlenmelidir.
