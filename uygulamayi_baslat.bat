@echo off
title ✈️ Yolcu Memnuniyeti Tahmin Uygulamasi
color 0b

echo.
echo ========================================================
echo   ✈️ Havayolu Yolcu Memnuniyeti - Makine Ogrenmesi
echo   CRISP-DM Deployment Simulasyonu Baslatiliyor...
echo ========================================================
echo.
echo Lutfen bekleyin, tarayici otomatik olarak acilacaktir...
echo (Bu siyah pencereyi uygulama acikken kapatmayin)
echo.

streamlit run app/streamlit_app.py

pause
