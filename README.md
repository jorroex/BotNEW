# 🤖 Bot de Telegram + Real-ESRGAN (anime_6B)

Mejora imágenes de anime con **Real-ESRGAN (x4plus_anime_6B)**.

## 🚀 Instalación
```bash
pip install -r requirements.txt
```

## 🔑 Variables de entorno
- `TELEGRAM_TOKEN`: token del bot (BotFather)

## ▶️ Ejecución
```bash
export TELEGRAM_TOKEN=tu_token
python bot.py
```

El script descarga automáticamente el modelo `RealESRGAN_x4plus_anime_6B.pth`
en `experiments/pretrained_models/` si no existe.

> Nota: El modelo anime usa RRDBNet con **6 bloques**.
