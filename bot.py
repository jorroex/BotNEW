import os
import uuid
import cv2
import urllib.request
import torch
import nest_asyncio
import asyncio
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ===============================
# Configuración del modelo
# ===============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('🖥️ Usando dispositivo:', DEVICE)

MODEL_DIR = os.path.join('experiments', 'pretrained_models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'RealESRGAN_x4plus_anime_6B.pth')
MODEL_URL  = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'

# Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    print('⬇️ Descargando modelo (anime_6B)...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('✅ Modelo descargado:', MODEL_PATH)

# RRDBNet con 6 bloques (anime_6B)
model_rrdb = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64,
    num_block=6,  # anime_6B usa 6 bloques
    num_grow_ch=32, scale=4
)

upscaler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model_rrdb,
    tile=200,          # si falta VRAM, baja a 120/100/60
    tile_pad=10,
    pre_pad=0,
    half=(DEVICE == 'cuda'),
    device=DEVICE
)

# ===============================
# Configuración del bot
# ===============================
TOKEN = os.environ.get('TELEGRAM_TOKEN')
if not TOKEN:
    raise ValueError('❌ Define la variable de entorno TELEGRAM_TOKEN.')

async def procesar_imagen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.photo:
            return
        file = await context.bot.get_file(update.message.photo[-1].file_id)

        # Archivos temporales únicos
        input_path = f'input_{uuid.uuid4().hex}.jpg'
        output_path = f'output_{uuid.uuid4().hex}.png'

        await file.download_to_drive(input_path)
        await update.message.reply_text('⏳ Procesando tu imagen con Real-ESRGAN...')

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            await update.message.reply_text('❌ No se pudo leer la imagen.')
            return

        try:
            result, _ = upscaler.enhance(img, outscale=4)
        except Exception as e:
            await update.message.reply_text(f'⚠️ Error en Real-ESRGAN:\n{str(e)}')
            return

        cv2.imwrite(output_path, result)
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(output_path, 'rb'))

    except Exception as e:
        await update.message.reply_text(f'⚠️ Error inesperado:\n{str(e)}')
    finally:
        # Limpieza
        try:
            if 'input_path' in locals() and os.path.exists(input_path): os.remove(input_path)
            if 'output_path' in locals() and os.path.exists(output_path): os.remove(output_path)
        except Exception:
            pass

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, procesar_imagen))
    nest_asyncio.apply()
    print('✅ Bot listo. Envíale una foto en Telegram.')
    asyncio.run(app.run_polling())

if __name__ == '__main__':
    main()