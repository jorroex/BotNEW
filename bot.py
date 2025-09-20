import os
import uuid
import cv2
import urllib.request
import torch
import nest_asyncio
import asyncio
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

# --- Parche automático basicsr para TorchVision recientes ---
import importlib.util
_spec = importlib.util.find_spec("basicsr")
if _spec and _spec.submodule_search_locations:
    _deg_path = os.path.join(_spec.submodule_search_locations[0], "data", "degradations.py")
    if os.path.exists(_deg_path):
        with open(_deg_path, "r", encoding="utf-8") as _f:
            _code = _f.read()
        if "functional_tensor" in _code:
            _code = _code.replace(
                "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                "from torchvision.transforms.functional import rgb_to_grayscale"
            )
            with open(_deg_path, "w", encoding="utf-8") as _f:
                _f.write(_code)
            print("✅ Parche basicsr aplicado en:", _deg_path)
# --- fin parche ---

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from torch.backends import cudnn

# ===============================
# Configuración del modelo
# ===============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("🖥️ Usando dispositivo:", DEVICE)

if DEVICE == 'cuda':
    cudnn.benchmark = True  # acelerar convoluciones con tamaños estables
else:
    # Acelerar CPU usando más hilos (Colab suele tener 2 vCPU)
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    except Exception:
        pass

MODEL_DIR = os.path.join("experiments", "pretrained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "RealESRGAN_x4plus_anime_6B.pth")
MODEL_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"

# Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    print("⬇️ Descargando modelo (anime_6B)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Modelo descargado:", MODEL_PATH)

# RRDBNet con 6 bloques (anime_6B)
model_rrdb = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64,
    num_block=6,  # clave para anime_6B
    num_grow_ch=32, scale=4
)

# Tiles: GPU -> más grandes (menos overhead); CPU -> sin tiles (si hay RAM) para evitar coste de mosaico
DEFAULT_TILE = 400 if DEVICE == 'cuda' else 0

upscaler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model_rrdb,
    tile=DEFAULT_TILE,
    tile_pad=10,
    pre_pad=0,
    half=(DEVICE == 'cuda'),
    device=DEVICE
)

def enhance_autotile(img, outscale=4):
    """Intenta con el tile actual; si OOM, reduce automáticamente."""
    # Secuencia de fallback de tiles
    candidates = [upscaler.tile] if upscaler.tile else [0]
    # Si falla, probamos valores más pequeños
    candidates += [300, 200, 120, 100, 60]
    tried = set()
    last_err = None
    for t in candidates:
        if t in tried: 
            continue
        tried.add(t)
        upscaler.tile = t
        try:
            return upscaler.enhance(img, outscale=outscale)
        except RuntimeError as e:
            msg = str(e).lower()
            last_err = e
            # Si no es por memoria, no tiene sentido seguir intentando
            if "out of memory" not in msg and "cuda" not in msg:
                raise
            print(f"⚠️ OOM con tile={t}, probando uno más pequeño...")
    # Si llegamos aquí, no hubo forma
    raise last_err if last_err else RuntimeError("Fallo de inferencia")

# ===============================
# Configuración del bot
# ===============================
TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("❌ Define la variable de entorno TELEGRAM_TOKEN.")

async def procesar_imagen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.photo:
            return
        tg_file = await context.bot.get_file(update.message.photo[-1].file_id)

        input_path = f"input_{uuid.uuid4().hex}.jpg"
        output_path = f"output_{uuid.uuid4().hex}.png"

        await tg_file.download_to_drive(input_path)
        await update.message.reply_text(
            f"⏳ Procesando en **{DEVICE.upper()}** (tile={upscaler.tile or 'full'})..."
        )

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            await update.message.reply_text("❌ No se pudo leer la imagen.")
            return

        try:
            result, _ = enhance_autotile(img, outscale=4)
        except Exception as e:
            await update.message.reply_text(f"⚠️ Error en Real-ESRGAN:\n{str(e)}")
            return

        cv2.imwrite(output_path, result)
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(output_path, "rb"))

    except Exception as e:
        await update.message.reply_text(f"⚠️ Error inesperado:\n{str(e)}")
    finally:
        for p in (locals().get("input_path"), locals().get("output_path")):
            try:
                if p and os.path.exists(p): os.remove(p)
            except Exception:
                pass

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, procesar_imagen))
    nest_asyncio.apply()
    print("✅ Bot listo. Envíale una foto en Telegram.")
    asyncio.run(app.run_polling())

if __name__ == "__main__":
    main()
