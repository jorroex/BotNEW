import os
import uuid
import cv2
import urllib.request
import torch
import nest_asyncio
import asyncio
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
from telegram.constants import ChatAction

# ===============================
# 0) Parche automático para basicsr (torchvision recientes)
# ===============================
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

# ===============================
# 1) Imports de Real-ESRGAN
# ===============================
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from torch.backends import cudnn

# ===============================
# 2) Configuración global
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🖥️ Usando dispositivo:", DEVICE)

if DEVICE == "cuda":
    cudnn.benchmark = True
else:
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    except Exception:
        pass

# Límite de Telegram para sendPhoto: 10 MB
MAX_TG_PHOTO = 10 * 1024 * 1024  # 10 MiB aprox.

# Dónde guardar / descargar el modelo
MODEL_DIR = os.path.join("experiments", "pretrained_models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "RealESRGAN_x4plus_anime_6B.pth")
MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
    "RealESRGAN_x4plus_anime_6B.pth"
)

# Descargar modelo si falta
if not os.path.exists(MODEL_PATH):
    print("⬇️ Descargando modelo (anime_6B)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Modelo descargado:", MODEL_PATH)

# Arquitectura correcta para anime_6B (RRDB con 6 bloques)
model_rrdb = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64,
    num_block=6,  # ¡clave para anime_6B!
    num_grow_ch=32, scale=4
)

# Estrategia de tiles:
# - En GPU: 400, 300, 200, 120, 100, 60
# - En CPU: 0 (full), 200, 120, 100, 60
TILE_CANDIDATES = [400, 300, 200, 120, 100, 60] if DEVICE == "cuda" else [0, 200, 120, 100, 60]
DEFAULT_TILE = TILE_CANDIDATES[0]

# Cache de instancias por tile (evita recargar pesos)
_UPSCALERS = {}
def get_upscaler(tile: int) -> RealESRGANer:
    if tile not in _UPSCALERS:
        _UPSCALERS[tile] = RealESRGANer(
            scale=4,
            model_path=MODEL_PATH,
            model=model_rrdb,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=(DEVICE == "cuda"),
            device=DEVICE
        )
    return _UPSCALERS[tile]

def enhance_autotile(img, outscale=4):
    """Prueba varios tiles si hay OOM. Devuelve (result, tile_usado)."""
    last_err = None
    for t in TILE_CANDIDATES:
        try:
            res, _ = get_upscaler(t).enhance(img, outscale=outscale)
            return res, t
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ OOM con tile={t}, probando uno más pequeño...")
                last_err = e
                continue
            raise
    raise last_err if last_err else RuntimeError("Fallo de inferencia")

# ===============================
# 3) Barra de progreso (mensaje editable)
# ===============================
async def start_progress(context: ContextTypes.DEFAULT_TYPE, chat_id: int, header="Procesando imagen…"):
    bars = ["▱▱▱▱▱", "▰▱▱▱▱", "▰▰▱▱▱", "▰▰▰▱▱", "▰▰▰▰▱", "▰▰▰▰▰"]
    msg = await context.bot.send_message(chat_id, f"{header}\n{bars[0]}")
    stop = asyncio.Event()

    async def updater():
        i = 0
        try:
            while not stop.is_set():
                i = (i + 1) % len(bars)
                try:
                    await msg.edit_text(f"{header}\n{bars[i]}")
                    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
                except Exception:
                    pass
                await asyncio.sleep(0.8)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(updater())
    return msg, stop, task

# ===============================
# 4) Bot
# ===============================
TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("❌ Define la variable de entorno TELEGRAM_TOKEN.")

async def procesar_imagen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    progress_msg = None
    progress_task = None
    stop_event = None
    try:
        if not update.message or not update.message.photo:
            return

        tg_file = await context.bot.get_file(update.message.photo[-1].file_id)
        input_path  = f"input_{uuid.uuid4().hex}.jpg"
        output_path = f"output_{uuid.uuid4().hex}.png"  # siempre PNG de alta calidad

        # Barra de progreso (no menciona CUDA ni tile)
        chat_id = update.effective_chat.id
        progress_msg, stop_event, progress_task = await start_progress(context, chat_id)

        await tg_file.download_to_drive(input_path)

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            await context.bot.send_message(chat_id, "❌ No se pudo leer la imagen.")
            return

        try:
            result, used_tile = enhance_autotile(img, outscale=4)
        except Exception as e:
            await context.bot.send_message(chat_id, f"⚠️ Error en Real-ESRGAN:\n{str(e)}")
            return

        # Guardar PNG (máxima calidad)
        cv2.imwrite(output_path, result)

        # Enviar:
        # - Si PNG <= 10 MB -> sendPhoto
        # - Si PNG  > 10 MB -> sendDocument (sin recomprimir)
        size_png = os.path.getsize(output_path)
        if size_png <= MAX_TG_PHOTO:
            with open(output_path, "rb") as f:
                await context.bot.send_photo(chat_id=chat_id, photo=f, caption="✅ Imagen mejorada")
        else:
            await context.bot.send_document(
                chat_id=chat_id,
                document=open(output_path, "rb"),
                filename=os.path.basename(output_path),
                caption="✅ Imagen mejorada (enviada como documento, >10 MB)"
            )

    except Exception as e:
        await update.message.reply_text(f"⚠️ Error inesperado:\n{str(e)}")
    finally:
        # Apagar barra de progreso
        try:
            if stop_event:
                stop_event.set()
            if progress_task:
                progress_task.cancel()
            if progress_msg:
                try:
                    await progress_msg.delete()
                except Exception:
                    try:
                        await progress_msg.edit_text("✅ Listo")
                    except Exception:
                        pass
        except Exception:
            pass

        # Limpieza de temporales
        for p in (locals().get("input_path"), locals().get("output_path")):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
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
