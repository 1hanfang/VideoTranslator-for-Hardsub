"""全局路径与常量配置。"""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
RUNTIME_DIR = os.path.join(ROOT, "runtime")
MODELS_DIR = os.path.join(ROOT, "models")
WORK_DIR = os.path.join(ROOT, "work")
UPLOAD_DIR = os.path.join(WORK_DIR, "uploads")
LOG_DIR = os.path.join(WORK_DIR, "logs")
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Videos", "OCRtranslator")  # 拖拽且找不到原路径时的兜底输出位置

SERVER_PORT = 8760
LLAMA_PORT = 8761

GGUF_NAME = "Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf"
GGUF_PATH = os.path.join(MODELS_DIR, GGUF_NAME)
GGUF_URL = ("https://hf-mirror.com/SakuraLLM/Sakura-GalTransl-7B-v3.7/"
            "resolve/main/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf")
GGUF_MIN_BYTES = 4_000_000_000  # 完整性粗校验

LLAMA_DIR = os.path.join(RUNTIME_DIR, "llama")
LLAMA_SERVER = os.path.join(LLAMA_DIR, "llama-server.exe")
LLAMA_ZIP_URL = ("https://github.com/ggml-org/llama.cpp/releases/download/"
                 "b9873/llama-b9873-bin-win-vulkan-x64.zip")

# ---- 流水线参数 ----
CROP_RATIO = 0.45          # 只在画面下方 45% 区域检测字幕
SAMPLE_STEP_SEC = 0.3      # 粗采样间隔（秒）
THUMB_SCALE = 8            # 逐帧缩略图相对原始分辨率的缩小倍数
DET_SCALE = 2              # 文本检测使用的缩小倍数（半分辨率）
MIN_EVENT_FRAMES = 5       # 短于此帧数的字幕事件丢弃
MERGE_GAP_FRAMES = 10      # 小于此间隔且内容相似的事件合并
OCR_MIN_CONF = 0.60        # 低于此置信度的行丢弃
TRANS_BATCH = 8            # 每次请求翻译的字幕条数（官方推荐 7-10）
TRANS_HISTORY = 5          # 携带的历史译文条数
LLAMA_CTX = 4096

def ffmpeg_exe() -> str:
    local = os.path.join(RUNTIME_DIR, "ffmpeg", "bin", "ffmpeg.exe")
    return local if os.path.exists(local) else "ffmpeg"

def ffprobe_exe() -> str:
    local = os.path.join(RUNTIME_DIR, "ffmpeg", "bin", "ffprobe.exe")
    return local if os.path.exists(local) else "ffprobe"

for _d in (WORK_DIR, UPLOAD_DIR, LOG_DIR, MODELS_DIR, RUNTIME_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)
