"""首次运行引导：仅用标准库，负责创建 venv、装依赖、下载 ffmpeg/llama.cpp/模型，
然后拉起 Web 服务并打开浏览器。由 启动.bat 调用，也可手动执行。"""
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

try:  # 控制台编码保险：GBK 控制台打印中文不崩溃
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
RUNTIME = os.path.join(ROOT, "runtime")
MODELS = os.path.join(ROOT, "models")
VENV_PY = os.path.join(RUNTIME, "venv", "Scripts", "python.exe")
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
REQ = os.path.join(APP, "requirements.txt")
REQ_MARK = os.path.join(RUNTIME, "venv", ".req_hash")

GGUF = os.path.join(MODELS, "Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf")
GGUF_URLS = [
    "https://hf-mirror.com/SakuraLLM/Sakura-GalTransl-7B-v3.7/resolve/main/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf",
    "https://huggingface.co/SakuraLLM/Sakura-GalTransl-7B-v3.7/resolve/main/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf",
]
GGUF_MIN = 4_000_000_000

LLAMA_DIR = os.path.join(RUNTIME, "llama")
LLAMA_EXE = os.path.join(LLAMA_DIR, "llama-server.exe")
LLAMA_URLS = [
    "https://github.com/ggml-org/llama.cpp/releases/download/b9873/llama-b9873-bin-win-vulkan-x64.zip",
    "https://ghproxy.net/https://github.com/ggml-org/llama.cpp/releases/download/b9873/llama-b9873-bin-win-vulkan-x64.zip",
]

FFMPEG_URLS = [
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip",
]

SERVER_PORT = 8760


def log(msg):
    print(f"[setup] {msg}", flush=True)


def download(urls, dest, min_bytes=1, tries_per_url=30):
    """带断点续传与多源回退的下载。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for url in urls:
        for attempt in range(tries_per_url):
            have = os.path.getsize(part) if os.path.exists(part) else 0
            try:
                req = urllib.request.Request(url)
                if have:
                    req.add_header("Range", f"bytes={have}-")
                with urllib.request.urlopen(req, timeout=30) as r:
                    total = have + int(r.headers.get("Content-Length") or 0)
                    mode = "ab" if have and r.status == 206 else "wb"
                    if mode == "wb":
                        have = 0
                    with open(part, mode) as f:
                        t0 = time.time()
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            have += len(chunk)
                            if time.time() - t0 > 3:
                                t0 = time.time()
                                pct = f"{have / total * 100:.1f}%" if total else f"{have >> 20}MB"
                                log(f"下载 {os.path.basename(dest)}: {pct}")
                if have >= min_bytes:
                    os.replace(part, dest)
                    return True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"下载中断（{type(e).__name__}），3 秒后重试 [{attempt + 1}/{tries_per_url}]")
                time.sleep(3)
    return False


def ensure_venv():
    global VENV_PY
    if not os.path.exists(VENV_PY):
        log("创建 Python 虚拟环境…")
        subprocess.check_call([sys.executable, "-m", "venv", os.path.join(RUNTIME, "venv")])
    req_hash = hashlib.md5(open(REQ, "rb").read()).hexdigest()
    old = open(REQ_MARK).read().strip() if os.path.exists(REQ_MARK) else ""
    if old != req_hash:
        log("安装 Python 依赖（首次约 2-5 分钟）…")
        subprocess.check_call([VENV_PY, "-m", "pip", "install", "-r", REQ,
                               "-i", PIP_MIRROR, "--quiet"])
        with open(REQ_MARK, "w") as f:
            f.write(req_hash)


def ensure_ffmpeg():
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    local = os.path.join(RUNTIME, "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.exists(local):
        return
    log("本机没有 ffmpeg，下载中（约 80MB）…")
    zpath = os.path.join(RUNTIME, "ffmpeg.zip")
    if not download(FFMPEG_URLS, zpath, min_bytes=20_000_000):
        raise RuntimeError("ffmpeg 下载失败，请检查网络后重新运行")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        top = names[0].split("/")[0]
        z.extractall(RUNTIME)
    src = os.path.join(RUNTIME, top)
    dst = os.path.join(RUNTIME, "ffmpeg")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.rename(src, dst)
    os.remove(zpath)


def ensure_llama():
    if os.path.exists(LLAMA_EXE):
        return
    zpath = os.path.join(RUNTIME, "llama-vulkan.zip")
    if not (os.path.exists(zpath) and os.path.getsize(zpath) > 10_000_000):
        log("下载 llama.cpp 推理引擎（约 32MB）…")
        if not download(LLAMA_URLS, zpath, min_bytes=10_000_000):
            raise RuntimeError("llama.cpp 下载失败，请检查网络后重新运行")
    log("解压 llama.cpp …")
    os.makedirs(LLAMA_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(LLAMA_DIR)
    if not os.path.exists(LLAMA_EXE):  # zip 可能有一层目录
        for root, _, files in os.walk(LLAMA_DIR):
            if "llama-server.exe" in files and root != LLAMA_DIR:
                for f in os.listdir(root):
                    shutil.move(os.path.join(root, f), LLAMA_DIR)
                break
    if os.path.exists(LLAMA_EXE):
        try:
            os.remove(zpath)  # 解压成功后安装包无用
        except OSError:
            pass


def ensure_gguf():
    if os.path.exists(GGUF) and os.path.getsize(GGUF) >= GGUF_MIN:
        return
    log("下载翻译模型 Sakura-GalTransl-7B（约 4.2GB，视网速需要几分钟到几十分钟，"
        "支持断点续传，中断后重新运行即可继续）…")
    # 兼容早期直接下到目标名的半成品
    if os.path.exists(GGUF) and os.path.getsize(GGUF) < GGUF_MIN:
        os.replace(GGUF, GGUF + ".part")
    if not download(GGUF_URLS, GGUF, min_bytes=GGUF_MIN):
        raise RuntimeError("翻译模型下载失败，请检查网络后重新运行")


def ensure_ocr_models():
    mark = os.path.join(MODELS, "rapidocr", ".ready")
    if os.path.exists(mark):
        return
    log("下载 OCR 模型（首次约 30MB）…")
    code = ("import numpy as np\n"
            "from rapidocr import RapidOCR\n"
            f"root = r'{os.path.join(MODELS, 'rapidocr')}'\n"
            "for lang in ('japan', 'en'):\n"
            "    eng = RapidOCR(params={'Rec.lang_type': lang, 'Global.model_root_dir': root,\n"
            "                           'Global.log_level': 'error'})\n"
            "    eng(np.full((48, 320, 3), 255, np.uint8), use_det=True, use_cls=False, use_rec=True)\n")
    subprocess.check_call([VENV_PY, "-c", code])
    with open(mark, "w") as f:
        f.write("ok")


def main():
    os.chdir(ROOT)
    if sys.version_info < (3, 9):
        log("需要 Python 3.9 及以上版本")
        sys.exit(1)
    ensure_venv()
    ensure_ffmpeg()
    ensure_llama()
    ensure_gguf()
    ensure_ocr_models()
    log(f"启动服务 http://127.0.0.1:{SERVER_PORT} …")
    subprocess.call([VENV_PY, os.path.join(APP, "main.py"), "--open-browser"])


if __name__ == "__main__":
    main()
