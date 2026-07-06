"""本地 Web 服务：前端页面 + 任务控制 API + WebSocket 进度推送。"""
import asyncio
import os
import subprocess
import sys
import threading
import time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import SERVER_PORT, UPLOAD_DIR, ROOT
from pipeline import probe as probe_mod
from pipeline.job import Job

app = FastAPI()
CREATE_NO_WINDOW = 0x08000000


def _is_under(path, parent):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(parent)]) == os.path.abspath(parent)
    except ValueError:
        return False


def _candidate_video(path, size, upload_dest):
    try:
        return (os.path.exists(path)
                and os.path.getsize(path) == size
                and os.path.abspath(path) != os.path.abspath(upload_dest)
                and not _is_under(path, os.path.join(ROOT, "work")))
    except OSError:
        return False


def _drive_roots():
    roots = []
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZC":
        root = f"{letter}:\\"
        if os.path.isdir(root):
            roots.append(root)
    return roots


def _find_original_file(filename, size, upload_dest, timeout=12.0):
    """Dragged files do not expose their full browser path, so try to relocate by name+size."""
    home = os.path.expanduser("~")
    root_drive = os.path.splitdrive(ROOT)[0] + "\\" if os.path.splitdrive(ROOT)[0] else ROOT
    common = [ROOT, root_drive, os.path.join(home, "Desktop"), os.path.join(home, "Downloads"),
              os.path.join(home, "Videos"), os.path.join(home, "Documents")]
    seen = set()
    roots = []
    for root in common + _drive_roots():
        if root and os.path.isdir(root):
            key = os.path.normcase(os.path.abspath(root))
            if key not in seen:
                seen.add(key)
                roots.append(root)

    for root in roots:
        p = os.path.join(root, filename)
        if _candidate_video(p, size, upload_dest):
            return p

    skip_names = {"$Recycle.Bin", "System Volume Information", "Windows", "Program Files",
                  "Program Files (x86)", "ProgramData", "AppData", "__pycache__"}
    skip_paths = [os.path.join(ROOT, p) for p in ("work", "runtime", "models", "output")]
    deadline = time.monotonic() + timeout
    for root in roots:
        for cur, dirs, files in os.walk(root, topdown=True, onerror=lambda e: None):
            if time.monotonic() > deadline:
                return None
            dirs[:] = [d for d in dirs
                       if d not in skip_names
                       and not any(_is_under(os.path.join(cur, d), p) for p in skip_paths)]
            if filename in files:
                p = os.path.join(cur, filename)
                if _candidate_video(p, size, upload_dest):
                    return p
    return None

_job: Job | None = None
_job_thread: threading.Thread | None = None
_job_lock = threading.Lock()


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.post("/api/pick")
def pick_file():
    """服务端弹出原生文件选择框（本地应用特权），拿到完整路径。"""
    ps = ("Add-Type -AssemblyName System.Windows.Forms;"
          "$f = New-Object System.Windows.Forms.OpenFileDialog;"
          "$f.Filter = '视频文件|*.mp4;*.mkv;*.avi;*.mov;*.ts;*.webm|所有文件|*.*';"
          "$f.Title = '选择要翻译的视频';"
          "if ($f.ShowDialog() -eq 'OK') { [Console]::Out.Write($f.FileName) }")
    try:
        out = subprocess.run(["powershell", "-STA", "-NoProfile", "-Command", ps],
                             capture_output=True, timeout=300,
                             creationflags=CREATE_NO_WINDOW)
        path = out.stdout.decode("gbk", "ignore").strip() or None
        if path and not os.path.exists(path):
            path = out.stdout.decode("utf-8", "ignore").strip() or None
    except Exception:
        path = None
    return {"path": path if path and os.path.exists(path) else None}


@app.post("/api/upload")
async def upload(file: UploadFile):
    """拖拽上传。若能在常见目录找到同名同大小的原文件则直接用原路径。"""
    filename = os.path.basename(file.filename)
    dest = os.path.join(UPLOAD_DIR, filename)
    tmp = dest + ".uploading"
    size = 0
    with open(tmp, "wb") as f:
        while chunk := await file.read(4 << 20):
            f.write(chunk)
            size += len(chunk)
    original = _find_original_file(filename, size, dest)
    if original:
        os.remove(tmp)
        return {"path": original, "relocated": True}
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        os.remove(tmp)  # 同名同大小：沿用旧副本，保持 mtime 不变以命中任务缓存
    else:
        os.replace(tmp, dest)
    return {"path": dest, "relocated": False}


class PathReq(BaseModel):
    path: str


@app.post("/api/probe")
def probe_video(req: PathReq):
    if not os.path.exists(req.path):
        return JSONResponse({"error": "文件不存在"}, status_code=404)
    try:
        return probe_mod.probe(req.path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


class StartReq(BaseModel):
    path: str
    src_lang: str = "auto"
    erase_mode: str = "fast"
    sub_mode: str = "replace"   # replace=替换原字幕 / append=原字幕旁加中文
    review: bool = False


@app.post("/api/start")
def start_job(req: StartReq):
    global _job, _job_thread
    with _job_lock:
        if _job and _job.snapshot()["status"] in ("running", "awaiting_review", "pending"):
            return JSONResponse({"error": "已有任务在运行"}, status_code=409)
        if not os.path.exists(req.path):
            return JSONResponse({"error": "文件不存在"}, status_code=404)
        _job = Job(req.path, {"src_lang": req.src_lang, "erase_mode": req.erase_mode,
                              "sub_mode": req.sub_mode, "review": req.review})
        _job_thread = threading.Thread(target=_job.run, daemon=True)
        _job_thread.start()
        return {"ok": True, "out_path": _job.out_path}


@app.post("/api/cancel")
def cancel_job():
    if _job:
        _job.cancel.set()
    return {"ok": True}


@app.get("/api/status")
def status():
    if _job is None:
        return {"status": "idle"}
    return _job.snapshot()


@app.get("/api/subs")
def get_subs():
    if _job is None:
        return []
    return _job.get_subs()


class SubsReq(BaseModel):
    trans: dict


@app.post("/api/subs")
def set_subs(req: SubsReq):
    if _job is None:
        return JSONResponse({"error": "没有任务"}, status_code=400)
    _job.set_trans(req.trans)
    return {"ok": True}


@app.post("/api/resume")
def resume_job():
    if _job is None:
        return JSONResponse({"error": "没有任务"}, status_code=400)
    _job.review_ready.set()
    return {"ok": True}


@app.websocket("/ws")
async def ws_status(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            snap = _job.snapshot() if _job else {"status": "idle"}
            await ws.send_json(snap)
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        pass


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    # 已有实例在运行（重复双击启动）：直接打开浏览器指向它
    try:
        import requests as _rq
        if _rq.get(f"http://127.0.0.1:{SERVER_PORT}/api/status", timeout=2).ok:
            print("程序已在运行，直接打开浏览器页面")
            webbrowser.open(f"http://127.0.0.1:{SERVER_PORT}")
            sys.exit(0)
    except Exception:
        pass
    if "--open-browser" in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(
            f"http://127.0.0.1:{SERVER_PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT, log_level="warning")
