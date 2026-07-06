"""阶段1：流式解码 + 粗采样文本检测。

单遍读完整个视频（裁剪为画面下方区域、缩至半分辨率），
- 每帧存一张 1/8 缩略图（memmap，供阶段2做逐帧边界精确定位）
- 每隔 sample_step 帧做一次文本检测（多线程 worker）
"""
import os
import queue
import subprocess
import threading

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ffmpeg_exe, CROP_RATIO, SAMPLE_STEP_SEC, THUMB_SCALE, DET_SCALE, MODELS_DIR

CREATE_NO_WINDOW = 0x08000000
_ENGINE_LOCK = threading.Lock()


def make_det_engine():
    from rapidocr import RapidOCR
    with _ENGINE_LOCK:  # 引擎初始化可能触发模型下载，避免并发写坏缓存
        return RapidOCR(params={
            "Rec.lang_type": "japan",
            "Global.model_root_dir": os.path.join(MODELS_DIR, "rapidocr"),
            "Global.log_level": "error",
            "Det.limit_type": "max",
            "Det.limit_side_len": 960,
            "Global.min_height": 12,
        })


def run_detect(video, info, workdir, progress=None, cancel=None, n_workers=4):
    """返回 dict：{n_frames, crop_y, crop_h, samples: [[idx, [[x1,y1,x2,y2],...]], ...]}
    坐标为整帧全分辨率绝对坐标。同时写出 thumbs.dat。"""
    W, H = info["width"], info["height"]
    crop_h = int(H * CROP_RATIO) // 2 * 2
    crop_y = H - crop_h
    half_w, half_h = W // DET_SCALE, crop_h // DET_SCALE
    th_w, th_h = W // THUMB_SCALE, crop_h // THUMB_SCALE
    step = max(1, round(SAMPLE_STEP_SEC * info["fps"]))

    thumbs_path = os.path.join(workdir, "thumbs.dat")
    frame_bytes = half_w * half_h * 3

    cmd = [ffmpeg_exe(), "-v", "error", "-i", video,
           "-vf", f"crop={W}:{crop_h}:0:{crop_y},scale={half_w}:{half_h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=frame_bytes * 8, creationflags=CREATE_NO_WINDOW)

    det_in = queue.Queue(maxsize=n_workers * 6)
    results = {}
    res_lock = threading.Lock()
    stop_flag = threading.Event()

    def worker():
        eng = make_det_engine()
        while not stop_flag.is_set():
            item = det_in.get()
            if item is None:
                det_in.task_done()
                break
            idx, frame = item
            try:
                out = eng(frame, use_det=True, use_cls=False, use_rec=False)
                boxes = []
                if out.boxes is not None:
                    for b in out.boxes:
                        x1, y1 = b.min(axis=0) * DET_SCALE
                        x2, y2 = b.max(axis=0) * DET_SCALE
                        boxes.append([int(x1), int(y1 + crop_y), int(x2), int(y2 + crop_y)])
                with res_lock:
                    results[idx] = boxes
            except Exception:
                with res_lock:
                    results[idx] = []
            det_in.task_done()

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for w_ in workers:
        w_.start()

    n_est = info["n_frames_est"] or 1
    idx = 0
    try:
        with open(thumbs_path, "wb") as tf:
            while True:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("cancelled")
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                frame = np.frombuffer(buf, np.uint8).reshape(half_h, half_w, 3)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tf.write(cv2.resize(gray, (th_w, th_h)).tobytes())
                if idx % step == 0:
                    det_in.put((idx, frame.copy()))
                if progress and idx % 300 == 0:
                    progress(min(idx / n_est, 0.999), f"扫描帧 {idx}/{n_est}")
                idx += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        for _ in workers:
            det_in.put(None)
        for w_ in workers:
            w_.join(timeout=120)
        stop_flag.set()

    n_frames = idx
    samples = [[i, results.get(i, [])] for i in sorted(results)]
    return {
        "n_frames": n_frames, "step": step,
        "crop_y": crop_y, "crop_h": crop_h,
        "thumb_w": th_w, "thumb_h": th_h,
        "samples": samples,
    }


def open_thumbs(workdir, det_meta):
    """以 memmap 打开阶段1的逐帧缩略图。"""
    path = os.path.join(workdir, "thumbs.dat")
    n, th, tw = det_meta["n_frames"], det_meta["thumb_h"], det_meta["thumb_w"]
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(n, th, tw))
