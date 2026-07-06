"""阶段5+6：擦除原字幕、绘制中文字幕、单遍重编码合成。

每个事件预计算：擦除掩码、临近干净帧的时域背景中值（静止像素直接回填，
残余动态像素用 TELEA 修复）、以及预光栅化的中文字幕 RGBA 覆盖层。
帧循环用线程池并行处理，写出端严格按帧序送入 ffmpeg(NVENC)。
"""
import bisect
import os
import queue
import subprocess
import sys
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ffmpeg_exe

CREATE_NO_WINDOW = 0x08000000

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]
_font_cache = {}
APPEND_FONT_SCALE = 0.50


def _font(size):
    if size not in _font_cache:
        for p in FONT_CANDIDATES:
            if os.path.exists(p):
                _font_cache[size] = ImageFont.truetype(p, size)
                break
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _wrap(text, font, max_w):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if font.getlength(cur + ch) > max_w and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines or [""]


def build_overlay(ev, zh_text, W, H, mode="replace"):
    """把译文光栅化为 RGBA 覆盖层，返回 (rect, bgr, alpha)。

    mode="replace"：画在原字幕位置；mode="append"：画在原字幕下方（放不下则上方）。"""
    st = ev.get("style", {})
    line_h = st.get("line_h", 40)
    size = int(min(max(round(line_h * 0.82), 16), 72))
    if mode == "append":
        size = int(min(max(round(line_h * APPEND_FONT_SCALE), 14), 58))
    font = _font(size)
    stroke = max(2, size // 12)
    fill = tuple(st.get("fill", [255, 255, 255])[::-1])       # BGR -> RGB
    outline = tuple(st.get("outline", [16, 16, 16])[::-1])

    x1, y1, x2, y2 = ev["bbox"]
    margin = 48
    max_w = min(W - 2 * margin, max((x2 - x1) * 1.25, W * 0.4))
    lines = _wrap(zh_text, font, max_w)
    gap = round(size * 1.24)
    text_w = max(int(font.getlength(l)) for l in lines)
    text_h = gap * (len(lines) - 1) + size

    centered = abs((x1 + x2) / 2 - W / 2) < W * 0.06
    ox = int((W - text_w) / 2) if centered else x1
    ox = max(8, min(ox, W - text_w - 8))
    if mode == "append":
        oy = y2 + 8                      # 原字幕正下方
        if oy + text_h > H - 8:
            oy = y1 - text_h - 10        # 下方放不下 → 原字幕上方
        oy = max(4, oy)
    else:
        oy = y1
        if oy + text_h > H - 10:
            oy = max(0, H - 10 - text_h)

    pad = stroke + 4
    canvas = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad + size // 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, l in enumerate(lines):
        draw.text((pad, pad + i * gap), l, font=font, fill=fill + (255,),
                  stroke_width=stroke, stroke_fill=outline + (255,))
    arr = np.asarray(canvas)
    rgb = arr[..., :3][..., ::-1].copy()   # -> BGR
    alpha = arr[..., 3].copy()
    rx1, ry1 = ox - pad, oy - pad
    rx2, ry2 = rx1 + arr.shape[1], ry1 + arr.shape[0]
    # 裁剪到画面内
    cx1, cy1 = max(0, rx1), max(0, ry1)
    cx2, cy2 = min(W, rx2), min(H, ry2)
    rgb = rgb[cy1 - ry1:cy2 - ry1, cx1 - rx1:cx2 - rx1]
    alpha = alpha[cy1 - ry1:cy2 - ry1, cx1 - rx1:cx2 - rx1]
    return (cx1, cy1, cx2, cy2), rgb, alpha


def _clean_frames_near(ev, intervals, n_frames, max_dist=150, want=4):
    """在事件前后找不属于任何字幕事件的帧号。intervals 为 (start,end) 有序列表。

    起点离事件边界留 6 帧余量：边界定位误差 ±3 帧，太近会采到残留字幕。"""
    starts = [iv[0] for iv in intervals]

    def in_event(f):
        i = bisect.bisect_right(starts, f) - 1
        return i >= 0 and intervals[i][0] <= f <= intervals[i][1]

    out = []
    for base, direction in ((ev["start"] - 6, -1), (ev["end"] + 6, 1)):
        got = 0
        f = base
        for _ in range(max_dist):
            if f < 0 or f >= n_frames:
                break
            if not in_event(f):
                out.append(f)
                got += 1
                f += direction * 5
                if got >= want // 2:
                    break
            else:
                f += direction
    return out[:want]


_val_engine = None


def _bg_has_text(bg_med):
    """对时域背景中值图跑一次文本检测：采样帧里混进了字幕会在中值里留下字形。"""
    global _val_engine
    try:
        if _val_engine is None:
            from pipeline.detect import make_det_engine
            _val_engine = make_det_engine()
        out = _val_engine(bg_med, use_det=True, use_cls=False, use_rec=False)
        return out.boxes is not None and len(out.boxes) > 0
    except Exception:
        return False  # 校验失败不阻塞流程，逐帧环形校验仍能兜底


def _finalize_bg(plan):
    """凑齐干净参考帧后计算时域背景：中值、静态掩码、外圈校验数据。"""
    pe = plan["pe"]
    stack = np.stack([p.astype(np.int16) for p in plan["patches"]])
    plan["patches"] = []
    mask = pe["mask"]
    bg_med = np.median(stack, axis=0).astype(np.uint8)
    spread = (stack.max(axis=0) - stack.min(axis=0)).max(axis=2)
    thresh = 14 if plan["erase_mode"] == "fast" else 10
    bg_static = (spread < thresh) & (mask > 0)
    if bg_static.sum() < (mask > 0).sum() * 0.05:
        return
    if _bg_has_text(bg_med):
        return  # 采样帧混入了未检出的字幕 → 参考背景不可信
    ring = (cv2.dilate(mask, np.ones((15, 15), np.uint8)) > 0) & (mask == 0)
    if ring.sum() < 200:
        return
    pe["bg_med"] = bg_med
    pe["bg_static"] = bg_static
    dyn = mask.copy()
    dyn[bg_static] = 0
    pe["dyn_mask"] = dyn
    pe["ring"] = ring
    pe["bg_ring"] = bg_med[ring].astype(np.int16)


def _collect_backgrounds(video, plans, n_frames, progress=None, cancel=None):
    """单遍顺序扫描视频，为所有事件收集干净参考帧。

    不做任何跳帧寻址（seek）：长 GOP 的 1080p/60fps 视频上 seek 一次要
    从关键帧重解码数百帧，逐事件 seek 会慢到像卡死；顺序 grab 则总耗时
    恒定为一遍解码。"""
    need = {}
    for pl in plans:
        for f in pl["frames"]:
            need.setdefault(f, []).append(pl)
    if not need:
        return
    last = max(need)
    cap = cv2.VideoCapture(video)
    idx = 0
    try:
        while idx <= last:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            plans_at = need.get(idx)
            if plans_at:
                ok, frame = cap.read()
                if not ok:
                    break
                for pl in plans_at:
                    rx1, ry1, rx2, ry2 = pl["pe"]["roi"]
                    patch = frame[ry1:ry2, rx1:rx2]
                    if patch.shape[:2] == (ry2 - ry1, rx2 - rx1):
                        pl["patches"].append(patch.copy())
                    pl["pending"] -= 1
                    if pl["pending"] == 0 and len(pl["patches"]) >= 2:
                        try:
                            _finalize_bg(pl)
                        except Exception:
                            pass  # 背景回填是锦上添花，失败退回纯修复即可
            else:
                if not cap.grab():
                    break
            idx += 1
            if progress and idx % 900 == 0:
                progress(idx / (last + 1), f"分析字幕背景 {idx}/{last + 1} 帧")
    finally:
        cap.release()
    # 视频尾部截断等原因没凑齐的，够 2 帧也算
    for pl in plans:
        if pl["pending"] > 0 and len(pl["patches"]) >= 2:
            try:
                _finalize_bg(pl)
            except Exception:
                pass


def prepare_events(video, info, events, trans, erase_mode="fast", sub_mode="replace",
                   workdir=None, progress=None, cancel=None):
    W, H = info["width"], info["height"]
    intervals = [(ev["start"], ev["end"]) for ev in events]
    n_frames = info["n_frames_est"] or 10 ** 9
    prepared = []
    plans = []
    for n, ev in enumerate(events):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
        zh = trans.get(str(ev["id"]), "").strip()
        if sub_mode == "append":
            # 双字幕模式：不擦除原字幕，只在其下方（或上方）叠加中文
            prepared.append({
                "start": ev["start"], "end": ev["end"], "roi": None,
                "mask": None, "dyn_mask": None, "bg_med": None, "bg_static": None,
                "ring": None, "bg_ring": None,
                "overlay": build_overlay(ev, zh, W, H, mode="append") if zh else None,
            })
            continue
        # 擦除掩码（全帧坐标，行框外扩）
        pads_x, pads_y = 10, 7
        rx1 = max(0, min(l["box"][0] for l in ev["lines"]) - pads_x - 12)
        ry1 = max(0, min(l["box"][1] for l in ev["lines"]) - pads_y - 12)
        rx2 = min(W, max(l["box"][2] for l in ev["lines"]) + pads_x + 12)
        ry2 = min(H, max(l["box"][3] for l in ev["lines"]) + pads_y + 12)
        mask = None
        # 优先用字形级掩码（OCR 阶段提取），修复面积小、痕迹轻
        if workdir and ev.get("glyph_rect") and ev.get("mask_key") is not None:
            try:
                mp = os.path.join(workdir, "masks", f"{ev['mask_key']}.png")
                g = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.exists(mp) else None
                gx1, gy1, gx2, gy2 = ev["glyph_rect"]
                if g is not None and g.shape == (gy2 - gy1, gx2 - gx1):
                    # 与 ROI 求交集写入；掩码与行框坐标可能来自不同来源，不假设包含关系
                    ix1, iy1 = max(gx1, rx1), max(gy1, ry1)
                    ix2, iy2 = min(gx2, rx2), min(gy2, ry2)
                    if ix2 > ix1 and iy2 > iy1:
                        mask = np.zeros((ry2 - ry1, rx2 - rx1), np.uint8)
                        mask[iy1 - ry1:iy2 - ry1, ix1 - rx1:ix2 - rx1] = \
                            g[iy1 - gy1:iy2 - gy1, ix1 - gx1:ix2 - gx1]
                        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
                        if not mask.any():
                            mask = None  # 交集内没有字形像素 → 回退整行矩形
            except Exception:
                mask = None
        if mask is None:
            mask = np.zeros((ry2 - ry1, rx2 - rx1), np.uint8)
            for l in ev["lines"]:
                x1, y1, x2, y2 = l["box"]
                mask[max(0, y1 - pads_y - ry1):y2 + pads_y - ry1,
                     max(0, x1 - pads_x - rx1):x2 + pads_x - rx1] = 255
        overlay = build_overlay(ev, zh, W, H) if zh else None
        pe = {
            "start": ev["start"], "end": ev["end"],
            "roi": (rx1, ry1, rx2, ry2), "mask": mask, "dyn_mask": mask,
            "bg_med": None, "bg_static": None,
            "ring": None, "bg_ring": None, "overlay": overlay,
        }
        prepared.append(pe)
        clean = _clean_frames_near(ev, intervals, n_frames)
        if len(clean) >= 2:
            plans.append({"pe": pe, "frames": sorted(set(clean)),
                          "pending": len(set(clean)), "patches": [],
                          "erase_mode": erase_mode})
    if plans:
        _collect_backgrounds(video, plans, n_frames, progress=progress, cancel=cancel)
    return prepared


RING_DIFF_MAX = 11.0  # 外圈平均色差超过此值视为背景已移动，放弃时域回填


def _process_frame(frame, pe, inpaint_radius):
    if pe["roi"] is not None:  # 需要擦除（replace 模式）
        rx1, ry1, rx2, ry2 = pe["roi"]
        roi = frame[ry1:ry2, rx1:rx2]
        use_bg = pe["bg_med"] is not None
        if use_bg and pe["ring"] is not None:
            cur = roi[pe["ring"]].astype(np.int16)
            if np.abs(cur - pe["bg_ring"]).mean() > RING_DIFF_MAX:
                use_bg = False  # 本帧背景相对参考帧已变化（镜头移动/淡入淡出/转场）
        if use_bg:
            roi[pe["bg_static"]] = pe["bg_med"][pe["bg_static"]]
            if pe["dyn_mask"].any():
                roi[:] = cv2.inpaint(roi, pe["dyn_mask"], inpaint_radius, cv2.INPAINT_TELEA)
        else:
            roi[:] = cv2.inpaint(roi, pe["mask"], inpaint_radius, cv2.INPAINT_TELEA)
    if pe["overlay"] is not None:
        (ox1, oy1, ox2, oy2), rgb, alpha = pe["overlay"]
        sub = frame[oy1:oy2, ox1:ox2]
        a = alpha[..., None].astype(np.uint16)
        sub[:] = ((rgb.astype(np.uint16) * a + sub.astype(np.uint16) * (255 - a) + 127) // 255).astype(np.uint8)
    return frame


def nvenc_available():
    cmd = [ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
           "-c:v", "h264_nvenc", "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30,
                              creationflags=CREATE_NO_WINDOW).returncode == 0
    except Exception:
        return False


def run_render(video, info, prepared, out_path, erase_mode="fast",
               progress=None, cancel=None, n_workers=4):
    W, H = info["width"], info["height"]
    frame_bytes = W * H * 3
    n_est = info["n_frames_est"] or 1
    inpaint_radius = 3 if erase_mode == "fast" else 5

    starts = [pe["start"] for pe in prepared]

    def event_for(idx):
        i = bisect.bisect_right(starts, idx) - 1
        if i >= 0 and prepared[i]["start"] <= idx <= prepared[i]["end"]:
            return prepared[i]
        return None

    dec = subprocess.Popen(
        [ffmpeg_exe(), "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=frame_bytes * 4, creationflags=CREATE_NO_WINDOW)

    if nvenc_available():
        vcodec = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                  "-cq", "23", "-b:v", "0", "-maxrate", "10M", "-bufsize", "20M"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
    enc_cmd = [ffmpeg_exe(), "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
               "-r", info["fps_str"], "-i", "-",
               "-i", video,
               "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
               *vcodec, "-pix_fmt", "yuv420p", out_path]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                           creationflags=CREATE_NO_WINDOW)

    in_q = queue.Queue(maxsize=48)
    out_buf = {}
    out_lock = threading.Condition()
    total_frames = [None]  # reader 结束后写入实际总帧数
    abort = threading.Event()

    def reader():
        idx = 0
        while not abort.is_set():
            buf = dec.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            in_q.put((idx, buf))
            idx += 1
        with out_lock:
            total_frames[0] = idx
            out_lock.notify_all()
        for _ in range(n_workers):
            in_q.put(None)

    def worker():
        while True:
            item = in_q.get()
            if item is None:
                break
            idx, buf = item
            pe = event_for(idx)
            if pe is not None:
                frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
                try:
                    frame = _process_frame(frame, pe, inpaint_radius)
                except Exception:
                    pass
                buf = frame.tobytes()
            with out_lock:
                out_buf[idx] = buf
                out_lock.notify_all()

    threads = [threading.Thread(target=reader, daemon=True)]
    threads += [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()

    written = 0
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            with out_lock:
                while written not in out_buf:
                    if total_frames[0] is not None and written >= total_frames[0]:
                        break
                    out_lock.wait(timeout=1.0)
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError("cancelled")
                if total_frames[0] is not None and written >= total_frames[0]:
                    break  # 全部写完
                buf = out_buf.pop(written)
            enc.stdin.write(buf)
            written += 1
            if progress and written % 150 == 0:
                progress(min(written / n_est, 0.999), f"合成 {written}/{n_est} 帧")
    finally:
        abort.set()
        if dec.poll() is None:
            dec.kill()
        try:
            enc.stdin.close()
        except OSError:
            pass
        enc_err = enc.stderr.read().decode("utf-8", "ignore") if enc.stderr else ""
        rc = enc.wait()
        if cancel is not None and cancel.is_set():
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            raise InterruptedError("cancelled")
        if rc != 0:
            raise RuntimeError(f"编码失败（exit {rc}）：{enc_err[:400]}")
    return written
