"""阶段2：把采样检测结果切分为字幕事件，并用逐帧缩略图精确定位起止帧。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import THUMB_SCALE, MIN_EVENT_FRAMES, MERGE_GAP_FRAMES

MIN_BOX_H = 16      # 全分辨率下，低于此高度的检测框视为噪声
MIN_BOX_W = 40
SAME_THRESH = 14.0  # 缩略图 bbox 区域平均灰度差，低于此视为同一条字幕
JUMP_MIN = 2.5      # 逐帧差分峰值低于此认为是渐变，退回采样点边界


def _valid_boxes(boxes):
    return [b for b in boxes if (b[3] - b[1]) >= MIN_BOX_H and (b[2] - b[0]) >= MIN_BOX_W]


def _union(boxes):
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _thumb_rect(bbox, crop_y, th_w, th_h):
    x1 = max(0, bbox[0] // THUMB_SCALE)
    y1 = max(0, (bbox[1] - crop_y) // THUMB_SCALE)
    x2 = min(th_w, (bbox[2] + THUMB_SCALE - 1) // THUMB_SCALE)
    y2 = min(th_h, (bbox[3] - crop_y + THUMB_SCALE - 1) // THUMB_SCALE)
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def _region_diff(thumbs, f1, f2, rect):
    x1, y1, x2, y2 = rect
    a = thumbs[f1, y1:y2, x1:x2].astype(np.int16)
    b = thumbs[f2, y1:y2, x1:x2].astype(np.int16)
    return float(np.abs(a - b).mean())


def _snap(thumbs, lo, hi, rect):
    """在 (lo, hi] 中找相邻帧差分最大的位置，返回突变后的帧号。"""
    if hi <= lo + 1:
        return hi
    diffs = [( _region_diff(thumbs, f - 1, f, rect), f) for f in range(lo + 1, hi + 1)]
    best_d, best_f = max(diffs)
    return best_f if best_d >= JUMP_MIN else hi


def run_events(det_meta, thumbs):
    crop_y = det_meta["crop_y"]
    th_w, th_h = det_meta["thumb_w"], det_meta["thumb_h"]
    n_frames = det_meta["n_frames"]
    samples = [(idx, _valid_boxes(bx)) for idx, bx in det_meta["samples"]]

    # 1) 按采样点切分成 run（连续且内容相似的采样序列）
    runs = []          # 每个 run: {"s": [采样帧号...], "bbox": 并集}
    cur = None
    prev_idx = None
    for idx, boxes in samples:
        has = len(boxes) > 0
        if not has:
            if cur:
                runs.append(cur)
                cur = None
            prev_idx = idx
            continue
        bbox = _union(boxes)
        if cur is None:
            cur = {"s": [idx], "bbox": bbox, "prev_empty": prev_idx}
        else:
            rect = _thumb_rect(_union([cur["bbox"], bbox]), crop_y, th_w, th_h)
            d = _region_diff(thumbs, cur["s"][-1], idx, rect)
            if d < SAME_THRESH:
                cur["s"].append(idx)
                cur["bbox"] = _union([cur["bbox"], bbox])
            else:  # 内容变了：结束当前 run，新开一个
                runs.append(cur)
                cur = {"s": [idx], "bbox": bbox, "prev_empty": None,
                       "split_from": cur["s"][-1]}
        prev_idx = idx
    if cur:
        runs.append(cur)

    # 2) 精确定位每个 run 的起止帧
    step = det_meta["step"]
    events = []
    for i, r in enumerate(runs):
        first_s, last_s = r["s"][0], r["s"][-1]
        rect = _thumb_rect(r["bbox"], crop_y, th_w, th_h)
        # 起点
        if r.get("split_from") is not None:
            start = _snap(thumbs, r["split_from"], first_s, rect)
        elif r.get("prev_empty") is not None:
            start = _snap(thumbs, r["prev_empty"], first_s, rect)
        else:
            start = max(0, first_s - step + 1) if first_s > 0 else 0
            start = _snap(thumbs, start - 1, first_s, rect) if start > 0 else 0
        # 终点：下一个采样点前
        nxt = min(last_s + step, n_frames - 1)
        # 若下一个 run 是从本 run 分裂出来的，其起点即本 run 终点
        if i + 1 < len(runs) and runs[i + 1].get("split_from") == last_s:
            end = None  # 之后由下一 run 的 start 决定
        else:
            end = _snap(thumbs, last_s, nxt, rect) - 1
            if end < last_s:
                end = last_s
        events.append({"start": int(start), "end": end, "bbox": r["bbox"]})

    for i, ev in enumerate(events):
        if ev["end"] is None:
            ev["end"] = int(events[i + 1]["start"]) - 1 if i + 1 < len(events) else n_frames - 1
        ev["end"] = int(min(ev["end"], n_frames - 1))

    # 3) 合并小间隔且区域相似的事件（处理转场闪断）
    merged = []
    for ev in events:
        if merged:
            p = merged[-1]
            gap = ev["start"] - p["end"]
            if 0 <= gap <= MERGE_GAP_FRAMES:
                rect = _thumb_rect(_union([p["bbox"], ev["bbox"]]), crop_y, th_w, th_h)
                mid_p = (p["start"] + p["end"]) // 2
                mid_e = (ev["start"] + ev["end"]) // 2
                if _region_diff(thumbs, mid_p, mid_e, rect) < SAME_THRESH:
                    p["end"] = ev["end"]
                    p["bbox"] = _union([p["bbox"], ev["bbox"]])
                    continue
        merged.append(ev)

    # 4) 丢弃过短事件，编号
    out = [ev for ev in merged if ev["end"] - ev["start"] + 1 >= MIN_EVENT_FRAMES]
    for i, ev in enumerate(out):
        ev["id"] = i
        ev["bbox"] = [int(v) for v in ev["bbox"]]
    return {"events": out}
