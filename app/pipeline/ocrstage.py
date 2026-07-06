"""阶段3：对每个字幕事件取代表帧做全分辨率 OCR，并采样字幕颜色样式。"""
import os
import sys
from difflib import SequenceMatcher

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, OCR_MIN_CONF, MERGE_GAP_FRAMES

_JP = None


def make_ocr_engine(lang="japan"):
    from rapidocr import RapidOCR
    return RapidOCR(params={
        "Rec.lang_type": lang,
        "Global.model_root_dir": os.path.join(MODELS_DIR, "rapidocr"),
        "Global.log_level": "error",
        "Det.limit_type": "max",
        "Det.limit_side_len": 1280,
        "Global.min_height": 12,
    })


def _boxes_from(res, crop_y):
    lines = []
    if res.txts:
        for txt, score, box in zip(res.txts, res.scores, res.boxes):
            x1, y1 = box.min(axis=0)
            x2, y2 = box.max(axis=0)
            lines.append({"text": txt.strip(), "conf": float(score),
                          "box": [int(x1), int(y1 + crop_y), int(x2), int(y2 + crop_y)]})
    return lines


def _overlaps(box, bbox, margin=24):
    return not (box[2] < bbox[0] - margin or box[0] > bbox[2] + margin or
                box[3] < bbox[1] - margin or box[1] > bbox[3] + margin)


def analyze_style(frame, lines, crop_y):
    """采样字幕填充色与描边色，并提取字形像素掩码。
    返回 (fill_bgr, outline_bgr, glyph_mask 或 None)。glyph_mask 为全帧大小 uint8。

    思路：文字像素是行框内亮度的两个极端（描边为对比设计）；
    腐蚀后存活多的一类是笔画芯（填充色），另一类是描边。"""
    mask = np.zeros(frame.shape[:2], np.uint8)
    for ln in lines:
        x1, y1, x2, y2 = ln["box"]
        mask[y1:y2, x1:x2] = 255
    px = mask > 0
    if px.sum() < 100:
        return (255, 255, 255), (16, 16, 16), None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = ((gray < 70) & px).astype(np.uint8)
    bright = ((gray > 190) & px).astype(np.uint8)
    if dark.sum() < 80 or bright.sum() < 80:
        return (255, 255, 255), (16, 16, 16), None
    k = np.ones((3, 3), np.uint8)
    dark_core = cv2.erode(dark, k, iterations=2).sum()
    bright_core = cv2.erode(bright, k, iterations=2).sum()
    if dark_core >= bright_core:
        fill_m, out_m = dark.astype(bool), bright.astype(bool)
    else:
        fill_m, out_m = bright.astype(bool), dark.astype(bool)
    fill = tuple(int(v) for v in np.median(frame[fill_m], axis=0))
    outline = tuple(int(v) for v in np.median(frame[out_m], axis=0))
    glyph = ((dark | bright) * 255).astype(np.uint8)
    # 字形像素应占行框的合理比例，否则说明分割不可靠（如背景大面积极亮/极暗）
    ratio = (glyph > 0).sum() / px.sum()
    if not 0.08 <= ratio <= 0.85:
        glyph = None
    return fill, outline, glyph


def run_ocr(video, info, det_meta, events, lang="japan", workdir=None,
            progress=None, cancel=None):
    """填充每个事件的 lines/text/conf/style。返回新的事件列表（丢弃无文字事件）。
    若给定 workdir，同时把每个事件的字形掩码存为 masks/{id}.png。"""
    engine = make_ocr_engine(lang)
    crop_y, crop_h = det_meta["crop_y"], det_meta["crop_h"]
    mask_dir = None
    if workdir:
        mask_dir = os.path.join(workdir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    out_events = []
    try:
        for n, ev in enumerate(events):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            length = ev["end"] - ev["start"] + 1
            tries = [ev["start"] + length // 2]
            best = None
            best_frame = None
            for attempt in range(3):
                if attempt == 1:
                    tries.append(ev["start"] + max(1, length // 4))
                elif attempt == 2:
                    tries.append(ev["end"] - max(1, length // 4))
                fidx = tries[-1]
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ok, frame = cap.read()
                if not ok:
                    continue
                crop = frame[crop_y:crop_y + crop_h]
                res = engine(crop, use_det=True, use_cls=False, use_rec=True)
                lines = [ln for ln in _boxes_from(res, crop_y)
                         if ln["conf"] >= OCR_MIN_CONF and _overlaps(ln["box"], ev["bbox"])
                         and len(ln["text"]) > 0]
                if lines:
                    conf = sum(l["conf"] for l in lines) / len(lines)
                    if best is None or conf > best[0]:
                        best = (conf, lines)
                        best_frame = frame
                    if conf >= 0.75:
                        break
                elif best is not None:
                    break
            if best is None:
                continue  # 误检事件，丢弃
            conf, lines = best
            lines.sort(key=lambda l: (l["box"][1], l["box"][0]))
            ev = dict(ev)
            ev["lines"] = lines
            ev["text"] = "\n".join(l["text"] for l in lines)
            ev["conf"] = round(conf, 3)
            fill, outline, glyph = analyze_style(best_frame, lines, crop_y)
            ev["style"] = {"fill": list(fill), "outline": list(outline),
                           "line_h": int(np.median([l["box"][3] - l["box"][1] for l in lines]))}
            ev["bbox"] = [int(min(l["box"][0] for l in lines)),
                          int(min(l["box"][1] for l in lines)),
                          int(max(l["box"][2] for l in lines)),
                          int(max(l["box"][3] for l in lines))]
            if mask_dir is not None and glyph is not None:
                x1, y1, x2, y2 = ev["bbox"]
                pad = 4
                gx1, gy1 = max(0, x1 - pad), max(0, y1 - pad)
                gx2, gy2 = min(glyph.shape[1], x2 + pad), min(glyph.shape[0], y2 + pad)
                cv2.imwrite(os.path.join(mask_dir, f"{ev['start']}.png"),
                            glyph[gy1:gy2, gx1:gx2])
                ev["glyph_rect"] = [int(gx1), int(gy1), int(gx2), int(gy2)]
                ev["mask_key"] = int(ev["start"])
            out_events.append(ev)
            if progress:
                progress((n + 1) / max(len(events), 1),
                         f"OCR {n + 1}/{len(events)}：{ev['text'][:24]}")
    finally:
        cap.release()

    # 相邻事件文本相似且几乎连续 → 合并（同一句被背景动画切碎/OCR 微小差异）
    merged = []
    for ev in out_events:
        if merged:
            p = merged[-1]
            gap = ev["start"] - p["end"]
            if 0 <= gap <= 12 and \
                    SequenceMatcher(None, p["text"], ev["text"]).ratio() > 0.85:
                keep = ev if ev["conf"] > p["conf"] else p
                end = ev["end"]
                # 整组字段必须来自同一侧：混用（如 lines 取 A、glyph_rect 留 B）
                # 会导致坐标彼此不一致，渲染时掩码切片越界
                for k in ("text", "conf", "lines", "style", "bbox", "glyph_rect", "mask_key"):
                    if k in keep:
                        p[k] = keep[k]
                    elif k in p:
                        del p[k]
                p["end"] = end
                continue
        merged.append(dict(ev))
    for i, ev in enumerate(merged):
        ev["id"] = i
    return {"events": merged}
