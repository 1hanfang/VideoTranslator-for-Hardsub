"""任务调度：串联六个阶段，逐阶段落盘缓存，支持断点续跑与取消。"""
import hashlib
import gc
import json
import os
import shutil
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WORK_DIR, UPLOAD_DIR, OUTPUT_DIR
from pipeline import probe as probe_mod
from pipeline import detect, events as events_mod, ocrstage, translate, render

STAGES = ["probe", "detect", "events", "ocr", "translate", "render"]
STAGE_NAMES = {"probe": "读取视频信息", "detect": "扫描字幕帧", "events": "切分字幕事件",
               "ocr": "识别原文", "translate": "翻译", "render": "擦除与合成"}


def work_dir_for(video):
    st = os.stat(video)
    key = f"{os.path.abspath(video)}|{st.st_size}|{int(st.st_mtime)}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    d = os.path.join(WORK_DIR, h)
    os.makedirs(d, exist_ok=True)
    return d


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _release_memmap(mm):
    if mm is None:
        return
    try:
        mm._mmap.close()
    except Exception:
        pass


def _rmtree_with_retries(path, attempts=6):
    if not os.path.exists(path):
        return True

    def onerror(func, p, exc_info):
        try:
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass

    for attempt in range(attempts):
        try:
            shutil.rmtree(path, onerror=onerror)
        except OSError:
            pass
        if not os.path.exists(path):
            return True
        gc.collect()
        time.sleep(0.35 * (attempt + 1))
    return not os.path.exists(path)


class Job:
    def __init__(self, video, opts=None):
        self.video = video
        self.opts = {"src_lang": "auto", "erase_mode": "fast", "review": False,
                     "sub_mode": "replace", **(opts or {})}
        self.workdir = work_dir_for(video)
        self.is_upload = os.path.abspath(video).startswith(
            os.path.abspath(UPLOAD_DIR) + os.sep)
        if self.is_upload:  # 拖拽上传的副本：原始位置未知，输出到用户视频目录兜底
            name, _ = os.path.splitext(os.path.basename(video))
            self.out_path = os.path.join(OUTPUT_DIR, name + ".zh.mp4")
        else:
            self.out_path = os.path.splitext(video)[0] + ".zh.mp4"
        self.cancel = threading.Event()
        self.review_ready = threading.Event()   # 校对完成信号
        self.state = {
            "video": video, "out_path": self.out_path, "status": "pending",
            "stage": None, "stage_name": "", "progress": 0.0, "message": "",
            "error": None, "started": None, "finished": None, "eta": None,
            "stage_index": 0, "n_stages": len(STAGES),
        }
        self._stage_t0 = None
        self._lock = threading.Lock()

    # ---- 状态 ----
    def snapshot(self):
        with self._lock:
            return dict(self.state)

    def _set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def _progress(self, frac, msg):
        eta = None
        if self._stage_t0 and frac > 0.02:
            elapsed = time.time() - self._stage_t0
            eta = elapsed / frac * (1 - frac)
        self._set(progress=round(frac, 4), message=msg,
                  eta=round(eta) if eta else None)

    def _begin(self, stage):
        if self.cancel.is_set():
            raise InterruptedError("cancelled")
        self._stage_t0 = time.time()
        self._set(stage=stage, stage_name=STAGE_NAMES[stage],
                  stage_index=STAGES.index(stage), progress=0.0, message="", eta=None)

    # ---- 事件/译文数据访问（供校对界面） ----
    def get_subs(self):
        p = os.path.join(self.workdir, "ocr.json")
        t = os.path.join(self.workdir, "trans.json")
        if not os.path.exists(p):
            return []
        evs = _load(p)["events"]
        trans = _load(t) if os.path.exists(t) else {}
        fps = _load(os.path.join(self.workdir, "meta.json"))["fps"]
        return [{"id": ev["id"], "start": ev["start"], "end": ev["end"],
                 "time": round(ev["start"] / fps, 1), "text": ev["text"],
                 "conf": ev.get("conf"), "trans": trans.get(str(ev["id"]), "")}
                for ev in evs]

    def set_trans(self, mapping):
        p = os.path.join(self.workdir, "trans.json")
        cur = _load(p) if os.path.exists(p) else {}
        cur.update({str(k): v for k, v in mapping.items()})
        _save(p, cur)

    # ---- 主流程 ----
    def run(self):
        try:
            self._set(status="running", started=time.time())
            wd = self.workdir

            # 1 probe
            self._begin("probe")
            meta_p = os.path.join(wd, "meta.json")
            if os.path.exists(meta_p):
                info = _load(meta_p)
            else:
                info = probe_mod.probe(self.video)
                _save(meta_p, info)

            # 2 detect
            self._begin("detect")
            det_p = os.path.join(wd, "samples.json")
            if os.path.exists(det_p):
                det_meta = _load(det_p)
            else:
                det_meta = detect.run_detect(self.video, info, wd,
                                             progress=self._progress, cancel=self.cancel)
                _save(det_p, det_meta)
            info["n_frames_est"] = det_meta["n_frames"]  # 实测帧数更准

            # 3 events
            self._begin("events")
            ev_p = os.path.join(wd, "events.json")
            if os.path.exists(ev_p):
                raw_events = _load(ev_p)["events"]
            else:
                thumbs = None
                try:
                    thumbs = detect.open_thumbs(wd, det_meta)
                    raw_events = events_mod.run_events(det_meta, thumbs)["events"]
                finally:
                    _release_memmap(thumbs)
                    thumbs = None
                    gc.collect()
                _save(ev_p, {"events": raw_events})
            self._progress(1.0, f"共 {len(raw_events)} 个候选字幕事件")

            # 4 ocr
            self._begin("ocr")
            ocr_p = os.path.join(wd, "ocr.json")
            if os.path.exists(ocr_p):
                events = _load(ocr_p)["events"]
            else:
                lang = "en" if self.opts["src_lang"] == "en" else "japan"
                events = ocrstage.run_ocr(self.video, info, det_meta, raw_events,
                                          lang=lang, workdir=wd, progress=self._progress,
                                          cancel=self.cancel)["events"]
                _save(ocr_p, {"events": events})
            if not events:
                raise RuntimeError("没有识别到任何字幕，请确认视频包含内嵌硬字幕")

            # 语言自动判断（决定翻译提示词）
            src_lang = self.opts["src_lang"]
            if src_lang == "auto":
                all_text = "".join(ev["text"] for ev in events)
                ascii_ratio = sum(c.isascii() for c in all_text) / max(len(all_text), 1)
                src_lang = "en" if ascii_ratio > 0.7 else "ja"

            # 5 translate
            self._begin("translate")
            trans = translate.run_translate(events, src_lang, wd,
                                            progress=self._progress, cancel=self.cancel)

            # 5.5 可选人工校对：暂停等待前端确认
            if self.opts.get("review"):
                self._set(status="awaiting_review", message="等待校对确认")
                while not self.review_ready.wait(timeout=1.0):
                    if self.cancel.is_set():
                        raise InterruptedError("cancelled")
                trans = _load(os.path.join(wd, "trans.json"))
                self._set(status="running")

            # 6 render
            self._begin("render")
            prepared = render.prepare_events(self.video, info, events, trans,
                                             erase_mode=self.opts["erase_mode"],
                                             sub_mode=self.opts["sub_mode"],
                                             workdir=wd,
                                             progress=lambda f, m: self._progress(f * 0.12, m),
                                             cancel=self.cancel)
            def render_prog(f, m):
                self._progress(0.12 + f * 0.88, m)
            n = render.run_render(self.video, info, prepared, self.out_path,
                                  erase_mode=self.opts["erase_mode"],
                                  progress=render_prog, cancel=self.cancel)

            prepared = None
            gc.collect()
            cleanup_ok = self._cleanup()  # 成功后清理缓存与上传副本，避免占用存储
            if not cleanup_ok:
                threading.Thread(target=self._cleanup, daemon=True).start()
            cleanup_msg = "" if cleanup_ok else "；缓存稍后继续清理"
            self._set(status="done", finished=time.time(), progress=1.0,
                      message=f"完成：{self.out_path}（{n} 帧）{cleanup_msg}")
        except InterruptedError:
            self._set(status="cancelled", message="已取消", finished=time.time())
        except Exception as e:
            traceback.print_exc()
            self._set(status="error", error=str(e), finished=time.time())

    def _cleanup(self):
        """任务成功后删除中间缓存（缩略图/掩码/识别结果）与拖拽上传的视频副本。
        取消/出错时保留缓存以便断点续跑。"""
        ok = _rmtree_with_retries(self.workdir)
        if self.is_upload and os.path.exists(self.video):
            try:
                os.remove(self.video)
            except OSError:
                ok = False
        return ok


# ---- 命令行调试入口 ----
if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--until", default="render", choices=STAGES)
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--fresh", action="store_true", help="清空缓存重跑")
    args = ap.parse_args()

    job = Job(args.video, {"src_lang": args.lang})
    if args.fresh:
        import shutil
        shutil.rmtree(job.workdir, ignore_errors=True)
        os.makedirs(job.workdir, exist_ok=True)

    # 截断流程：把 until 之后的阶段砍掉
    cut = STAGES.index(args.until)
    orig_begin = job._begin
    def begin_or_stop(stage):
        if STAGES.index(stage) > cut:
            raise KeyboardInterrupt(f"stopped after {args.until}")
        orig_begin(stage)
    job._begin = begin_or_stop

    def watcher():
        while True:
            s = job.snapshot()
            eta = f" eta {s['eta']}s" if s.get("eta") else ""
            print(f"[{s['stage']}] {s['progress']*100:5.1f}% {s['message']}{eta}")
            if s["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(2)
    threading.Thread(target=watcher, daemon=True).start()
    try:
        job.run()
    except KeyboardInterrupt as e:
        print(e)
    s = job.snapshot()
    print("status:", s["status"], s.get("error") or "")
