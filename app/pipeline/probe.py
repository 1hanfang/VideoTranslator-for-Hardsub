"""ffprobe 视频信息探测。"""
import json
import subprocess
from fractions import Fraction

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ffprobe_exe

CREATE_NO_WINDOW = 0x08000000


def probe(path: str) -> dict:
    cmd = [ffprobe_exe(), "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", path]
    out = subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {out.stderr.decode('utf-8', 'ignore')[:300]}")
    data = json.loads(out.stdout.decode("utf-8", "ignore"))
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise RuntimeError("文件中没有视频流")
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    fps = Fraction(v.get("r_frame_rate", "30/1"))
    duration = float(v.get("duration") or data["format"].get("duration") or 0)
    nb = v.get("nb_frames")
    n_frames = int(nb) if nb and nb.isdigit() else int(duration * fps) if duration else 0
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps_str": v.get("r_frame_rate", "30/1"),
        "fps": float(fps),
        "duration": duration,
        "n_frames_est": n_frames,
        "has_audio": a is not None,
        "vcodec": v.get("codec_name", ""),
        "bitrate": int(v.get("bit_rate") or data["format"].get("bit_rate") or 0),
    }
