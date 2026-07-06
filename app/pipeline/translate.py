"""阶段4：llama.cpp 拉起 Sakura-GalTransl 模型，按官方 v3 协议批量翻译。"""
import json
import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (GGUF_PATH, LLAMA_SERVER, LLAMA_PORT, LLAMA_CTX,
                    TRANS_BATCH, TRANS_HISTORY, LOG_DIR)

CREATE_NO_WINDOW = 0x08000000

SYSTEM_PROMPT = ("你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，"
                 "并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，"
                 "不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。")


class LlamaServer:
    def __init__(self, port=LLAMA_PORT):
        self.port = port
        self.proc = None
        self.log_path = os.path.join(LOG_DIR, "llama-server.log")

    def start(self, timeout=180):
        if self.proc and self.proc.poll() is None:
            return
        try:  # 端口上已有存活的 server（上次残留）则直接复用
            if requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        if not os.path.exists(LLAMA_SERVER):
            raise RuntimeError(f"缺少 llama-server：{LLAMA_SERVER}")
        if not os.path.exists(GGUF_PATH):
            raise RuntimeError(f"缺少翻译模型：{GGUF_PATH}")
        cmd = [LLAMA_SERVER, "-m", GGUF_PATH, "-c", str(LLAMA_CTX),
               "-ngl", "99", "--host", "127.0.0.1", "--port", str(self.port),
               "--no-webui"]
        self.log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     creationflags=CREATE_NO_WINDOW)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server 启动失败，日志：{self.log_path}")
            try:
                r = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1.5)
        self.stop()
        raise RuntimeError(f"llama-server 健康检查超时，日志：{self.log_path}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None

    def chat(self, user_text, max_tokens=1024, temperature=0.3, top_p=0.8):
        payload = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_text}],
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        }
        r = requests.post(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                          json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def _build_prompt(lines, history, src_name, glossary=""):
    parts = []
    if history:
        parts.append("历史翻译：" + "\n".join(history))
        parts.append("")
    parts.append("参考以下术语表（可为空，格式为src->dst #备注）：")
    parts.append(glossary)
    parts.append(f"根据以上术语表的对应关系和备注，结合历史剧情和上下文，"
                 f"将下面的文本从{src_name}翻译成简体中文：")
    parts.append("\n".join(lines))
    return "\n".join(parts)


def _build_glossary(server, events):
    """收集说话人前缀（"名字「"格式），让模型统一定名，保证全程人名一致。"""
    import re
    names = []
    for ev in events:
        m = re.match(r"^([^「『\s]{1,8})[「『]", ev["text"])
        if m and m.group(1) not in names:
            names.append(m.group(1))
    entries = []
    for name in names[:20]:
        if re.fullmatch(r"[一-鿿]+", name):
            zh = name  # 纯汉字名直接沿用
        else:
            try:
                reply = server.chat(
                    f"将下面的日文人名翻译成简体中文，只输出译名本身：\n{name}",
                    max_tokens=32, temperature=0.1)
                zh = reply.splitlines()[0].strip().strip("「」『』\"'：:")
            except Exception:
                continue
            if not zh or len(zh) > 12:
                continue
        entries.append(f"{name}->{zh} #人名")
    return "\n".join(entries)


def _clean_output(text, n_expect):
    lines = [l for l in (s.strip() for s in text.splitlines()) if l]
    return lines if len(lines) == n_expect else None


def run_translate(events, src_lang, workdir, progress=None, cancel=None):
    """返回 {event_id(str): 译文}。逐批落盘，可断点续跑。"""
    src_name = "英文" if src_lang == "en" else "日文"
    cache_path = os.path.join(workdir, "trans.json")
    done = {}
    if os.path.exists(cache_path):
        try:
            done = json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            done = {}

    todo = [ev for ev in events if str(ev["id"]) not in done]
    if not todo:
        return done

    server = LlamaServer()
    server.start()
    history = []
    try:
        gloss_p = os.path.join(workdir, "glossary.txt")
        if os.path.exists(gloss_p):
            glossary = open(gloss_p, encoding="utf-8").read()
        else:
            if progress:
                progress(0.0, "统一人名术语表…")
            glossary = _build_glossary(server, events)
            with open(gloss_p, "w", encoding="utf-8") as f:
                f.write(glossary)
        for bi in range(0, len(todo), TRANS_BATCH):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            batch = todo[bi:bi + TRANS_BATCH]
            # 事件内部换行折为一行，保证行数与事件一一对应
            src_lines = [ev["text"].replace("\n", "　") for ev in batch]
            prompt = _build_prompt(src_lines, history[-TRANS_HISTORY:], src_name, glossary)
            out_lines = None
            for attempt, temp in enumerate((0.3, 0.1)):
                try:
                    reply = server.chat(prompt, temperature=temp)
                except requests.RequestException as e:
                    if attempt == 1:
                        raise RuntimeError(f"翻译请求失败：{e}")
                    server.start()
                    continue
                out_lines = _clean_output(reply, len(src_lines))
                if out_lines:
                    break
            if out_lines is None:  # 行数对不上：退化为逐条翻译
                out_lines = []
                for line in src_lines:
                    reply = server.chat(_build_prompt([line], history[-TRANS_HISTORY:], src_name, glossary),
                                        temperature=0.1, max_tokens=512)
                    cand = [l for l in (s.strip() for s in reply.splitlines()) if l]
                    out_lines.append(cand[0] if cand else line)
            for ev, zh in zip(batch, out_lines):
                done[str(ev["id"])] = zh
            history.extend(out_lines)
            tmp = cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(done, f, ensure_ascii=False, indent=0)
            os.replace(tmp, cache_path)
            if progress:
                n_done = min(bi + TRANS_BATCH, len(todo))
                progress(n_done / len(todo),
                         f"翻译 {n_done}/{len(todo)}：{out_lines[-1][:24]}")
    finally:
        server.stop()
    return done
