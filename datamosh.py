"""
DATAMOSH // Loop507
Simulazione di datamosh in puro DSP (OpenCV + NumPy). Nessuna AI / rete neurale.

Idea: il datamosh reale nasce rimuovendo gli I-frame, cosi' i P-frame applicano
i vettori di movimento a un'immagine "vecchia". Qui lo simuliamo:
  1. stimiamo il moto del video sorgente (optical flow Farneback)
  2. lo applichiamo (warp) al frame "mosh" accumulato, senza mai rinfrescarlo
  3. rinfresco (I-frame) solo quando lo decidono i parametri / il seed

Minimalismo Computazionale / Glitch Brutalista.
"""

import json
import os
import random
import struct
import subprocess
import tempfile

import cv2
import librosa
import numpy as np
import streamlit as st

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # fallback: ffmpeg di sistema
    FFMPEG = "ffmpeg"


# ----------------------------------------------------------------------------
# DSP
# ----------------------------------------------------------------------------
def at(arr, k):
    return float(arr[min(int(k), len(arr) - 1)])


def fit_frame(frame, w, h):
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def compute_flow(prev_gray, cur_gray, dis=None):
    """Flow cur -> prev: serve per il backward warp (out(x) = mosh(x + flow))."""
    if dis is not None:
        return dis.calc(cur_gray, prev_gray, None)
    return cv2.calcOpticalFlowFarneback(
        cur_gray, prev_gray, None,
        pyr_scale=0.5, levels=4, winsize=21, iterations=3,
        poly_n=5, poly_sigma=1.1, flags=0,
    )


def to_block_mv(flow, bs, rng, mv_noise):
    """Simula i macroblocchi: un vettore (intero) per blocco, con rumore opzionale."""
    h, w = flow.shape[:2]
    bw, bh = max(1, w // bs), max(1, h // bs)
    small = cv2.resize(flow, (bw, bh), interpolation=cv2.INTER_AREA)
    if mv_noise > 0:
        small = small + rng.normal(0.0, mv_noise, small.shape).astype(np.float32)
    small = np.round(small)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def warp(img, flow, grid_x, grid_y):
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def datamosh_video(src_a, src_b, out_path, p, progress_cb=None):
    """
    src_a : video che fornisce il MOVIMENTO (e i frame reali fuori dalla finestra mosh)
    src_b : video opzionale che fornisce la TEXTURE (transfer: moto di A su immagine di B)
    """
    rng = np.random.default_rng(p["seed"])
    for k, v in (("a_strength", 0.0), ("a_bloom", False), ("a_refresh", False),
                 ("onset_thr", 0.6), ("refresh_thr", 0.85), ("flow_engine", "Farneback")):
        p.setdefault(k, v)
    au = p.get("au")
    dis = (cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
           if str(p["flow_engine"]).startswith("DIS") else None)

    cap = cv2.VideoCapture(src_a)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, p["max_w"] / float(ow))
    w = max(16, int(ow * scale) // 2 * 2)
    h = max(16, int(oh * scale) // 2 * 2)
    n_max = min(total, int(p["max_sec"] * fps))

    cap_b = cv2.VideoCapture(src_b) if src_b else None

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))

    start_f = int(n_max * p["mosh_start"] / 100.0)
    end_f = int(n_max * p["mosh_end"] / 100.0)

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Impossibile leggere il video / cannot read video")
    first = fit_frame(first, w, h)
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    mosh = first.copy().astype(np.float32)
    writer.write(first)

    stats = {"frames": 1, "iframes": 1, "holds": 0, "blooms": 0}
    bloom_flow, bloom_left = None, 0

    for t in range(1, n_max):
        ok, frame = cap.read()
        if not ok:
            break
        frame = fit_frame(frame, w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        in_window = start_f <= t < end_f

        if not in_window:
            # fuori dalla finestra: video reale, e il mosh riparte dall'ultimo frame reale
            mosh = frame.astype(np.float32)
            out = frame
        else:
            # entrando nella finestra: se c'e' B, la texture di partenza e' B
            if t == start_f and cap_b is not None:
                okb, fb = cap_b.read()
                if okb:
                    mosh = fit_frame(fb, w, h).astype(np.float32)

            # I-frame di rinfresco
            refresh = (p["iframe_every"] > 0 and (t - start_f) % p["iframe_every"] == 0
                       and t != start_f) or (rng.random() < p["iframe_prob"]) \
                or bool(au and p["a_refresh"] and at(au["trig_refresh"], t) > p["refresh_thr"])
            if refresh:
                mosh = frame.astype(np.float32)
                stats["iframes"] += 1
                bloom_left = 0

            # flow del frame
            k_str = p["strength"] * (1.0 + p["a_strength"] * at(au["rms"], t) if au else 1.0)
            flow = compute_flow(prev_gray, gray, dis) * k_str
            if p["mode"] == "Block MV":
                flow = to_block_mv(flow, p["block"], rng,
                                p["mv_noise"] + (at(au["noise"], t) * 4.0 if au else 0.0))

            # hold: freeze (P-frame vuoto)
            if rng.random() < p["hold_prob"]:
                flow = np.zeros_like(flow)
                stats["holds"] += 1

            # bloom: ripete lo stesso flow per k frame
            if bloom_left > 0 and bloom_flow is not None:
                flow = bloom_flow
                bloom_left -= 1
            elif (au and p["a_bloom"] and at(au["trig_bloom"], t) > p["onset_thr"]) or rng.random() < p["bloom_prob"]:
                bloom_flow = flow.copy() * p["bloom_gain"]
                bloom_left = int(rng.integers(p["bloom_len"][0], p["bloom_len"][1] + 1))
                stats["blooms"] += 1

            mosh = warp(mosh, flow, grid_x, grid_y)

            # residuo: quanto del frame reale "trapela" nel mosh
            if p["residual"] > 0:
                mosh = cv2.addWeighted(mosh, 1.0 - p["residual"],
                                       frame.astype(np.float32), p["residual"], 0)

            out = np.clip(mosh, 0, 255).astype(np.uint8)

        writer.write(out)
        prev_gray = gray
        stats["frames"] += 1
        if progress_cb and t % 5 == 0:
            progress_cb(t / max(1, n_max - 1))

    writer.release()
    cap.release()
    if cap_b is not None:
        cap_b.release()
    return fps, stats


def finalize_h264(raw_path, audio_src, final_path, keep_audio, max_sec):
    """Re-encode in H.264 (compatibile browser) + audio opzionale dalla sorgente."""
    cmd = [FFMPEG, "-y", "-err_detect", "ignore_err", "-i", raw_path]
    if keep_audio:
        cmd += ["-t", str(max_sec), "-i", audio_src, "-map", "0:v:0", "-map", "1:a:0?"]
        cmd += ["-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-preset", "veryfast", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", final_path]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ----------------------------------------------------------------------------
# AUDIO
# ----------------------------------------------------------------------------
def zero_au(n):
    keys = ("rms", "onset", "beat", "low", "mid", "high", "trig_bloom", "trig_drop",
            "trig_refresh", "corr", "noise")
    au = {k: np.zeros(n) for k in keys}
    au.update({"bpm": 0.0, "ok": False})
    return au


def audio_features(src, fps, n_frames, max_sec, bpm_manual=0.0, accent=4):
    """RMS, onset, beat e 3 bande (basso/medio/acuto) per frame, normalizzati 0-1 (librosa)."""
    wav = os.path.join(tempfile.mkdtemp(), "a.wav")
    try:
        subprocess.run([FFMPEG, "-y", "-i", src, "-t", str(max_sec), "-vn", "-ac", "1",
                        "-ar", "22050", wav], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        y, sr = librosa.load(wav, sr=22050, mono=True)
    except Exception:  # noqa: BLE001
        return zero_au(n_frames)
    if y.size < 4096:
        return zero_au(n_frames)
    hop = 512
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    fq = librosa.fft_frequencies(sr=sr, n_fft=2048)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    ft = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    vt = np.arange(n_frames) / fps

    def norm(v):
        v = np.interp(vt, ft, v[:len(ft)])
        v = np.where(vt <= ft[-1], v, 0.0)  # oltre la fine dell'audio: silenzio
        return np.clip(v / (np.percentile(v, 99) + 1e-9), 0, 1)

    def band(lo, hi):
        e = np.log1p(spec[(fq >= lo) & (fq < hi)].sum(axis=0))
        return norm(np.maximum(0.0, np.diff(e, prepend=e[0])))

    tempo, bt = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
    bt_t = librosa.frames_to_time(bt, sr=sr, hop_length=hop)
    if bpm_manual > 0:  # griglia a BPM fissi, con la fase del primo beat rilevato
        period = 60.0 / bpm_manual
        t0 = (bt_t[0] % period) if len(bt_t) else 0.0
        bt_t, bpm = np.arange(t0, ft[-1], period), float(bpm_manual)
    else:
        bpm = float(np.atleast_1d(tempo)[0])
    beat = np.zeros(n_frames)
    for i, tb in enumerate(bt_t):
        j = int(round(tb * fps))
        if 0 <= j < n_frames:
            beat[j] = 1.0 if i % accent == 0 else 0.75  # accento sul primo battito
    au = zero_au(n_frames)
    au.update({"rms": norm(rms), "onset": norm(onset), "beat": beat, "bpm": bpm, "ok": True,
               "low": band(0, 150), "mid": band(150, 2000), "high": band(2000, sr / 2)})
    return au


def route_audio(au, trigger, bands):
    """Decide quali segnali pilotano cosa (bloom, drop I-frame, refresh, corruzione, rumore MV)."""
    def mix(x):
        if trigger == "Beat (a tempo)":
            return au["beat"]
        return np.maximum(x, au["beat"]) if trigger == "Onset + Beat" else x

    if bands:  # bassi -> bloom/I-frame, medi -> refresh, acuti -> corruzione/rumore
        au["trig_bloom"] = au["trig_drop"] = mix(au["low"])
        au["trig_refresh"] = mix(au["mid"])
        au["corr"] = au["high"] + 0.3 * au["rms"]
        au["noise"] = au["high"].copy()
    else:
        au["trig_bloom"] = au["trig_drop"] = au["trig_refresh"] = mix(au["onset"])
        au["corr"] = au["rms"] + mix(au["onset"])
        au["noise"] = np.zeros_like(au["rms"])
    au["onset"] = mix(au["onset"])
    return au


# ----------------------------------------------------------------------------
# BYTE-LEVEL DATAMOSH (AVI / MPEG-4 part 2) -- parser e muxer scritti a mano
# ----------------------------------------------------------------------------
def vop_type(b):
    i = b.find(b"\x00\x00\x01\xb6")
    return -1 if i < 0 or i + 4 >= len(b) else b[i + 4] >> 6  # 0=I 1=P 2=B


def riff_chunks(buf, pos, end):
    while pos + 8 <= end:
        cid = buf[pos:pos + 4]
        size = struct.unpack("<I", buf[pos + 4:pos + 8])[0]
        if cid in (b"RIFF", b"LIST"):
            yield from riff_chunks(buf, pos + 12, min(end, pos + 8 + size))
        else:
            yield cid, pos + 8, size
        pos += 8 + size + (size & 1)


def read_avi_frames(path):
    with open(path, "rb") as f:
        buf = f.read()
    return [buf[o:o + s] for cid, o, s in riff_chunks(buf, 0, len(buf))
            if cid in (b"00dc", b"00db")]


def write_avi(path, frames, w, h, fps):
    def ck(i, d):
        return i + struct.pack("<I", len(d)) + d + (b"\0" if len(d) & 1 else b"")

    def ls(t, d):
        return b"LIST" + struct.pack("<I", len(d) + 4) + t + d

    n, mx = len(frames), max(len(f) for f in frames)
    movi, idx = b"".join(ck(b"00dc", f) for f in frames), b""
    off = 4
    for f in frames:
        idx += b"00dc" + struct.pack("<III", 0x10 if vop_type(f) == 0 else 0, off, len(f))
        off += 8 + len(f) + (len(f) & 1)
    avih = struct.pack("<IIIIIIIIII4I", int(1e6 / fps), 0, 0, 0x10, n, 0, 1, mx, w, h, 0, 0, 0, 0)
    strh = b"vids" + b"XVID" + struct.pack("<IHHIIIIIIIIhhhh", 0, 0, 0, 0, 1000, int(fps * 1000),
                                           0, n, mx, 0xFFFFFFFF, 0, 0, 0, w, h)
    strf = struct.pack("<IiiHH", 40, w, h, 1, 24) + b"XVID" + struct.pack("<IiiII", w * h * 3, 0, 0, 0, 0)
    hdrl = ls(b"hdrl", ck(b"avih", avih) + ls(b"strl", ck(b"strh", strh) + ck(b"strf", strf)))
    body = b"AVI " + hdrl + ls(b"movi", movi) + ck(b"idx1", idx)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def corrupt_bytes(f, n, rng):
    if n <= 0 or len(f) < 64:
        return f
    b = bytearray(f)
    for i in rng.integers(32, len(b), n):
        b[int(i)] = int(rng.integers(2, 256))  # mai 0/1: non creiamo start code
    return bytes(b)


def byte_mosh(frames, au, p, rng):
    """Una slot di output per frame: la durata resta quella dell'audio."""
    n = len(frames)
    p.setdefault("kf_set", set())
    p.setdefault("protect", -1)
    types = [vop_type(f) for f in frames]
    start_f, end_f = int(n * p["mosh_start"] / 100), int(n * p["mosh_end"] / 100)
    out, s, hold, last, last_t = [], 0, 0, None, 0
    st = {"frames": 0, "dropped_i": 0, "dups": 0, "skips": 0, "corrupt": 0}
    for k in range(n):
        dr, bl, en = at(au["trig_drop"], k), at(au["trig_bloom"], k), at(au["rms"], k)
        win = start_f <= k < end_f and k > 0
        if hold > 0 and last is not None:
            out.append(last); hold -= 1; st["dups"] += 1; continue
        s = min(s, n - 1)
        if win:
            if types[s] == 0 and s != p["protect"] and (p["drop_mode"] == "Tutti" or
                    (p["drop_mode"] == "Su onset" and (dr > p["drop_thr"] or s in p["kf_set"]))):
                s = min(s + 1, n - 1); st["dropped_i"] += 1
            if last is not None and last_t != 0 and (bl > p["onset_thr"] or rng.random() < p["bloom_prob"]):
                hold = int(max(bl, 0.3) * p["max_dup"]); out.append(last); st["dups"] += 1; continue
            if en < p["quiet_thr"] and last is not None and last_t != 0:
                if k - s > 0:
                    s = min(s + 1, n - 1); st["skips"] += 1
                elif k - s < 0:
                    out.append(last); st["dups"] += 1; continue
        f, t = frames[s], types[s]
        if win and t != 0:
            nb = int(len(f) * p["corrupt"] * at(au["corr"], k))
            if nb > 0:
                f = corrupt_bytes(f, nb, rng); st["corrupt"] += nb
        out.append(f); last, last_t = f, t; s += 1
    st["frames"] = len(out)
    return out, st


def audio_keyframes(sig, thr, gap=3):
    """Indici dei picchi locali del segnale sopra soglia (a distanza minima)."""
    idx, last = [], -99
    for k in range(1, len(sig) - 1):
        if sig[k] > thr and sig[k] >= sig[k - 1] and sig[k] > sig[k + 1] and k - last >= gap:
            idx.append(k)
            last = k
    return idx


def transcode_avi(src, out, w, h, fps, p, kf_times=()):
    cmd = [FFMPEG, "-y", "-i", src, "-t", str(p["max_sec"]), "-an",
           "-vf", f"scale={w}:{h},fps={fps}", "-c:v", "mpeg4", "-vtag", "XVID",
           "-q:v", str(p["qscale"]), "-g", str(p["kf"]), "-bf", "0"]
    if len(kf_times):
        cmd += ["-force_key_frames", ",".join(f"{t:.3f}" for t in kf_times)]
    subprocess.run(cmd + ["-f", "avi", out], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def byte_datamosh(src, src_b, out_avi, p):
    rng = np.random.default_rng(p["seed"])
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ow, oh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    w = max(16, int(min(p["max_w"], ow)) // 2 * 2)
    h = max(16, int(oh * w / ow) // 2 * 2)
    n_est = int(p["max_sec"] * fps)
    au = p["au"] or zero_au(n_est + 2)
    p["au"] = au

    # keyframe forzati sui picchi dell'audio: ogni colpo ha un I-frame da togliere
    kfi = audio_keyframes(au["trig_drop"][:n_est], p["drop_thr"]) if p.get("kf_on_audio") else []
    p["kf_set"], p["protect"] = set(kfi), -1
    tmp = tempfile.mkdtemp()
    path_a = os.path.join(tmp, "a.avi")
    transcode_avi(src, path_a, w, h, fps, p, [k / fps for k in kfi])
    frames = read_avi_frames(path_a)
    if len(frames) < 3:
        raise RuntimeError("Troppi pochi frame / too few frames")
    n = len(frames)

    if src_b:  # transfer: l'I-frame di B (texture) + i P-frame di A (movimento)
        path_b = os.path.join(tmp, "b.avi")
        transcode_avi(src_b, path_b, w, h, fps, p)
        fb = read_avi_frames(path_b)
        tyb = [vop_type(f) for f in fb]
        start_f = min(n - 1, int(n * p["mosh_start"] / 100))
        cand = [i for i, t in enumerate(tyb) if t == 0 and i <= start_f] or \
               [i for i, t in enumerate(tyb) if t == 0]
        if cand:
            frames[start_f] = fb[cand[-1]]
            p["protect"] = start_f
            if p["drop_mode"] != "Mai":
                p["drop_mode"] = "Tutti"

    out, st = byte_mosh(frames, au, p, rng)
    write_avi(out_avi, out, w, h, fps)
    st["kf_audio"] = len(kfi)
    st["transfer"] = bool(src_b)
    return fps, st


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="DATAMOSH // Loop507", layout="wide")
st.title("DATAMOSH // Loop507")
st.caption("Minimalismo Computazionale / Glitch Brutalista :: audio-reactive :: puro DSP, nessuna AI")

ACC = {"4/4": 4, "3/4": 3, "6/8": 6, "2/4": 2}
DEFAULTS = {
    "seed": 507, "mosh_range": (15, 100), "max_w": 640, "max_sec": 15, "prev_sec": 4,
    "keep_audio": True, "onset_thr": 0.6, "bloom_prob": 0.02, "trigger": "Onset",
    "bands": False, "bpm_manual": 0.0, "meter": "4/4", "flow_engine": "DIS (veloce)",
    "kf": 12, "qscale": 5, "kf_on_audio": True, "drop_mode": "Su onset", "drop_thr": 0.5,
    "max_dup": 12, "corrupt": 0.002, "quiet_thr": 0.25,
    "mode": "Block MV", "block": 16, "strength": 1.5, "mv_noise": 0.0, "hold_prob": 0.05,
    "bloom_gain": 2.5, "bloom_len": (6, 24), "residual": 0.0, "a_strength": 1.0,
    "a_bloom": True, "a_refresh": False, "refresh_thr": 0.85,
}
KEEP = ("seed", "max_w", "max_sec", "prev_sec", "keep_audio", "bpm_manual", "meter", "flow_engine")
OPTS = {"trigger": ["Onset", "Beat (a tempo)", "Onset + Beat"], "drop_mode": ["Su onset", "Tutti", "Mai"],
        "mode": ["Block MV", "Optical Flow"], "block": [8, 16, 32, 64], "meter": list(ACC),
        "max_w": [320, 480, 640, 854, 1280], "flow_engine": ["DIS (veloce)", "Farneback"]}
LIM = {"seed": (0, 2**31 - 1), "mosh_range": (0, 100), "max_sec": (2, 60), "prev_sec": (2, 8),
       "onset_thr": (0.1, 1.0), "bloom_prob": (0.0, 0.3), "bpm_manual": (0.0, 300.0),
       "kf": (6, 120), "qscale": (2, 15), "drop_thr": (0.1, 1.0), "max_dup": (1, 40),
       "corrupt": (0.0, 0.01), "quiet_thr": (0.0, 0.6), "strength": (0.2, 4.0), "mv_noise": (0.0, 8.0),
       "hold_prob": (0.0, 0.5), "bloom_gain": (1.0, 5.0), "bloom_len": (2, 60), "residual": (0.0, 0.5),
       "a_strength": (0.0, 3.0), "refresh_thr": (0.5, 1.0)}
MANUAL = "— manuale —"
PRESETS = {
    "Glitch ritmico a tempo": ("Tab 1 Byte :: i colpi tolgono gli I-frame e spalmano il movimento, a tempo col brano.",
        {"trigger": "Beat (a tempo)", "kf": 8, "drop_mode": "Su onset", "max_dup": 12, "corrupt": 0.001}),
    "Caos totale": ("Tab 1 Byte :: mosh continuo e corruzione pesante.",
        {"drop_mode": "Tutti", "max_dup": 25, "corrupt": 0.005, "qscale": 8}),
    "Minimale": ("Tab 1 Byte :: pochi duplicati sugli attacchi forti, quasi nessuna corruzione. Effetto sottile.",
        {"drop_mode": "Su onset", "drop_thr": 0.7, "max_dup": 6, "corrupt": 0.0, "onset_thr": 0.7}),
    "Liquido e morbido": ("Tab 2 Optical :: deformazioni fluide, l'immagine resta leggibile.",
        {"mode": "Optical Flow", "strength": 1.0, "residual": 0.1, "a_strength": 1.5, "hold_prob": 0.0}),
    "Codec rotto": ("Tab 2 Optical :: macro-blocchi che sbandano a ogni colpo.",
        {"mode": "Block MV", "block": 32, "strength": 2.0, "mv_noise": 2.0, "a_bloom": True}),
    "Mosh a ondate": ("Tab 2 Optical :: il movimento si accumula, sui colpi forti l'immagine si rinfresca. A ondate col brano.",
        {"trigger": "Beat (a tempo)", "a_refresh": True, "refresh_thr": 0.9, "bloom_len": (8, 30)}),
    "Un video mangia l'altro": ("Tab 1 o 2 :: carica Video A (movimento) e Video B (immagine): A deforma B.",
        {"mode": "Block MV", "block": 16, "mosh_range": (0, 100), "residual": 0.0}),
    "Cassa, rullante, hi-hat": ("Tab 1 Byte :: modo bande: la cassa toglie gli I-frame, il rullante rinfresca, gli hi-hat corrompono.",
        {"bands": True, "drop_mode": "Su onset", "corrupt": 0.003, "max_dup": 14, "kf": 8}),
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
for _k, _v in (("preset", MANUAL), ("dm_out", None), ("dm_report", None)):
    st.session_state.setdefault(_k, _v)


def apply_preset():
    name = st.session_state["preset"]
    if name in PRESETS:
        for k, v in {**DEFAULTS, **PRESETS[name][1]}.items():
            if k not in KEEP:
                st.session_state[k] = v


def valid(k, v):
    d = DEFAULTS[k]
    if k in OPTS:
        return v in OPTS[k]
    if isinstance(d, bool):
        return isinstance(v, bool)
    num = (int, float)
    if isinstance(d, tuple):
        lo, hi = LIM[k]
        return (isinstance(v, (list, tuple)) and len(v) == 2 and v[0] <= v[1] and
                all(isinstance(x, num) and not isinstance(x, bool) and lo <= x <= hi for x in v))
    lo, hi = LIM[k]
    return isinstance(v, num) and not isinstance(v, bool) and lo <= v <= hi


def load_params():
    up = st.session_state.get("pjson")
    if up is None:
        return
    try:
        data, n = json.loads(up.getvalue().decode("utf-8")), 0
        for k, v in data.items():
            if k in DEFAULTS and valid(k, v):
                d = DEFAULTS[k]
                if isinstance(d, tuple):
                    v = tuple(v)
                elif isinstance(d, float):
                    v = float(v)
                elif isinstance(d, int) and not isinstance(d, bool):
                    v = int(v)
                st.session_state[k] = v
                n += 1
        st.session_state["pmsg"] = f"Caricati {n} parametri / loaded {n} params"
    except Exception:  # noqa: BLE001
        st.session_state["pmsg"] = "File JSON non valido / invalid JSON"


def params_json():
    return json.dumps({k: st.session_state[k] for k in DEFAULTS}, indent=1)


st.selectbox("Preset (riporta i controlli ai valori del preset, poi puoi ritoccarli)",
             [MANUAL] + list(PRESETS), key="preset", on_change=apply_preset)
if st.session_state["preset"] in PRESETS:
    st.caption(PRESETS[st.session_state["preset"]][0])

VT = ["mp4", "mov", "avi", "mkv", "webm"]
c1, c2 = st.columns(2)
up_a = c1.file_uploader("Video A (movimento + audio) / motion + audio", type=VT, key="a")
up_b = c2.file_uploader("Video B opzionale (immagine/texture che A deforma)", type=VT, key="b")
up_au = st.file_uploader("Audio esterno opzionale (il tuo brano: guida il mosh e diventa la colonna sonora)",
                         type=["wav", "mp3", "flac", "ogg", "m4a", "aac"], key="au")

with st.sidebar:
    st.header("Comuni / Common")
    seed = st.number_input("Seed", 0, 2**31 - 1, key="seed")
    st.button("Seed casuale / Random seed",
              on_click=lambda: st.session_state.update(seed=random.randint(0, 2**31 - 1)))
    mosh_range = st.slider("Finestra mosh / window (%)", 0, 100, key="mosh_range")
    max_w = st.select_slider("Larghezza max / Max width", OPTS["max_w"], key="max_w")
    max_sec = st.slider("Durata max (s) / Max seconds", 2, 60, key="max_sec")
    prev_sec = st.slider("Secondi anteprima / Preview seconds", 2, 8, key="prev_sec")
    keep_audio = st.checkbox("Mantieni audio (A o esterno) / Keep audio", key="keep_audio")
    trigger = st.radio("Trigger audio", OPTS["trigger"], key="trigger",
                       help="Beat = scatti precisi sul tempo del brano (battere piu' forte).")
    onset_thr = st.slider("Soglia onset / Onset threshold", 0.1, 1.0, step=0.05, key="onset_thr")
    bloom_prob = st.slider("Bloom casuale / Random bloom prob", 0.0, 0.3, step=0.01, key="bloom_prob")
    with st.expander("Audio avanzato / Advanced audio"):
        bands = st.checkbox("Modo bande: bassi = I-frame/bloom, medi = refresh, acuti = corruzione", key="bands")
        bpm_manual = st.number_input("BPM manuale (0 = automatico)", 0.0, 300.0, step=1.0, key="bpm_manual")
        meter = st.radio("Battuta / Meter", OPTS["meter"], key="meter", horizontal=True)
    with st.expander("Salva / carica parametri"):
        st.download_button("Salva parametri (JSON)", params_json(),
                           file_name="datamosh_params.json", mime="application/json")
        st.file_uploader("Carica parametri (JSON)", type=["json"], key="pjson", on_change=load_params)
        if st.session_state.get("pmsg"):
            st.caption(st.session_state["pmsg"])


def save_uploads():
    tmp = tempfile.mkdtemp()
    out = []
    for up in (up_a, up_b, up_au):
        if up is None:
            out.append(None)
            continue
        pth = os.path.join(tmp, up.name)
        with open(pth, "wb") as f:
            f.write(up.getbuffer())
        out.append(pth)
    return tmp, out[0], out[1], out[2]


def common(preview=False):
    return {"seed": int(seed), "mosh_start": mosh_range[0], "mosh_end": mosh_range[1],
            "max_w": int(min(max_w, 320) if preview else max_w),
            "max_sec": int(prev_sec if preview else max_sec),
            "onset_thr": onset_thr, "trigger": trigger, "bloom_prob": bloom_prob,
            "bands": bands, "bpm_manual": float(bpm_manual), "meter": meter}


def au_for(pa, pau, msec):
    cap = cv2.VideoCapture(pa)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    au = audio_features(pau or pa, fps, int(msec * fps) + 2, msec, float(bpm_manual), ACC[meter])
    return route_audio(au, trigger, bands)


def publish(final, rep):
    with open(final, "rb") as f:
        st.session_state["dm_out"] = f.read()
    st.session_state["dm_report"] = rep


def run_byte(preview):
    tmp, pa, pb, pau = save_uploads()
    try:
        p = {**common(preview), "kf": kf, "qscale": qscale, "drop_mode": drop_mode,
             "drop_thr": drop_thr, "max_dup": max_dup, "corrupt": corrupt,
             "quiet_thr": quiet_thr, "kf_on_audio": kf_on_audio}
        p["au"] = au_for(pa, pau, p["max_sec"])
        avi, final = os.path.join(tmp, "mosh.avi"), os.path.join(tmp, "out.mp4")
        with st.spinner("Moshing bytes..."):
            fps, stt = byte_datamosh(pa, pb, avi, p)
            finalize_h264(avi, pau or pa, final, keep_audio, p["max_sec"])
        publish(final, ("BYTE" + (" (anteprima)" if preview else ""), p, fps, stt, p["au"]["ok"]))
    except Exception as e:  # noqa: BLE001
        st.error(f"Errore / Error :: {e}")


def run_optical(preview):
    tmp, pa, pb, pau = save_uploads()
    try:
        p = {**common(preview), "mode": mode, "block": int(block), "strength": strength,
             "mv_noise": mv_noise, "iframe_every": 0, "iframe_prob": 0.0,
             "hold_prob": hold_prob, "bloom_gain": bloom_gain,
             "bloom_len": (int(bloom_len[0]), int(bloom_len[1])), "residual": residual,
             "a_strength": a_strength, "a_bloom": a_bloom, "a_refresh": a_refresh,
             "refresh_thr": refresh_thr, "flow_engine": flow_engine}
        p["au"] = au_for(pa, pau, p["max_sec"])
        raw, final = os.path.join(tmp, "raw.mp4"), os.path.join(tmp, "out.mp4")
        bar = st.progress(0.0)
        fps, stt = datamosh_video(pa, pb, raw, p, progress_cb=lambda v: bar.progress(min(1.0, v)))
        finalize_h264(raw, pau or pa, final, keep_audio, p["max_sec"])
        bar.empty()
        publish(final, ("OPTICAL" + (" (anteprima)" if preview else ""), p, fps, stt, p["au"]["ok"]))
    except Exception as e:  # noqa: BLE001
        st.error(f"Errore / Error :: {e}")


tab1, tab2 = st.tabs(["1 // BYTE (AVI)", "2 // OPTICAL FLOW"])

with tab1:
    st.write("Elimina I-frame, duplica P-frame e corrompe byte :: guidato dall'audio. "
             "Con il Video B: l'immagine e' di B, il movimento di A.")
    a, b, c = st.columns(3)
    drop_mode = a.radio("Elimina I-frame / Drop I-frames", OPTS["drop_mode"], key="drop_mode")
    max_dup = b.slider("Max duplicati P per onset / Max P dups", 1, 40, key="max_dup")
    corrupt = c.slider("Corruzione byte / Byte corruption", 0.0, 0.01, step=0.0005, format="%.4f", key="corrupt")
    with st.expander("Avanzati / Advanced"):
        a2, b2, c2 = st.columns(3)
        kf = a2.slider("Keyframe ogni N / Keyframe every N", 6, 120, key="kf")
        qscale = a2.slider("Qualita' (basso = migliore) / q-scale", 2, 15, key="qscale")
        kf_on_audio = a2.checkbox("Keyframe sui picchi audio (consigliato)", key="kf_on_audio")
        drop_thr = b2.slider("Soglia drop I / Drop threshold", 0.1, 1.0, step=0.05, key="drop_thr")
        quiet_thr = c2.slider("Soglia quiete (recupero sync) / Quiet thr", 0.0, 0.6, step=0.05, key="quiet_thr")
    r1, r2 = st.columns(2)
    go1 = r1.button("GENERA BYTE / GENERATE", type="primary", disabled=up_a is None)
    pv1 = r2.button("ANTEPRIMA / PREVIEW", disabled=up_a is None, key="pv1")
    if (go1 or pv1) and up_a is not None:
        run_byte(pv1)

with tab2:
    st.write("Optical flow :: transfer A->B opzionale :: guidato dall'audio.")
    a, b, c = st.columns(3)
    mode = a.radio("Modo / Mode", OPTS["mode"], key="mode")
    strength = b.slider("Forza / Strength", 0.2, 4.0, step=0.1, key="strength")
    residual = c.slider("Residuo / Leak", 0.0, 0.5, step=0.01, key="residual")
    with st.expander("Avanzati / Advanced"):
        a2, b2, c2 = st.columns(3)
        block = a2.select_slider("Blocco / Block", OPTS["block"], key="block")
        a_strength = a2.slider("RMS -> forza / strength", 0.0, 3.0, step=0.1, key="a_strength")
        a_bloom = a2.checkbox("Onset -> bloom", key="a_bloom")
        a_refresh = a2.checkbox("Onset forte -> I-frame / strong onset refresh", key="a_refresh")
        refresh_thr = a2.slider("Soglia refresh / Refresh thr", 0.5, 1.0, step=0.05, key="refresh_thr")
        mv_noise = b2.slider("Rumore MV / MV noise", 0.0, 8.0, step=0.5, key="mv_noise")
        hold_prob = b2.slider("Hold prob", 0.0, 0.5, step=0.01, key="hold_prob")
        flow_engine = b2.radio("Motore flow / Flow engine", OPTS["flow_engine"], key="flow_engine")
        bloom_gain = c2.slider("Bloom gain", 1.0, 5.0, step=0.1, key="bloom_gain")
        bloom_len = c2.slider("Bloom durata / length", 2, 60, key="bloom_len")
    r1, r2 = st.columns(2)
    go2 = r1.button("GENERA OPTICAL / GENERATE", type="primary", disabled=up_a is None)
    pv2 = r2.button("ANTEPRIMA / PREVIEW", disabled=up_a is None, key="pv2")
    if (go2 or pv2) and up_a is not None:
        run_optical(pv2)

if st.session_state["dm_out"] is not None:
    st.video(st.session_state["dm_out"])
    st.download_button("SCARICA MP4 / DOWNLOAD MP4", st.session_state["dm_out"],
                       file_name="datamosh_loop507.mp4", mime="video/mp4")
    kind, p, fps, stt, au_ok = st.session_state["dm_report"]
    lines = ["DATAMOSH REPORT // LOOP507", f"approccio / approach :: {kind}",
             f"preset :: {st.session_state['preset']}",
             f"seed :: {p['seed']}  fps :: {fps:.2f}",
             f"finestra / window :: {p['mosh_start']}% -> {p['mosh_end']}%",
             f"audio-reactive :: {'si / yes' if au_ok else 'no (audio assente / missing)'}",
             f"trigger :: {p.get('trigger', 'Onset')}  bande/bands :: {p.get('bands', False)}",
             f"bpm :: {p['au'].get('bpm', 0):.1f}  battuta/meter :: {p.get('meter', '4/4')}"
             + (" (BPM manuale)" if p.get("bpm_manual", 0) > 0 else "")]
    lines += [f"{k} :: {v}" for k, v in stt.items()]
    lines.append("dsp :: " + ("AVI/MPEG-4 byte-level" if kind.startswith("BYTE") else "OpenCV " + str(p.get("flow_engine")) + " + remap")
                 + " + librosa :: no AI")
    st.code("\n".join(lines), language="text")
