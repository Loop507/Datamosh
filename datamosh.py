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

import hashlib
import json
import os
import random
import re
import struct
import subprocess
import tempfile
import wave

from scipy.signal import lfilter
import cv2
import librosa
import numpy as np
import streamlit as st


# ============================================================================
# AUDIO: analisi (RMS, onset, beat, bande) e instradamento dei segnali
# ============================================================================

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # fallback: ffmpeg di sistema
    FFMPEG = "ffmpeg"


def at(arr, k):
    return float(arr[min(int(k), len(arr) - 1)])


def zero_au(n):
    keys = ("rms", "onset", "beat", "low", "mid", "high", "trig_bloom", "trig_drop",
            "trig_refresh", "corr", "noise")
    au = {k: np.zeros(n) for k in keys}
    au.update({"bpm": 0.0, "ok": False, "dur": 0.0, "beats": []})
    return au


def analyze_audio(src, max_sec):
    """Parte pesante (da tenere in cache): decodifica e analisi. Ritorna None se non c'e' audio."""
    wav = os.path.join(tempfile.mkdtemp(), "a.wav")
    try:
        subprocess.run([FFMPEG, "-y", "-i", src, "-t", str(max_sec), "-vn", "-ac", "1",
                        "-ar", "22050", wav], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        y, sr = librosa.load(wav, sr=22050, mono=True)
    except Exception:  # noqa: BLE001
        return None
    if y.size < 4096:
        return None
    hop = 512
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    fq = librosa.fft_frequencies(sr=sr, n_fft=2048)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    def band(lo, hi):
        e = np.log1p(spec[(fq >= lo) & (fq < hi)].sum(axis=0))
        return np.maximum(0.0, np.diff(e, prepend=e[0]))

    tempo, bt = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
    return {"ft": librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop), "rms": rms,
            "onset": onset, "low": band(0, 150), "mid": band(150, 2000), "high": band(2000, sr / 2.0),
            "tempo": float(np.atleast_1d(tempo)[0]),
            "bt_t": librosa.frames_to_time(bt, sr=sr, hop_length=hop)}


def features_from_analysis(an, fps, n_frames, bpm_manual=0.0, accent=4, offset=0.0, downbeat=0):
    """Parte leggera: porta l'analisi a un valore per frame e costruisce la griglia dei beat.
    offset = spostamento dei beat in secondi; downbeat = quale battito della battuta e' il primo (0-based)."""
    if an is None:
        return zero_au(n_frames)
    ft = an["ft"]
    vt = np.arange(n_frames) / fps

    def norm(v):
        v = np.interp(vt, ft, v[:len(ft)])
        v = np.where(vt <= ft[-1], v, 0.0)  # oltre la fine dell'audio: silenzio
        return np.clip(v / (np.percentile(v, 99) + 1e-9), 0, 1)

    bt_t = an["bt_t"]
    if bpm_manual > 0:  # griglia a BPM fissi, con la fase del primo beat rilevato
        period = 60.0 / bpm_manual
        t0 = (bt_t[0] % period) if len(bt_t) else 0.0
        bt_t, bpm = np.arange(t0, ft[-1], period), float(bpm_manual)
    else:
        bpm = an["tempo"]
        if len(bt_t) >= 4:  # dai beat rilevati: piu' preciso della stima grezza (es. 130,0 invece di 129,2)
            bpm = float(60.0 / np.median(np.diff(bt_t)))
    beat, beats = np.zeros(n_frames), []
    for i, tb in enumerate(np.asarray(bt_t) + offset):
        is_acc = (i - downbeat) % accent == 0
        beats.append((float(tb), bool(is_acc)))
        j = int(round(tb * fps))
        if 0 <= j < n_frames:
            beat[j] = 1.0 if is_acc else 0.75  # accento sul primo battito della battuta
    au = zero_au(n_frames)
    au.update({"rms": norm(an["rms"]), "onset": norm(an["onset"]), "beat": beat, "bpm": bpm, "ok": True,
               "low": norm(an["low"]), "mid": norm(an["mid"]), "high": norm(an["high"]),
               "dur": float(ft[-1]), "beats": beats})
    return au


def audio_features(src, fps, n_frames, max_sec, bpm_manual=0.0, accent=4, offset=0.0, downbeat=0):
    """RMS, onset, beat e 3 bande (basso/medio/acuto) per frame, normalizzati 0-1 (librosa)."""
    return features_from_analysis(analyze_audio(src, max_sec), fps, n_frames, bpm_manual, accent, offset, downbeat)


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


DRIVERS = ["Fisso", "Volume", "Bassi", "Medi", "Acuti", "Colpi/Beat"]


def envelope(sig, decay):
    out, e = np.zeros(len(sig)), 0.0
    for i, v in enumerate(sig):
        e = max(float(v), e * decay)
        out[i] = e
    return out


def driver_env(au, name, decay):
    key = {"Volume": "rms", "Bassi": "low", "Medi": "mid", "Acuti": "high", "Colpi/Beat": "onset"}.get(name)
    return None if key is None else envelope(au[key], decay)

# ============================================================================
# MOTORI VIDEO: byte-level, optical flow, residuo, databending, pixel sorting
# ============================================================================

def make_writer(path, fps, w, h):
    """File .avi = MJPG (quasi senza perdite, per i passaggi intermedi); altrimenti mp4v."""
    code = "MJPG" if str(path).lower().endswith(".avi") else "mp4v"
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, (w, h))


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


MASKS = ["Ovunque", "Solo dove si muove", "Solo dove e' fermo", "Luminosita' alta", "Luminosita' bassa"]


MT_DEFAULTS = {"gx": 0.0, "gy": 0.0, "zoom": 0.0, "zoom_drv": "Colpi/Beat", "rot": 0.0, "shear": 0.0,
               "fluid": 0.0, "shift": 0.0, "mirror": "No", "mask": "Ovunque", "mthr": 1.0}


MT_LIM = {"gx": (-8.0, 8.0), "gy": (-8.0, 8.0), "zoom": (-3.0, 3.0), "rot": (-3.0, 3.0), "shear": (-3.0, 3.0),
          "fluid": (0.0, 1.0), "shift": (0.0, 1.0), "mthr": (0.2, 6.0)}


def transform_flow(flow, mt, zenv_t, fs, rng, gx, gy, block):
    """Gravita', zoom, rotazione, shear, specchio, blocchi sfasati e 'fluido' sul campo di moto."""
    h, w = flow.shape[:2]
    f = flow.copy()
    if mt["fluid"] > 0:  # movimento medio: sfocatura spaziale + media nel tempo
        k = 1 + 2 * int(25 * mt["fluid"])
        if k > 1:
            f = cv2.GaussianBlur(f, (k, k), 0)
        a = 0.85 * mt["fluid"]
        if fs.get("flow") is not None:
            f = (1 - a) * f + a * fs["flow"]
        fs["flow"] = f.copy()
    cx, cy, sc = w / 2.0, h / 2.0, w / 2.0
    if mt["zoom"]:
        f -= mt["zoom"] * zenv_t * 6.0 * np.stack([gx - cx, gy - cy], -1) / sc
    if mt["rot"]:
        f += mt["rot"] * 6.0 * np.stack([-(gy - cy), gx - cx], -1) / sc
    if mt["shear"]:
        f[..., 0] += mt["shear"] * 6.0 * (gy - cy) / cy
    f[..., 0] -= mt["gx"]
    f[..., 1] -= mt["gy"]
    if mt["mirror"] == "Orizzontale":
        half = w // 2
        f[:, w - half:] = f[:, :half][:, ::-1] * np.array([-1.0, 1.0], np.float32)
    elif mt["mirror"] == "Verticale":
        half = h // 2
        f[h - half:] = f[:half][::-1] * np.array([1.0, -1.0], np.float32)
    if mt["shift"] > 0:
        nby, nbx = h // block, w // block
        if nby > 0 and nbx > 0:
            sel = rng.random((nby, nbx)) < mt["shift"]
            vec = rng.integers(-12, 13, (nby, nbx, 2)).astype(np.float32)
            field = np.where(sel[..., None], vec, 0.0).astype(np.float32)
            f[:nby * block, :nbx * block] += cv2.resize(field, (nbx * block, nby * block),
                                                        interpolation=cv2.INTER_NEAREST)
    return f


def motion_mask(mt, raw_flow, gray):
    """Mappa 0-1 di dove agisce il mosh (None = ovunque)."""
    kind = mt["mask"]
    if kind == "Ovunque":
        return None
    if kind in ("Solo dove si muove", "Solo dove e' fermo"):
        mag = cv2.GaussianBlur(np.linalg.norm(raw_flow, axis=2).astype(np.float32), (0, 0), 6)
        m = np.clip((mag - mt["mthr"]) / max(0.2, mt["mthr"]), 0, 1)
        return m if kind == "Solo dove si muove" else 1.0 - m
    g = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (0, 0), 6)
    m = np.clip((g - 0.5) * 6 + 0.5, 0, 1)
    return m if kind == "Luminosita' alta" else 1.0 - m


PS_KEYS = ["Luminanza", "Tinta", "Saturazione", "Rosso", "Verde", "Blu"]


PS_MODES = ["Soglia", "Bordi", "Casuale"]


def ps_key(img, name):
    if name == "Luminanza":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if name in ("Tinta", "Saturazione"):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[..., 0].astype(np.float32) / 179.0 if name == "Tinta" else hsv[..., 1].astype(np.float32) / 255.0
    return img[..., {"Blu": 0, "Verde": 1, "Rosso": 2}[name]].astype(np.float32) / 255.0


def sort_rows(img, key, sortable, maxlen, rng, reverse, skip, breaks=None):
    """Ordina i pixel dentro ogni intervallo di una riga (intervallo = serie di pixel 'sortable')."""
    h, w = key.shape
    start = sortable.copy()
    start[:, 1:] = sortable[:, 1:] & ~sortable[:, :-1]
    if breaks is not None:
        start |= breaks & sortable
    run_id = np.cumsum(start.ravel()).reshape(h, w)
    if skip > 0 and run_id.max() > 0:  # una parte degli intervalli resta com'e'
        sortable = sortable & (rng.random(int(run_id.max()) + 1) >= skip)[run_id]
    gid = run_id
    if maxlen > 0:
        idx = np.arange(w)[None, :]
        rs = np.maximum.accumulate(np.where(start, idx, 0), axis=1)
        gid = run_id * (w // maxlen + 2) + (idx - rs) // maxlen
    sel = np.flatnonzero(sortable.ravel())
    if sel.size == 0:
        return img
    k = key.ravel()[sel]
    order = np.lexsort((-k if reverse else k, gid.ravel()[sel]))
    flat = img.reshape(-1, 3)
    out = flat.copy()
    out[sel] = flat[sel[order]]
    return out.reshape(img.shape)


def pixelsort_frame(frame, p, env, rng):
    ang = int(p["angle"]) % 180
    h, w = frame.shape[:2]
    valid = None
    if ang == 0:
        img = frame
    elif ang == 90:
        img = np.ascontiguousarray(frame.transpose(1, 0, 2))
    else:
        # tela quadrata abbastanza grande da non tagliare gli angoli quando si ruota
        dd = int(np.ceil(np.hypot(w, h)))
        py, px = (dd - h) // 2, (dd - w) // 2
        canvas = cv2.copyMakeBorder(frame, py, dd - h - py, px, dd - w - px, cv2.BORDER_REPLICATE)
        mtx = cv2.getRotationMatrix2D((dd / 2.0, dd / 2.0), ang, 1.0)
        img = cv2.warpAffine(canvas, mtx, (dd, dd), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        inside = np.zeros((dd, dd), np.uint8)
        inside[py:py + h, px:px + w] = 255
        valid = cv2.warpAffine(inside, mtx, (dd, dd)) > 200  # si ordina solo l'immagine vera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lo, hi = p["range"]
    ex = p["expand"] * env
    ih, iw = gray.shape
    breaks, skip, maxlen = None, p["skip"], int(p["maxlen"])
    if p["imode"] == "Soglia":
        g = gray.astype(np.float32) / 255.0
        sortable = (g >= lo - ex) & (g <= hi + ex)
    elif p["imode"] == "Bordi":
        sc = 1.0 - 0.7 * min(1.0, ex / 0.6)  # con l'audio: piu' bordi, intervalli piu' corti
        t1 = max(5, int(lo * 255 * sc))
        edges = cv2.dilate(cv2.Canny(gray, t1, max(t1 + 10, int(hi * 255 * sc * 0.6))), np.ones((3, 3), np.uint8))
        sortable = edges == 0
    else:  # intervalli di lunghezza casuale
        length = maxlen or 60
        breaks = rng.random((ih, iw)) < 1.0 / length
        sortable = np.ones((ih, iw), bool)
        skip, maxlen = max(0.0, skip - ex), 0
    if valid is not None:
        sortable = sortable & valid
    res = sort_rows(img, ps_key(img, p["key"]), sortable, maxlen, rng, p["reverse"], skip, breaks)
    if ang == 0:
        out = res
    elif ang == 90:
        out = np.ascontiguousarray(res.transpose(1, 0, 2))
    else:
        back = cv2.warpAffine(res, mtx, (dd, dd), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
        out = np.ascontiguousarray(back[py:py + h, px:px + w])
    return out if p["mix"] >= 1 else cv2.addWeighted(frame, 1 - p["mix"], out, p["mix"], 0)


def pixelsort_video(src_video, src_image, out_path, p, progress_cb=None):
    """Video, oppure immagine fissa che si anima col brano (serve l'audio)."""
    rng = np.random.default_rng(p["seed"])
    au, cap = p["au"], None
    if src_image:
        img = cv2.imread(src_image)
        if img is None:
            raise RuntimeError("Immagine non leggibile / unreadable image")
        fps = float(p["fps_img"])
        oh, ow = img.shape[:2]
        n = min(int(p["max_sec"] * fps), int(au["dur"] * fps))
    else:
        cap = cv2.VideoCapture(src_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        ow, oh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1, int(p["max_sec"] * fps))
    if n < 2:
        raise RuntimeError("Audio o video troppo corto / source too short")
    scale = min(1.0, p["max_w"] / float(ow))
    w, h = max(16, int(ow * scale) // 2 * 2), max(16, int(oh * scale) // 2 * 2)
    if src_image:
        img = fit_frame(img, w, h)
    writer = make_writer(out_path, fps, w, h)
    env = driver_env(au, p["drv"], p["release"])
    start_f, end_f = int(n * p["mosh_start"] / 100), int(n * p["mosh_end"] / 100)
    done = sorted_n = 0
    levels = []  # forza dell'effetto per fotogramma (serve all'audio)
    for t in range(n):
        if src_image:
            frame = img
        else:
            ok, fr = cap.read()
            if not ok:
                break
            frame = fit_frame(fr, w, h)
        if start_f <= t < end_f:
            lv = 1.0 if env is None else at(env, t)
            out = pixelsort_frame(frame, p, lv, rng)
            sorted_n += 1
        else:
            out, lv = frame, 0.0
        writer.write(out)
        levels.append(float(lv))
        done += 1
        if progress_cb and t % 5 == 0:
            progress_cb(t / max(1, n - 1))
    writer.release()
    if cap is not None:
        cap.release()
    return fps, {"frames": done, "sorted": sorted_n, "source": "image" if src_image else "video", "env": levels}


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
    mt, fs = p.get("mt"), {}
    zenv = driver_env(au, mt["zoom_drv"], 0.8) if (mt and au) else None

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

    writer = make_writer(out_path, fps, w, h)
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

    stats = {"frames": 1, "iframes": 1, "holds": 0, "blooms": 0, "ev": [0]}
    bloom_flow, bloom_left, b_loaded = None, 0, False

    for t in range(1, n_max):
        ok, frame = cap.read()
        if not ok:
            break
        frame = fit_frame(frame, w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        in_window = start_f <= t < end_f
        code = 0  # evento del fotogramma per l'audio: 0 reale, 1 mosh, 2 hold, 3/4 bloom, 5 I-frame

        if not in_window:
            # fuori dalla finestra: video reale, e il mosh riparte dall'ultimo frame reale
            mosh = frame.astype(np.float32)
            out = frame
        else:
            code = 1
            # entrando nella finestra: se c'e' B, la texture di partenza e' B
            if cap_b is not None and not b_loaded:
                okb, fb = cap_b.read()
                if okb:
                    mosh = fit_frame(fb, w, h).astype(np.float32)
                b_loaded = True

            # I-frame di rinfresco
            refresh = (p["iframe_every"] > 0 and (t - start_f) % p["iframe_every"] == 0
                       and t != start_f) or (rng.random() < p["iframe_prob"]) \
                or bool(au and p["a_refresh"] and at(au["trig_refresh"], t) > p["refresh_thr"])
            if refresh:
                mosh = frame.astype(np.float32)
                stats["iframes"] += 1
                code = 5
                bloom_left = 0

            # flow del frame
            k_str = p["strength"] * (1.0 + p["a_strength"] * at(au["rms"], t) if au else 1.0)
            raw = compute_flow(prev_gray, gray, dis)
            flow = raw * k_str
            if p["mode"] == "Block MV":
                flow = to_block_mv(flow, p["block"], rng,
                                p["mv_noise"] + (at(au["noise"], t) * 4.0 if au else 0.0))

            # hold: freeze (P-frame vuoto)
            if rng.random() < p["hold_prob"]:
                flow = np.zeros_like(flow)
                stats["holds"] += 1
                code = 2

            # bloom: ripete lo stesso flow per k frame
            if bloom_left > 0 and bloom_flow is not None:
                flow = bloom_flow
                bloom_left -= 1
                code = 4
            elif (au and p["a_bloom"] and at(au["trig_bloom"], t) > p["onset_thr"]) or rng.random() < p["bloom_prob"]:
                bloom_flow = flow.copy() * p["bloom_gain"]
                bloom_left = int(rng.integers(p["bloom_len"][0], p["bloom_len"][1] + 1))
                stats["blooms"] += 1
                code = 3

            if mt:
                flow = transform_flow(flow, mt, 1.0 if zenv is None else at(zenv, t), fs, rng,
                                      grid_x, grid_y, p["block"])
            mosh = warp(mosh, flow, grid_x, grid_y)

            # residuo: quanto del frame reale "trapela" nel mosh
            if p["residual"] > 0:
                mosh = cv2.addWeighted(mosh, 1.0 - p["residual"],
                                       frame.astype(np.float32), p["residual"], 0)

            if mt:
                mk = motion_mask(mt, raw, gray)
                if mk is not None:  # fuori dalla mappa torna il video reale
                    mosh = mosh * mk[..., None] + frame.astype(np.float32) * (1.0 - mk[..., None])
            out = np.clip(mosh, 0, 255).astype(np.uint8)

        writer.write(out)
        stats["ev"].append(code)
        prev_gray = gray
        stats["frames"] += 1
        if progress_cb and t % 5 == 0:
            progress_cb(t / max(1, n_max - 1))

    writer.release()
    cap.release()
    if cap_b is not None:
        cap_b.release()
    return fps, stats


def residual_video(src_a, src_b, out_path, p, progress_cb=None):
    """Simula un decoder P-frame: out = warp(out_prev, moto) + guadagno * residuo (accumulo + clipping).
    Con guadagno > 1 e quantizzazione l'errore si accumula: colori saturi e coriandoli sui bordi."""
    rng = np.random.default_rng(p["seed"])
    au = p.get("au")
    dis = (cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
           if str(p["flow_engine"]).startswith("DIS") else None)
    cap = cv2.VideoCapture(src_a)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ow, oh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, p["max_w"] / float(ow))
    w, h = max(16, int(ow * scale) // 2 * 2), max(16, int(oh * scale) // 2 * 2)
    n_max = min(total, int(p["max_sec"] * fps))
    cap_b = cv2.VideoCapture(src_b) if src_b else None
    writer = make_writer(out_path, fps, w, h)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    start_f, end_f = int(n_max * p["mosh_start"] / 100), int(n_max * p["mosh_end"] / 100)

    def ycc(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Impossibile leggere il video / cannot read video")
    first = fit_frame(first, w, h)
    prev_gray, prev_f = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), ycc(first)
    out_f = prev_f.copy()
    writer.write(first)
    gain = np.array([p["res_luma"], p["res_chroma"], p["res_chroma"]], np.float32)
    mt, fs = p.get("mt"), {}
    zenv = driver_env(au, mt["zoom_drv"], 0.8) if (mt and au) else None
    step = float(p["res_quant"])
    stt = {"frames": 1, "iframes": 0, "holds": 0, "blooms": 0, "ev": [0]}
    bloom_flow, bloom_left, b_loaded = None, 0, False
    for t in range(1, n_max):
        ok, fr = cap.read()
        if not ok:
            break
        frame = fit_frame(fr, w, h)
        gray, cur_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), ycc(frame)
        code = 0  # evento del fotogramma per l'audio: 0 reale, 1 mosh, 2 hold, 3/4 bloom, 5 I-frame
        if not (start_f <= t < end_f):
            out_f, out = cur_f.copy(), frame
        else:
            code = 1
            if cap_b is not None and not b_loaded:
                okb, fb = cap_b.read()
                if okb:
                    out_f = ycc(fit_frame(fb, w, h))
                b_loaded = True
            if au and p["a_refresh"] and at(au["trig_refresh"], t) > p["refresh_thr"]:
                out_f = cur_f.copy()  # "I-frame": azzera l'accumulo
                bloom_left = 0
                stt["iframes"] += 1
                code = 5
            k_str = p["strength"] * (1.0 + p["a_strength"] * at(au["rms"], t) if au else 1.0)
            raw = compute_flow(prev_gray, gray, dis)
            true = to_block_mv(raw, p["block"], rng, 0.0)  # moto "vero" a macroblocchi
            mflow = to_block_mv(raw * k_str, p["block"], rng,
                                p["mv_noise"] + (at(au["noise"], t) * 4.0 if au else 0.0))
            if rng.random() < p["hold_prob"]:
                mflow = np.zeros_like(mflow)
                stt["holds"] += 1
                code = 2
            if bloom_left > 0 and bloom_flow is not None:
                mflow, bloom_left = bloom_flow, bloom_left - 1
                code = 4
            elif (au and p["a_bloom"] and at(au["trig_bloom"], t) > p["onset_thr"]) or \
                    rng.random() < p["bloom_prob"]:
                bloom_flow = mflow * p["bloom_gain"]
                bloom_left = int(rng.integers(p["bloom_len"][0], p["bloom_len"][1] + 1))
                stt["blooms"] += 1
                code = 3
            if mt:
                mflow = transform_flow(mflow, mt, 1.0 if zenv is None else at(zenv, t), fs, rng,
                                       gx, gy, p["block"])
            r = cur_f - warp(prev_f, true, gx, gy)  # il residuo che manderebbe l'encoder
            if p.get("grain", 0) > 0:  # grana: rumore anche dove nulla si muove, cosi' i coriandoli arrivano ovunque
                r = r + rng.normal(0.0, p["grain"], r.shape).astype(np.float32)
            if step > 0:
                r = np.round(r / step) * step
            gk = 1.0 + (p["a_strength"] * at(au["rms"], t) if au else 0.0)
            out_f = np.clip(warp(out_f, mflow, gx, gy) + (1.0 + (gain - 1.0) * min(gk, 3.0)) * r, 0, 255)
            if p.get("decay", 0) > 0:  # l'errore svanisce piano piano verso il video reale
                out_f = (1.0 - p["decay"]) * out_f + p["decay"] * cur_f
            if mt:
                mk = motion_mask(mt, raw, gray)
                if mk is not None:
                    out_f = out_f * mk[..., None] + cur_f * (1.0 - mk[..., None])
            out = cv2.cvtColor(out_f.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        writer.write(out)
        stt["ev"].append(code)
        prev_gray, prev_f = gray, cur_f
        stt["frames"] += 1
        if progress_cb and t % 5 == 0:
            progress_cb(t / max(1, n_max - 1))
    writer.release()
    cap.release()
    if cap_b is not None:
        cap_b.release()
    return fps, stt


FORMATS = ["Originale", "16:9", "9:16", "1:1"]
FIT_MODES = ["Ritaglia", "Bordi neri"]


def format_size(fmt, max_w):
    """Dimensioni (pari) per un formato social, con lo stesso numero di pixel in tutti i formati:
    16:9 usa max_w come larghezza, 9:16 come altezza, 1:1 come 0,75 del lato."""
    mw = int(max_w)
    if fmt == "16:9":
        tw, th = mw, mw * 9 / 16
    elif fmt == "9:16":
        tw, th = mw * 9 / 16, mw
    else:
        tw = th = mw * 0.75
    return max(16, int(round(tw / 2)) * 2), max(16, int(round(th / 2)) * 2)


def format_video(src, out_avi, tw, th, fit, max_sec):
    """Porta un video al formato scelto (ritaglio al centro o bordi neri), solo per i secondi che servono."""
    if fit == "Bordi neri":
        vf = (f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
              f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
    else:
        vf = f"scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},setsar=1"
    subprocess.run([FFMPEG, "-y", "-i", src, "-t", str(max_sec), "-an", "-vf", vf, "-c:v", "mjpeg", "-q:v", "2",
                    out_avi], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_avi


def format_image(src, out_png, tw, th, fit):
    img = cv2.imread(src)
    if img is None:
        raise RuntimeError("Immagine non leggibile / unreadable image")
    h, w = img.shape[:2]
    if fit == "Bordi neri":
        s = min(tw / w, th / h)
        rw, rh = min(tw, max(1, int(round(w * s)))), min(th, max(1, int(round(h * s))))
    else:
        s = max(tw / w, th / h)
        rw, rh = max(tw, int(np.ceil(w * s))), max(th, int(np.ceil(h * s)))
    r = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((th, tw, 3), np.uint8)
    if fit == "Bordi neri":
        y0, x0 = (th - rh) // 2, (tw - rw) // 2
        canvas[y0:y0 + rh, x0:x0 + rw] = r
    else:
        y0, x0 = (rh - th) // 2, (rw - tw) // 2
        canvas = np.ascontiguousarray(r[y0:y0 + th, x0:x0 + tw])
    cv2.imwrite(out_png, canvas)
    return out_png


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


def decode_pcm(path, sr, max_sec):
    """Audio della sorgente come float32 stereo (n, 2), oppure None se non c'e' una traccia audio."""
    r = subprocess.run([FFMPEG, "-v", "error", "-t", str(max_sec), "-i", path, "-vn", "-ac", "2", "-ar", str(sr),
                        "-f", "f32le", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    a = np.frombuffer(r.stdout, np.float32)
    a = a[:a.size // 2 * 2].reshape(-1, 2).copy()
    return a if len(a) else None


def slice_audio(a, smap, fps, sr, fade_ms=3):
    """Ogni slot ripete o salta la fettina di suono del fotogramma mostrato; dove e' continuo non c'e' nessun taglio."""
    n, spf = len(smap), sr / float(fps)
    total = int(round(n * spf))

    def b(i):
        return int(round(i * spf))

    a = np.vstack([a, np.zeros((max(0, total + 4 - len(a)), 2), np.float32)])
    out, fade = np.zeros((total, 2), np.float32), max(1, int(sr * fade_ms / 1000))
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
    prev, src_pos = None, 0
    for k, sf in enumerate(smap):
        lo, hi = b(k), b(k + 1)
        sf = int(sf)
        joined = prev is not None and sf == prev + 1  # continuo: niente salto, niente dissolvenza
        start = src_pos if joined else b(sf)
        chunk = a[start:start + (hi - lo)]
        out[lo:lo + len(chunk)] = chunk
        if not joined and prev is not None and hi - lo > 2 * fade:
            out[lo - fade:lo] *= ramp[::-1]
            out[lo:lo + fade] *= ramp
        prev, src_pos = sf, start + (hi - lo)
    return out, a[:total]


def bend_audio(a, amt, delay_s, sr, rng):
    """Gli effetti del databending applicati al suono (a: float32 (n, 2)): stessi nomi, stesse quantita'."""
    a = np.array(a, np.float32)  # copia: burn e invert scrivono sul posto e non devono toccare il flusso originale
    n = len(a)

    def g(k):
        return float(amt.get(k, 0.0))

    def band(frac):
        h = max(2, int(n * frac))
        y0 = int(rng.integers(0, max(1, n - h)))
        return y0, min(n, y0 + h)

    if g("slip") > 0.03:  # cancella campioni: tutto il resto scivola
        k = int(g("slip") * 400) + 1
        for _ in range(1 + int(2 * g("slip"))):
            pos = int(rng.integers(0, max(1, n - k)))
            a = np.concatenate([a[:pos], a[pos + k:], np.zeros((k, 2), np.float32)])
    if g("echo") > 0.03:
        gain, d = 0.85 * g("echo"), max(8, int(delay_s * sr))
        y, norm = a.copy(), 1.0
        for k in (1, 2, 3):
            sh = k * d
            if sh < n:
                y[sh:] += (gain ** k) * a[:-sh]
                norm += gain ** k
        a = y / norm
    if g("reverb") > 0.03:
        al = 1.0 - 10.0 ** (-(0.3 + 2.4 * g("reverb")))
        wet = lfilter([1 - al], [1, -al], a, axis=0).astype(np.float32)
        wet = np.where(np.abs(wet) > np.abs(a), wet, a)
        a = a * (1 - g("reverb")) + wet * g("reverb")
    if g("hp") > 0.03:
        low = lfilter([0.1], [1, -0.9], a, axis=0).astype(np.float32)
        a = a * (1 - g("hp")) + (a - low) * 2 * g("hp")
    if g("dist") > 0.03:
        drive = 1 + 14 * g("dist")
        a = np.tanh(drive * a) / np.tanh(drive)
    if g("burn") > 0.03:
        gain = 1 + 7 * g("burn")
        for _ in range(1 + int(3 * g("burn"))):
            s0, s1 = band(0.05 + 0.25 * g("burn"))
            a[s0:s1] = np.clip(a[s0:s1] * np.linspace(1, gain, s1 - s0, dtype=np.float32)[:, None], -1, 1)
    if g("invert") > 0.05:  # nel suono la polarita' non si sente: la fascia viene letta al contrario
        for _ in range(1 + int(3 * g("invert"))):
            s0, s1 = band(0.04 + 0.2 * g("invert"))
            a[s0:s1] = a[s0:s1][::-1].copy()
    if g("crush") > 0.03:
        lv = 2 + int(30 * (1 - g("crush")))
        a = np.round(a * lv) / lv
        if g("crush") > 0.3:
            k = 3 * (1 + int(g("crush") * 10))
            a = np.repeat(a[::k], k, axis=0)[:n]
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def apply_bend(x, amts, fps, sr, rng, delay_s, edge_ms=3):
    """Applica bend_audio a ogni slot con le quantita' di quel fotogramma; i raccordi sono dissolvenze complementari."""
    spf, f = sr / float(fps), max(1, int(sr * edge_ms / 1000))
    total, ctx = len(x), int(sr / float(fps) * 0.5)
    acc, wsum = np.zeros_like(x), np.zeros(total, np.float32)

    def b(i):
        return int(round(i * spf))

    for k, am in enumerate(amts):
        if not am or max(am.values()) <= 0.03:
            continue
        s0, s1 = max(0, b(k) - f), min(total, b(k + 1) + f)
        if s1 - s0 < 4 * f:
            continue
        cs, ce = max(0, s0 - ctx), min(total, s1 + ctx)
        seg = bend_audio(x[cs:ce], am, delay_s, sr, rng)[s0 - cs:s1 - cs]
        w = np.ones(s1 - s0, np.float32)
        w[:2 * f] = np.linspace(0.0, 1.0, 2 * f, dtype=np.float32)
        w[-2 * f:] = np.minimum(w[-2 * f:], np.linspace(1.0, 0.0, 2 * f, dtype=np.float32))
        acc[s0:s1] += seg * w[:, None]
        wsum[s0:s1] += w
    wsum = np.minimum(wsum, 1.0)
    return acc + x * (1.0 - wsum)[:, None]


def destroy_audio(audio_src, out_wav, smap, amts, fps, max_sec, amount=1.0, sr=44100, delay_s=0.04, seed=507):
    """Audio rovinato come il video: riordino delle fettine (smap) + effetti per fotogramma (amts)."""
    a = decode_pcm(audio_src, sr, max_sec)
    if a is None or not (smap or amts):
        return None
    smap = list(smap) if smap else list(range(len(amts)))
    moshed, orig = slice_audio(a, smap, fps, sr)
    if amts:
        moshed = apply_bend(moshed, amts, fps, sr, np.random.default_rng(int(seed)), delay_s)
    mixed = np.clip((1.0 - amount) * orig + amount * moshed, -1.0, 1.0)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((mixed * 32767).astype(np.int16).tobytes())
    return out_wav


def mosh_audio(audio_src, out_wav, smap, fps, max_sec, amount=1.0, sr=44100, fade_ms=3):
    """Solo il riordino delle fettine (motore Byte)."""
    return destroy_audio(audio_src, out_wav, smap, None, fps, max_sec, amount, sr)


STRONG = {"crush": 0.6, "dist": 0.5, "echo": 0.35, "slip": 0.3, "burn": 0.2}


def audio_plan(kind, p, stt):
    """Cosa fa l'effetto fotogramma per fotogramma, tradotto per il suono: (mappa dei fotogrammi, effetti per slot)."""
    if kind == "BYTE":
        return stt.get("smap"), None
    if kind == "DATABEND":
        return None, [dict(a) for a in stt.get("amts", [])]
    if kind == "PIXELSORT":
        m = float(p.get("mix", 1.0))
        return None, [{} if e <= 0 else {"reverb": 0.8 * e * m, "echo": 0.5 * e * m, "slip": 0.5 * e * m,
                                          "crush": 0.45 * e * m, "dist": 0.3 * e * m}
                      for e in stt.get("env", [])]
    ev = stt.get("ev")
    if not ev:
        return None, None
    keep = 1.0 - float(p.get("residual", 0.0))  # il residuo fa trapelare il reale: anche l'audio
    gain = min(1.0, float(p.get("bloom_gain", 2.5)) / 3.0)
    base = {"crush": 0.12 + 0.3 * min(1.0, float(p.get("res_quant", 0.0)) / 24.0), "dist": 0.15, "slip": 0.08}
    smap, amts, last_s, loop0 = [], [], 0, 0
    for t, code in enumerate(ev):
        if code in (0, 5):  # video reale o I-frame: l'audio e' pulito
            s, a = t, {}
        elif code == 1:
            s, a = t, {k: v * keep for k, v in base.items()}
        elif code == 2:  # hold: l'immagine si congela, il suono si ripete
            s, a = last_s, {"crush": 0.5 * keep, "dist": 0.25 * keep}
        else:  # bloom: lo stesso movimento si ripete, e il suono gira in un anello di 3 fotogrammi
            if code == 3:
                loop0 = t
            s = loop0 + ((t - loop0) % 3)
            a = {k: v * gain * keep for k, v in STRONG.items()}
        smap.append(s)
        amts.append(a)
        last_s = s
    return smap, amts


def audio_plans(kind, stt):
    """Unisce i piani di tutti i passi (anche di una catena): mappe composte in ordine, effetti sommati."""
    steps = stt["steps"] if kind == "CHAIN" else [(kind, stt.get("p") or {}, stt)]
    smap, amts = None, None
    for k, p, s in steps:
        sm, am = audio_plan(k, p, s)
        if sm:
            smap = list(sm) if smap is None else [smap[min(int(i), len(smap) - 1)] for i in sm]
        if am:
            if amts is None:
                amts = [dict(a) for a in am]
            else:
                for i, a in enumerate(am):
                    if i >= len(amts):
                        amts.append({})
                    for fx, v in a.items():
                        amts[i][fx] = min(1.0, amts[i].get(fx, 0.0) + v)
    return smap, amts


def audio_with_mosh(kind, ss, audio_src, stt, fps, max_sec, tmp, p=None):
    """Se l'opzione e' attiva, ritorna l'audio rovinato come l'effetto (altrimenti l'audio originale)."""
    amt = float(ss["audio_mosh_amt"])
    if not (ss["keep_audio"] and ss["audio_mosh"]) or amt <= 0:
        return audio_src
    st1 = stt if kind == "CHAIN" else dict(stt, p=p or {})
    smap, amts = audio_plans(kind, st1)
    if not (smap or amts):
        return audio_src
    steps = stt["steps"] if kind == "CHAIN" else [(kind, p or {}, stt)]
    dl = next((float(q.get("delay", 0.37)) for k, q, _s in steps if k == "DATABEND"), 0.37)
    out = destroy_audio(audio_src, os.path.join(tmp, "audio_mosh.wav"), smap, amts, fps, max_sec, amt,
                        delay_s=max(0.008, min(0.18, dl * 0.12)), seed=int((steps[0][1] or {}).get("seed", 507)))
    if not out:
        return audio_src
    for _k, _q, s0 in steps:
        s0["audio_mosh"] = amt
    return out


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
    smap, last_s = [], 0  # smap[k] = fotogramma sorgente mostrato nello slot k (serve all'audio)
    st = {"frames": 0, "dropped_i": 0, "dups": 0, "skips": 0, "corrupt": 0}
    for k in range(n):
        dr, bl, en = at(au["trig_drop"], k), at(au["trig_bloom"], k), at(au["rms"], k)
        win = start_f <= k < end_f and k > 0
        if hold > 0 and last is not None:
            out.append(last); smap.append(last_s); hold -= 1; st["dups"] += 1; continue
        s = min(max(s, k - p.get("max_lag", 8)), n - 1)  # resincronia: l'immagine non resta mai troppo indietro rispetto all'audio
        if win:
            if types[s] == 0 and s != p["protect"] and (p["drop_mode"] == "Tutti" or
                    (p["drop_mode"] == "Su onset" and (dr > p["drop_thr"] or s in p["kf_set"]))):
                s = min(s + 1, n - 1); st["dropped_i"] += 1
            if last is not None and last_t != 0 and (bl > p["onset_thr"] or rng.random() < p["bloom_prob"]):
                hold = int(max(bl, 0.3) * p["max_dup"]); out.append(last); smap.append(last_s); st["dups"] += 1; continue
            if en < p["quiet_thr"] and last is not None and last_t != 0:
                if k - s > 0:
                    s = min(s + 1, n - 1); st["skips"] += 1
                elif k - s < 0:
                    out.append(last); smap.append(last_s); st["dups"] += 1; continue
        f, t = frames[s], types[s]
        if win and t != 0:
            nb = int(len(f) * p["corrupt"] * at(au["corr"], k))
            if nb > 0:
                f = corrupt_bytes(f, nb, rng); st["corrupt"] += nb
        out.append(f); smap.append(s); last, last_t, last_s = f, t, s; s += 1
    st["frames"] = len(out)
    st["smap"] = smap
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
    p["max_lag"] = max(2, int(round(fps * 0.3)))  # ritardo massimo dell'immagine sull'audio: 0,3 secondi
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


FX = ("echo", "reverb", "invert", "burn", "crush", "hp", "dist", "slip")


FX_IT = {"echo": "eco", "reverb": "riverbero", "invert": "inverti", "burn": "amplifica",
         "crush": "bitcrush", "hp": "passa-alto", "dist": "distorsione", "slip": "slittamento"}


FX_EN = {"echo": "echo", "reverb": "reverb", "invert": "invert", "burn": "amplify",
         "crush": "bitcrush", "hp": "high-pass", "dist": "distortion", "slip": "byte slip"}


def bend_frame(frame, amt, p, rng):
    """Il frame (BGR) e' un segnale di byte centrato a 127.5: ci applico gli effetti e lo riconverto."""
    h, w = frame.shape[:2]
    rb = w * 3
    r0, r1 = int(h * p["zone"][0] / 100), int(h * p["zone"][1] / 100)
    if r1 - r0 < 2:
        return frame
    out = frame.copy()
    flat = out.reshape(-1)
    a = (flat[r0 * rb:r1 * rb].astype(np.float32) - 127.5) / 127.5
    n = a.size

    def bands(k, frac):
        for _ in range(k):
            hgt = max(1, int((r1 - r0) * frac))
            y0 = int(rng.integers(0, max(1, r1 - r0 - hgt)))
            yield y0 * rb, min(n, (y0 + hgt) * rb)

    if amt["slip"] > 0.03:  # cancella k byte: tutto il resto scivola
        k = int(amt["slip"] * 90) + 1
        for _ in range(1 + int(2 * amt["slip"])):
            pos = int(rng.integers(0, max(1, n - k)))
            a = np.concatenate([a[:pos], a[pos + k:], np.zeros(k, np.float32)])
    if amt["echo"] > 0.03:
        g = 0.85 * amt["echo"]
        d = max(3, int(p["delay"] * w) * 3 if p["align"] else int(p["delay"] * rb))
        y, norm = a.copy(), 1.0
        for k in (1, 2, 3):
            s = k * d
            if s < n:
                y[s:] += (g ** k) * a[:-s]
                norm += g ** k
        a = y / norm
    if amt["reverb"] > 0.03:  # coda esponenziale: pixel trascinati come vernice
        al = 1.0 - 10.0 ** (-(0.3 + 2.4 * amt["reverb"]))  # da 0.5 a 0.998: coda fino a ~200 pixel
        wet = lfilter([1 - al], [1, -al], a).astype(np.float32)
        wet = np.where(np.abs(wet) > np.abs(a), wet, a)  # la coda resta solo dove supera l'originale: il contrasto non si spegne
        a = a * (1 - amt["reverb"]) + wet * amt["reverb"]
    if amt["hp"] > 0.03:  # passa-alto: solo i bordi
        low = lfilter([0.1], [1, -0.9], a).astype(np.float32)
        a = a * (1 - amt["hp"]) + (a - low) * 2 * amt["hp"]
    if amt["dist"] > 0.03:
        drive = 1 + 14 * amt["dist"]
        a = np.tanh(drive * a) / np.tanh(drive)
    if amt["burn"] > 0.03:  # amplificazione con fade dentro fasce di righe
        g = 1 + 7 * amt["burn"]
        for s0, s1 in bands(1 + int(3 * amt["burn"]), 0.05 + 0.25 * amt["burn"]):
            a[s0:s1] = np.clip(a[s0:s1] * np.linspace(1, g, s1 - s0, dtype=np.float32), -1, 1)
    if amt["invert"] > 0.05:  # inverti polarita' = negativo a fasce
        for s0, s1 in bands(1 + int(3 * amt["invert"]), 0.04 + 0.2 * amt["invert"]):
            a[s0:s1] *= -1
    if amt["crush"] > 0.03:
        lv = 2 + int(30 * (1 - amt["crush"]))
        a = np.round(a * lv) / lv
        if amt["crush"] > 0.3:
            k = 3 * (1 + int(amt["crush"] * 10))
            a = np.repeat(a[::k], k)[:n]
    flat[r0 * rb:r1 * rb] = np.clip(a * 127.5 + 127.5, 0, 255).astype(np.uint8)
    return out


def to_tiles(f, t):
    """Riordina l'immagine a tessere t x t: il segnale attraversa una tessera dopo l'altra."""
    h2, w2 = f.shape[0] // t * t, f.shape[1] // t * t
    g = f[:h2, :w2].reshape(h2 // t, t, w2 // t, t, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(g).reshape(h2, w2, 3)


def from_tiles(g, t, out):
    h2, w2 = g.shape[:2]
    out[:h2, :w2] = g.reshape(h2 // t, w2 // t, t, t, 3).transpose(0, 2, 1, 3, 4).reshape(h2, w2, 3)
    return out


def bend_scan(frame, amt, p, rng):
    """Direzione di lettura (orizzontale/verticale/blocchi) e spazio colore: cambiano il disegno dell'effetto."""
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) if p["space"] == "YCrCb" else frame
    if p["scan"] == "Verticale":
        g = bend_frame(np.ascontiguousarray(f.transpose(1, 0, 2)), amt, p, rng)
        f = np.ascontiguousarray(g.transpose(1, 0, 2))
    elif p["scan"] == "Blocchi" and min(f.shape[:2]) >= 2 * int(p["tile"]):
        t = int(p["tile"])
        f = from_tiles(bend_frame(to_tiles(f, t), amt, p, rng), t, f.copy())
    else:
        f = bend_frame(f, amt, p, rng)
    return cv2.cvtColor(f, cv2.COLOR_YCrCb2BGR) if p["space"] == "YCrCb" else f


def databend_video(src_video, src_image, out_path, p, progress_cb=None):
    """Sorgente: un video oppure un'immagine fissa che si anima col brano (serve l'audio)."""
    rng = np.random.default_rng(p["seed"])
    au, cap = p["au"], None
    if src_image:
        img = cv2.imread(src_image)
        if img is None:
            raise RuntimeError("Immagine non leggibile / unreadable image")
        fps = float(p["fps_img"])
        oh, ow = img.shape[:2]
        n = min(int(p["max_sec"] * fps), int(au["dur"] * fps))
    else:
        cap = cv2.VideoCapture(src_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        ow, oh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1, int(p["max_sec"] * fps))
    if n < 2:
        raise RuntimeError("Audio o video troppo corto / source too short")
    scale = min(1.0, p["max_w"] / float(ow))
    w, h = max(16, int(ow * scale) // 2 * 2), max(16, int(oh * scale) // 2 * 2)
    if src_image:
        img = fit_frame(img, w, h)
    writer = make_writer(out_path, fps, w, h)
    env = {k: driver_env(au, p["db_drv"][k], p["release"]) for k in FX}
    start_f, end_f = int(n * p["mosh_start"] / 100), int(n * p["mosh_end"] / 100)
    prev, bent, done = None, 0, 0
    amts = []  # quantita' degli effetti per fotogramma (servono all'audio)
    for t in range(n):
        if src_image:
            frame = img
        else:
            ok, fr = cap.read()
            if not ok:
                break
            frame = fit_frame(fr, w, h)
        if start_f <= t < end_f:
            amt = {k: min(1.0, p["db_fx"][k] * (1.0 if env[k] is None else at(env[k], t))) for k in FX}
            base = frame if prev is None or p["mem"] <= 0 else \
                cv2.addWeighted(frame, 1 - p["mem"], prev, p["mem"], 0)
            out = bend_scan(base, amt, p, rng)
            prev, bent = out, bent + 1
            amts.append(dict(amt))
        else:
            out, prev = frame, None
            amts.append({})
        writer.write(out)
        done += 1
        if progress_cb and t % 5 == 0:
            progress_cb(t / max(1, n - 1))
    writer.release()
    if cap is not None:
        cap.release()
    return fps, {"frames": done, "bent": bent, "source": "image" if src_image else "video",
                 "fx": dict(p["db_fx"]), "scan": p["scan"], "space": p["space"], "amts": amts}

# ============================================================================
# PARAMETRI E PRESET (nessuna interfaccia)
# ============================================================================

ACC = {"4/4": 4, "3/4": 3, "6/8": 6, "2/4": 2}

DEFAULTS = {
    "seed": 507, "mosh_range": (15, 100), "max_w": 640, "max_sec": 30, "prev_sec": 3,
    "keep_audio": True, "audio_mosh": False, "audio_mosh_amt": 1.0, "onset_thr": 0.6, "bloom_prob": 0.02, "trigger": "Onset",
    "bands": False, "bpm_manual": 0.0, "meter": "4/4", "beat_offset": 0, "downbeat": 1, "show_wave": False, "fmt": "Originale", "fit": "Ritaglia", "flow_engine": "DIS (veloce)",
    "kf": 12, "qscale": 5, "kf_on_audio": True, "drop_mode": "Su onset", "drop_thr": 0.5,
    "max_dup": 12, "corrupt": 0.002, "quiet_thr": 0.25,
    "mode": "Block MV", "block": 16, "strength": 1.5, "mv_noise": 0.0, "hold_prob": 0.05,
    "bloom_gain": 2.5, "bloom_len": (6, 24), "residual": 0.0, "a_strength": 1.0,
    "a_bloom": True, "a_refresh": False, "refresh_thr": 0.85,
    "rs_luma": 1.25, "rs_chroma": 2.0, "rs_quant": 8.0, "rs_block": 16, "rs_strength": 1.0, "rs_mv_noise": 0.0,
    "rs_hold": 0.0, "rs_bloom": True, "rs_bloom_gain": 2.5, "rs_bloom_len": (6, 24), "rs_asr": 0.5,
    "rs_refresh": False, "rs_refresh_thr": 0.85, "rs_decay": 0.0, "rs_grain": 2.0,
    "ps_key": "Luminanza", "ps_imode": "Soglia", "ps_range": (0.25, 0.85), "ps_angle": 0, "ps_maxlen": 0,
    "ps_skip": 0.0, "ps_reverse": False, "ps_mix": 1.0, "ps_expand": 0.35, "ps_drv": "Colpi/Beat",
    "ps_release": 0.8, "ps_fps": 30,
    "db_echo": 0.5, "db_reverb": 0.4, "db_invert": 0.3, "db_burn": 0.3, "db_crush": 0.2, "db_hp": 0.0,
    "db_dist": 0.0, "db_slip": 0.0,
    "db_d_echo": "Bassi", "db_d_reverb": "Volume", "db_d_invert": "Colpi/Beat", "db_d_burn": "Medi",
    "db_d_crush": "Acuti", "db_d_hp": "Acuti", "db_d_dist": "Volume", "db_d_slip": "Colpi/Beat",
    "db_scan": "Orizzontale", "db_space": "BGR", "db_tile": 32,
    "db_delay": 0.37, "db_align": False, "db_release": 0.8, "db_mem": 0.0, "db_zone": (0, 100), "db_fps": 30,
}

for _p in ("mt2", "mt4"):  # trasformazioni del moto: stesse impostazioni, una copia per tab
    DEFAULTS.update({f"{_p}_{_k}": _v for _k, _v in MT_DEFAULTS.items()})

KEEP = ("fmt", "fit", "beat_offset", "downbeat", "show_wave", "seed", "max_w", "max_sec", "prev_sec", "keep_audio", "audio_mosh", "audio_mosh_amt", "bpm_manual", "meter", "flow_engine")

OPTS = {"trigger": ["Onset", "Beat (a tempo)", "Onset + Beat"], "drop_mode": ["Su onset", "Tutti", "Mai"],
        "mode": ["Block MV", "Optical Flow"], "rs_block": [8, 16, 32, 64], "block": [8, 16, 32, 64], "meter": list(ACC),
        "max_w": [320, 480, 640, 854, 1280], "fmt": FORMATS, "fit": FIT_MODES, "flow_engine": ["DIS (veloce)", "Farneback"], "ps_fps": [24, 25, 30], "ps_key": PS_KEYS, "ps_imode": PS_MODES, "ps_drv": DRIVERS,
        "db_fps": [24, 25, 30], "db_scan": ["Orizzontale", "Verticale", "Blocchi"],
        "db_space": ["BGR", "YCrCb"], "db_tile": [16, 32, 64],
        **{"db_d_" + k: DRIVERS for k in FX}}

LIM = {"seed": (0, 2**31 - 1), "audio_mosh_amt": (0.0, 1.0), "mosh_range": (0, 100), "max_sec": (2, 180), "prev_sec": (2, 6),
       "onset_thr": (0.1, 1.0), "bloom_prob": (0.0, 0.3), "bpm_manual": (0.0, 300.0), "beat_offset": (-300, 300), "downbeat": (1, 6),
       "kf": (6, 120), "qscale": (2, 15), "drop_thr": (0.1, 1.0), "max_dup": (1, 40),
       "corrupt": (0.0, 0.01), "quiet_thr": (0.0, 0.6), "strength": (0.2, 4.0), "mv_noise": (0.0, 8.0),
       "hold_prob": (0.0, 0.5), "bloom_gain": (1.0, 5.0), "bloom_len": (2, 60), "residual": (0.0, 0.5),
       "a_strength": (0.0, 3.0), "refresh_thr": (0.5, 1.0), "db_delay": (0.01, 1.5), "db_release": (0.0, 0.95),
       "db_mem": (0.0, 0.95), "db_zone": (0, 100), "rs_luma": (0.0, 3.0), "rs_chroma": (0.0, 4.0), "rs_quant": (0.0, 48.0), "rs_strength": (0.2, 4.0),
       "rs_mv_noise": (0.0, 8.0), "rs_hold": (0.0, 0.5), "rs_bloom_gain": (1.0, 5.0), "rs_bloom_len": (2, 60),
       "rs_asr": (0.0, 3.0), "rs_refresh_thr": (0.5, 1.0), "rs_decay": (0.0, 0.3), "rs_grain": (0.0, 12.0),
       "ps_range": (0.0, 1.0), "ps_angle": (0, 175), "ps_maxlen": (0, 400), "ps_skip": (0.0, 0.95),
       "ps_mix": (0.0, 1.0), "ps_expand": (0.0, 0.6), "ps_release": (0.0, 0.95), **{"db_" + k: (0.0, 1.0) for k in FX}}

for _p in ("mt2", "mt4"):
    OPTS.update({f"{_p}_zoom_drv": DRIVERS, f"{_p}_mirror": ["No", "Orizzontale", "Verticale"], f"{_p}_mask": MASKS})
    LIM.update({f"{_p}_{_k}": _v for _k, _v in MT_LIM.items()})

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
    "Codec rotto": (
        "Tab 2 Optical :: macro-blocchi che sbandano a ogni colpo e si ricompongono sul primo battito di ogni battuta.",
        {
            'trigger': 'Beat (a tempo)',
            'mode': 'Block MV',
            'block': 32,
            'strength': 1.5,
            'mv_noise': 1.0,
            'a_bloom': True,
            'a_refresh': True,
            'refresh_thr': 0.9,
        }),
    "Mosh a ondate": ("Tab 2 Optical :: il movimento si accumula, sui colpi forti l'immagine si rinfresca. A ondate col brano.",
        {"trigger": "Beat (a tempo)", "a_refresh": True, "refresh_thr": 0.9, "bloom_len": (8, 30)}),
    "Un video mangia l'altro": ("Tab 2 Optical (vale anche per 1 e 4) :: carica Video A (movimento) e Video B (immagine): A deforma B.",
        {"mode": "Block MV", "block": 16, "mosh_range": (0, 100), "residual": 0.0}),
    "Vernice fresca": (
        "Tab 3 Databending :: pixel trascinati come vernice fresca, a ritmo di volume.",
        {
            'db_reverb': 1.0,
            'db_echo': 0.2,
            'db_invert': 0.0,
            'db_burn': 0.0,
            'db_crush': 0.0,
            'db_release': 0.9,
            'db_mem': 0.6,
        }),
    "Schermo bruciato": (
        "Tab 3 Databending :: luminosita' bruciata, negativi a fasce e distorsione sui colpi.",
        {
            'db_burn': 0.6,
            'db_dist': 0.5,
            'db_invert': 0.4,
            'db_echo': 0.0,
            'db_reverb': 0.1,
            'db_crush': 0.0,
            'db_d_burn': 'Bassi',
            'db_d_invert': 'Colpi/Beat',
        }),
    "Fantasma a righe": ("Tab 3 Databending :: sdoppiamenti diagonali e bordi: ghosting a righe.",
        {"db_echo": 0.9, "db_hp": 0.6, "db_delay": 0.33, "db_reverb": 0.0, "db_invert": 0.0, "db_burn": 0.0,
         "db_crush": 0.0, "db_d_echo": "Volume"}),
    "Pioggia digitale": (
        "Tab 3 Databending :: lettura verticale: colonne che scivolano e si invertono, con code luminose.",
        {
            'db_scan': 'Verticale',
            'db_reverb': 1.0,
            'db_echo': 0.5,
            'db_crush': 0.3,
            'db_slip': 0.5,
            'db_invert': 0.2,
            'db_burn': 0.0,
            'db_d_reverb': 'Volume',
            'db_mem': 0.3,
        }),
    "Blocchi corrotti": ("Tab 3 Databending :: lettura a tessere: mosaico di blocchi sbagliati, colori a parte (YCrCb).",
        {"db_scan": "Blocchi", "db_tile": 32, "db_space": "YCrCb", "db_echo": 0.6, "db_invert": 0.5,
         "db_crush": 0.4, "db_reverb": 0.1, "db_burn": 0.0}),
    "Confetti saturo": (
        "Tab 4 Residuo :: i residui dei P-frame si accumulano: colori saturi e coriandoli ovunque (stile del video di riferimento).",
        {
            'rs_luma': 1.25,
            'rs_chroma': 2.0,
            'rs_quant': 8.0,
            'rs_block': 16,
            'rs_strength': 1.0,
            'rs_hold': 0.0,
            'rs_asr': 0.5,
            'rs_grain': 2.0,
        }),
    "Confetti fine": (
        "Tab 4 Residuo :: coriandoli piccoli e fitti, immagine piu' leggibile.",
        {
            'rs_luma': 1.15,
            'rs_chroma': 1.8,
            'rs_quant': 4.0,
            'rs_block': 8,
            'rs_asr': 0.5,
            'rs_grain': 1.5,
        }),
    "Confetti a ondate": (
        "Tab 4 Residuo :: la saturazione sale e sui colpi forti si azzera: cicli a tempo col brano.",
        {
            'trigger': 'Beat (a tempo)',
            'rs_luma': 1.3,
            'rs_chroma': 2.4,
            'rs_quant': 10.0,
            'rs_refresh': True,
            'rs_refresh_thr': 0.9,
            'rs_asr': 1.0,
            'rs_grain': 2.5,
        }),
    "Residuo estremo": (
        "Tab 4 Residuo :: guadagni alti e grana grossa: immagine quasi distrutta, colori impazziti.",
        {
            'rs_luma': 1.5,
            'rs_chroma': 3.5,
            'rs_quant': 24.0,
            'rs_hold': 0.05,
            'rs_asr': 0.5,
            'rs_grain': 5.0,
        }),
    "Zoom a pugno": (
        "Tab 2 Optical :: lo zoom pulsa sui colpi: il mosh si gonfia verso di te e si azzera sul primo battito.",
        {
            'trigger': 'Beat (a tempo)',
            'mode': 'Block MV',
            'block': 16,
            'strength': 1.0,
            'mt2_zoom': 0.8,
            'mt2_zoom_drv': 'Colpi/Beat',
            'mt2_fluid': 0.2,
            'a_bloom': True,
            'a_refresh': True,
            'refresh_thr': 0.9,
        }),
    "Colata": (
        "Tab 2 Optical :: l'immagine cola verso il basso come cera e si ricompone a ogni battuta.",
        {
            'trigger': 'Beat (a tempo)',
            'mode': 'Optical Flow',
            'strength': 1.0,
            'mt2_gy': 1.5,
            'mt2_fluid': 0.3,
            'residual': 0.04,
            'a_refresh': True,
            'refresh_thr': 0.9,
        }),
    "Vortice": (
        "Tab 2 Optical :: il movimento ruota e si avvita: mosh a spirale che si distende a ogni battuta.",
        {
            'trigger': 'Beat (a tempo)',
            'mode': 'Optical Flow',
            'mt2_rot': 0.6,
            'mt2_fluid': 0.5,
            'mt2_zoom': 0.2,
            'mt2_zoom_drv': 'Fisso',
            'a_refresh': True,
            'refresh_thr': 0.9,
        }),
    "Specchio": (
        "Tab 2 Optical :: il moto di sinistra viene specchiato a destra, con zoom a tempo e ripartenza a ogni battuta.",
        {
            'trigger': 'Beat (a tempo)',
            'mode': 'Block MV',
            'mt2_mirror': 'Orizzontale',
            'mt2_zoom': 0.3,
            'mt2_zoom_drv': 'Colpi/Beat',
            'a_refresh': True,
            'refresh_thr': 0.9,
        }),
    "Confetti sui movimenti": (
        "Tab 4 Residuo :: coriandoli solo dove qualcosa si muove, il resto resta pulito.",
        {
            'mt4_mask': 'Solo dove si muove',
            'mt4_mthr': 1.0,
            'rs_grain': 2.5,
        }),
    "Confetti a decadimento": (
        "Tab 4 Residuo :: l'errore svanisce piano piano verso il video reale: ondate di colore.",
        {
            'rs_decay': 0.06,
            'rs_luma': 1.3,
            'rs_chroma': 2.4,
            'rs_grain': 3.0,
        }),
    "Cascata": ("Tab 5 Pixel sort :: ordinamento verticale: cascate di colore dall'alto, la soglia si allarga sui bassi.",
        {"ps_angle": 90, "ps_key": "Luminanza", "ps_range": (0.3, 0.9), "ps_expand": 0.4, "ps_drv": "Bassi"}),
    "Fiamma": ("Tab 5 Pixel sort :: colonne ordinate per tinta e rovesciate: effetto fuoco, scatta sui colpi.",
        {"ps_angle": 90, "ps_key": "Tinta", "ps_range": (0.2, 0.8), "ps_reverse": True, "ps_expand": 0.35}),
    "Bordi sciolti": ("Tab 5 Pixel sort :: gli intervalli sono delimitati dai bordi: le forme si sciolgono in orizzontale.",
        {"ps_imode": "Bordi", "ps_key": "Tinta", "ps_range": (0.2, 0.8), "ps_expand": 0.4}),
    "Pioggia a tempo": ("Tab 5 Pixel sort :: gocce verticali corte e a tratti, sul tempo del brano.",
        {"trigger": "Beat (a tempo)", "ps_angle": 90, "ps_maxlen": 80, "ps_skip": 0.3, "ps_expand": 0.5}),
    "Diagonale a righe": (
        "Tab 5 Pixel sort :: ordinamento inclinato di 35 gradi per il canale rosso: righe diagonali.",
        {
            'ps_angle': 35,
            'ps_key': 'Rosso',
            'ps_range': (0.3, 0.8),
            'ps_maxlen': 80,
            'ps_expand': 0.15,
            'ps_drv': 'Volume',
        }),
    "Cassa, rullante, hi-hat": ("Tab 1 Byte :: modo bande: la cassa toglie gli I-frame, il rullante rinfresca, gli hi-hat corrompono.",
        {"bands": True, "drop_mode": "Su onset", "corrupt": 0.003, "max_dup": 14, "kf": 8}),
}

TAB_LABELS = ["1 // BYTE (AVI)", "2 // OPTICAL FLOW", "3 // DATABENDING", "4 // RESIDUO", "5 // PIXEL SORT"]

def tab_of(name):
    """Numero del tab (1-4) per cui e' pensato il preset: lo ricavo dalla descrizione ('Tab 4 Residuo :: ...')."""
    m = re.match(r"Tab (\d)", PRESETS[name][0])
    return int(m.group(1)) if m else 1

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



VAR_PREFIX = {"BYTE": ("kf", "qscale", "drop_mode", "drop_thr", "max_dup", "corrupt", "quiet_thr", "kf_on_audio"),
              "OPTICAL": ("mode", "block", "strength", "mv_noise", "hold_prob", "bloom_gain", "bloom_len", "residual",
                          "a_strength", "a_bloom", "a_refresh", "refresh_thr", "mt2_"),
              "RESIDUAL": ("rs_", "mt4_"), "DATABEND": ("db_",), "PIXELSORT": ("ps_",)}
VAR_COMMON = ("mosh_range", "onset_thr", "bloom_prob", "trigger")
VAR_SKIP = ("db_fps", "ps_fps", "db_tile")


def variant_keys(kind):
    """Controlli che le varianti possono cambiare per un effetto (la catena usa quelli dei passi scelti)."""
    kinds = list(VAR_PREFIX) if kind == "CHAIN" else [kind]
    pre = tuple(x for k in kinds for x in VAR_PREFIX[k])
    keys = [k for k in DEFAULTS if k not in KEEP and k not in VAR_SKIP and
            (k in VAR_COMMON or any(k == x or (x.endswith("_") and k.startswith(x)) for x in pre))]
    return [k for k in keys if k in LIM or k in OPTS or isinstance(DEFAULTS[k], bool)]


def vary_params(base, kind, rng, amount=0.75):
    """Una variante vera: sposta i controlli verso valori casuali (amount 0 = uguale, 1 = a caso nei limiti)."""
    out = dict(base)
    for k in variant_keys(kind):
        v = base[k]
        if k in OPTS:
            if rng.random() < 0.5 * amount + 0.15:
                out[k] = rng.choice(OPTS[k])
        elif isinstance(DEFAULTS[k], bool):
            if rng.random() < 0.35 * amount + 0.1:
                out[k] = not v
        else:
            lo, hi = LIM[k]
            is_int = isinstance(lo, int) and isinstance(hi, int) and not isinstance(DEFAULTS[k], float)

            def pick(x):
                t = rng.uniform(lo, hi)
                r = x + (t - x) * amount
                return int(round(r)) if is_int else float(r)

            if isinstance(DEFAULTS[k], tuple):
                a, b = sorted((pick(v[0]), pick(v[1])))
                out[k] = (a, b)
            else:
                out[k] = pick(v)
        if not valid(k, out[k]):
            out[k] = base[k]
    return out

# ----------------------------------------------------------------------------
# CATENA DI EFFETTI + COSTRUZIONE DEI PARAMETRI
# ----------------------------------------------------------------------------
NONE = "— nessuno —"
STEPS = {"Byte": "BYTE", "Optical": "OPTICAL", "Databending": "DATABEND", "Residuo": "RESIDUAL",
         "Pixel sort": "PIXELSORT"}
DEFAULTS.update({"chain_0": "Databending", "chain_1": "Residuo", "chain_2": NONE})
OPTS.update({f"chain_{i}": [NONE] + list(STEPS) for i in range(3)})
TAB_LABELS.append("6 // CATENA")
PRESETS["Catena sporca"] = (
    "Tab 6 Catena :: databending leggero, poi pixel sort, poi residuo: tutto insieme sul tuo audio, con l'immagine ancora leggibile.",
    {
        'chain_0': 'Databending',
        'chain_1': 'Pixel sort',
        'chain_2': 'Residuo',
        'db_echo': 0.1,
        'db_reverb': 0.15,
        'db_invert': 0.05,
        'db_burn': 0.0,
        'db_crush': 0.0,
        'ps_angle': 90,
        'ps_range': (0.4, 0.75),
        'ps_expand': 0.05,
        'ps_skip': 0.7,
        'ps_maxlen': 80,
        'rs_luma': 1.1,
        'rs_chroma': 1.5,
        'rs_quant': 6.0,
        'rs_grain': 1.0,
    })
PRESETS["Catena a tempo"] = (
    "Tab 6 Catena :: byte-level e poi pixel sort, agganciati al beat del brano.",
    {
        'chain_0': 'Byte',
        'chain_1': 'Pixel sort',
        'chain_2': '— nessuno —',
        'trigger': 'Beat (a tempo)',
        'kf': 8,
        'drop_mode': 'Su onset',
        'corrupt': 0.001,
        'ps_angle': 90,
        'ps_range': (0.3, 0.8),
        'ps_maxlen': 80,
        'ps_skip': 0.5,
        'ps_expand': 0.25,
    })


def mt_from_state(ss, pfx):
    return {k: ss[f"{pfx}_{k}"] for k in MT_DEFAULTS}


def common_params(ss, preview=False):
    return {"seed": int(ss["seed"]), "mosh_start": ss["mosh_range"][0], "mosh_end": ss["mosh_range"][1],
            "max_w": int(min(ss["max_w"], 240) if preview else ss["max_w"]),
            "max_sec": int(ss["prev_sec"] if preview else ss["max_sec"]),
            "onset_thr": ss["onset_thr"], "trigger": ss["trigger"], "bloom_prob": ss["bloom_prob"],
            "bands": ss["bands"], "bpm_manual": float(ss["bpm_manual"]), "meter": ss["meter"],
            "fmt": ss["fmt"], "fit": ss["fit"]}


def build_params(kind, ss, preview=False):
    """Parametri di un motore a partire dai valori dei controlli (st.session_state o un dict)."""
    p = common_params(ss, preview)
    if kind == "BYTE":
        p.update({k: ss[k] for k in ("kf", "qscale", "drop_mode", "drop_thr", "max_dup", "corrupt",
                                     "quiet_thr", "kf_on_audio")})
    elif kind == "OPTICAL":
        p.update({"mode": ss["mode"], "block": int(ss["block"]), "strength": ss["strength"],
                  "mv_noise": ss["mv_noise"], "iframe_every": 0, "iframe_prob": 0.0, "hold_prob": ss["hold_prob"],
                  "bloom_gain": ss["bloom_gain"], "bloom_len": (int(ss["bloom_len"][0]), int(ss["bloom_len"][1])),
                  "residual": ss["residual"], "a_strength": ss["a_strength"], "a_bloom": ss["a_bloom"],
                  "a_refresh": ss["a_refresh"], "refresh_thr": ss["refresh_thr"], "flow_engine": ss["flow_engine"],
                  "mt": mt_from_state(ss, "mt2")})
    elif kind == "RESIDUAL":
        p.update({"res_luma": ss["rs_luma"], "res_chroma": ss["rs_chroma"], "res_quant": ss["rs_quant"],
                  "block": int(ss["rs_block"]), "strength": ss["rs_strength"], "mv_noise": ss["rs_mv_noise"],
                  "hold_prob": ss["rs_hold"], "a_bloom": ss["rs_bloom"], "bloom_gain": ss["rs_bloom_gain"],
                  "bloom_len": (int(ss["rs_bloom_len"][0]), int(ss["rs_bloom_len"][1])),
                  "a_strength": ss["rs_asr"], "a_refresh": ss["rs_refresh"], "refresh_thr": ss["rs_refresh_thr"],
                  "flow_engine": ss["flow_engine"], "decay": ss["rs_decay"], "grain": ss["rs_grain"],
                  "mt": mt_from_state(ss, "mt4")})
    elif kind == "DATABEND":
        p.update({"db_fx": {k: ss["db_" + k] for k in FX}, "db_drv": {k: ss["db_d_" + k] for k in FX},
                  "delay": ss["db_delay"], "align": ss["db_align"], "release": ss["db_release"],
                  "mem": ss["db_mem"], "zone": ss["db_zone"], "fps_img": ss["db_fps"], "scan": ss["db_scan"],
                  "space": ss["db_space"], "tile": ss["db_tile"]})
    elif kind == "PIXELSORT":
        p.update({"key": ss["ps_key"], "imode": ss["ps_imode"], "range": tuple(ss["ps_range"]),
                  "angle": int(ss["ps_angle"]), "maxlen": int(ss["ps_maxlen"]), "skip": ss["ps_skip"],
                  "reverse": ss["ps_reverse"], "mix": ss["ps_mix"], "expand": ss["ps_expand"],
                  "drv": ss["ps_drv"], "release": ss["ps_release"], "fps_img": ss["ps_fps"]})
    else:
        raise ValueError(f"effetto sconosciuto: {kind}")
    return p

# ============================================================================
# ESECUZIONE DI EFFETTI E CATENE
# ============================================================================

def video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def compute_au(ss, src_audio, msec, fps, features=audio_features):
    """Segnali audio per frame (RMS, onset, beat, bande) gia' instradati secondo trigger e modo bande."""
    au = features(src_audio, fps, int(msec * fps) + 2, msec, float(ss["bpm_manual"]), ACC[ss["meter"]],
                   ss["beat_offset"] / 1000.0, int(ss["downbeat"]) - 1)
    return route_audio(au, ss["trigger"], ss["bands"])


def decode_mid(avi, out):
    """Decodifica l'AVI byte-level (anche rotto) in un AVI MJPG leggibile dal passo successivo."""
    subprocess.run([FFMPEG, "-y", "-err_detect", "ignore_err", "-i", avi, "-an", "-c:v", "mjpeg", "-q:v", "2", out],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_engine(kind, p, src_video, src_image, src_b, out_path, progress_cb=None):
    """Esegue un motore. BYTE scrive un AVI MPEG-4 byte-level; gli altri un video (mp4 o avi MJPG)."""
    if kind == "BYTE":
        return byte_datamosh(src_video, src_b, out_path, p)
    if kind == "DATABEND":
        return databend_video(src_video, src_image, out_path, p, progress_cb)
    if kind == "PIXELSORT":
        return pixelsort_video(src_video, src_image, out_path, p, progress_cb)
    fn = datamosh_video if kind == "OPTICAL" else residual_video
    fps, stt = fn(src_video, src_b, out_path, p, progress_cb)
    stt["transfer"] = src_b is not None
    return fps, stt


def run_chain(steps, ss, preview, src_video, src_image, src_b, au, tmp, progress_cb=None):
    """steps = nomi degli effetti (chiavi di STEPS) nell'ordine. Ogni passo legge il video del precedente.
    Ritorna {"fps", "steps": [(kind, p, stt)], "raw"}: raw e' il file da passare a finalize_h264."""
    kinds = [STEPS[s] for s in steps]
    if not kinds:
        raise RuntimeError("Scegli almeno un effetto / pick at least one effect")
    if src_image and kinds[0] not in ("DATABEND", "PIXELSORT"):
        raise RuntimeError("L'immagine fissa funziona solo se il primo passo e' Databending o Pixel sort / "
                           "a still image needs Databending or Pixel sort as the first step")
    results, cur_video, cur_image, fps, out = [], src_video, src_image, 25.0, None
    for i, kind in enumerate(kinds):
        last = i == len(kinds) - 1
        p = build_params(kind, ss, preview)
        p["seed"] += 101 * i  # un seme diverso per passo
        p["au"] = au
        ext = "avi" if (kind == "BYTE" or not last) else "mp4"
        out = os.path.join(tmp, f"step{i}.{ext}")
        cb = (lambda v, i=i: progress_cb((i + v) / len(kinds))) if progress_cb else None
        fps, stt = run_engine(kind, p, cur_video, cur_image, src_b if i == 0 else None, out, cb)
        results.append((kind, p, stt))
        if not last:
            if kind == "BYTE":
                dec = os.path.join(tmp, f"step{i}_dec.avi")
                decode_mid(out, dec)
                out = dec
            cur_video, cur_image = out, None
    return {"fps": fps, "steps": results, "raw": out}

# ============================================================================
# REPORT BILINGUE (IT/EN), HASHTAG E TAG YOUTUBE
# ============================================================================

PRESET_EN = {"Manuale": "Manual", "Glitch ritmico a tempo": "Rhythmic Glitch", "Caos totale": "Total Chaos",
             "Minimale": "Minimal", "Liquido e morbido": "Liquid & Soft", "Codec rotto": "Broken Codec",
             "Mosh a ondate": "Wave Mosh", "Un video mangia l'altro": "One Video Eats the Other",
             "Cassa, rullante, hi-hat": "Kick, Snare, Hi-hat", "Vernice fresca": "Fresh Paint",
             "Schermo bruciato": "Burnt Screen", "Fantasma a righe": "Striped Ghost",
             "Confetti saturo": "Saturated Confetti", "Zoom a pugno": "Slam Zoom", "Colata": "Melt Down",
             "Vortice": "Vortex", "Specchio": "Mirror", "Confetti sui movimenti": "Motion Confetti",
             "Confetti a decadimento": "Decaying Confetti", "Cascata": "Waterfall", "Fiamma": "Flame",
             "Bordi sciolti": "Melted Edges", "Pioggia a tempo": "Beat Rain", "Diagonale a righe": "Striped Diagonal", "Confetti fine": "Fine Confetti",
             "Confetti a ondate": "Confetti Waves", "Catena sporca": "Dirty Chain", "Catena a tempo": "Beat Chain", "Residuo estremo": "Extreme Residual", "Pioggia digitale": "Digital Rain", "Blocchi corrotti": "Corrupted Blocks"}


SCAN_EN = {"Orizzontale": "Horizontal", "Verticale": "Vertical", "Blocchi": "Blocks"}


FIT_EN = {"Ritaglia": "crop", "Bordi neri": "black bars"}
TRIG_EN = {"Onset": "Onset", "Beat (a tempo)": "Beat (tempo-locked)", "Onset + Beat": "Onset + Beat"}


KEY_EN = {"Luminanza": "Luminance", "Tinta": "Hue", "Saturazione": "Saturation", "Rosso": "Red",
          "Verde": "Green", "Blu": "Blue"}


IMODE_EN = {"Soglia": "Threshold", "Bordi": "Edges", "Casuale": "Random"}


MT_NAMES = {"zoom": ("zoom", "zoom"), "rot": ("rotazione", "rotation"), "shear": ("shear", "shear"),
            "fluid": ("fluido", "fluid"), "shift": ("blocchi sfasati", "shifted blocks")}


def mt_desc(mt, it):
    """Elenco delle trasformazioni del moto attive (stringa vuota se nessuna)."""
    if not mt:
        return ""
    i = 0 if it else 1
    parts = [f"{MT_NAMES[k][i]} {mt[k]:+.1f}" for k in MT_NAMES if mt[k]]
    if mt["gx"] or mt["gy"]:
        parts.append(f"{'gravita' if it else 'gravity'} {mt['gx']:+.1f}/{mt['gy']:+.1f}")
    if mt["mirror"] != "No":
        parts.append(f"{'specchio' if it else 'mirror'} {mt['mirror']}")
    if mt["mask"] != "Ovunque":
        parts.append(f"{'mappa' if it else 'map'} {mt['mask']}")
    return ", ".join(parts)


TAGS = {  # (hashtag centrali, tag YouTube centrale, hashtag finali, tag YouTube finali)
    "BYTE": (["byte"], ["byte"], ["mpeg4", "avi"], ["mpeg-4", "avi"]),
    "OPTICAL": (["opticalflow"], ["optical flow"], ["motionvectors"], ["motion vectors"]),
    "DATABEND": (["databending"], ["databending"], ["numpy", "scipy"], ["numpy", "scipy"]),
    "RESIDUAL": (["residualmosh"], ["residual mosh"], ["pframe"], ["p-frame"]),
    "PIXELSORT": (["pixelsorting"], ["pixel sorting"], ["numpy"], ["numpy"]),
}


def step_rows(kind, p, stt, it):
    """Righe specifiche di un effetto + descrizione DSP."""
    def yn(v):
        return ("Sì" if it else "Yes") if v else "No"

    eng = str(p.get("flow_engine", "")).split(" ")[0]
    if kind == "BYTE":
        rows = [("Dropped I", stt["dropped_i"]), ("Dups", stt["dups"]), ("Skips", stt["skips"]),
                ("Corrupt", stt["corrupt"]), ("Keyframe Audio", stt["kf_audio"]), ("Transfer", yn(stt["transfer"]))]
        dsp = "AVI/MPEG-4 byte-level + librosa"
    elif kind == "OPTICAL":
        rows = [("Modo" if it else "Mode", p["mode"]), ("Motore flow" if it else "Flow engine", eng),
                ("I-frame", stt["iframes"]), ("Hold", stt["holds"]), ("Bloom", stt["blooms"]),
                ("Transfer", yn(stt["transfer"]))]
        dsp = f"OpenCV {eng} optical flow + librosa"
    elif kind == "RESIDUAL":
        rows = [("Blocco" if it else "Block", p["block"]), ("Motore flow" if it else "Flow engine", eng),
                ("Guadagno luma" if it else "Luma gain", p["res_luma"]),
                ("Guadagno colore" if it else "Chroma gain", p["res_chroma"]),
                ("Quantizzazione" if it else "Quantization", p["res_quant"]),
                ("I-frame", stt["iframes"]), ("Hold", stt["holds"]), ("Bloom", stt["blooms"]),
                ("Transfer", yn(stt["transfer"]))]
        dsp = f"OpenCV {eng} flow + residual accumulation + librosa"
    elif kind == "PIXELSORT":
        src = "Video" if stt["source"] == "video" else ("Immagine" if it else "Still image")
        lo, hi = p["range"]
        rows = [("Sorgente" if it else "Source", src),
                ("Chiave" if it else "Key", p["key"] if it else KEY_EN[p["key"]]),
                ("Intervalli" if it else "Intervals", p["imode"] if it else IMODE_EN[p["imode"]]),
                ("Soglie" if it else "Range", f"{lo:.2f}-{hi:.2f}"),
                ("Angolo" if it else "Angle", f"{p['angle']} deg"),
                ("Lunghezza max" if it else "Max length", p["maxlen"] or "-"),
                ("Espansione audio" if it else "Audio expansion", f"{p['expand']:.2f} ({p['drv']})"),
                ("Frame ordinati" if it else "Sorted frames", stt["sorted"])]
        dsp = "NumPy vectorized pixel sorting + librosa"
    else:
        nm = FX_IT if it else FX_EN
        fx = ", ".join(f"{nm[k]} {v:.2f}" for k, v in stt["fx"].items() if v > 0)
        src = "Video" if stt["source"] == "video" else ("Immagine" if it else "Still image")
        rows = [("Sorgente" if it else "Source", src), ("Effetti" if it else "Effects", fx or "-"),
                ("Scansione" if it else "Scan", stt["scan"] if it else SCAN_EN[stt["scan"]]),
                ("Spazio colore" if it else "Color space", stt["space"]),
                ("Frame piegati" if it else "Bent frames", stt["bent"])]
        dsp = "NumPy/SciPy signal databending + librosa"
    d = mt_desc(p.get("mt"), it) if kind in ("OPTICAL", "RESIDUAL") else ""
    if d:
        rows.append(("Moto" if it else "Motion", d))
    if kind == "RESIDUAL" and p.get("decay", 0) > 0:
        rows.append(("Decadimento" if it else "Decay", p["decay"]))
    if kind == "RESIDUAL" and p.get("grain", 0) > 0:
        rows.append(("Grana" if it else "Grain", p["grain"]))
    rows.append(("Audio rovinato" if it else "Audio destroyed", f"{stt['audio_mosh']:.2f}" if stt.get("audio_mosh") else "No"))
    return rows, dsp


def build_report(kind, p, fps, stt, au_ok, preset, num):
    """Report bilingue (IT + EN), hashtag e tag YouTube in inglese.
    kind = un effetto, oppure "CHAIN" con stt = {"steps": [(kind, p, stt), ...]} (p = parametri comuni)."""
    chain = kind == "CHAIN"
    steps = stt["steps"] if chain else [(kind, p, stt)]
    frames = steps[-1][2]["frames"]
    seed = p["seed"]
    preset = "Manuale" if str(preset).startswith("—") else preset

    def block(lang):
        it = lang == "it"

        def yn(v):
            return ("Sì" if it else "Yes") if v else "No"

        note = (" (manuale)" if it else " (manual)") if p.get("bpm_manual", 0) > 0 else ""
        label = kind if not chain else ("CATENA" if it else "CHAIN") + " (" + " > ".join(k for k, _, _ in steps) + ")"
        rows = [("Approccio" if it else "Approach", label),
                ("Preset", preset if it else PRESET_EN.get(preset, preset)),
                ("Seed", seed), ("FPS", f"{fps:.2f}"),
                ("Finestra" if it else "Window", f"{p['mosh_start']}% -> {p['mosh_end']}%"),
                ("Audio-reattivo" if it else "Audio-reactive", yn(au_ok)),
                ("Trigger", p["trigger"] if it else TRIG_EN.get(p["trigger"], p["trigger"])),
                ("Bande" if it else "Bands", yn(p["bands"])),
                ("BPM", f"{p['au'].get('bpm', 0):.1f}{note}"),
                ("Battuta" if it else "Meter", p["meter"]), ("Frames", frames)]
        if p.get("fmt", "Originale") != "Originale":
            rows.insert(5, ("Formato" if it else "Format",
                            f"{p['fmt']} ({p['fit'] if it else FIT_EN[p['fit']]})"))
        out = [f"DATAMOSH // N.{num:03d}"] + [f"{k} :: {v}" for k, v in rows]
        dsps = []
        for i, (k, pp, ss) in enumerate(steps, 1):
            r, dsp = step_rows(k, pp, ss, it)
            dsps.append(dsp)
            if chain:
                out.append(f"--- {'Passo' if it else 'Step'} {i}: {k} ---")
            out += [f"{a} :: {b}" for a, b in r]
        dsp = dsps[0] if not chain else " > ".join(d.replace(" + librosa", "") for d in dsps) + " + librosa"
        return out + [f"DSP :: {dsp}", "Nessun Modello AI/neurale" if it else "No AI/Neural Models",
                      "Direction & Algorithm :: Loop507"]

    def uniq(items):
        return list(dict.fromkeys(items))

    kinds = [k for k, _, _ in steps]
    mid = uniq([t for k in kinds for t in TAGS[k][0]] + (["datamoshchain"] if chain else []))
    end = uniq([t for k in kinds for t in TAGS[k][2]])
    ymid = uniq([t for k in kinds for t in TAGS[k][1]] + (["datamosh chain"] if chain else []))
    yend = uniq([t for k in kinds for t in TAGS[k][3]])
    hashtags = ["datamosh", "loop507"] + mid + ["glitchart", "noai", "generativeaudio",
        "motiondesign", "algorithmicart", "proceduralart", "digitalart",
        "experimentalvideo", "experimentalsound"] + end + ["datamoshing", "compressionart"]
    yt = ["datamosh", "loop507"] + ymid + ["glitch art", "no ai", "generative audio",
          "motion design", "algorithmic art", "procedural art", "digital art",
          "experimental video", "experimental sound"] + yend + ["datamoshing", "compression art"]
    sep = "=" * 40
    return "\n".join([sep, "[ ITALIANO ]", sep, *block("it"), "", sep, "[ ENGLISH ]", sep, *block("en"), "",
                      " ".join("#" + t for t in hashtags), "", sep, "[ TAG YOUTUBE — ENGLISH ONLY ]", sep,
                      ", ".join(yt)]) + "\n"

# ============================================================================
# INTERFACCIA STREAMLIT
# ============================================================================

def _in_streamlit():
    """True solo quando il file gira dentro Streamlit: cosi' i motori si possono importare nei test senza avviare l'interfaccia."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001
        return False


if _in_streamlit():
    # ----------------------------------------------------------------------------
    # UI
    # ----------------------------------------------------------------------------
    st.set_page_config(page_title="DATAMOSH // Loop507", layout="wide")
    st.title("DATAMOSH // Loop507")
    st.caption("Minimalismo Computazionale / Glitch Brutalista :: audio-reactive :: puro DSP, nessuna AI")

    for _k, _v in DEFAULTS.items():
        st.session_state.setdefault(_k, _v)
    for _k, _v in (("preset", MANUAL), ("dm_out", None), ("dm_report", None), ("counter", 0), ("counter_pending", False)):
        st.session_state.setdefault(_k, _v)
    if st.session_state["counter_pending"]:  # dopo un render finale: prossimo numero progressivo
        st.session_state["counter"] += 1
        st.session_state["counter_pending"] = False






    def apply_preset():
        name = st.session_state["preset"]
        if name in PRESETS:
            st.session_state["active_tab"] = TAB_LABELS[tab_of(name) - 1]  # apre il tab del preset
            for k, v in {**DEFAULTS, **PRESETS[name][1]}.items():
                if k not in KEEP:
                    st.session_state[k] = v




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


    st.selectbox("Preset (imposta i controlli e apre il tab giusto; poi puoi ritoccarli)",
                 [MANUAL] + sorted(PRESETS, key=tab_of), key="preset", on_change=apply_preset,
                 format_func=lambda n: n if n == MANUAL else f"[{tab_of(n)}] {n}")
    if st.session_state["preset"] in PRESETS:
        st.caption(PRESETS[st.session_state["preset"]][0])

    VT = ["mp4", "mov", "avi", "mkv", "webm"]
    c1, c2 = st.columns(2)
    up_a = c1.file_uploader("Video A (movimento + audio) / motion + audio", type=VT, key="a")
    up_b = c2.file_uploader("Video B opzionale (immagine/texture che A deforma)", type=VT, key="b")
    up_au = st.file_uploader("Audio esterno opzionale (il tuo brano: guida il mosh e diventa la colonna sonora)",
                             type=["wav", "mp3", "flac", "ogg", "m4a", "aac"], key="au")
    up_img = st.file_uploader("Immagine fissa opzionale (tab 3 Databending, 5 Pixel sort, 6 Catena: si anima col brano, serve l'audio esterno)",
                              type=["png", "jpg", "jpeg", "bmp", "webp"], key="img")

    with st.sidebar:
        st.header("Comuni / Common")
        seed = st.number_input("Seed", 0, 2**31 - 1, key="seed")
        st.button("Seed casuale / Random seed",
                  on_click=lambda: st.session_state.update(seed=random.randint(0, 2**31 - 1)))
        st.number_input("Numero progressivo N. / Counter (nome file e report)", 0, 9999, step=1, key="counter")
        mosh_range = st.slider("Finestra mosh / window (%)", 0, 100, key="mosh_range")
        max_w = st.select_slider("Larghezza max / Max width", OPTS["max_w"], key="max_w",
                                 help="Per il formato 9:16 e' l'altezza, per 1:1 il lato vale 0,75 volte: stessi pixel in tutti i formati.")
        st.radio("Formato / Format", OPTS["fmt"], key="fmt", horizontal=True,
                 help="Originale = come la sorgente. Gli altri portano video, Video B e immagine al formato scelto prima degli effetti.")
        st.radio("Adattamento al formato / Fit", OPTS["fit"], key="fit", horizontal=True,
                 help="Ritaglia = riempie il formato tagliando i bordi; Bordi neri = mostra tutta l'immagine.")
        max_sec = st.slider("Durata max (s) / Max seconds", 2, 180, key="max_sec")
        if max_sec > 60:
            st.caption("Oltre 60 s il render puo' richiedere diversi minuti: prova prima con l'anteprima.")
        prev_sec = st.slider("Secondi anteprima / Preview seconds", 2, 6, key="prev_sec")
        keep_audio = st.checkbox("Mantieni audio (A o esterno) / Keep audio", key="keep_audio")
        audio_mosh = st.checkbox("L'audio viene rovinato come l'effetto / Audio destroyed like the effect", key="audio_mosh",
                                 disabled=not keep_audio,
                                 help="Byte: dove l'immagine si congela o salta, si ripete o salta anche il suono. Optical e Residuo: stutter e bitcrush su hold e bloom. Databending: gli stessi effetti (eco, riverbero, bitcrush, distorsione...) con gli stessi pilotaggi. Pixel sort: smear e crush sulla forza dell'effetto. Vale anche per le catene.")
        audio_mosh_amt = st.slider("Quantita' audio rovinato / Audio mosh amount", 0.0, 1.0, step=0.05,
                                   key="audio_mosh_amt", disabled=not (keep_audio and audio_mosh))
        trigger = st.radio("Trigger audio", OPTS["trigger"], key="trigger",
                           help="Beat = scatti precisi sul tempo del brano (battere piu' forte).")
        onset_thr = st.slider("Soglia onset / Onset threshold", 0.1, 1.0, step=0.05, key="onset_thr")
        bloom_prob = st.slider("Bloom casuale / Random bloom prob", 0.0, 0.3, step=0.01, key="bloom_prob")
        with st.expander("Audio avanzato / Advanced audio"):
            bands = st.checkbox("Modo bande: bassi = I-frame/bloom, medi = refresh, acuti = corruzione", key="bands")
            bpm_manual = st.number_input("BPM manuale (0 = automatico)", 0.0, 300.0, step=1.0, key="bpm_manual")
            meter = st.radio("Battuta / Meter", OPTS["meter"], key="meter", horizontal=True)
            st.slider("Spostamento beat (ms) / Beat offset", -300, 300, step=5, key="beat_offset")
            st.slider("Battito accentato della battuta / Downbeat", 1, 6, key="downbeat")
        with st.expander("Salva / carica parametri"):
            st.download_button("Salva parametri (JSON)", params_json(),
                               file_name="datamosh_params.json", mime="application/json")
            st.file_uploader("Carica parametri (JSON)", type=["json"], key="pjson", on_change=load_params)
            if st.session_state.get("pmsg"):
                st.caption(st.session_state["pmsg"])


    def save_uploads():
        tmp = tempfile.mkdtemp()
        out = []
        for up in (up_a, up_b, up_au, up_img):
            if up is None:
                out.append(None)
                continue
            pth = os.path.join(tmp, up.name)
            with open(pth, "wb") as f:
                f.write(up.getbuffer())
            out.append(pth)
        return tmp, out[0], out[1], out[2], out[3]



    @st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
    def cached_analysis(digest, _path, max_sec):
        """Analisi audio pesante in cache: la chiave e' il contenuto del file, non i parametri del tempo."""
        return analyze_audio(_path, max_sec)


    def file_digest(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


    def au_for(pa, pau, msec, fps=None):
        return compute_au(st.session_state, pau or pa, msec, fps or video_fps(pa),
                          features=lambda s, f, n, m, b, a, o, d: features_from_analysis(
                              cached_analysis(file_digest(s), s, m), f, n, b, a, o, d))


    def publish(final, kind, p, fps, stt, preview):
        num = int(st.session_state["counter"])
        name = f"DATAMOSH_N.{num:03d}" + ("_anteprima" if preview else "")
        with open(final, "rb") as f:
            st.session_state["dm_out"] = f.read()
        txt = build_report(kind, p, fps, stt, p["au"]["ok"], st.session_state["preset"], num)
        st.session_state["dm_report"] = {"name": name, "txt": txt, "preview": preview}


    def finish(ok, preview):
        if ok and not preview:
            st.session_state["counter_pending"] = True
            st.rerun()


    def prepare(kind, ss, preview):
        """Salva i file, li porta al formato scelto, controlla le combinazioni e calcola l'audio (dall'originale)."""
        tmp, pa, pb, pau, pi = save_uploads()
        pi = pi if kind in ("DATABEND", "PIXELSORT", "CHAIN") else None  # l'immagine fissa serve solo a questi
        if pi and not pau:
            raise RuntimeError("Per animare l'immagine serve un audio esterno / an audio file is required")
        first = kind
        if kind == "CHAIN":
            names = [ss[f"chain_{i}"] for i in range(3) if ss[f"chain_{i}"] != NONE]
            if not names:
                raise RuntimeError("Scegli almeno un effetto / pick at least one effect")
            first = STEPS[names[0]]
        img_fps = float(ss["db_fps"] if first == "DATABEND" else ss["ps_fps"]) if pi else None
        cp = common_params(ss, preview)
        au = au_for(pa, pau, cp["max_sec"], img_fps)
        if pi and not au["ok"]:
            raise RuntimeError("Audio non leggibile / audio unreadable")
        src = {"tmp": tmp, "video": pa, "b": pb, "image": pi, "audio": pau or pa}
        if cp["fmt"] != "Originale":
            tw, th = format_size(cp["fmt"], cp["max_w"])
            if pi:
                src["image"] = format_image(pi, os.path.join(tmp, "fmt_img.png"), tw, th, cp["fit"])
            elif pa:
                src["video"] = format_video(pa, os.path.join(tmp, "fmt_a.avi"), tw, th, cp["fit"], cp["max_sec"])
            if pb:
                src["b"] = format_video(pb, os.path.join(tmp, "fmt_b.avi"), tw, th, cp["fit"], cp["max_sec"])
        return src, au


    def render(kind, ss, preview, src, au, cb=None):
        """Esegue un effetto (o la catena) e ritorna (video finale, tipo, parametri, fps, statistiche)."""
        tmp = src["tmp"]
        video = None if src["image"] else src["video"]
        msec = common_params(ss, preview)["max_sec"]
        final = os.path.join(tmp, "out.mp4")
        if kind == "CHAIN":
            names = [ss[f"chain_{i}"] for i in range(3) if ss[f"chain_{i}"] != NONE]
            res = run_chain(names, ss, preview, video, src["image"], src["b"], au, tmp, cb)
            stt = {"frames": res["steps"][-1][2]["frames"], "steps": res["steps"]}
            audio = audio_with_mosh(kind, ss, src["audio"], stt, res["fps"], msec, tmp)
            finalize_h264(res["raw"], audio, final, ss["keep_audio"], msec)
            return final, kind, res["steps"][0][1], res["fps"], stt
        p = build_params(kind, ss, preview)
        p["au"] = au
        raw = os.path.join(tmp, "raw." + ("avi" if kind == "BYTE" else "mp4"))
        fps, stt = run_engine(kind, p, video, src["image"], src["b"], raw, cb)
        audio = audio_with_mosh(kind, ss, src["audio"], stt, fps, msec, tmp, p)
        finalize_h264(raw, audio, final, ss["keep_audio"], msec)
        return final, kind, p, fps, stt


    def run_effect(kind, preview):
        """Un effetto (tab 1-5) o la catena (kind = "CHAIN", tab 6): render, download e report."""
        try:
            ss = st.session_state
            src, au = prepare(kind, ss, preview)
            bar = st.progress(0.0)
            with st.spinner("Elaborazione / Processing..."):
                final, k, p, fps, stt = render(kind, ss, preview, src, au, lambda v: bar.progress(min(1.0, v)))
            bar.empty()
            publish(final, k, p, fps, stt, preview)
            return True
        except Exception as e:  # noqa: BLE001
            st.error(f"Errore / Error :: {e}")
            return False


    def run_variants(kind):
        """Quattro anteprime con semi diversi (il primo e' quello attuale): scegli la migliore con un clic."""
        try:
            ss = st.session_state
            src, au = prepare(kind, ss, True)
            rng = random.Random(int(ss["seed"]))
            seeds = [int(ss["seed"])] + [rng.randrange(2 ** 31) for _ in range(3)]
            base = {k: ss[k] for k in DEFAULTS}
            bar, items = st.progress(0.0), []
            with st.spinner("Genero 4 varianti / Generating 4 variants..."):
                for i, sd in enumerate(seeds):
                    sub = os.path.join(src["tmp"], f"v{i}")
                    os.makedirs(sub, exist_ok=True)
                    vp = base if i == 0 else vary_params(base, kind, random.Random(sd))  # la prima resta quella attuale
                    final = render(kind, dict(vp, seed=sd), True, dict(src, tmp=sub), au,
                                   lambda v, i=i: bar.progress(min(1.0, (i + v) / 4)))[0]
                    with open(final, "rb") as f:
                        items.append({"seed": sd, "video": f.read(), "params": {k: vp[k] for k in variant_keys(kind)}})
            bar.empty()
            st.session_state["dm_variants"] = {"kind": kind, "items": items}
        except Exception as e:  # noqa: BLE001
            st.error(f"Errore / Error :: {e}")


    def use_seed(seed):
        st.session_state["seed"] = seed


    def show_variants():
        v = st.session_state.get("dm_variants")
        if not v:
            return
        st.subheader("4 varianti / 4 variants :: " + v["kind"])
        for i, (col, it) in enumerate(zip(st.columns(4), v["items"])):
            col.video(it["video"])
            col.caption(f"seed {it['seed']}")
            col.button("Usa questo seed / Use this seed", key=f"useseed{i}", on_click=use_seed, args=(it["seed"],))


    def waveform_view():
        """Forma d'onda con i beat rilevati: si vede subito se il tempo e' giusto."""
        up = up_au or up_a
        if up is None:
            st.info("Carica un audio o un video per vedere la forma d'onda.")
            return
        ss = st.session_state
        path = os.path.join(tempfile.mkdtemp(), up.name)
        with open(path, "wb") as f:
            f.write(up.getbuffer())
        fps, msec = 30.0, int(ss["max_sec"])
        au = features_from_analysis(cached_analysis(file_digest(path), path, msec), fps, int(msec * fps) + 2,
                                    float(ss["bpm_manual"]), ACC[ss["meter"]], ss["beat_offset"] / 1000.0,
                                    int(ss["downbeat"]) - 1)
        if not au["ok"]:
            st.warning("Audio non leggibile / audio unreadable")
            return
        import altair as alt
        import pandas as pd
        n = int(min(au["dur"], msec) * fps)
        df = pd.DataFrame({"t": np.arange(n) / fps, "volume": au["rms"][:n]})
        bdf = pd.DataFrame(au["beats"], columns=["t", "accent"])
        bdf = bdf[(bdf["t"] >= 0) & (bdf["t"] <= n / fps)]
        area = alt.Chart(df).mark_area(opacity=0.45).encode(x=alt.X("t:Q", title="secondi / seconds"),
                                                            y=alt.Y("volume:Q", title="volume"))
        rules = alt.Chart(bdf).mark_rule().encode(
            x="t:Q", color=alt.Color("accent:N", scale=alt.Scale(domain=[True, False], range=["#ff3366", "#888888"]),
                                     legend=None),
            strokeWidth=alt.condition(alt.datum.accent, alt.value(2.5), alt.value(1)))
        chart = (area + rules).properties(height=170)
        try:
            st.altair_chart(chart, width="stretch")
        except TypeError:
            st.altair_chart(chart, use_container_width=True)
        st.caption(f"BPM {au['bpm']:.1f} :: battuta {ss['meter']} :: {len(bdf)} battiti (in rosso il primo di ogni "
                   f"battuta) :: spostamento {ss['beat_offset']} ms")




    def motion_controls(pfx):
        st.caption("Trasformazioni del campo di moto e mappa: dove agisce il mosh.")
        a, b, c = st.columns(3)
        a.slider("Gravita' X (+ = destra)", -8.0, 8.0, step=0.5, key=f"{pfx}_gx")
        a.slider("Gravita' Y (+ = verso il basso)", -8.0, 8.0, step=0.5, key=f"{pfx}_gy")
        a.radio("Specchio / Mirror", ["No", "Orizzontale", "Verticale"], key=f"{pfx}_mirror", horizontal=True)
        b.slider("Zoom (+ avanti / - indietro)", -3.0, 3.0, step=0.1, key=f"{pfx}_zoom")
        b.selectbox("Lo zoom pulsa con", DRIVERS, key=f"{pfx}_zoom_drv")
        b.slider("Rotazione / Vortice", -3.0, 3.0, step=0.1, key=f"{pfx}_rot")
        c.slider("Shear", -3.0, 3.0, step=0.1, key=f"{pfx}_shear")
        c.slider("Fluido (movimento medio)", 0.0, 1.0, step=0.05, key=f"{pfx}_fluid")
        c.slider("Blocchi sfasati", 0.0, 1.0, step=0.05, key=f"{pfx}_shift")
        d, e = st.columns(2)
        d.selectbox("Dove agisce il mosh (mappa)", MASKS, key=f"{pfx}_mask")
        e.slider("Soglia di movimento della mappa (px)", 0.2, 6.0, step=0.1, key=f"{pfx}_mthr")


    st.toggle("Mostra forma d'onda e beat / Show waveform and beats", key="show_wave")
    if st.session_state["show_wave"]:
        waveform_view()

    try:  # Streamlit recente: i tab ricordano la selezione e si possono cambiare da codice
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(TAB_LABELS, key="active_tab", on_change="rerun")
    except TypeError:  # versione vecchia: nessun cambio automatico di tab
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(TAB_LABELS)

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
        r1, r2, r3 = st.columns(3)
        go1 = r1.button("GENERA BYTE / GENERATE", type="primary", disabled=up_a is None)
        pv1 = r2.button("ANTEPRIMA / PREVIEW", disabled=up_a is None, key="pv1")
        var1 = r3.button("4 VARIANTI / 4 VARIANTS", disabled=up_a is None, key="var1")
        if (go1 or pv1) and up_a is not None:
            finish(run_effect("BYTE", pv1), pv1)
        if var1 and up_a is not None:
            run_variants("BYTE")

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
        with st.expander("Trasformazioni del moto / Motion transforms"):
            motion_controls("mt2")
        r1, r2, r3 = st.columns(3)
        go2 = r1.button("GENERA OPTICAL / GENERATE", type="primary", disabled=up_a is None)
        pv2 = r2.button("ANTEPRIMA / PREVIEW", disabled=up_a is None, key="pv2")
        var2 = r3.button("4 VARIANTI / 4 VARIANTS", disabled=up_a is None, key="var2")
        if (go2 or pv2) and up_a is not None:
            finish(run_effect("OPTICAL", pv2), pv2)
        if var2 and up_a is not None:
            run_variants("OPTICAL")

    with tab3:
        st.write("I pixel diventano un segnale audio e passano negli effetti sonori. Funziona su un video (Video A) oppure su "
                 "un'immagine fissa che si anima col brano (Immagine + Audio esterno). Se carichi entrambi, usa l'immagine.")
        labels = {"echo": "Eco / Echo", "reverb": "Riverbero / Reverb", "invert": "Inverti / Invert", "burn": "Amplifica / Burn",
                  "crush": "Bitcrush", "hp": "Passa-alto / Edge", "dist": "Distorsione / Distortion", "slip": "Slittamento byte / Slip"}
        s1, s2, s3 = st.columns(3)
        s1.radio("Scansione / Scan", OPTS["db_scan"], key="db_scan", horizontal=True,
                 help="Orizzontale = righe; Verticale = colate; Blocchi = mosaico di tessere.")
        s2.radio("Spazio colore / Color space", OPTS["db_space"], key="db_space", horizontal=True,
                 help="YCrCb separa luminosita' e colore: le distorsioni cambiano aspetto.")
        s3.select_slider("Tessera (solo Blocchi) / Tile", OPTS["db_tile"], key="db_tile")
        cols = st.columns(4)
        for i, k in enumerate(FX):
            cols[i % 4].slider(labels[k], 0.0, 1.0, step=0.05, key="db_" + k)
        with st.expander("Driver audio per effetto / Audio driver per effect"):
            dcols = st.columns(4)
            for i, k in enumerate(FX):
                dcols[i % 4].selectbox(labels[k], DRIVERS, key="db_d_" + k)
        with st.expander("Avanzati / Advanced"):
            a3, b3, c3 = st.columns(3)
            a3.slider("Ritardo eco (frazione della larghezza)", 0.01, 1.5, step=0.01, key="db_delay")
            a3.checkbox("Eco allineato ai pixel (multiplo di 3 byte)", key="db_align")
            b3.slider("Decadimento effetti / Release", 0.0, 0.95, step=0.05, key="db_release")
            b3.slider("Memoria (accumulo frame) / Feedback", 0.0, 0.95, step=0.05, key="db_mem")
            c3.slider("Zona righe (%) / Row zone", 0, 100, key="db_zone")
            c3.select_slider("FPS (solo immagine fissa)", OPTS["db_fps"], key="db_fps")
        r1, r2, r3 = st.columns(3)
        nodata = up_a is None and up_img is None
        go3 = r1.button("GENERA DATABEND / GENERATE", type="primary", disabled=nodata)
        pv3 = r2.button("ANTEPRIMA / PREVIEW", disabled=nodata, key="pv3")
        var3 = r3.button("4 VARIANTI / 4 VARIANTS", disabled=nodata, key="var3")
        if go3 or pv3:
            finish(run_effect("DATABEND", pv3), pv3)
        if var3:
            run_variants("DATABEND")


    with tab4:
        st.write("Simula un decoder che accumula i residui dei P-frame senza mai azzerare l'errore: colori saturi, coriandoli "
                 "sui bordi e code dove ci si muove. Il Video B (opzionale) diventa l'immagine di partenza.")
        a, b, c = st.columns(3)
        a.slider("Guadagno luma / Luma gain", 0.0, 3.0, step=0.05, key="rs_luma")
        b.slider("Guadagno colore / Chroma gain", 0.0, 4.0, step=0.05, key="rs_chroma")
        c.slider("Quantizzazione (grana) / Quantization", 0.0, 48.0, step=1.0, key="rs_quant")
        with st.expander("Avanzati / Advanced"):
            a4, b4, c4 = st.columns(3)
            a4.select_slider("Blocco / Block", OPTS["rs_block"], key="rs_block")
            a4.slider("Forza movimento / Motion strength", 0.2, 4.0, step=0.1, key="rs_strength")
            a4.slider("Rumore MV / MV noise", 0.0, 8.0, step=0.5, key="rs_mv_noise")
            b4.slider("RMS -> guadagno / gain", 0.0, 3.0, step=0.1, key="rs_asr")
            b4.checkbox("Onset -> bloom", key="rs_bloom")
            b4.checkbox("Onset forte -> azzera accumulo / refresh", key="rs_refresh")
            b4.slider("Soglia refresh / Refresh thr", 0.5, 1.0, step=0.05, key="rs_refresh_thr")
            c4.slider("Hold prob", 0.0, 0.5, step=0.01, key="rs_hold")
            c4.slider("Bloom gain", 1.0, 5.0, step=0.1, key="rs_bloom_gain")
            c4.slider("Bloom durata / length", 2, 60, key="rs_bloom_len")
            c4.slider("Decadimento dell'errore / Decay", 0.0, 0.3, step=0.01, key="rs_decay")
            c4.slider("Grana / Grain (coriandoli ovunque)", 0.0, 12.0, step=0.5, key="rs_grain")
        with st.expander("Trasformazioni del moto / Motion transforms"):
            motion_controls("mt4")
        r1, r2, r3 = st.columns(3)
        go4 = r1.button("GENERA RESIDUO / GENERATE", type="primary", disabled=up_a is None)
        pv4 = r2.button("ANTEPRIMA / PREVIEW", disabled=up_a is None, key="pv4")
        var4 = r3.button("4 VARIANTI / 4 VARIANTS", disabled=up_a is None, key="var4")
        if (go4 or pv4) and up_a is not None:
            finish(run_effect("RESIDUAL", pv4), pv4)
        if var4 and up_a is not None:
            run_variants("RESIDUAL")


    with tab5:
        st.write("Ordina i pixel di righe o colonne dentro intervalli scelti da una soglia: colate, cascate e righe. "
                 "La soglia si allarga sui colpi. Funziona su un video (Video A) oppure su un'immagine fissa che si anima col brano "
                 "(Immagine + Audio esterno).")
        a5, b5, c5 = st.columns(3)
        a5.selectbox("Ordina per / Sort key", PS_KEYS, key="ps_key")
        b5.radio("Intervalli / Intervals", PS_MODES, key="ps_imode", horizontal=True,
                 help="Soglia = in base alla luminosita'; Bordi = tra un bordo e l'altro; Casuale = tratti di lunghezza casuale.")
        c5.slider("Angolo / Angle (0 = righe, 90 = colonne)", 0, 175, step=5, key="ps_angle")
        a5.slider("Soglie luminosita' (min - max)", 0.0, 1.0, step=0.05, key="ps_range")
        b5.slider("Espansione sui colpi / Audio expansion", 0.0, 0.6, step=0.05, key="ps_expand")
        c5.selectbox("L'espansione segue", DRIVERS, key="ps_drv")
        with st.expander("Avanzati / Advanced"):
            a6, b6, c6 = st.columns(3)
            a6.slider("Lunghezza max intervallo (px, 0 = libera)", 0, 400, step=10, key="ps_maxlen")
            a6.slider("Intervalli lasciati intatti / Skip", 0.0, 0.95, step=0.05, key="ps_skip")
            b6.checkbox("Ordine inverso / Reverse", key="ps_reverse")
            b6.slider("Mix con l'originale", 0.0, 1.0, step=0.05, key="ps_mix")
            c6.slider("Decadimento / Release", 0.0, 0.95, step=0.05, key="ps_release")
            c6.select_slider("FPS (solo immagine fissa)", OPTS["ps_fps"], key="ps_fps")
        r1, r2, r3 = st.columns(3)
        nodata5 = up_a is None and up_img is None
        go5 = r1.button("GENERA PIXEL SORT / GENERATE", type="primary", disabled=nodata5)
        pv5 = r2.button("ANTEPRIMA / PREVIEW", disabled=nodata5, key="pv5")
        var5 = r3.button("4 VARIANTI / 4 VARIANTS", disabled=nodata5, key="var5")
        if go5 or pv5:
            finish(run_effect("PIXELSORT", pv5), pv5)
        if var5:
            run_variants("PIXELSORT")


    with tab6:
        st.write("Concatena fino a 3 effetti: ogni passo lavora sul video del precedente, con lo stesso audio. "
                 "Ogni effetto usa le impostazioni attuali del suo tab: regolale prima, oppure scegli un preset. "
                 "Con un'immagine fissa il primo passo deve essere Databending o Pixel sort.")
        ccols = st.columns(3)
        for i, col in enumerate(ccols):
            col.selectbox(f"Passo {i + 1}", OPTS[f"chain_{i}"], key=f"chain_{i}")
        chain_names = [st.session_state[f"chain_{i}"] for i in range(3) if st.session_state[f"chain_{i}"] != NONE]
        st.caption(" > ".join(chain_names) if chain_names else "Scegli almeno un effetto.")
        if len(chain_names) >= 2 and st.session_state["max_sec"] > 60:
            st.warning("Catena + durata oltre 60 s: il render puo' richiedere molti minuti. Prova prima con l'anteprima.")
        r1, r2, r3 = st.columns(3)
        nodata6 = (up_a is None and up_img is None) or not chain_names
        go6 = r1.button("GENERA CATENA / GENERATE", type="primary", disabled=nodata6)
        pv6 = r2.button("ANTEPRIMA / PREVIEW", disabled=nodata6, key="pv6")
        var6 = r3.button("4 VARIANTI / 4 VARIANTS", disabled=nodata6, key="var6")
        if go6 or pv6:
            finish(run_effect("CHAIN", pv6), pv6)
        if var6:
            run_variants("CHAIN")


    def dl_button(*args, **kw):
        try:  # senza rerun dell'app (Streamlit recente)
            return st.download_button(*args, on_click="ignore", **kw)
        except TypeError:
            return st.download_button(*args, **kw)


    def show_results():
        r = st.session_state["dm_report"]
        vcol, tcol = st.columns([1, 2] if r["preview"] else [3, 2])
        vcol.video(st.session_state["dm_out"])
        with tcol:
            d1, d2 = st.columns(2)
            with d1:
                dl_button("SCARICA VIDEO / DOWNLOAD VIDEO", st.session_state["dm_out"],
                          file_name=r["name"] + ".mp4", mime="video/mp4", key="dl_video")
            with d2:
                dl_button("SCARICA REPORT / DOWNLOAD REPORT", r["txt"],
                          file_name=r["name"] + ".txt", mime="text/plain", key="dl_report")
            st.caption(r["name"] + ".mp4  +  " + r["name"] + ".txt")
            st.code(r["txt"], language="text")


    show_results = getattr(st, "fragment", lambda f: f)(show_results)

    if st.session_state["dm_out"] is not None and st.session_state["dm_report"] is not None:
        show_results()

    show_variants()
