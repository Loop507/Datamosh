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

import os
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


def compute_flow(prev_gray, cur_gray):
    """Flow cur -> prev: serve per il backward warp (out(x) = mosh(x + flow))."""
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
                 ("onset_thr", 0.6), ("refresh_thr", 0.85)):
        p.setdefault(k, v)
    au = p.get("au")

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
                or bool(au and p["a_refresh"] and at(au["onset"], t) > p["refresh_thr"])
            if refresh:
                mosh = frame.astype(np.float32)
                stats["iframes"] += 1
                bloom_left = 0

            # flow del frame
            k_str = p["strength"] * (1.0 + p["a_strength"] * at(au["rms"], t) if au else 1.0)
            flow = compute_flow(prev_gray, gray) * k_str
            if p["mode"] == "Block MV":
                flow = to_block_mv(flow, p["block"], rng, p["mv_noise"])

            # hold: freeze (P-frame vuoto)
            if rng.random() < p["hold_prob"]:
                flow = np.zeros_like(flow)
                stats["holds"] += 1

            # bloom: ripete lo stesso flow per k frame
            if bloom_left > 0 and bloom_flow is not None:
                flow = bloom_flow
                bloom_left -= 1
            elif (au and p["a_bloom"] and at(au["onset"], t) > p["onset_thr"]) or rng.random() < p["bloom_prob"]:
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
def audio_features(src, fps, n_frames, max_sec):
    """RMS + onset per frame normalizzati 0-1 (librosa). Zeri se manca l'audio."""
    zeros = {"rms": np.zeros(n_frames), "onset": np.zeros(n_frames),
             "beat": np.zeros(n_frames), "bpm": 0.0, "ok": False}
    wav = os.path.join(tempfile.mkdtemp(), "a.wav")
    try:
        subprocess.run([FFMPEG, "-y", "-i", src, "-t", str(max_sec), "-vn", "-ac", "1",
                        "-ar", "22050", wav], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        y, sr = librosa.load(wav, sr=22050, mono=True)
    except Exception:  # noqa: BLE001
        return zeros
    if y.size < 4096:
        return zeros
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    ft = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    vt = np.arange(n_frames) / fps
    tempo, bt = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
    beat = np.zeros(n_frames)
    for i, tb in enumerate(librosa.frames_to_time(bt, sr=sr, hop_length=hop)):
        j = int(round(tb * fps))
        if 0 <= j < n_frames:
            beat[j] = 1.0 if i % 4 == 0 else 0.75  # accento sul battere (4/4)

    def norm(v):
        v = np.interp(vt, ft, v[:len(ft)])
        v = np.where(vt <= ft[-1], v, 0.0)  # oltre la fine dell'audio: silenzio
        return np.clip(v / (np.percentile(v, 99) + 1e-9), 0, 1)

    return {"rms": norm(rms), "onset": norm(onset), "beat": beat,
            "bpm": float(np.atleast_1d(tempo)[0]), "ok": True}


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
    types = [vop_type(f) for f in frames]
    start_f, end_f = int(n * p["mosh_start"] / 100), int(n * p["mosh_end"] / 100)
    out, s, hold, last, last_t = [], 0, 0, None, 0
    st = {"frames": 0, "dropped_i": 0, "dups": 0, "skips": 0, "corrupt": 0}
    for k in range(n):
        on, en = at(au["onset"], k), at(au["rms"], k)
        win = start_f <= k < end_f and k > 0
        if hold > 0 and last is not None:
            out.append(last); hold -= 1; st["dups"] += 1; continue
        s = min(s, n - 1)
        if win:
            if types[s] == 0 and (p["drop_mode"] == "Tutti" or
                                  (p["drop_mode"] == "Su onset" and on > p["drop_thr"])):
                s = min(s + 1, n - 1); st["dropped_i"] += 1
            if last is not None and last_t != 0 and (on > p["onset_thr"] or rng.random() < p["bloom_prob"]):
                hold = int(max(on, 0.3) * p["max_dup"]); out.append(last); st["dups"] += 1; continue
            if en < p["quiet_thr"] and last is not None and last_t != 0:
                if k - s > 0:
                    s = min(s + 1, n - 1); st["skips"] += 1
                elif k - s < 0:
                    out.append(last); st["dups"] += 1; continue
        f, t = frames[s], types[s]
        if win and t != 0:
            nb = int(len(f) * p["corrupt"] * (en + on))
            if nb > 0:
                f = corrupt_bytes(f, nb, rng); st["corrupt"] += nb
        out.append(f); last, last_t = f, t; s += 1
    st["frames"] = len(out)
    return out, st


def byte_datamosh(src, out_avi, p):
    rng = np.random.default_rng(p["seed"])
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ow, oh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    w = max(16, int(min(p["max_w"], ow)) // 2 * 2)
    h = max(16, int(oh * w / ow) // 2 * 2)
    tmp = os.path.join(tempfile.mkdtemp(), "src.avi")
    subprocess.run([FFMPEG, "-y", "-i", src, "-t", str(p["max_sec"]), "-an",
                    "-vf", f"scale={w}:{h},fps={fps}", "-c:v", "mpeg4", "-vtag", "XVID",
                    "-q:v", str(p["qscale"]), "-g", str(p["kf"]), "-bf", "0", "-f", "avi", tmp],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = read_avi_frames(tmp)
    if len(frames) < 3:
        raise RuntimeError("Troppi pochi frame / too few frames")
    au = p["au"] or {"rms": np.zeros(len(frames)), "onset": np.zeros(len(frames))}
    out, st = byte_mosh(frames, au, p, rng)
    write_avi(out_avi, out, w, h, fps)
    return fps, st


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="DATAMOSH // Loop507", layout="wide")
st.title("DATAMOSH // Loop507")
st.caption("Minimalismo Computazionale / Glitch Brutalista :: audio-reactive :: puro DSP, nessuna AI")
for key in ("dm_out", "dm_report"):
    st.session_state.setdefault(key, None)

VT = ["mp4", "mov", "avi", "mkv", "webm"]
c1, c2 = st.columns(2)
up_a = c1.file_uploader("Video A (movimento + audio) / motion + audio", type=VT, key="a")
up_b = c2.file_uploader("Video B opzionale (texture, solo Optical Flow)", type=VT, key="b")
up_au = st.file_uploader("Audio esterno opzionale (il tuo brano: guida il mosh e diventa la colonna sonora)",
                         type=["wav", "mp3", "flac", "ogg", "m4a", "aac"], key="au")

with st.sidebar:
    st.header("Comuni / Common")
    seed = st.number_input("Seed", 0, 2**31 - 1, 507)
    mosh_range = st.slider("Finestra mosh / window (%)", 0, 100, (15, 100))
    max_w = st.select_slider("Larghezza max / Max width", [320, 480, 640, 854, 1280], 640)
    max_sec = st.slider("Durata max (s) / Max seconds", 2, 60, 15)
    keep_audio = st.checkbox("Mantieni audio (A o esterno) / Keep audio", True)
    onset_thr = st.slider("Soglia onset / Onset threshold", 0.1, 1.0, 0.6, 0.05)
    bloom_prob = st.slider("Bloom casuale / Random bloom prob", 0.0, 0.3, 0.02, 0.01)
    trigger = st.radio("Trigger audio", ["Onset", "Beat (a tempo)", "Onset + Beat"],
                       help="Beat = scatti precisi sul tempo del brano (battere piu' forte).")


def save_uploads():
    tmp = tempfile.mkdtemp()
    out = []
    for up in (up_a, up_b, up_au):
        if up is None:
            out.append(None); continue
        pth = os.path.join(tmp, up.name)
        with open(pth, "wb") as f:
            f.write(up.getbuffer())
        out.append(pth)
    return tmp, out[0], out[1], out[2]


def common():
    return {"seed": int(seed), "mosh_start": mosh_range[0], "mosh_end": mosh_range[1],
            "max_w": int(max_w), "max_sec": int(max_sec), "onset_thr": onset_thr, "trigger": trigger,
            "bloom_prob": bloom_prob}


def au_for(pa, pau=None):
    cap = cv2.VideoCapture(pa)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    au = audio_features(pau or pa, fps, int(max_sec * fps) + 2, max_sec)
    if trigger == "Beat (a tempo)":
        au["onset"] = au["beat"]
    elif trigger == "Onset + Beat":
        au["onset"] = np.maximum(au["onset"], au["beat"])
    return au


def publish(final, rep):
    with open(final, "rb") as f:
        st.session_state["dm_out"] = f.read()
    st.session_state["dm_report"] = rep


tab1, tab2 = st.tabs(["1 // BYTE (AVI)", "2 // OPTICAL FLOW"])

with tab1:
    st.write("Elimina I-frame, duplica P-frame e corrompe byte :: guidato dall'audio.")
    a, b, c = st.columns(3)
    kf = a.slider("Keyframe ogni N / Keyframe every N", 6, 120, 12)
    qscale = a.slider("Qualita' (basso = migliore) / q-scale", 2, 15, 5)
    drop_mode = b.radio("Elimina I-frame / Drop I-frames", ["Su onset", "Tutti", "Mai"])
    drop_thr = b.slider("Soglia drop I / Drop threshold", 0.1, 1.0, 0.5, 0.05)
    max_dup = c.slider("Max duplicati P per onset / Max P dups", 1, 40, 12)
    corrupt = c.slider("Corruzione byte / Byte corruption", 0.0, 0.01, 0.002, 0.0005, format="%.4f")
    quiet_thr = a.slider("Soglia quiete (recupero sync) / Quiet thr", 0.0, 0.6, 0.25, 0.05)
    go1 = st.button("GENERA BYTE / GENERATE", type="primary", disabled=up_a is None)
    if go1 and up_a is not None:
        tmp, pa, _, pau = save_uploads()
        try:
            p = {**common(), "kf": kf, "qscale": qscale, "drop_mode": drop_mode,
                 "drop_thr": drop_thr, "max_dup": max_dup, "corrupt": corrupt,
                 "quiet_thr": quiet_thr}
            p["au"] = au_for(pa, pau)
            avi, final = os.path.join(tmp, "mosh.avi"), os.path.join(tmp, "out.mp4")
            with st.spinner("Moshing bytes..."):
                fps, stt = byte_datamosh(pa, avi, p)
                finalize_h264(avi, pau or pa, final, keep_audio, max_sec)
            publish(final, ("BYTE", p, fps, stt, p["au"]["ok"]))
        except Exception as e:  # noqa: BLE001
            st.error(f"Errore / Error :: {e}")

with tab2:
    st.write("Optical flow Farneback :: transfer A->B opzionale :: guidato dall'audio.")
    a, b, c = st.columns(3)
    mode = a.radio("Modo / Mode", ["Block MV", "Optical Flow"])
    block = a.select_slider("Blocco / Block", [8, 16, 32, 64], 16)
    strength = b.slider("Forza / Strength", 0.2, 4.0, 1.5, 0.1)
    mv_noise = b.slider("Rumore MV / MV noise", 0.0, 8.0, 0.0, 0.5)
    hold_prob = b.slider("Hold prob", 0.0, 0.5, 0.05, 0.01)
    bloom_gain = c.slider("Bloom gain", 1.0, 5.0, 2.5, 0.1)
    bloom_len = c.slider("Bloom durata / length", 2, 60, (6, 24))
    residual = c.slider("Residuo / Leak", 0.0, 0.5, 0.0, 0.01)
    a_strength = a.slider("RMS -> forza / strength", 0.0, 3.0, 1.0, 0.1)
    a_bloom = a.checkbox("Onset -> bloom", True)
    a_refresh = a.checkbox("Onset forte -> I-frame / strong onset refresh", False)
    refresh_thr = a.slider("Soglia refresh / Refresh thr", 0.5, 1.0, 0.85, 0.05)
    go2 = st.button("GENERA OPTICAL / GENERATE", type="primary", disabled=up_a is None)
    if go2 and up_a is not None:
        tmp, pa, pb, pau = save_uploads()
        try:
            p = {**common(), "mode": mode, "block": int(block), "strength": strength,
                 "mv_noise": mv_noise, "iframe_every": 0, "iframe_prob": 0.0,
                 "hold_prob": hold_prob, "bloom_gain": bloom_gain,
                 "bloom_len": (int(bloom_len[0]), int(bloom_len[1])), "residual": residual,
                 "a_strength": a_strength, "a_bloom": a_bloom, "a_refresh": a_refresh,
                 "refresh_thr": refresh_thr}
            p["au"] = au_for(pa, pau)
            raw, final = os.path.join(tmp, "raw.mp4"), os.path.join(tmp, "out.mp4")
            bar = st.progress(0.0)
            fps, stt = datamosh_video(pa, pb, raw, p, progress_cb=lambda v: bar.progress(min(1.0, v)))
            finalize_h264(raw, pau or pa, final, keep_audio, max_sec)
            bar.empty()
            publish(final, ("OPTICAL", p, fps, stt, p["au"]["ok"]))
        except Exception as e:  # noqa: BLE001
            st.error(f"Errore / Error :: {e}")

if st.session_state["dm_out"] is not None:
    st.video(st.session_state["dm_out"])
    st.download_button("SCARICA MP4 / DOWNLOAD MP4", st.session_state["dm_out"],
                       file_name="datamosh_loop507.mp4", mime="video/mp4")
    kind, p, fps, stt, au_ok = st.session_state["dm_report"]
    lines = ["DATAMOSH REPORT // LOOP507", f"approccio / approach :: {kind}",
             f"seed :: {p['seed']}  fps :: {fps:.2f}",
             f"finestra / window :: {p['mosh_start']}% -> {p['mosh_end']}%",
             f"audio-reactive :: {'si / yes' if au_ok else 'no (audio assente / missing)'}"]
    lines.append(f"trigger :: {p.get('trigger', 'Onset')}  bpm :: {p['au'].get('bpm', 0):.1f}")
    lines += [f"{k} :: {v}" for k, v in stt.items()]
    lines.append("dsp :: " + ("AVI/MPEG-4 byte-level" if kind == "BYTE" else "OpenCV Farneback + remap")
                 + " + librosa :: no AI")
    st.code("\n".join(lines), language="text")
