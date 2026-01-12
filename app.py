from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid, os, subprocess, requests, csv, math, shutil
import yt_dlp
import base64
import cv2
import numpy as np
import time

app = FastAPI()

# CORS (pentru Bubble / test)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # pentru test; în prod pui domeniul tău
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servește fișierele generate: /files/<job_id>/...
app.mount("/files", StaticFiles(directory="."), name="files")


class ExtractRequest(BaseModel):
    video_url: str

    # Câte screenshot-uri finale
    max_images: int = 10

    # Câte GIF-uri să facă (separat de max_images)
    max_gifs: int = 4

    # Câte cadre candidate să evalueze înainte (mai mare = selecție mai bună, dar mai lent)
    candidates: int = 140

    # GIF settings
    make_gifs: bool = True
    gif_fps: int = 15          # (compat Bubble) dar backend-ul îl forțează la 15
    gif_seconds: int = 10      # tu alegi, backend clamp 3..10
    gif_width: int = 480       # reduce dacă vrei fișiere mai mici


def sh(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


@app.get("/health")
def health():
    def status(cmd):
        code, out, err = sh(cmd)
        return {"cmd": " ".join(cmd), "code": code, "out": out[:200], "err": err[:200]}

    return {
        "cwd": os.getcwd(),
        "ffmpeg": status(["ffmpeg", "-version"]),
        "ffprobe": status(["ffprobe", "-version"]),
        "scenedetect": status(["scenedetect", "-h"]),  # la tine nu merge -version
        "opencv": cv2.__version__,
    }


def ensure_tools():
    for tool in ["ffmpeg", "ffprobe"]:
        code, _, err = sh([tool, "-version"])
        if code != 0:
            raise RuntimeError(f"{tool} nu este disponibil în PATH: {err}")

    code, _, err = sh(["scenedetect", "-h"])
    if code != 0:
        raise RuntimeError(f"scenedetect nu este disponibil în PATH: {err}")


def download_video(url: str, output_path: str):
    # direct mp4
    if url.lower().endswith(".mp4"):
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(r.content)
        return

    cookies_file = os.environ.get("YTDLP_COOKIES", "").strip()

    ydl_opts = {
        "outtmpl": output_path,
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "quiet": False,
        "noplaylist": True,
        "merge_output_format": "mp4",
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
    }

    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file
        print(f"[yt] using cookies file: {cookies_file}")
    else:
        print("[yt] cookies file missing; continuing without cookies")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def require_key(x_api_key: str | None):
    expected = os.environ.get("TAGGLE_API_KEY", "")
    # dacă nu ai setat cheia în Render, nu blochează (mod dev)
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def parse_timecode(t: str) -> float:
    t = str(t).strip()
    if not t:
        return 0.0
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(t)


def find_scenes_csv(folder: str) -> str:
    for f in os.listdir(folder):
        lf = f.lower()
        if lf.endswith("-scenes.csv") or lf == "scenes.csv":
            return os.path.join(folder, f)
    raise FileNotFoundError("Scenes CSV not found (ex: input-Scenes.csv)")


def read_timecode_list_csv(csv_path: str):
    """
    Uneori scenedetect exportă un CSV de tip “Timecode List”.
    Extragem orice celulă care pare timecode.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    timecodes = []
    for r in rows:
        for cell in r:
            c = str(cell).strip()
            if ":" in c and any(ch.isdigit() for ch in c):
                try:
                    timecodes.append(parse_timecode(c))
                except:
                    pass

    return sorted(set([round(x, 3) for x in timecodes if x >= 0]))


def read_scenes_or_timecodes(csv_path: str):
    """
    Returnează:
      ("scenes", list_of_(dur,start,end)) sau ("timecodes", list_of_seconds)
    """
    # try scenes format
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            start_col = None
            end_col = None
            for h in headers:
                lh = h.lower()
                if start_col is None and "start" in lh:
                    start_col = h
                if end_col is None and "end" in lh:
                    end_col = h

            if start_col and end_col:
                scenes = []
                for row in reader:
                    try:
                        start = parse_timecode(row[start_col])
                        end = parse_timecode(row[end_col])
                        if end > start:
                            scenes.append((end - start, start, end))
                    except:
                        continue
                return "scenes", scenes
    except:
        pass

    # fallback timecode list
    return "timecodes", read_timecode_list_csv(csv_path)


def video_duration_seconds(video_path: str) -> float:
    code, out, err = sh([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        video_path
    ])
    if code != 0:
        raise RuntimeError(f"ffprobe error: {err}")
    return float(out.strip())


def extract_frame(video_path: str, ts: float, out_path: str):
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = (p.returncode == 0) and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    return ok, p.stderr[-600:]


def pick_evenly(items, k):
    if not items:
        return []
    if k <= 1:
        return [items[len(items)//2]]
    if len(items) <= k:
        return items
    idxs = [round(i * (len(items)-1) / (k-1)) for i in range(k)]
    out = []
    last = None
    for idx in idxs:
        val = items[int(idx)]
        if last is None or val != last:
            out.append(val)
            last = val
    return out[:k]


# --- Cinematic scoring ---
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _clipping_score(gray: np.ndarray) -> float:
    """
    Penalizează highlights/shadows tăiate (clip).
    Returnează un scor mic când e ok, mare când e clip.
    """
    if gray.size == 0:
        return 0.0
    # procente de pixeli aproape de 0 sau 255
    dark = float((gray <= 5).mean())
    bright = float((gray >= 250).mean())
    return dark + bright


def _motion_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Proxy simplu pentru "energie/motion": diferență medie între cadre.
    (Bun pt reclame: acțiune, whip-pan, lumină schimbătoare, match-action etc.)
    """
    if img_a is None or img_b is None:
        return 0.0
    a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    a = cv2.resize(a, (256, 144), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (256, 144), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(a, b)
    return float(diff.mean())


def score_frame(img_bgr: np.ndarray) -> dict:
    """
    Scor “director-friendly”:
    - fețe / mimică (mai multă greutate + close-up)
    - sharpness (overall + în zona feței)
    - contrast / lumină (și penalizare clip highlights/shadows)
    - colorfulness
    Plus hist vector pentru diversitate.
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return {"score": -1e9, "vec": None}

    small = cv2.resize(img_bgr, (320, int(320*h/w))) if w > 320 else img_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # focus / sharpness
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # lumină
    mean = float(gray.mean())
    std = float(gray.std())
    exposure_pen = abs(mean - 130.0)
    clip_pen = _clipping_score(gray)  # highlights/shadows tăiate

    # colorfulness
    (B, G, R) = cv2.split(small.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    colorfulness = float(np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2))

    # faces: număr + mărimea celei mai mari fețe (close-up) + sharpness în ROI
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    face_score = 0.0
    max_face_area = 0.0
    face_roi_sharp = 0.0

    for (x, y, fw, fh) in faces[:3]:
        area = (fw * fh) / float(gray.shape[0] * gray.shape[1])
        max_face_area = max(max_face_area, area)

        roi = gray[y:y+fh, x:x+fw]
        if roi.size > 0:
            face_roi_sharp = max(face_roi_sharp, float(cv2.Laplacian(roi, cv2.CV_64F).var()))

        # preferăm fețe vizibile (mimică)
        face_score += area * 12.0

    # histogramă pt diversitate
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # compunere scor
    score = (
        2.0 * math.log1p(sharp) +
        1.1 * math.log1p(std) +
        1.0 * math.log1p(colorfulness) +
        5.2 * face_score +                 # fețe/mimică mai important
        0.6 * math.log1p(face_roi_sharp) + # față bine focusată
        1.8 * math.log1p(max_face_area * 1000.0 + 1.0) -  # close-up bonus
        0.015 * exposure_pen -
        6.0 * clip_pen                     # penalizează hard clip
    )

    return {
        "score": float(score),
        "vec": hist,
        "faces": int(len(faces)),
        "mean": mean,
        "sharp": sharp,
        "clip": float(clip_pen),
        "face_max_area": float(max_face_area),
    }


def hist_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))


# --- Perceptual dedup (dHash) ---
def dhash64(img_bgr: np.ndarray) -> int:
    """
    dHash 8x8 => 64-bit int. Robust pt duplicate vizuale (cadre similare).
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]  # 8x8 boolean
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(bool(b))
    return h


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def is_unique_hash(new_hash: int, used_hashes: list, min_dist: int) -> bool:
    for h in used_hashes:
        if hamming64(new_hash, h) < min_dist:
            return False
    return True

def ensure_cookies_file():
    """
    Scrie cookies.txt pe disk din env var YTDLP_COOKIES_B64 (base64).
    Folosit de yt-dlp prin optiunea 'cookiefile'.
    """
    b64 = os.environ.get("YTDLP_COOKIES_B64", "").strip()
    path = os.environ.get("YTDLP_COOKIES", "/app/cookies.txt").strip()

    if not b64:
        print("[cookies] YTDLP_COOKIES_B64 not set; skipping")
        return

    try:
        data = base64.b64decode(b64.encode("utf-8"))
        # dacă path nu are folder (rare), nu încercăm să creăm
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(path, "wb") as f:
            f.write(data)

        print(f"[cookies] written to {path} ({len(data)} bytes)")
    except Exception as e:
        print(f"[cookies] failed to write: {e}")

def select_diverse_unique_top(candidates: list, k: int, hash_min_dist: int = 12, min_ts_gap: float = 0.0) -> list:
    """
    Greedy selection:
    - începe cu cel mai bun scor
    - apoi adaugă cadre care maximizează scor + diversitate
    - refuză duplicate vizuale (dHash)
    - opțional: refuză cadre prea apropiate în timp (ca să nu iei 3 cadre din aceeași scenă)
    """
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    selected = []
    used_hashes = []
    used_ts = []

    for c in candidates:
        if c.get("vec") is not None:
            selected.append(c)
            used_hashes.append(int(c.get("hash", 0)))
            used_ts.append(float(c.get("ts", 0.0)))
            break

    if not selected:
        return []

    remaining = [c for c in candidates if c is not selected[0]]
    local_hash_dist = hash_min_dist

    while len(selected) < k and remaining:
        best = None
        best_val = -1e18

        for c in remaining[:700]:
            if c.get("vec") is None:
                continue

            ts = float(c.get("ts", 0.0))
            if min_ts_gap > 0 and any(abs(ts - t0) < min_ts_gap for t0 in used_ts):
                continue

            ch = int(c.get("hash", 0))
            if not is_unique_hash(ch, used_hashes, local_hash_dist):
                continue

            d = min(hist_dist(c["vec"], s["vec"]) for s in selected)
            val = c["score"] + 1.6 * d

            if val > best_val:
                best_val = val
                best = c

        if best is None:
            # relaxăm treptat ca să umplem k, dar păstrăm totuși anti-duplicate decent
            local_hash_dist = max(6, local_hash_dist - 2)
            if local_hash_dist <= 6:
                break
            continue

        selected.append(best)
        used_hashes.append(int(best.get("hash", 0)))
        used_ts.append(float(best.get("ts", 0.0)))
        remaining.remove(best)

    return selected


def make_gif(video_path: str, center_ts: float, out_gif: str, fps: int, seconds: int, width: int):
    """
    GIF fps, seconds, cu palettegen/paletteuse (calitate bună).
    start = center_ts - seconds/2
    """
    start = max(0.0, center_ts - (seconds / 2.0))
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(seconds),
        "-i", video_path,
        "-vf", vf + ",split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop", "0",
        out_gif
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = (p.returncode == 0) and os.path.exists(out_gif) and os.path.getsize(out_gif) > 0
    return ok, p.stderr[-800:]


@app.post("/extract")
def extract(req: ExtractRequest, request: Request,  x_api_key: str | None = Header(default=None)):
    try:
        ensure_tools()
        ensure_cookies_file()
        require_key(x_api_key)
        job_id = str(uuid.uuid4())
        os.makedirs(job_id, exist_ok=True)
        video_path = f"{job_id}/input.mp4"

        print(f"\n[Job {job_id}] DOWNLOAD...")
        download_video(req.video_url, video_path)

        dur = video_duration_seconds(video_path)
        print(f"[Job {job_id}] duration={dur:.2f}s")

        print(f"[Job {job_id}] SCENEDETECT...")
        code, out, err = sh(["scenedetect", "-i", video_path, "detect-content", "list-scenes", "-o", job_id])
        if code != 0:
            raise RuntimeError(f"scenedetect failed:\n{err}")

        scenes_csv = find_scenes_csv(job_id)
        mode, data = read_scenes_or_timecodes(scenes_csv)
        print(f"[Job {job_id}] csv={os.path.basename(scenes_csv)} mode={mode}")

        max_images = max(1, int(req.max_images))
        max_gifs = max(0, int(req.max_gifs))
        candidates_n = max(max_images * 10, int(req.candidates))  # puțin mai mulți candidați -> mai bun pt cut-uri

        # 1) candidate timestamps
        timestamps = []

        if mode == "scenes":
            scenes = data
            if not scenes:
                raise RuntimeError("No scenes parsed from CSV.")
            scenes = sorted(scenes, key=lambda x: x[1])

            # Pentru "cut/match cut": luăm nu doar mid, ci și near-start / near-end
            eps = 0.18  # secunde (frame offset) ca să prindem before/after cut
            scene_ts = []
            for (_, s, e) in scenes:
                mid = (s + e) / 2.0
                scene_ts.extend([
                    max(0.0, s + eps),         # aproape de început (după cut)
                    max(0.0, mid),             # reprezentativ
                    max(0.0, e - eps),         # aproape de sfârșit (înainte de cut)
                ])

            scene_ts = [max(0.0, min(ts, max(0.0, dur - 0.3))) for ts in scene_ts]
            timestamps = pick_evenly(scene_ts, min(candidates_n, len(scene_ts)))
        else:
            timecodes = data
            if not timecodes:
                raise RuntimeError("CSV did not contain any timecodes.")
            timestamps = pick_evenly(timecodes, min(candidates_n, len(timecodes)))

        # fallback: completează uniform dacă e nevoie
        if len(timestamps) < candidates_n:
            extra = pick_evenly(list(np.linspace(0, max(0.0, dur - 0.5), num=candidates_n)), candidates_n)
            timestamps = sorted(set([round(x, 2) for x in (timestamps + extra)]))

        timestamps = [max(0.0, min(ts, max(0.0, dur - 0.3))) for ts in timestamps]
        print(f"[Job {job_id}] candidates timestamps={len(timestamps)}")

        # 2) extract + score candidates
        tmp_dir = os.path.join(job_id, "_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        scored = []
        motion_dt = 0.20  # secunde pentru motion proxy

        for idx, ts in enumerate(timestamps, 1):
            tmp_img = os.path.join(tmp_dir, f"c-{idx:04d}.jpg")
            ok, _ = extract_frame(video_path, ts, tmp_img)
            if not ok:
                continue
            img = cv2.imread(tmp_img)
            if img is None:
                continue

            # motion proxy (2nd frame la ts + dt)
            tmp_img2 = os.path.join(tmp_dir, f"m-{idx:04d}.jpg")
            ok2, _ = extract_frame(video_path, min(ts + motion_dt, max(0.0, dur - 0.3)), tmp_img2)
            img2 = cv2.imread(tmp_img2) if ok2 else None
            motion = _motion_score(img, img2)

            info = score_frame(img)

            # Bonus ușor pentru energie/motion (bune pt idei de filmare, camera movement, match-action)
            # (nu exagerăm ca să nu alegem doar blur-uri)
            info["score"] = float(info["score"] + 0.35 * math.log1p(motion))
            info["motion"] = float(motion)

            info.update({
                "ts": float(ts),
                "tmp": tmp_img,
                "hash": dhash64(img),
            })
            scored.append(info)

        if not scored:
            raise RuntimeError("Nu am putut extrage niciun frame candidat.")

        # min_ts_gap: evită să iei prea multe din aceeași zonă (mai "story coverage")
        # (dacă vrei ultra-dens pe spoturi scurte, poți reduce)
        min_ts_gap_frames = max(0.8, (dur / max_images) * 0.25)

        selected = select_diverse_unique_top(
            scored,
            max_images,
            hash_min_dist=12,
            min_ts_gap=min_ts_gap_frames
        )

        # 3) save final keyframes
        saved_frames = []
        for i, c in enumerate(selected, 1):
            out_img = f"{job_id}/keyframe-{i:02d}.jpg"
            shutil.copyfile(c["tmp"], out_img)
            saved_frames.append(os.path.basename(out_img))

        # 4) GIF selection (separat din ALL scored ca să ajungi cât ai cerut, fără duplicate)
        saved_gifs = []
        gif_errors = []

        gif_fps = 15
        gif_seconds = max(3, min(10, int(req.gif_seconds)))

        gif_count = 0
        gif_used_hashes = []
        gif_used_ts = []
        min_ts_gap_gif = max(1.0, gif_seconds * 0.60)

        if req.make_gifs and max_gifs > 0:
            gif_candidates = sorted(scored, key=lambda x: x["score"], reverse=True)

            for c in gif_candidates:
                if gif_count >= max_gifs:
                    break

                ts = float(c["ts"])

                if any(abs(ts - t0) < min_ts_gap_gif for t0 in gif_used_ts):
                    continue

                ch = int(c.get("hash", 0))
                if not is_unique_hash(ch, gif_used_hashes, min_dist=12):
                    continue

                out_gif = f"{job_id}/clip-{gif_count + 1:02d}.gif"
                ok, ferr = make_gif(
                    video_path, ts, out_gif,
                    fps=gif_fps,
                    seconds=gif_seconds,
                    width=int(req.gif_width)
                )
                if ok:
                    gif_count += 1
                    saved_gifs.append(os.path.basename(out_gif))
                    gif_used_hashes.append(ch)
                    gif_used_ts.append(ts)
                else:
                    gif_errors.append(f"gif {gif_count + 1} failed: {ferr}")

        # cleanup tmp
        shutil.rmtree(tmp_dir, ignore_errors=True)

        base = str(request.base_url).rstrip("/")
        frame_urls = [f"{base}/files/{job_id}/{name}" for name in saved_frames]
        gif_urls = [f"{base}/files/{job_id}/{name}" for name in saved_gifs]

        meta = [{
            "file": saved_frames[i],
            "ts": float(selected[i]["ts"]),
            "score": float(selected[i]["score"]),
            "faces": int(selected[i]["faces"]),
            "mean": float(selected[i]["mean"]),
            "sharp": float(selected[i]["sharp"]),
            "motion": float(selected[i].get("motion", 0.0)),
            "clip": float(selected[i].get("clip", 0.0)),
            "face_max_area": float(selected[i].get("face_max_area", 0.0)),
        } for i in range(len(saved_frames))]

        return {
            "job_id": job_id,
            "frames": frame_urls,
            "gifs": gif_urls,
            "mode": mode,
            "meta": meta,
            "gif_settings": {
                "fps": gif_fps,
                "seconds": gif_seconds,
                "width": int(req.gif_width),
                "max_gifs": int(req.max_gifs),
            },
            "gif_errors_preview": gif_errors[:2],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
