"""
Phase 7c, step 1: duplicate-frame check between training.zip and our val / test scenes.

Every frame of training.zip and every 10th frame of the val / test scenes of val.zip is reduced to a 64x36 grey thumbnail
(and a 16x16 difference hash). For every sampled frame the nearest training frame is found: mean absolute grey difference
(0-255) and Hamming distance of the 256-bit hash. Context distances: adjacent frames inside one scene (a near-duplicate
looks like this) and the nearest frame of a different scene. Writes results/phase7c/duplicate_check.json.
"""
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

import fw_uav_io as io

TRAIN_ZIP = os.path.join(io.DATA_DIR, "training.zip")
CACHE = os.path.join(io.DATA_DIR, "thumbs_7c.npz")
OUT = os.path.join("results", "phase7c")

# Tunable settings (same values as before, now named instead of repeated as bare numbers).
THUMB_SIZE = (64, 36)    # (width, height) of the grey thumbnail used for the mean-absolute-difference check
HASH_RESIZE = (17, 16)   # one extra column, so each row gives 16 left/right comparisons -> 16x16 = 256 bits
N_WORKERS = 6            # processes used to read frames out of the zip files
SAMPLE_STEP = 10         # keep every 10th frame of the val / test scenes
N_CONTEXT = 300          # random training frames used for the context distances
CONTEXT_SEED = 0         # seed for those random frames, so the numbers are reproducible


def thumb(raw):
    """Return a grey thumbnail and a 256-bit difference hash for one encoded image."""
    g = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    t = cv2.resize(g, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    h = cv2.resize(g, HASH_RESIZE, interpolation=cv2.INTER_AREA)
    return t, (h[:, 1:] > h[:, :-1]).ravel()


def job(args):
    """Worker: thumbnail every `step`-th frame of one scene inside one zip file."""
    path, prefix, scene, step = args
    out = []
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if n.startswith(f"{prefix}/{scene}/rgb/") and n.endswith(".png"))
        for n in names[::step]:
            t, h = thumb(zf.read(n))
            out.append((scene, int(os.path.basename(n)[:-4]), t, h))
    return out


def collect(path, prefix, scenes, step):
    """Thumbnail the given scenes in parallel; returns (scene ids, frame numbers, thumbnails, hashes)."""
    with Pool(N_WORKERS) as p:
        res = p.map(job, [(path, prefix, s, step) for s in scenes])
    flat = [r for rs in res for r in rs]
    return (np.array([r[0] for r in flat]), np.array([r[1] for r in flat]), np.stack([r[2] for r in flat]),
            np.stack([r[3] for r in flat]))


def nearest_training_frames(t, h, train_t, train_h):
    """For each sampled frame, distances to every training frame and the index of the closest one.

    Returns (mad, ham, best_by_mad, best_by_hamming); mad and ham have shape (n_sampled, n_train).
    """
    mad = np.abs(t[:, None] - train_t[None]).mean(axis=(2, 3))
    ham = (h[:, None] != train_h[None]).sum(axis=2)
    return mad, ham, mad.argmin(1), ham.argmin(1)


def context_stats(tr):
    """What a near-duplicate looks like: adjacent frames of one scene, vs the nearest frame of another scene."""
    adj, other = [], []
    rng = np.random.default_rng(CONTEXT_SEED)
    for k in rng.choice(len(tr[0]), N_CONTEXT, replace=False):
        same = tr[0] == tr[0][k]
        nb = np.nonzero(same & (np.abs(tr[1].astype(int) - tr[1][k]) == 1))[0]
        if len(nb):
            adj.append(float(np.abs(tr[2][nb[0]] - tr[2][k]).mean()))
        d = np.abs(tr[2][~same] - tr[2][k]).mean(axis=(1, 2))
        other.append(float(d.min()))
    return {"adjacent_frame_mad_median": float(np.median(adj)), "adjacent_frame_mad_max": float(np.max(adj)),
            "other_scene_nearest_mad_min": float(np.min(other)), "other_scene_nearest_mad_median": float(np.median(other))}


def main():
    with open("fw_uav_split.json") as f:
        split = json.load(f)
    with zipfile.ZipFile(TRAIN_ZIP) as zf:
        train_scenes = sorted({n.split("/")[1] for n in zf.namelist() if n.count("/") >= 2})
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        tr = (z["sc"], z["fr"], z["t"], z["h"])
    else:
        tr = collect(TRAIN_ZIP, "training", train_scenes, 1)
        np.savez_compressed(CACHE, sc=tr[0], fr=tr[1], t=tr[2], h=tr[3])
    print("training frames hashed:", len(tr[0]))
    res = {}
    for part in ("test", "val"):
        sc, fr, t, h = collect(io.ZIP_PATH, "val", split[part], SAMPLE_STEP)
        mad, ham, i, j = nearest_training_frames(t, h, tr[2], tr[3])
        rows = [{"scene": s, "frame": int(f), "min_mad": float(mad[k, i[k]]), "nearest_mad": f"{tr[0][i[k]]}_{tr[1][i[k]]:06d}",
                 "min_hamming": int(ham[k, j[k]])} for k, (s, f) in enumerate(zip(sc, fr))]
        res[part] = rows
        print(f"{part}: {len(rows)} sampled frames; min MAD to training {mad.min(1).min():.2f}..{mad.min(1).max():.2f}, "
              f"min Hamming {ham.min(1).min()}..{ham.min(1).max()} (of 256)")
        for s in split[part]:
            r = [x for x in rows if x["scene"] == s]
            print(f"   {s}: min MAD {min(x['min_mad'] for x in r):.2f}, min Hamming {min(x['min_hamming'] for x in r)}")
    ctx = context_stats(tr)
    print("context:", ctx)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "duplicate_check.json"), "w") as f:
        json.dump({"context": ctx, **res}, f, indent=1)


if __name__ == "__main__":
    main()