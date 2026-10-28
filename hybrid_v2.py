"""Reproducible U-Net/YOLO instance fusion using only competition-trained models.

Local: python sfseg/exp/hybrid_v2.py
Kaggle: the companion notebook attaches the four original training kernels.
The YOLO model held out network fold 0. All meta-model validation is therefore
restricted to those 177 unseen images, grouped by observation day.
"""
import concurrent.futures
import glob
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import zipfile

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from sklearn.model_selection import GroupKFold

H = W = 2048
ROOT = Path(__file__).resolve().parents[2] if "/kaggle/" not in __file__ else Path("/kaggle/working")
LOCAL = (ROOT / "work/v123/meta.json").exists()
OUT = ROOT / "work/hybrid_v2" if LOCAL else ROOT / "hybrid_v2"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
FEATURES = ["source", "conf", "area", "bw", "bh", "fill", "aspect", "radius",
            "p0_mean", "p0_q25", "p0_max", "p1_mean", "p1_q25", "p1_max",
            "cc_iou", "yolo_iou", "yolo_conf", "boundary_change", "image_count"]


def log(*args):
    print(f"[{(time.time()-T0)/60:.1f}m]", *args, flush=True)


def encode_crop(mask, x, y):
    """Encode a small bounding-box mask directly into full-image COCO RLE."""
    yy, xx = np.nonzero(mask.T)
    if not len(xx):
        return None
    idx = (yy.astype(np.int64) + x) * H + xx + y
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks] + 1, idx[-1] + 1]
    counts = np.empty(2 * len(starts) + 1, dtype=np.int64)
    counts[0] = starts[0]
    counts[1:-1:2] = ends - starts
    counts[2:-1:2] = starts[1:] - ends[:-1]
    counts[-1] = H * W - ends[-1]
    return mu.frPyObjects({"size": [H, W], "counts": counts.tolist()}, H, W)


def geometry(rle):
    x, y, w, h = map(int, mu.toBbox(rle))
    # Full decode occurs only once per original proposal, not once per setting.
    mask = mu.decode(rle)[y:y+h, x:x+w]
    return x, y, w, h, mask


def counts(gt_sets, pred):
    acc = np.zeros(4, dtype=float)
    for gt in gt_sets:
        if not gt:
            acc[2] += len(pred)
        elif not pred:
            acc[3] += len(gt)
        else:
            iou = np.asarray(mu.iou(pred, gt, [0] * len(gt)))
            hit = iou > .5
            acc += [iou[hit].sum(), hit.sum(), (~hit.any(1)).sum(), (~hit.any(0)).sum()]
    return acc


def pq(c):
    return float(c[0] / max(c[1] + .5 * (c[2] + c[3]), 1))


def discover():
    if LOCAL:
        roots = [ROOT / "work" / v for v in ("v1", "v2", "v3")]
        meta = json.load(open(ROOT / "work/v123/meta.json"))
        gt = pickle.load(open(ROOT / "work/gt.pkl", "rb"))
        yp = ROOT / "work/yolo_v1"
        data = ROOT / "MAGFiLO_1.0_Kaggle_2026"
    else:
        roots = [Path(glob.glob(f"/kaggle/input/**/sf-seg-train-{v}", recursive=True)[0]) for v in ("v1", "v2", "v3")]
        meta = json.load(open(roots[0] / "meta.json"))
        yp = Path(glob.glob("/kaggle/input/**/pred_hold.pkl", recursive=True)[0]).parent
        js = Path(glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)[0])
        data = js.parent.parent
        coco = json.load(open(js))
        annotations = {}
        for a in coco["annotations"]:
            annotations.setdefault(a["image_id"], []).append(mu.merge(mu.frPyObjects(a["segmentation"], H, W)))
        gt = {}
        for im in coco["images"]:
            gt.setdefault(Path(im["file_name"]).stem, []).append(annotations.get(im["id"], []))
    hold = pickle.load(open(yp / "pred_hold.pkl", "rb"))
    test = pickle.load(open(yp / "pred_test.pkl", "rb"))
    fold = dict(zip(meta["stems"], meta["fold"]))
    assert set(hold) == {s for s, f in fold.items() if f == 0}, "YOLO holdout/fold mismatch"
    return roots, meta, gt, hold, test, data


def one_image(job):
    stem, items, paths, gt_sets = job
    p = np.zeros((H, W, 2), np.float32)
    for path in paths:
        with zipfile.ZipFile(path) as z:
            a = cv2.imdecode(np.frombuffer(z.read(stem + ".png"), np.uint8), cv2.IMREAD_COLOR)
        p += a[:, :, :2]
    p /= 255 * len(paths)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((p[:, :, 0] >= .5).astype(np.uint8), 8)
    proposals = []
    for k in range(1, n):
        x, y, w, h, area = map(int, stats[k])
        if area >= 60:
            proposals.append((0, 0., encode_crop(lab[y:y+h, x:x+w] == k, x, y), 0.))
    cc = [r[2] for r in proposals]
    ys = [{"size": [H, W], "counts": c.encode() if isinstance(c, str) else c}
          for score, c in items if score >= .03]
    scores = [score for score, _ in items if score >= .03]
    for score, rle in zip(scores, ys):
        if mu.area(rle) < 60:
            continue
        proposals.append((1, score, rle, 0.))
        x, y, w, h, mask = geometry(rle)
        x0, y0, x1, y1 = max(0, x-3), max(0, y-3), min(W, x+w+3), min(H, y+h+3)
        m = np.zeros((y1-y0, x1-x0), np.uint8)
        m[y-y0:y-y0+h, x-x0:x-x0+w] = mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = (cv2.erode(m, kernel) > 0) | ((cv2.dilate(m, kernel) > 0) & (p[y0:y1, x0:x1, 0] >= .5))
        area = refined.sum()
        if area >= 60 and .7 < area / max(m.sum(), 1) < 1.4:
            change = float(np.logical_xor(m, refined).sum() / max(m.sum(), 1))
            proposals.append((2, score, encode_crop(refined, x0, y0), change))
    if not proposals:
        return []
    rles = [r[2] for r in proposals]
    ci = mu.iou(rles, cc, [0] * len(cc)) if cc else np.zeros((len(rles), 0))
    yi = mu.iou(rles, ys, [0] * len(ys)) if ys else np.zeros((len(rles), 0))
    values = np.zeros(len(rles))
    if gt_sets is not None:
        for gs in gt_sets:
            if gs:
                iou = np.asarray(mu.iou(rles, gs, [0] * len(gs)))
                values += np.where(iou > .5, iou, 0).sum(1)
    rows = []
    for i, (source, score, rle, change) in enumerate(proposals):
        x, y, w, h, m = geometry(rle)
        fg = m > 0
        v = p[y:y+h, x:x+w][fg]
        area = int(fg.sum())
        row = dict(stem=stem, source=source, conf=score, rle=rle, area=area,
                   bw=w, bh=h, fill=area / (w*h), aspect=max(w,h)/max(1,min(w,h)),
                   radius=float(np.hypot(x+w/2-1024,y+h/2-1024)/1000),
                   p0_mean=float(v[:,0].mean()), p0_q25=float(np.quantile(v[:,0],.25)), p0_max=float(v[:,0].max()),
                   p1_mean=float(v[:,1].mean()), p1_q25=float(np.quantile(v[:,1],.25)), p1_max=float(v[:,1].max()),
                   cc_iou=float(ci[i].max()) if len(cc) else 0.,
                   yolo_iou=float(yi[i].max()) if len(ys) else 0.,
                   yolo_conf=float(max((s for s,j in zip(scores,yi[i]) if j >= .5),default=0)),
                   boundary_change=change, image_count=len(cc), v=values[i], A=len(gt_sets) if gt_sets is not None else 0)
        rows.append(row)
    return rows


def build(roots, items, gt, split):
    cache = OUT / f"candidates_{split}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    mapping = {}
    for root in roots:
        for path in root.glob(f"{'oof' if split == 'hold' else 'test'}_fold*.zip"):
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    mapping.setdefault(name[:-4], []).append(str(path))
    jobs = [(s, sorted(v, reverse=True), mapping[s], gt[s] if split == "hold" else None) for s,v in sorted(items.items())]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i, result in enumerate(pool.map(one_image, jobs)):
            rows.extend(result)
            if i % 20 == 0:
                log(split, i, "of", len(jobs), "candidates", len(rows))
    df = pd.DataFrame(rows)
    df.to_pickle(cache)
    return df


def select(df, cut, sources):
    """Greedy RLE selection, strict disjointness, no full-frame mask copies."""
    result = {}
    for stem, group in df[(df.score >= cut) & df.source.isin(sources)].groupby("stem", sort=False):
        chosen = []
        for row in group.sort_values("score", ascending=False, kind="stable").itertuples():
            r = row.rle
            if chosen:
                occupied = mu.merge(chosen)
                overlap = float(mu.area(mu.merge([r, occupied], intersect=True)))
                if overlap > .1 * row.area:
                    continue
                if overlap:
                    mask = mu.decode(r).astype(bool) & ~mu.decode(occupied).astype(bool)
                    if mask.sum() < 60:
                        continue
                    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
            chosen.append(r)
        result[stem] = chosen
    return result


def evaluate(gt, pred, stems):
    total = sum((counts(gt[s], pred.get(s, [])) for s in stems), np.zeros(4))
    return dict(PQ=pq(total), iou_sum=total[0], TP=int(total[1]), FP=int(total[2]), FN=int(total[3]))


def write_csv(pred, path):
    rows = [(f"{s}_{i}", r["counts"].decode("ascii")) for s, rs in sorted(pred.items()) for i,r in enumerate(rs,1)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(path, index=False)
    log("wrote", path, "instances", len(rows))


def main():
    cv2.setNumThreads(1)
    roots, meta, gt, hold, test_items, data = discover()
    oof = build(roots, hold, gt, "hold")
    test = build(roots, test_items, gt, "test")
    stems = sorted(hold)
    # All candidates of an observation/day stay in the same meta-validation fold.
    groups = oof.stem.str[:8]
    oof["score"] = 0.
    oof["meta_fold"] = -1
    test["score"] = 0.
    params = dict(objective="regression", learning_rate=.035, num_leaves=15,
                  min_data_in_leaf=60, lambda_l2=10., feature_fraction=.85,
                  verbosity=-1, num_threads=3, seed=20261028)
    for f, (tr, va) in enumerate(GroupKFold(4).split(oof, groups=groups)):
        model = lgb.train(params, lgb.Dataset(oof.iloc[tr][FEATURES],
                          oof.iloc[tr].v / oof.iloc[tr].A, weight=oof.iloc[tr].A), 250)
        oof.loc[va,"score"] = model.predict(oof.iloc[va][FEATURES])
        oof.loc[va,"meta_fold"] = f
        test["score"] += model.predict(test[FEATURES]) / 4
        model.save_model(str(OUT / f"selector_fold{f}.txt"))
    oof.to_pickle(OUT / "scored_hold.pkl")
    test.to_pickle(OUT / "scored_test.pkl")
    settings = []
    for name, sources in [("semantic",[0]),("raw_hybrid",[0,1]),("refined_hybrid",[0,2]),("all",[0,1,2])]:
        for cut in (.18,.22,.26,.30,.35,.40):
            pred = select(oof, cut, sources)
            exact = evaluate(gt,pred,stems)
            fold_counts = []
            for f in range(4):
                ss = oof.loc[oof.meta_fold == f,"stem"].unique()
                fold_counts.append(sum((counts(gt[s],pred.get(s,[])) for s in ss), np.zeros(4)).tolist())
            settings.append(dict(name=name,sources=sources,cut=cut,**exact,fold_counts=fold_counts))
            log(name,cut,round(exact["PQ"],5))
    # Choose settings using the other three meta-folds, then evaluate held-out fold.
    nested_counts = np.zeros(4)
    for f in range(4):
        choice = max(settings,key=lambda r:pq(np.sum([c for j,c in enumerate(r["fold_counts"]) if j != f],axis=0)))
        nested_counts += choice["fold_counts"][f]
    best = max(settings,key=lambda r:r["PQ"])
    chosen = select(test,best["cut"],best["sources"])
    write_csv(chosen,OUT / "submission_model.csv")
    # Controlled high-precision supplement to the existing reference: preserve all
    # reference masks; add only new masks supported by both model families.
    ref_file = ROOT / "work/champ055.csv"
    fusion = None
    if ref_file.exists():
        ref = {}
        for row in pd.read_csv(ref_file).itertuples():
            ref.setdefault(row.filament_id.rsplit("_",1)[0],[]).append({"size":[H,W],"counts":row.segmentation_rle.encode()})
        supported = test[(test.cc_iou >= .5) & (test.yolo_iou >= .5) & (test.yolo_conf >= .15)]
        additions = select(supported,max(.40,best["cut"]),best["sources"])
        n_add = 0
        for stem, rs in additions.items():
            base = ref.setdefault(stem,[])
            for r in rs:
                overlap = float(mu.area(mu.merge([r,mu.merge(base)],intersect=True))) if base else 0.
                if overlap > .05 * float(mu.area(r)):
                    continue
                if overlap:
                    m = mu.decode(r).astype(bool) & ~mu.decode(mu.merge(base)).astype(bool)
                    if m.sum() < 100:
                        continue
                    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
                base.append(r)
                n_add += 1
        write_csv(ref,OUT / "submission_reference_consensus.csv")
        fusion = dict(added=n_add,provenance="lamhuy8904 embedded prediction CSV + independently trained cross-model consensus; reference masks are not reproducible from available weights")
    # Fast RLE validation: areas, bounds, and exact pairwise intersections.
    audits = {}
    for path in OUT.glob("submission*.csv"):
        frame = pd.read_csv(path)
        assert frame.columns.tolist() == ["filament_id","segmentation_rle"]
        assert frame.filament_id.is_unique and frame.notna().all().all()
        overlap = 0
        for stem,g in frame.groupby(frame.filament_id.str.rsplit("_",n=1).str[0]):
            assert stem in test_items
            rles = [{"size":[H,W],"counts":c.encode()} for c in g.segmentation_rle]
            areas = mu.area(rles)
            assert (areas > 0).all()
            overlap += int(areas.sum()) - int(mu.area(mu.merge(rles)))
        assert overlap == 0, (path,overlap)
        audits[path.name] = dict(rows=len(frame),overlap_pixels=overlap,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report = dict(chosen=best,nested_meta_PQ=pq(nested_counts),holdout_images=len(stems),
                  network_holdout_fold=0,meta_validation="4 folds grouped by observation day",
                  selection_validation_limit="Network training is independent of these images. Meta-model cross-fitting and settings selection share meta-training data; this is not a fully nested retraining estimate.",
                  settings=settings,fusion=fusion,audits=audits,
                  sources=["ranjeet258/sf-seg-train-v1","ranjeet258/sf-seg-train-v2","ranjeet258/sf-seg-train-v3","ranjeet258/sf-yolo-v1"])
    (OUT / "report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    log("chosen",best["name"],best["cut"],"holdout PQ",best["PQ"],"nested settings estimate",pq(nested_counts))
    log("audits",audits)


if __name__ == "__main__":
    main()
