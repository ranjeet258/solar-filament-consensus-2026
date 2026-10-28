"""Solar filament segmentation - grouped K-fold training of a 2-channel soft-label U-Net.

Targets (both in [0, 1], built from the per-annotator polygons):
  ch0 "shape":    (#annotators covering the pixel) / (#annotators who drew anything in this filament cluster)
  ch1 "presence": (#annotators covering the pixel) / (#annotators of the image)
A filament cluster is an 8-connected component of the union of all annotators' masks.

Outputs (in /kaggle/working): oof_fold{k}.zip, test_fold{k}.zip (PNG probability maps, uint8, B=ch0 G=ch1),
model_fold{k}.pt, meta.json. Folds are trained in parallel, one process per GPU.
"""
import glob
import io
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import zipfile

import numpy as np

CFG = dict(folds=4, crop=768, bs=8, epochs=36, crops_per_img=3, lr=3e-4, wd=1e-4, enc="resnext50_32x4d",
           seed=42, ema=0.998, loader_workers=2)
CACHE = "/kaggle/temp/cache"
OUT = "/kaggle/working"
H = W = 2048
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f}m]", *a, flush=True)


# ----------------------------------------------------------------------------- data cache (CPU, parent process)
def prepare():
    import cv2
    from pycocotools import mask as mu

    os.makedirs(CACHE, exist_ok=True)
    js = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)[0]
    root = os.path.dirname(os.path.dirname(js))
    train_dir = os.path.join(root, "train", "train_images")
    test_dir = os.path.join(root, "test", "test_images")
    coco = json.load(open(js))
    log("labels", js)

    sets_of, anns_of = {}, {}
    for im in coco["images"]:
        sets_of.setdefault(os.path.splitext(im["file_name"])[0], []).append(im["id"])
    for a in coco["annotations"]:
        anns_of.setdefault(a["image_id"], []).append(a)
    stems = sorted(sets_of)
    test_stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(test_dir, "*.jpeg")))
    n = len(stems)
    log(f"{n} train images, {len(coco['images'])} annotator-sets, {len(test_stems)} test images")

    imgs = np.lib.format.open_memmap(f"{CACHE}/imgs.npy", "w+", np.uint8, (n, H, W))
    tgt = np.lib.format.open_memmap(f"{CACHE}/tgt.npy", "w+", np.uint8, (n, 2, H, W))
    centers = []
    for i, s in enumerate(stems):
        im = cv2.imread(os.path.join(train_dir, s + ".jpeg"), cv2.IMREAD_GRAYSCALE)
        assert im is not None and im.shape == (H, W), (s, None if im is None else im.shape)
        imgs[i] = im
        per_annot, cs = [], []
        for aset in sets_of[s]:
            m = np.zeros((H, W), np.uint8)
            for a in anns_of.get(aset, []):
                m |= mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], H, W)))
                x, y, w, h = a["bbox"]
                cs.append([x + w / 2, y + h / 2])
            per_annot.append(m)
        count = np.sum(per_annot, axis=0).astype(np.float32)
        ncc, lab = cv2.connectedComponents((count > 0).astype(np.uint8), connectivity=8)
        k = np.zeros(ncc, np.float32)
        for m in per_annot:
            k[np.unique(lab[m > 0])] += 1
        k[0] = 1
        tgt[i, 0] = np.rint(255 * count / np.maximum(k[lab], 1)).astype(np.uint8)
        tgt[i, 1] = np.rint(255 * count / len(per_annot)).astype(np.uint8)
        centers.append(cs)
        if i % 100 == 0:
            log("cache train", i)
    imgs.flush(); tgt.flush(); del imgs, tgt

    test = np.lib.format.open_memmap(f"{CACHE}/test.npy", "w+", np.uint8, (len(test_stems), H, W))
    for i, s in enumerate(test_stems):
        im = cv2.imread(os.path.join(test_dir, s + ".jpeg"), cv2.IMREAD_GRAYSCALE)
        assert im is not None and im.shape == (H, W), s
        test[i] = im
    test.flush(); del test

    perm = np.random.RandomState(CFG["seed"]).permutation(n)
    fold = np.zeros(n, int)
    fold[perm] = np.arange(n) % CFG["folds"]
    meta = dict(stems=stems, test_stems=test_stems, fold=fold.tolist(), centers=centers,
                n_sets=[len(sets_of[s]) for s in stems], cfg=CFG)
    json.dump(meta, open(f"{CACHE}/meta.json", "w"))
    json.dump({k: v for k, v in meta.items() if k != "centers"}, open(f"{OUT}/meta.json", "w"))
    log("cache done")


# ----------------------------------------------------------------------------- worker (one per GPU)
def worker(gpu, folds):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import copy
    import cv2
    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp

    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = True
    meta = json.load(open(f"{CACHE}/meta.json"))
    fold_of = np.array(meta["fold"])
    C = CFG["crop"]

    class DS(torch.utils.data.Dataset):
        def __init__(self, idx):
            self.idx, self.imgs, self.tgt = idx, None, None

        def __len__(self):
            return len(self.idx) * CFG["crops_per_img"]

        def __getitem__(self, i):
            if self.imgs is None:
                self.imgs = np.load(f"{CACHE}/imgs.npy", mmap_mode="r")
                self.tgt = np.load(f"{CACHE}/tgt.npy", mmap_mode="r")
            j = int(self.idx[i % len(self.idx)])
            r = np.random
            win = int(round(C / math.exp(r.uniform(math.log(0.7), math.log(1.4)))))
            cs = meta["centers"][j]
            if cs and r.rand() < 0.6:
                cx, cy = cs[r.randint(len(cs))]
                cx += r.uniform(-win / 3, win / 3); cy += r.uniform(-win / 3, win / 3)
            else:
                cx, cy = r.uniform(0, W), r.uniform(0, H)
            x0 = int(np.clip(cx - win / 2, 0, W - win)); y0 = int(np.clip(cy - win / 2, 0, H - win))
            im = np.array(self.imgs[j, y0:y0 + win, x0:x0 + win])
            tg = np.array(self.tgt[j, :, y0:y0 + win, x0:x0 + win])
            if win != C:
                interp = cv2.INTER_AREA if win > C else cv2.INTER_LINEAR
                im = cv2.resize(im, (C, C), interpolation=interp)
                tg = np.stack([cv2.resize(t, (C, C), interpolation=cv2.INTER_LINEAR) for t in tg])
            k = r.randint(4)
            im, tg = np.rot90(im, k), np.rot90(tg, k, axes=(1, 2))
            if r.rand() < 0.5:
                im, tg = im[:, ::-1], tg[:, :, ::-1]
            x = im.astype(np.float32) / 255.0
            x = np.clip((x - 0.5) * r.uniform(0.8, 1.25) + 0.5 + r.uniform(-0.08, 0.08), 0, 1) ** r.uniform(0.8, 1.25)
            return (torch.from_numpy(np.ascontiguousarray(x))[None],
                    torch.from_numpy(np.ascontiguousarray(tg)).float() / 255.0)

    def norm(x):
        return (x - 0.45) / 0.225

    @torch.no_grad()
    def predict(model, im):
        """uint8 (H, W) -> uint8 (2, H, W) probabilities, 8-view dihedral TTA."""
        x0 = norm(torch.from_numpy(np.ascontiguousarray(im)).cuda().float()[None, None] / 255.0)
        acc = 0
        for tr in (False, True):
            x = x0.transpose(2, 3) if tr else x0
            for fl in ([], [2], [3], [2, 3]):
                xx = torch.flip(x, fl) if fl else x
                with torch.autocast("cuda", dtype=torch.float16):
                    p = torch.sigmoid(model(xx.contiguous(memory_format=torch.channels_last)).float())
                p = torch.flip(p, fl) if fl else p
                acc = acc + (p.transpose(2, 3) if tr else p)
        return (acc[0] / 8 * 255).round().byte().cpu().numpy()

    def dump(zpath, names, arrays_iter):
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for name, p in zip(names, arrays_iter):
                ok, buf = cv2.imencode(".png", np.dstack([p[0], p[1], np.zeros_like(p[0])]))
                assert ok
                z.writestr(name + ".png", buf.tobytes())

    def run_fold(f, smoke=False):
        tag = f"smoke{gpu}" if smoke else f"fold{f}"
        tr = np.flatnonzero(fold_of != f); va = np.flatnonzero(fold_of == f)
        torch.manual_seed(CFG["seed"] + f)
        model = smp.Unet(CFG["enc"], encoder_weights="imagenet", in_channels=1, classes=2)
        model = model.cuda().to(memory_format=torch.channels_last)
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        dl = torch.utils.data.DataLoader(
            DS(tr), batch_size=CFG["bs"], shuffle=True, drop_last=True, num_workers=CFG["loader_workers"],
            pin_memory=True, persistent_workers=True,
            worker_init_fn=lambda w: np.random.seed((torch.initial_seed() + w) % 2 ** 31))
        total = 3 if smoke else CFG["epochs"] * len(dl)
        warm = min(300, total // 10 + 1)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        scaler = torch.amp.GradScaler("cuda")
        step, run, t0 = 0, 0.0, time.time()
        model.train()
        while step < total:
            for x, y in dl:
                x = norm(x.cuda(non_blocking=True)).contiguous(memory_format=torch.channels_last)
                y = y.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logit = model(x).float()
                bce = F.binary_cross_entropy_with_logits(logit, y)
                p = torch.sigmoid(logit)
                dice = 1 - ((2 * (p * y).sum((0, 2, 3)) + 1) / (p.sum((0, 2, 3)) + y.sum((0, 2, 3)) + 1)).mean()
                loss = bce + 0.5 * dice
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt); scaler.update(); sched.step()
                d = min(CFG["ema"], (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - d)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
                step += 1
                run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
                if step % 200 == 0:
                    log(f"{tag} step {step}/{total} loss {run:.4f} bce {bce.item():.4f} "
                        f"{(time.time() - t0) / step:.2f}s/it")
                if step >= total:
                    break
        del dl

        imgs = np.load(f"{CACHE}/imgs.npy", mmap_mode="r")
        tgt = np.load(f"{CACHE}/tgt.npy", mmap_mode="r")
        test = np.load(f"{CACHE}/test.npy", mmap_mode="r")
        va_use = va[:2] if smoke else va
        te_use = range(2 if smoke else len(test))
        inter = np.zeros(2); den = np.zeros(2)

        def oof_iter():
            for j in va_use:
                p = predict(ema, imgs[j])
                t = np.asarray(tgt[j]) >= 128
                q = p >= 128
                inter[:] += (q & t).sum((1, 2)); den[:] += q.sum((1, 2)) + t.sum((1, 2))
                yield p

        dump(f"{OUT}/oof_{tag}.zip", [meta["stems"][j] for j in va_use], oof_iter())
        log(f"{tag} OOF pixel dice @0.5: shape {2 * inter[0] / max(den[0], 1):.4f} "
            f"presence {2 * inter[1] / max(den[1], 1):.4f}")
        dump(f"{OUT}/test_{tag}.zip", [meta["test_stems"][j] for j in te_use], (predict(ema, test[j]) for j in te_use))
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in ema.state_dict().items()},
                   f"{OUT}/model_{tag}.pt")
        if smoke:
            for fn in (f"oof_{tag}.zip", f"test_{tag}.zip", f"model_{tag}.pt"):
                os.remove(f"{OUT}/{fn}")
        log(f"{tag} done")
        del model, ema, opt
        torch.cuda.empty_cache()

    log(f"worker gpu {gpu} folds {folds}: {torch.cuda.get_device_name(0)}")
    run_fold(folds[0], smoke=True)      # end-to-end check of train / predict / save before spending GPU time
    for f in folds:
        run_fold(f)


def main():
    # The parent imports neither torch nor cv2, so the forked children start from a clean process.
    import importlib.util
    for mod, pkg in (("segmentation_models_pytorch", "segmentation-models-pytorch"), ("pycocotools", "pycocotools")):
        if importlib.util.find_spec(mod) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    n_gpu = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
    assert n_gpu >= 1, "no GPU"
    log("gpus", n_gpu, CFG)
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=prepare)]
    procs[0].start(); procs[0].join()
    assert procs[0].exitcode == 0, "prepare failed"
    procs = [ctx.Process(target=worker, args=(g, list(range(g, CFG["folds"], n_gpu)))) for g in range(n_gpu)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    log("all done", sorted(os.listdir(OUT)))


main()
