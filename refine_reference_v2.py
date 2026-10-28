"""Controlled boundary-band experiment; reference CSV provenance stays explicit."""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/hybrid_v2"
spec = importlib.util.spec_from_file_location("hybrid_v2",ROOT / "sfseg/exp/hybrid_v2.py")
hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid)


def main():
    cv2.setNumThreads(1)
    hold = pd.read_pickle(OUT / "scored_hold.pkl")
    pairs = hold[hold.source == 1].merge(hold[hold.source == 2],on=["stem","conf"],suffixes=("_raw","_ref"))
    evidence = pairs[(pairs.cc_iou_raw >= .75) & (pairs.conf >= .3)]
    gain = float((evidence.v_ref-evidence.v_raw).sum())
    assert len(evidence) >= 100 and gain > 0, "No held-out support for boundary refinement"
    print("Held-out strong-agreement proposal pairs:",len(evidence),"matched-IoU sum gain:",gain,flush=True)
    candidates = pd.read_pickle(OUT / "scored_test.pkl")
    source = OUT / "submission_reference_consensus.csv"
    frame = pd.read_csv(source)
    output, changed, altered_pixels = [], 0, 0
    for stem, group in frame.groupby(frame.filament_id.str.rsplit("_",n=1).str[0],sort=True):
        refs = [{"size":[2048,2048],"counts":c.encode()} for c in group.segmentation_rle]
        semantic = candidates[(candidates.stem == stem) & (candidates.source == 0) & (candidates.yolo_conf >= .3)]
        cc = semantic.rle.tolist()
        iou = np.asarray(mu.iou(refs,cc,[0]*len(cc))) if cc else np.zeros((len(refs),0))
        final = []
        for k, r in enumerate(refs):
            replacement = r
            if cc and iou[k].max() >= .75:
                match = cc[int(iou[k].argmax())]
                x,y,w,h = map(int,mu.toBbox(r))
                x0,y0,x1,y1 = max(0,x-3),max(0,y-3),min(2048,x+w+3),min(2048,y+h+3)
                original = mu.decode(r)[y0:y1,x0:x1]
                foreground = mu.decode(match)[y0:y1,x0:x1] > 0
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                proposal = (cv2.erode(original,kernel)>0) | ((cv2.dilate(original,kernel)>0)&foreground)
                # All other original instances retain priority over new boundary pixels.
                others = refs[:k]+refs[k+1:]+final
                if others:
                    occupied = mu.decode(mu.merge(others))[y0:y1,x0:x1] > 0
                    proposal &= ~occupied
                n_changed = int(np.logical_xor(original,proposal).sum())
                if 0 < n_changed <= .1 * int(original.sum()) and proposal.sum() >= 100:
                    replacement = hybrid.encode_crop(proposal,x0,y0)
                    changed += 1
                    altered_pixels += n_changed
            final.append(replacement)
        assert int(mu.area(final).sum()) == int(mu.area(mu.merge(final))), "Overlap after refinement"
        for k,r in enumerate(final,1):
            output.append((f"{stem}_{k}",r["counts"].decode("ascii")))
    path = OUT / "submission_reference_refined.csv"
    pd.DataFrame(output,columns=["filament_id","segmentation_rle"]).to_csv(path,index=False)
    report = dict(source=str(source.relative_to(ROOT)),output=str(path.relative_to(ROOT)),
                  changed_instances=changed,altered_pixels=altered_pixels,
                  max_relative_change=.1,minimum_reference_semantic_iou=.75,minimum_yolo_confidence=.3,
                  held_out_proposal_pairs=len(evidence),held_out_matched_iou_sum_gain=gain,
                  validation_limitation="Held-out evidence concerns YOLO proposals; it is not direct validation of the stronger public reference masks.",
                  reference_provenance="lamhuy8904 embedded prediction CSV; generating reference weights unavailable")
    (OUT / "refinement_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
