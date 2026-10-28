"""Build a lower-dose poison file from the main (dose 8) poison file.

Why not generate: the sampler stores the 8 selected texts IN SELECTION ORDER
(one relevance-best anchor per poison-text style first: faq, update, bio, def,
authority; then the MMR picks). "Dose k" is therefore the first k of them — the
same attack with fewer chunks written. Generating a fresh dose-2 file instead
changes TWO things at once (how many chunks AND which texts), which is exactly
the confound a dose ablation is supposed to avoid.

So: dose 2 / 4 / 6 take the first 2 / 4 / 6 texts per question from the main
table's poison file. No model calls, no GPU, a few seconds.

Usage
  cd experiments
  python derive_dose.py --from ../results/dc_ours_noleak_xlam-2-8b.json \
      --k 2 --out ../results/dc_ours_vol2_xlam-2-8b.json
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True,
                    help="dose-8 poison file to take the prefix from")
    ap.add_argument("--k", type=int, required=True, help="texts per question")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blob = json.load(open(args.src))
    variants = blob.get("variants") or {}
    if len(variants) != 1:
        raise SystemExit(f"[dose] expected exactly one variant in {args.src}, "
                         f"got {list(variants)}")
    (vname, vbody), = variants.items()
    writes = vbody.get("poison_writes") or []
    if not writes:
        raise SystemExit(f"[dose] no poison_writes in {args.src}")

    by_q: dict[str, list] = {}
    for w in writes:            # file order == sampler pick order
        by_q.setdefault(w.get("qid") or "", []).append(w)

    out_writes = []
    for qid, ws in by_q.items():
        if len(ws) < args.k:
            raise SystemExit(f"[dose] {qid}: only {len(ws)} texts, cannot take {args.k}")
        out_writes += ws[:args.k]

    meta = dict(blob.get("meta") or {})
    meta.update({
        "per_target_chunks": args.k,
        "n_writes": len(out_writes),
        "n_targets": len(by_q),
        "derived_from": os.path.basename(args.src),
        "derivation": f"first {args.k} of the {len(writes)//max(1,len(by_q))} "
                      f"selected texts per question (same order as the sampler)",
        "generation_only": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    out = {
        "meta": meta,
        "targets": blob.get("targets") or {},
        "variants": {vname: {"poison_writes": out_writes,
                             "derived_from_variant": vname}},
    }
    per_q = {len([w for w in out_writes if w["qid"] == q]) for q in by_q}
    if per_q != {args.k}:
        raise SystemExit(f"[dose] per-question count {sorted(per_q)} != k={args.k}")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"[dose] {os.path.basename(args.out)}: {len(out_writes)} texts, "
          f"{len(by_q)} questions, per-question {min(per_q)}-{max(per_q)} "
          f"(derived from {os.path.basename(args.src)}, k={args.k})")


if __name__ == "__main__":
    main()
