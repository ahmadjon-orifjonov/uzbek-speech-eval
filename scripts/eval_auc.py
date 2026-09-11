# -*- coding: utf-8 -*-
"""
AUC + 95% BOOTSTRAP CONFIDENCE INTERVALS over the perturbation study.

Reads results/perturbation.json and answers the question a real system
faces: with only ONE score per word (no "correct variant" to compare
against), how well does each measure separate correct from perturbed?

Two things matter here:

  1. The resampling unit is the UTTERANCE, not the word. Perturbations
     from the same recording share a speaker, a recording condition and
     an acoustic model state — treating them as independent would make
     the interval far too narrow.
  2. AUC is reported per measure AND per confusion group, because the
     measures do not agree: `path_norm` is the stable one overall,
     `mean_logpost` is stronger on q/k and x/h specifically.

Output: results/auc.json
"""
import sys
import io
import json
import random
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "results" / "perturbation.json"
OUT = ROOT / "results" / "auc.json"

MEASURES = ["mean_logpost", "path_norm", "margin", "blank_ratio"]
BOOTSTRAP = 2000
SEED = 20260911

# Which perturbation types belong to which linguistic group.
GROUPS = {
    "q/k and x/h": {"q->k", "k->q", "x->h", "h->x"},
    "o'/g'": {"o‘->o", "g‘->g"},
}


def auc(pos, neg):
    """Probability that a random 'correct' scores above a random 'perturbed'.

    Mann-Whitney U with tie correction (ties count as half), computed by
    rank sum so it stays O(n log n).
    """
    if not pos or not neg:
        return None
    hammasi = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    # average ranks for ties
    while i < len(hammasi):
        j = i
        while j + 1 < len(hammasi) and hammasi[j + 1][0] == hammasi[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks.setdefault(k, r)
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, lab) in enumerate(hammasi) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def bootstrap_ci(rows, measure, tanla=None, n=BOOTSTRAP, seed=SEED):
    """95% CI, resampling UTTERANCES with replacement (cluster bootstrap)."""
    by_utt = {}
    for r in rows:
        if tanla and r["tur"] not in tanla:
            continue
        by_utt.setdefault(r["id"], []).append(r)
    utts = list(by_utt)
    if len(utts) < 3:
        return None, None, 0, 0
    rnd = random.Random(seed)
    qiymatlar = []
    for _ in range(n):
        namuna = [rnd.choice(utts) for _ in utts]
        pos, neg = [], []
        for u in namuna:
            for r in by_utt[u]:
                pos.append(r["togri"][measure])
                neg.append(r["buzuq"][measure])
        a = auc(pos, neg)
        if a is not None:
            qiymatlar.append(a)
    qiymatlar.sort()
    if not qiymatlar:
        return None, None, 0, 0
    lo = qiymatlar[int(0.025 * len(qiymatlar))]
    hi = qiymatlar[min(int(0.975 * len(qiymatlar)), len(qiymatlar) - 1)]
    jami = sum(len(v) for v in by_utt.values())
    return lo, hi, len(utts), jami


def nuqta(rows, measure, tanla=None):
    pos, neg = [], []
    for r in rows:
        if tanla and r["tur"] not in tanla:
            continue
        pos.append(r["togri"][measure])
        neg.append(r["buzuq"][measure])
    return auc(pos, neg)


def main():
    if not IN.exists():
        print("perturbation.json yo'q — avval: python scripts/eval_perturbation.py")
        return 1
    d = json.loads(IN.read_text(encoding="utf-8"))
    rows = d["xom"]
    turlar = sorted({r["tur"] for r in rows})
    print("Yozuvlar: {}  |  gaplar: {}  |  perturbatsiya turlari: {}".format(
        len(rows), len({r["id"] for r in rows}), ", ".join(turlar)))
    print("Bootstrap: {} takror, birlik = GAP (klaster bootstrap)\n".format(BOOTSTRAP))

    kesimlar = [("overall", None)] + [(nom, t) for nom, t in GROUPS.items()]
    chiqish = {"bootstrap": BOOTSTRAP, "seed": SEED,
               "utterances": len({r["id"] for r in rows}),
               "perturbations": len(rows), "auc": {}}

    print("{:<14} {:<12} {:>7}  {:>16}  {:>7} {:>6}".format(
        "measure", "group", "AUC", "95% CI", "utts", "n"))
    print("-" * 70)
    for m in MEASURES:
        chiqish["auc"][m] = {}
        for nom, tanla in kesimlar:
            a = nuqta(rows, m, tanla)
            lo, hi, n_utt, n = bootstrap_ci(rows, m, tanla)
            chiqish["auc"][m][nom] = {
                "auc": round(a, 4) if a is not None else None,
                "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
                "utterances": n_utt, "perturbations": n,
            }
            print("{:<14} {:<12} {:>7}  {:>16}  {:>7} {:>6}".format(
                m, nom,
                "{:.3f}".format(a) if a is not None else "-",
                "[{:.3f}, {:.3f}]".format(lo, hi) if lo is not None else "-",
                n_utt, n))
        print()

    # negative control: untouched words must not move at all
    nc = [r for r in rows if r.get("nc_togri") and r.get("nc_buzuq")]
    buzilmagan = sum(1 for r in nc
                     if all(abs(r["nc_togri"][m] - r["nc_buzuq"][m]) < 1e-9 for m in MEASURES))
    chiqish["negative_control"] = {"checked": len(nc), "unchanged": buzilmagan}
    print("Negative control: {}/{} untouched words unchanged across all measures"
          .format(buzilmagan, len(nc)))

    OUT.write_text(json.dumps(chiqish, ensure_ascii=False, indent=1),
                   encoding="utf-8", newline="\n")
    print("\n->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
