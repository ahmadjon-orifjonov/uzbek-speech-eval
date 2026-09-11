# -*- coding: utf-8 -*-
"""
G2P QAMROV TESTI — 3-hafta "BAJARILDI" mezoni.

Shartlar (texnik/02_TEXNIK_REJA.md):
  - parser crash rate 0%
  - noaniq so'zlar `uncertain` deb belgilanadi
  - hech qanday noaniq qoida jim ishlatilmaydi

Chiqish: results/07_g2p_qamrov.json + data/g2p_lexicon.tsv
"""
import sys, io, json, collections
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from g2p import g2p, FONEMALAR, UNLILAR

IN = ROOT / "results" / "samples.json"
OUT = ROOT / "results" / "g2p_coverage.json"
LEX = ROOT / "data" / "g2p_lexicon.tsv"

yozuvlar = json.loads(IN.read_text(encoding="utf-8"))
sozlar = []
for r in yozuvlar:
    sozlar += r["etalon_norm"].split()
noyob = sorted(set(sozlar))
print(f"So'zlar: {len(sozlar)} ta, noyob: {len(noyob)} ta")

crash, natijalar = [], []
nomalum_sanoq = collections.Counter()
uncertain_turlari = collections.Counter()
fonema_sanoq = collections.Counter()

for w in noyob:
    try:
        r = g2p(w)
    except Exception as e:
        crash.append((w, f"{type(e).__name__}: {e}")); continue
    natijalar.append(r)
    for n in r["nomalum"]:
        nomalum_sanoq[n] += 1
    for u in r["uncertain"]:
        uncertain_turlari[u.split("(")[0].strip()] += 1
    for f in r["fonemalar"]:
        fonema_sanoq[f] += 1

n_uncertain = sum(1 for r in natijalar if r["uncertain"])
n_kop_variant = sum(1 for r in natijalar if len(r["kandidatlar"]) > 1)

# leksikon fayli: so'z <TAB> fonemalar <TAB> holat
satrlar = ["# O'zbek tili G2P leksikoni (qoida asosida yaratilgan)",
           "# ustunlar: so'z / fonemalar / holat (aniq | noaniq) / variantlar soni"]
for r in sorted(natijalar, key=lambda x: x["soz"]):
    holat = "noaniq" if r["uncertain"] else "aniq"
    satrlar.append(f"{r['soz']}\t{' '.join(r['fonemalar'])}\t{holat}\t{len(r['kandidatlar'])}")
LEX.write_text("\n".join(satrlar), encoding="utf-8")

xulosa = {
    "manba": "FLEURS uz_uz test, 30 namuna",
    "sozlar_jami": len(sozlar),
    "noyob_sozlar": len(noyob),
    "crash": len(crash),
    "crash_rate_%": round(100 * len(crash) / max(len(noyob), 1), 2),
    "noaniqlik_bor_%": round(100 * n_uncertain / max(len(natijalar), 1), 1),
    "kop_variantli_%": round(100 * n_kop_variant / max(len(natijalar), 1), 1),
    "nomalum_grafemalar": dict(nomalum_sanoq),
    "uncertain_turlari": dict(uncertain_turlari),
    "fonema_chastotasi": dict(fonema_sanoq.most_common()),
    "ishlatilmagan_fonemalar": [f for f in FONEMALAR if f not in fonema_sanoq],
    "crash_royxati": crash[:20],
}
OUT.write_text(json.dumps(xulosa, ensure_ascii=False, indent=2), encoding="utf-8")

print()
print(f"CRASH RATE          : {xulosa['crash_rate_%']}%   ({len(crash)} ta)")
print(f"Noaniqlik bor so'zlar: {xulosa['noaniqlik_bor_%']}%")
print(f"Ko'p variantli       : {xulosa['kop_variantli_%']}%")
print(f"Noma'lum grafemalar  : {dict(nomalum_sanoq) or 'YO`Q'}")
print()
print("NOANIQLIK TURLARI:")
for k, v in uncertain_turlari.most_common():
    print(f"  {v:4d}  {k}")
print()
print("FONEMA CHASTOTASI (top 12):")
for f, c in fonema_sanoq.most_common(12):
    print(f"  {f:3s} {c:5d}")
print(f"\nIshlatilmagan fonemalar: {xulosa['ishlatilmagan_fonemalar'] or 'yo`q'}")
print(f"\nLeksikon: {LEX}  ({len(natijalar)} so'z)")
print(f"Hisobot : {OUT}")
