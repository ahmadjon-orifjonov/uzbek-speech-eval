# -*- coding: utf-8 -*-
"""
TRANSKRIPT PERTURBATSIYASIGA SEZGIRLIK TESTI (sanity check + negative control)

NOMI MUHIM (review note): bu "talaffuz xatosini aniqlash tajribasi" EMAS.
Audio o'zgarmaydi — faqat etalon matn ataylab buziladi. Shuning uchun u
faqat quyidagini tekshiradi:

    Known-text alignment ball i transkript buzilishiga sezgirmi?

Nimani ISBOTLAMAYDI: tizim o'quvchining haqiqiy talaffuz xatosini topishini.

HAL QILUVCHI SAVOL (review note): agar to'g'ri va buzilgan transkript ballari
bir-biriga yaqin chiqsa — model bu vazifa uchun yetarlicha DISKRIMINATIV EMAS.

NEGATIVE CONTROL: buzilgan so'zdan boshqa, TEGILMAGAN so'zning balli ham
o'lchanadi. U o'zgarmasligi kerak. O'zgarsa — signal lokal emas, ya'ni
butun alignment siljigan va natija ishonchsiz.

Chiqish: results/05_perturbatsiya.json
"""
import sys, io, json, time, random
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "results" / "samples.json"
OUT = ROOT / "results" / "perturbation.json"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, soundfile as sf
from aligner import Aligner

SEED = 20260910
random.seed(SEED); np.random.seed(SEED)

APO = "‘"   # kanonik apostrof (00_normalize.py bilan bir xil)

# O'zbek tilida real chalkashadigan juftliklar
PERTURBATSIYALAR = [
    ("q->k",   lambda w: w.replace("q", "k", 1) if "q" in w else None),
    ("k->q",   lambda w: w.replace("k", "q", 1) if "k" in w else None),
    ("x->h",   lambda w: w.replace("x", "h", 1) if "x" in w else None),
    ("h->x",   lambda w: w.replace("h", "x", 1) if "h" in w else None),
    (f"o{APO}->o", lambda w: w.replace("o" + APO, "o", 1) if "o" + APO in w else None),
    (f"g{APO}->g", lambda w: w.replace("g" + APO, "g", 1) if "g" + APO in w else None),
    ("belgi_tushirish", None),   # pastda alohida ishlanadi
]

OLCHOVLAR = ["mean_logpost", "path_norm", "margin", "blank_ratio"]


def belgi_tushir(w, rnd):
    """So'z ichidan bitta undoshni tushiradi (birinchi/oxirgi emas)."""
    if len(w) < 4:
        return None
    unlilar = set("aeiou")
    nomzod = [i for i in range(1, len(w) - 1) if w[i] not in unlilar and w[i] != APO]
    if not nomzod:
        return None
    i = rnd.choice(nomzod)
    return w[:i] + w[i + 1:]


def main():
    print("Model yuklanmoqda...")
    al = Aligner()
    print(f"  tayyor. blank={al.blank}, delim={al.delim}")

    yozuvlar = json.loads(IN.read_text(encoding="utf-8"))
    rnd = random.Random(SEED)
    yozishlar = []
    t0 = time.time()

    for rec in yozuvlar:
        wav, sr = sf.read(rec["fayl"], dtype="float32")
        etalon = rec["etalon_norm"]
        sozlar = etalon.split()
        if len(sozlar) < 4:
            continue

        baza = al.align(wav, etalon, key=rec["id"])
        if "xato" in baza:
            print(f"  {rec['id']}: {baza['xato']}"); continue
        baza_map = {i: w for i, w in enumerate(baza["sozlar"])}

        for nom, fn in PERTURBATSIYALAR:
            # buzish mumkin bo'lgan so'zlarni topamiz
            nomzodlar = []
            for i, w in enumerate(sozlar):
                yangi = belgi_tushir(w, rnd) if fn is None else fn(w)
                if yangi and yangi != w:
                    nomzodlar.append((i, w, yangi))
            if not nomzodlar:
                continue
            i, eski, yangi = rnd.choice(nomzodlar)

            buzuq = list(sozlar); buzuq[i] = yangi
            buz = al.align(wav, " ".join(buzuq), key=rec["id"])
            if "xato" in buz:
                continue
            buz_map = {k: w for k, w in enumerate(buz["sozlar"])}
            if i not in baza_map or i not in buz_map:
                continue

            # negative control: buzilgan so'zdan uzoqdagi tegilmagan so'z
            boshqalar = [k for k in baza_map if abs(k - i) >= 2 and k in buz_map]
            nc = rnd.choice(boshqalar) if boshqalar else None

            yozishlar.append({
                "id": rec["id"], "tur": nom, "soz_idx": i,
                "soz_togri": eski, "soz_buzuq": yangi,
                "togri": {m: baza_map[i][m] for m in OLCHOVLAR},
                "buzuq": {m: buz_map[i][m] for m in OLCHOVLAR},
                "nc_soz": baza_map[nc]["soz"] if nc is not None else None,
                "nc_togri": {m: baza_map[nc][m] for m in OLCHOVLAR} if nc is not None else None,
                "nc_buzuq": {m: buz_map[nc][m] for m in OLCHOVLAR} if nc is not None else None,
            })
        print(f"  {rec['id']} tugadi ({len(yozishlar)} yozuv)")

    # ---- TAHLIL ----
    def delta(rows, a, b, m):
        v = [(r[b][m] - r[a][m]) for r in rows if r[a] and r[b] and r[a][m] is not None and r[b][m] is not None]
        return np.array(v, dtype=float)

    turlar = sorted({r["tur"] for r in yozishlar})
    jadval = {}
    for t in turlar:
        rows = [r for r in yozishlar if r["tur"] == t]
        blok = {"n": len(rows)}
        for m in OLCHOVLAR:
            d = delta(rows, "togri", "buzuq", m)
            nc = delta([r for r in rows if r["nc_togri"]], "nc_togri", "nc_buzuq", m)
            if len(d) == 0:
                continue
            # "aniqlash darajasi": buzilgan variant pastroq ball olgan hollar ulushi
            # (blank_ratio uchun teskari: yuqori = yomonroq)
            past = float(np.mean(d < 0)) if m != "blank_ratio" else float(np.mean(d > 0))
            blok[m] = {
                "delta_ortacha": round(float(d.mean()), 4),
                "delta_median": round(float(np.median(d)), 4),
                "delta_std": round(float(d.std()), 4),
                "kutilgan_yonalishda_%": round(past * 100, 1),
                "NC_delta_ortacha": round(float(nc.mean()), 4) if len(nc) else None,
                "NC_delta_median": round(float(np.median(nc)), 4) if len(nc) else None,
            }
        jadval[t] = blok

    # umumiy
    umumiy = {}
    for m in OLCHOVLAR:
        d = delta(yozishlar, "togri", "buzuq", m)
        nc = delta([r for r in yozishlar if r["nc_togri"]], "nc_togri", "nc_buzuq", m)
        past = float(np.mean(d < 0)) if m != "blank_ratio" else float(np.mean(d > 0))
        umumiy[m] = {
            "n": int(len(d)),
            "delta_ortacha": round(float(d.mean()), 4),
            "delta_median": round(float(np.median(d)), 4),
            "kutilgan_yonalishda_%": round(past * 100, 1),
            "NC_delta_ortacha": round(float(nc.mean()), 4) if len(nc) else None,
        }

    xulosa = {
        "tajriba": "Transkript perturbatsiyasiga sezgirlik testi (sanity check)",
        "OGOHLANTIRISH": "Audio o'zgarmagan. Bu talaffuz xatosini aniqlash tajribasi EMAS.",
        "model": al.model_id, "seed": SEED,
        "namunalar": len({r['id'] for r in yozishlar}),
        "yozishlar": len(yozishlar),
        "vaqt_s": round(time.time() - t0, 1),
        "umumiy": umumiy,
        "tur_boyicha": jadval,
        "xom": yozishlar,
    }
    OUT.write_text(json.dumps(xulosa, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print(f"YOZISHLAR: {len(yozishlar)}   VAQT: {xulosa['vaqt_s']}s")
    print()
    print(f"{'TUR':18s} {'n':>3s} {'Δmean_logpost':>14s} {'to`g`ri yo`n%':>13s} {'NC Δ':>9s}")
    print("-" * 78)
    for t, b in jadval.items():
        if "mean_logpost" not in b: continue
        x = b["mean_logpost"]
        print(f"{t:18s} {b['n']:3d} {x['delta_ortacha']:14.4f} "
              f"{x['kutilgan_yonalishda_%']:12.1f}% {str(x['NC_delta_ortacha']):>9s}")
    print("-" * 78)
    u = umumiy["mean_logpost"]
    print(f"{'UMUMIY':18s} {u['n']:3d} {u['delta_ortacha']:14.4f} "
          f"{u['kutilgan_yonalishda_%']:12.1f}% {str(u['NC_delta_ortacha']):>9s}")
    print()
    for m in ["path_norm", "margin", "blank_ratio"]:
        u = umumiy[m]
        print(f"  {m:14s} Δ={u['delta_ortacha']:+8.4f}  "
              f"kutilgan yo'nalishda {u['kutilgan_yonalishda_%']:5.1f}%  NC Δ={u['NC_delta_ortacha']}")
    print()
    print(f"Saqlandi: {OUT}")


if __name__ == "__main__":
    main()
