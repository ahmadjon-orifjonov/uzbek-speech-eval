# -*- coding: utf-8 -*-
"""
JONLI DEMO YADROSI — audio + ma'lum matn -> so'z bahosi + xato signali.

review note retsenziyasidan keyingi versiya. Asosiy tuzatish:
**HARD ABSTENTION GATE** — audio yoki alignment ishonchsiz bo'lsa, tizim
natija bermaydi. Sabab: known-text forced alignment butunlay boshqa
audioni ham "ishlov berilgan natija" ga aylantiradi. Pitchda mazmunsiz
"q → k" ko'rsatilishi crashdan xavfliroq.

Uch bosqich:

  0-BOSQICH — GATE (review note)
     Bo'sh/qisqa audio, klipping, past ovoz, noma'lum belgi, so'z
     indekslari nomuvofiqligi -> abstain=True, natija YO'Q.

  1-BOSQICH — known-text alignment (T-06 usuli)
     So'z darajasida path_norm va mean_logpost. Kelishmasa "noaniq".

  2-BOSQICH — XATO SIGNALI (forced choice)
     Shubhali so'z uchun almashtirilgan variant sinaladi:
     "boshqa" -> "boshka". Variant yuqoriroq ball olsa — q → k signali.

     Signal chiqishi uchun UCH shart (review note):
       1. asosiy baho "shubhali" yoki "kuchli_ogish"  ("noaniq" EMAS)
       2. eng yaxshi variant farqi ostonadan yuqori
       3. eng yaxshi va ikkinchi variant orasida ajratish bor
          (winner's curse himoyasi) VA boshqa so'zlar balli buzilmagan

     ⚠ o' / g' uchun ISHLATILMAYDI — T-06 da AUC 0.433 (mean_logpost).

DA'VO CHEGARASI (texnik/03_DAVO_DALIL_CHEGARA.md, Review note:
ekranda "q fonemasi noto'g'ri talaffuz qilindi" YOZILMAYDI. Yoziladigan
gap: "bu so'zda q o'rniga k eshitilgan bo'lishi mumkin" — ehtimoliy
signal, avtomatik tashxis emas.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from aligner import Aligner, normalize

# --- 1-bosqich chegaralari ---
# VAQTINCHALIK — 04_alignment.json taqsimotidan (30 FLEURS yozuvi).
# KALIBRLANMAGAN. Ekspert oldida "validatsiyalangan chegara" DEYILMAYDI
# (review note).
CHEGARA = {
    "path_norm": {"shubhali": -0.15, "aniq": -0.45},
    "mean_logpost": {"shubhali": -0.35, "aniq": -1.00},
}

# --- 2-bosqich ostonalari ---
# OSTONA: eng yaxshi variant asl balldan shuncha yuqori bo'lishi kerak.
# AJRATISH: eng yaxshi va ikkinchi variant orasidagi minimal farq —
#   winner's curse himoyasi (review note).
# BUZILISH: variant alignmentida tegilmagan so'zlar shundan ko'p
#   o'zgarsa, qayta tekislash butun gapni siljitgan degani -> signal yo'q.
OSTONA = 0.05
AJRATISH = 0.02
BUZILISH = 0.10

# --- 0-bosqich (gate) chegaralari ---
MIN_DAVOMIYLIK_S = 1.2
MIN_RMS = 0.008
MAX_PEAK = 0.999
MAX_DAVOMIYLIK_S = 60.0
# Gapning shuncha qismi kuchli og'ishda bo'lsa — bu alohida tovush xatosi
# emas, boshqa matn yoki shovqin. Alohida so'zga ishora qilinmaydi.
#
# 0.75 qiymati EMPIRIK, kalibrlanmagan. 0.6 sinab ko'rilgan edi va u
# qiyin (tez, uzun) yozuvlarni ham rad etdi: FLEURS_03 da ASL matnning
# o'zida 19 so'zdan 11 tasi kuchli og'ishda chiqadi. Butunlay boshqa
# matn holatida esa ulush 100% — 0.75 uni baribir ushlaydi.
MAX_KUCHLI_ULUSH = 0.75
# Va: uzunroq gapda hech bir so'z mos chiqmasa, bu ham mos kelmaslik belgisi.
MIN_MOS_SOZ = 1
MIN_SOZ_MOS_TEKSHIRUV = 4

ALMASHTIRISH = [
    ("q", "k", "q → k"),
    ("k", "q", "k → q"),
    ("x", "h", "x → h"),
    ("h", "x", "h → x"),
]

MASHQ = {
    "q → k": "Til orqa qismi mashqi: “qor — kor”, “qish — kish” juftliklari",
    "k → q": "“kitob — qitob”, “kul — qul” juftliklarini ajratib o‘qish",
    "x → h": "“xola — hola”, “xat — hat” juftliklari; bo‘g‘iz mashqi",
    "h → x": "“halol — xalol”, “hovli — xovli” juftliklarini taqqoslash",
}

BAHO_MATN = {
    "mos": "Kutilgan talaffuzga mos",
    "shubhali": "Yengil og‘ish — tekshirish tavsiya etiladi",
    "kuchli_ogish": "Etalondan sezilarli akustik og‘ish",
    "noaniq": "Noaniq — o‘lchovlar kelishmadi",
}


def baho(w):
    """Ikki o'lchov kelishmasa -> 'noaniq' (review note)."""
    pn, ml = w["path_norm_xom"], w["mean_logpost_xom"]

    def sinf(v, t):
        if v <= t["aniq"]:
            return 2
        if v <= t["shubhali"]:
            return 1
        return 0

    a, b = sinf(pn, CHEGARA["path_norm"]), sinf(ml, CHEGARA["mean_logpost"])
    if a == b:
        return ["mos", "shubhali", "kuchli_ogish"][a]
    if min(a, b) == 0 and max(a, b) == 2:
        return "noaniq"
    return ["mos", "shubhali", "kuchli_ogish"][max(a, b)]


def _variantlar(soz):
    """Har bir almashtiriladigan tovushning HAR BIR pozitsiyasi alohida.

    Review note: 2.4: pozitsiya saqlanadi va dublikatlar olib tashlanadi —
    "qishloq" da ikkita q bor, ular ikki xil gipoteza.
    """
    chiq, korilgan = [], set()
    for asl, yangi, nom in ALMASHTIRISH:
        for i, ch in enumerate(soz):
            if ch != asl:
                continue
            var = soz[:i] + yangi + soz[i + 1:]
            if var == soz or var in korilgan:
                continue
            korilgan.add(var)
            chiq.append({"soz": var, "nom": nom, "pozitsiya": i})
    return chiq


def audio_sifati(wav):
    """0-bosqich: audio o'lchovlari."""
    if wav is None or wav.size == 0:
        return {"bosh": True, "dav": 0.0, "peak": 0.0, "rms": 0.0, "jimlik": 1.0}
    dav = len(wav) / 16000
    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(wav ** 2)))
    # jimlik ulushi: 20 ms oynalarda energiya juda past bo'lgan qism
    n = max(int(0.02 * 16000), 1)
    oyna = wav[:len(wav) // n * n].reshape(-1, n)
    jim = float(np.mean(np.sqrt(np.mean(oyna ** 2, axis=1)) < 0.004)) if oyna.size else 1.0
    return {"bosh": False, "dav": dav, "peak": peak, "rms": rms, "jimlik": jim}


class Demo:
    def __init__(self):
        self.al = Aligner()

    # ---------- 0-BOSQICH ----------
    def _gate(self, wav, sifat, r, sozlar_norm):
        """Natija berish mumkinmi? (sabab, texnik_izoh) yoki None."""
        if sifat["bosh"]:
            return ("Yozuv bo‘sh — mikrofon ovoz olmadi", "wav.size == 0")
        if sifat["dav"] < MIN_DAVOMIYLIK_S:
            return ("Yozuv juda qisqa — matnni to‘liq o‘qing",
                    "davomiylik {:.2f} s < {}".format(sifat["dav"], MIN_DAVOMIYLIK_S))
        if sifat["dav"] > MAX_DAVOMIYLIK_S:
            return ("Yozuv juda uzun — qisqa matn o‘qing",
                    "davomiylik {:.1f} s > {}".format(sifat["dav"], MAX_DAVOMIYLIK_S))
        if sifat["peak"] >= MAX_PEAK:
            return ("Mikrofon juda baland (klipping) — ovozni pasaytiring",
                    "peak {:.3f}".format(sifat["peak"]))
        if sifat["rms"] < MIN_RMS:
            return ("Ovoz juda past — mikrofonga yaqinroq o‘qing",
                    "rms {:.4f} < {}".format(sifat["rms"], MIN_RMS))
        if sifat["jimlik"] > 0.85:
            return ("Yozuvda nutq deyarli yo‘q", "jimlik {:.0%}".format(sifat["jimlik"]))
        if r.get("nomalum_belgilar"):
            return ("Matnda model tanimaydigan belgi bor",
                    "noma’lum: " + " ".join(r["nomalum_belgilar"]))
        # so'z indekslari mosligi (review note) — eng jiddiy xato manbai
        sozlar = r.get("sozlar", [])
        if len(sozlar) != len(sozlar_norm):
            return ("Matn va tahlil mos kelmadi",
                    "alignment {} so‘z, etalon {} so‘z".format(len(sozlar), len(sozlar_norm)))
        for i, w in enumerate(sozlar):
            if w.get("soz_idx") != i or w["soz"] != sozlar_norm[i]:
                return ("Matn va tahlil mos kelmadi",
                        "indeks {}: '{}' != '{}'".format(i, w["soz"], sozlar_norm[i]))
        return None

    # ---------- 2-BOSQICH ----------
    def _xato_signali(self, wav, sozlar, sozlar_norm, i):
        """Forced choice. Uchta shart bajarilsagina signal qaytadi."""
        w = sozlar[i]
        varlar = _variantlar(w["soz"])
        if not varlar:
            return None
        asl = w["path_norm_xom"]
        ballar = []
        for v in varlar:
            yangi = list(sozlar_norm)
            yangi[i] = v["soz"]
            r2 = self.al.align(wav, " ".join(yangi))
            if "xato" in r2:
                continue
            s2 = r2["sozlar"]
            if len(s2) != len(sozlar) or s2[i]["soz"] != v["soz"]:
                continue      # qayta tekislash tuzilmani buzdi
            # tegilmagan so'zlar qanchalik siljidi (review note)
            siljish = max(
                (abs(s2[j]["path_norm_xom"] - sozlar[j]["path_norm_xom"])
                 for j in range(len(s2)) if j != i),
                default=0.0)
            ballar.append({"nom": v["nom"], "pozitsiya": v["pozitsiya"],
                           "ball": s2[i]["path_norm_xom"], "siljish": siljish})
        if not ballar:
            return None
        ballar.sort(key=lambda b: -b["ball"])
        eng, ikki = ballar[0], (ballar[1] if len(ballar) > 1 else None)
        farq = eng["ball"] - asl
        ajratish = (eng["ball"] - ikki["ball"]) if ikki else None

        natija = {
            "nom": eng["nom"],
            "pozitsiya": eng["pozitsiya"],
            "asl": round(asl, 4),
            "variant": round(eng["ball"], 4),
            "farq": round(farq, 4),
            "ostona": OSTONA,
            "ajratish": round(ajratish, 4) if ajratish is not None else None,
            "siljish": round(eng["siljish"], 4),
            "variant_soni": len(ballar),
            "qabul": False,
            "rad_sabab": None,
        }
        if farq <= OSTONA:
            natija["rad_sabab"] = "farq ostonadan past"
        elif eng["siljish"] > BUZILISH:
            natija["rad_sabab"] = "qayta tekislash boshqa so‘zlarni siljitdi"
        elif ajratish is not None and ajratish < AJRATISH:
            natija["rad_sabab"] = "variantlar bir-biriga juda yaqin"
        else:
            natija["qabul"] = True
        return natija

    # ---------- ASOSIY ----------
    def tahlil(self, wav, matn):
        """wav: 16 kHz float32 mono. matn: xom etalon matn."""
        sifat = audio_sifati(wav)
        norm = normalize(matn)
        sozlar_norm = norm.split()
        if not sozlar_norm:
            return {"abstain": True, "sabab": "Matn bo‘sh", "texnik": "norm bo‘sh"}

        if sifat["bosh"] or sifat["dav"] < MIN_DAVOMIYLIK_S:
            g = self._gate(wav, sifat, {}, sozlar_norm)
            return {"abstain": True, "sabab": g[0], "texnik": g[1], "sifat": sifat}

        r = self.al.align(wav, norm)
        if "xato" in r:
            return {"abstain": True, "sabab": "Audio matnga mos kelmadi",
                    "texnik": r["xato"], "sifat": sifat}

        g = self._gate(wav, sifat, r, sozlar_norm)
        if g:
            return {"abstain": True, "sabab": g[0], "texnik": g[1], "sifat": sifat}

        sozlar = r["sozlar"]
        for w in sozlar:
            w["baho"] = baho(w)
            w["xato_turi"] = None
            w["mashq"] = None
            w["signal"] = None

        # GLOBAL MOSLIK (review note): forced alignment butunlay boshqa
        # matnga ham yo'l topadi. Agar gapning katta qismi kuchli og'ishda
        # bo'lsa, bu "bir-ikki tovush xato" emas — boshqa matn o'qilgan,
        # yoki shovqin. Bunday holda alohida so'zga ishora qilinmaydi.
        kuchli = sum(1 for w in sozlar if w["baho"] == "kuchli_ogish")
        mos = sum(1 for w in sozlar if w["baho"] == "mos")
        ulush = kuchli / max(len(sozlar), 1)
        nomos = None
        if len(sozlar) >= 3 and ulush > MAX_KUCHLI_ULUSH:
            nomos = "so‘zlarning {:.0%} i kuchli og‘ishda (chegara {:.0%})".format(
                ulush, MAX_KUCHLI_ULUSH)
        elif len(sozlar) >= MIN_SOZ_MOS_TEKSHIRUV and mos < MIN_MOS_SOZ:
            nomos = "hech bir so‘z kutilgan talaffuzga mos chiqmadi"
        if nomos:
            return {"abstain": True,
                    "sabab": "Audio matnga mos kelmadi — boshqa matn yoki shovqin",
                    "texnik": nomos, "sifat": sifat}

        # forced choice — "noaniq" ga QO'LLANILMAYDI (review note)
        for i, w in enumerate(sozlar):
            if w["baho"] not in ("shubhali", "kuchli_ogish"):
                continue
            s = self._xato_signali(wav, sozlar, sozlar_norm, i)
            if not s:
                continue
            w["signal"] = s
            if s["qabul"]:
                w["xato_turi"] = s["nom"]
                w["mashq"] = MASHQ.get(s["nom"])

        sanoq = {k: sum(1 for w in sozlar if w["baho"] == k) for k in BAHO_MATN}
        turlar = {}
        for w in sozlar:
            if w["xato_turi"]:
                turlar[w["xato_turi"]] = turlar.get(w["xato_turi"], 0) + 1
        tavsiya = [{"tur": n, "soni": turlar[n], "mashq": MASHQ.get(n)}
                   for n in sorted(turlar, key=lambda k: -turlar[k])]

        return {
            "abstain": False,
            "sozlar": sozlar,
            "sanoq": sanoq,
            "turlar": turlar,
            "tavsiya": tavsiya,
            "davomiylik_s": round(sifat["dav"], 2),
            "sifat": {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in sifat.items()},
        }
