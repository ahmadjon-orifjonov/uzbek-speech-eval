# -*- coding: utf-8 -*-
"""
O'ZBEK TILI UCHUN QOIDA ASOSIDAGI G2P (grafema -> fonema)

Arxitektura (review note):
  normalizator -> digraf parser -> affiks/istisno qoidalari ->
  asosiy moslashtirish -> kandidat variantlar + noaniqlik bayrog'i

QAT'IY QOIDA: noaniq holat JIM o'tkazilmaydi. Har bir noaniqlik
`uncertain` ro'yxatiga yoziladi va bir nechta kandidat qaytariladi.

Fonema inventari 30 ta birlik. Bu inventar fonologik modelga bog'liq —
yakuniy shakli tilshunos bilan tasdiqlanishi kerak (TEKSHIRILADI).
"""
from pathlib import Path
import importlib.util, re

_spec = importlib.util.spec_from_file_location("normalize", Path(__file__).parent / "normalize.py")
_nm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_nm)
normalize = _nm.normalize
APO = _nm.KANONIK_APOSTROF     # ‘

# ---------------------------------------------------------------- inventar
UNLILAR = {"a", "e", "i", "o", "u", "ɵ"}
FONEMALAR = [
    # unlilar (6)
    "a", "e", "i", "o", "u", "ɵ",
    # undoshlar (24)
    "b", "d", "f", "g", "ʁ", "h", "dʒ", "ʒ", "k", "l", "m", "n", "ŋ",
    "p", "q", "r", "s", "ʃ", "t", "tʃ", "v", "x", "j", "z", "ʔ",
]

# ---------------------------------------------------------- asosiy jadval
# uzunroq grafemalar oldin tekshiriladi
GRAFEMA_FONEMA = [
    ("o" + APO, "ɵ"),      # oʻ
    ("g" + APO, "ʁ"),      # gʻ
    ("sh", "ʃ"),
    ("ch", "tʃ"),
    ("ng", "ŋ"),           # NOANIQ — pastda tekshiriladi
    ("a", "a"), ("b", "b"), ("d", "d"), ("e", "e"), ("f", "f"),
    ("g", "g"), ("h", "h"), ("i", "i"), ("j", "dʒ"),   # NOANIQ
    ("k", "k"), ("l", "l"), ("m", "m"), ("n", "n"), ("o", "o"),
    ("p", "p"), ("q", "q"), ("r", "r"), ("s", "s"), ("t", "t"),
    ("u", "u"), ("v", "v"), ("x", "x"), ("y", "j"), ("z", "z"),
    (APO, "ʔ"),            # tutuq belgisi — NOANIQ
    ("c", "s"),            # mustaqil c o'zbekchada yo'q; xorijiy so'zlarda
    ("w", "v"),            # xorijiy
]

# ------------------------------------------------------------- istisnolar
# j = /ʒ/ bo'ladigan (asosan rus/fransuz orqali kirgan) so'zlar
J_ZHE = {
    "jurnal", "jurnalist", "jurnalistika", "janr", "jest", "jaket", "jelatin",
    "jalyuzi", "jokey", "jonglyor", "drenaj", "kollaj", "massaj", "garaj",
    "vitraj", "bagaj", "pilotaj", "montaj", "tiraj", "abajur", "jiraf",
}
# ng = /n/+/g/ bo'ladigan (morfema chegarasi) holatlar
NG_AJRALGAN = {
    "kelinga", "tanga", "manga", "sanga", "ringa", "ovunga",
}
# to'liq qo'lda tekshirilgan istisnolar: so'z -> fonema ro'yxati
QOLDA = {
    # affiksda unli tushishi -> yozuvda ham aks etadi, shuning uchun
    # bu yerda faqat yozuvdan chiqmaydigan holatlar turadi
}

# --------------------------------------------------------------- yordamchi
_UZUNLIK = sorted({len(g) for g, _ in GRAFEMA_FONEMA}, reverse=True)
_MAP = dict(GRAFEMA_FONEMA)


def _parse(soz):
    """So'zni grafemalarga bo'ladi (uzun grafemalar ustuvor).

    MUHIM ISTISNO: `ng` dan keyin apostrof kelsa, bu `n` + `g‘` demakdir
    (masalan qo‘ng‘iroq = q o‘ n g‘ i r o q). Aks holda parser
    `ng` + `‘` deb noto‘g‘ri bo‘ladi. Bu xato 2026-09-10 da topildi.
    """
    i, out = 0, []
    while i < len(soz):
        for L in _UZUNLIK:
            g = soz[i:i + L]
            if g not in _MAP:
                continue
            # ng + apostrof  ->  n va gʻ alohida
            if g == "ng" and soz[i + 2:i + 3] == APO:
                out.append("n"); i += 1; break
            out.append(g); i += L; break
        else:
            out.append(soz[i]); i += 1   # noma'lum belgi — o'zi qoladi
    return out


def g2p(soz_xom):
    """Bitta so'z uchun fonema ketma-ketligi(lari).

    Qaytadi: dict(
        soz, grafemalar, fonemalar (asosiy variant),
        kandidatlar (list[list[str]]), uncertain (list[str]), nomalum (list[str])
    )
    """
    soz = normalize(soz_xom)
    if not soz or " " in soz:
        return {"soz": soz, "grafemalar": [], "fonemalar": [], "kandidatlar": [],
                "uncertain": ["bo'sh yoki ko'p so'zli kirish"], "nomalum": []}

    if soz in QOLDA:
        return {"soz": soz, "grafemalar": _parse(soz), "fonemalar": QOLDA[soz],
                "kandidatlar": [QOLDA[soz]], "uncertain": [], "nomalum": []}

    grafemalar = _parse(soz)
    uncertain, nomalum = [], []
    kandidatlar = [[]]

    for idx, g in enumerate(grafemalar):
        if g not in _MAP:
            nomalum.append(g)
            uncertain.append(f"noma'lum grafema {g!r} ({idx}-o'rin)")
            continue

        variantlar = [_MAP[g]]

        # --- j: /dʒ/ yoki /ʒ/
        if g == "j":
            if soz in J_ZHE:
                variantlar = ["ʒ"]
            else:
                variantlar = ["dʒ", "ʒ"]
                uncertain.append(f"j ({idx}) — /dʒ/ yoki /ʒ/; istisnolar lug'ati to'ldirilishi kerak")

        # --- ng: /ŋ/ yoki /n/+/g/ (morfema chegarasi)
        elif g == "ng":
            if soz in NG_AJRALGAN:
                variantlar = ["n g"]
            else:
                # so'z oxirida deyarli doim /ŋ/
                oxirida = idx == len(grafemalar) - 1
                if oxirida:
                    variantlar = ["ŋ"]
                else:
                    variantlar = ["ŋ", "n g"]
                    uncertain.append(f"ng ({idx}) — /ŋ/ yoki /n/+/g/ (morfema chegarasi)")

        # --- tutuq belgisi: unlidan keyin cho'zish, undoshdan keyin /ʔ/
        elif g == APO:
            oldingi = grafemalar[idx - 1] if idx > 0 else ""
            if oldingi in UNLILAR or _MAP.get(oldingi) in UNLILAR:
                variantlar = ["ʔ", "ː"]
                uncertain.append(f"tutuq belgisi ({idx}) — /ʔ/ yoki oldingi unlining cho'zilishi")
            else:
                variantlar = ["ʔ"]

        # --- c / w: o'zbekchada mustaqil ishlatilmaydi
        elif g in ("c", "w"):
            uncertain.append(f"{g!r} ({idx}) — o'zbek lotin yozuvida mustaqil ishlatilmaydi")

        # kandidatlarni kengaytirish (portlashni oldini olish uchun chegara)
        yangi = []
        for k in kandidatlar:
            for v in variantlar:
                yangi.append(k + v.split())
        kandidatlar = yangi[:8]

    return {
        "soz": soz,
        "grafemalar": grafemalar,
        "fonemalar": kandidatlar[0] if kandidatlar else [],
        "kandidatlar": kandidatlar,
        "uncertain": uncertain,
        "nomalum": nomalum,
    }


def g2p_matn(matn):
    return [g2p(w) for w in normalize(matn).split()]


if __name__ == "__main__":
    import sys, io, json
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print(f"Fonema inventari: {len(FONEMALAR)} birlik")
    print("  unlilar   :", " ".join(sorted(UNLILAR)))
    print("  undoshlar :", " ".join(f for f in FONEMALAR if f not in UNLILAR))
    print()

    testlar = [
        "shahar", "chiroq", "tong", "qishloq", "yaxshi", "hisob",
        "o'zbek", "g'alaba", "og'zi", "shahri", "jurnal", "jiyan",
        "ma'no", "san'at", "kelinga", "singil", "bahor", "xotira",
        "qo'ng'iroq", "o'qituvchi", "ehtiyot", "mustaqillik",
    ]
    for t in testlar:
        r = g2p(t)
        bayroq = f"   [{len(r['uncertain'])} noaniqlik]" if r["uncertain"] else ""
        print(f"{t:16s} -> /{' '.join(r['fonemalar'])}/"
              f"{'  (' + str(len(r['kandidatlar'])) + ' variant)' if len(r['kandidatlar'])>1 else ''}{bayroq}")
        for u in r["uncertain"]:
            print(f"                    ! {u}")
