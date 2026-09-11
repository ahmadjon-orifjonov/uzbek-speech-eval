# -*- coding: utf-8 -*-
"""
MATN NORMALIZATORI — hamma joyda shu ishlatiladi.

Sabab: T-01 Topilma 2 (texnik/04_TOPILMALAR.md).
Model vocabida faqat U+2018 (‘) va U+2019 (’) bor. Oddiy klaviatura
apostrofi (U+0027) [UNK] ga aylanadi va o'/g' bo'lgan HAR BIR so'z
jimgina xato deb hisoblanadi.

Bu modul etalon matn, G2P, annotatsiya va demo — hammasida bir xil
chaqiriladi. Ikki xil normalizatsiya = o'lchov yaroqsiz.
"""
import re
import unicodedata

# Model vocabidagi apostrof (o'/g' uchun tanlangan kanonik belgi).
# ESLATMA: USC/CV matnlarida ‘ va ’ dan qaysi biri ko'p uchrashi
# o'lchanishi kerak; hozircha U+2018 tanlandi va shu yerda bitta
# joyda o'zgartiriladi.
KANONIK_APOSTROF = "‘"  # ‘

# Apostrofga o'xshab ketadigan hamma belgi shu ro'yxatda
APOSTROF_VARIANTLARI = [
    "'",  # ' ASCII
    "‘",  # ‘ chap
    "’",  # ’ o'ng
    "ʼ",  # ʼ modifier letter apostrophe
    "ʻ",  # ʻ modifier letter turned comma (ko'p ishlatiladi!)
    "´",  # ´ acute
    "`",  # ` grave
    "′",  # ′ prime
    "＇",  # ＇ fullwidth
]

_APOSTROF_RE = re.compile("[" + "".join(APOSTROF_VARIANTLARI) + "]")

# Model vocabi: a-z + | + kanonik apostrof
RUXSAT_ETILGAN = set("abcdefghijklmnopqrstuvwxyz") | {KANONIK_APOSTROF, " "}


def normalize(matn: str) -> str:
    """Matnni model alifbosiga keltiradi. Deterministik."""
    if matn is None:
        return ""
    # 1. Unicode NFC
    matn = unicodedata.normalize("NFC", matn)
    # 2. Kichik harf
    matn = matn.lower()
    # 3. Barcha apostrof variantlarini kanonikka
    matn = _APOSTROF_RE.sub(KANONIK_APOSTROF, matn)
    # 4. Tire va uzun chiziqlarni bo'shliqqa
    matn = re.sub(r"[‐-―\-]", " ", matn)
    # 5. Ruxsat etilmagan belgilarni olib tashlash (tinish belgilari, raqamlar)
    matn = "".join(ch if ch in RUXSAT_ETILGAN else " " for ch in matn)
    # 6. Ko'p bo'shliqni bittaga
    matn = re.sub(r"\s+", " ", matn).strip()
    return matn


def tekshir(matn: str):
    """Normalizatsiyadan keyin model uchun begona belgi qolganmi?"""
    n = normalize(matn)
    begona = sorted({ch for ch in n if ch not in RUXSAT_ETILGAN})
    return n, begona


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    testlar = [
        "O'zbekiston",                 # ASCII apostrof
        "O‘zbekiston",                 # U+2018
        "O’zbekiston",                 # U+2019
        "Oʻzbekiston",                 # U+02BB — eng ko'p uchraydigan rasmiy variant
        "G'alaba, 2026-yil!",          # tinish + raqam
        "Shahar — katta.",             # uzun tire
        "Bugun havo yaxshi.",
        "Is'hoq va ishoq",             # muhim minimal juftlik
    ]
    print(f"Kanonik apostrof: U+{ord(KANONIK_APOSTROF):04X}")
    print()
    for t in testlar:
        n, begona = tekshir(t)
        bayroq = "" if not begona else f"   BEGONA: {begona}"
        print(f"{t!r:32s} -> {n!r}{bayroq}")

    # Barcha apostrof variantlari bir xil natija berishini tasdiqlash
    variantlar = ["o" + a + "zbek" for a in APOSTROF_VARIANTLARI]
    natijalar = {normalize(v) for v in variantlar}
    print()
    print(f"Apostrof variantlari soni: {len(APOSTROF_VARIANTLARI)}")
    print(f"Normalizatsiyadan keyin turli natijalar: {len(natijalar)} -> {natijalar}")
    assert len(natijalar) == 1, "XATO: apostroflar bir xil natija bermayapti!"
    print("TEST O'TDI: barcha apostrof variantlari bitta shaklga keldi.")
