# -*- coding: utf-8 -*-
"""
1-DARAJA: known-text so'z darajasidagi tahlil.

Kirish : data/audio/*.wav + results/02_namunalar.json (etalon matn)
Jarayon: CTC dekod -> normalizatsiya -> so'z darajasida Levenshtein tekislash
Chiqish: results/03_soz_darajasi.json + qisqa hisobot

MUHIM: bu TALAFFUZ tahlili EMAS. Bu tanish va matnga moslik tahlili.
(texnik/03_DAVO_DALIL_CHEGARA.md)
"""
import sys, io, json, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "results" / "samples.json"
OUT = ROOT / "results" / "free_decoding.json"
MODEL_ID = "lucio/xls-r-uzbek-cv8"

import importlib.util
spec = importlib.util.spec_from_file_location("normalize", Path(__file__).resolve().parent.parent / "src" / "normalize.py")
nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nm)
normalize = nm.normalize

import numpy as np, soundfile as sf, torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def tekislash(etalon, gipoteza):
    """So'z darajasida Levenshtein + amallar ro'yxati (backtrace)."""
    n, m = len(etalon), len(gipoteza)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            narx = 0 if etalon[i - 1] == gipoteza[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1,        # deletion
                          d[i, j - 1] + 1,        # insertion
                          d[i - 1, j - 1] + narx) # sub / match
    amallar, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if etalon[i-1] == gipoteza[j-1] else 1):
            amallar.append(("moslik" if etalon[i-1] == gipoteza[j-1] else "almashtirish",
                            etalon[i-1], gipoteza[j-1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            amallar.append(("tushirish", etalon[i-1], None)); i -= 1
        else:
            amallar.append(("qoshish", None, gipoteza[j-1])); j -= 1
    amallar.reverse()
    return int(d[n, m]), amallar


print(f"Model yuklanmoqda: {MODEL_ID}")
t0 = time.time()
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()
print(f"  yuklandi ({time.time()-t0:.1f}s), parametrlar: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

yozuvlar = json.loads(IN.read_text(encoding="utf-8"))
natijalar = []
jami_xato = jami_soz = 0
t_start = time.time()

for rec in yozuvlar:
    arr, sr = sf.read(rec["fayl"], dtype="float32")
    assert sr == 16000, f"sr={sr}"
    inputs = processor(arr, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    ids = torch.argmax(logits, dim=-1)
    xom = processor.batch_decode(ids)[0]
    gip = normalize(xom)

    e_soz, g_soz = rec["etalon_norm"].split(), gip.split()
    masofa, amallar = tekislash(e_soz, g_soz)
    jami_xato += masofa; jami_soz += len(e_soz)
    wer = masofa / max(len(e_soz), 1)

    sanoq = {"moslik": 0, "almashtirish": 0, "tushirish": 0, "qoshish": 0}
    for a, _, _ in amallar:
        sanoq[a] += 1

    natijalar.append({
        "id": rec["id"], "davomiylik_s": rec["davomiylik_s"],
        "etalon": rec["etalon_norm"], "gipoteza": gip,
        "wer": round(wer, 4), "masofa": masofa, "soz_soni": len(e_soz),
        "amallar_sanogi": sanoq,
        "xatolar": [{"tur": a, "etalon": e, "eshitilgan": g}
                    for a, e, g in amallar if a != "moslik"],
    })
    print(f"  {rec['id']}  WER={wer:5.1%}  S={sanoq['almashtirish']:2d} "
          f"D={sanoq['tushirish']:2d} I={sanoq['qoshish']:2d}")

umumiy_wer = jami_xato / max(jami_soz, 1)
xulosa = {
    "model": MODEL_ID,
    "manba": "google/fleurs uz_uz test (held-out: model CV8 ustida o'qitilgan)",
    "namunalar": len(natijalar),
    "jami_soz": jami_soz,
    "jami_xato": jami_xato,
    "umumiy_wer": round(umumiy_wer, 4),
    "ishlov_vaqti_s": round(time.time() - t_start, 1),
    "natijalar": natijalar,
}
OUT.write_text(json.dumps(xulosa, ensure_ascii=False, indent=2), encoding="utf-8")

print()
print(f"UMUMIY WER: {umumiy_wer:.1%}  ({jami_xato}/{jami_soz} so'z)")
print(f"Vaqt: {xulosa['ishlov_vaqti_s']}s  ({xulosa['ishlov_vaqti_s']/len(natijalar):.1f}s / namuna, CPU)")
print(f"Saqlandi: {OUT}")
