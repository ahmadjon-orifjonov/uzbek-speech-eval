# -*- coding: utf-8 -*-
"""
1-HAFTA: test namunalarini tayyorlash.

USC (issai/Uzbek_Speech_Corpus) bitta 9.47 GB tar.gz — streaming ishlamaydi,
to'liq yuklash kerak (Colab'da qilinadi). Shuning uchun mahalliy smoke-test
uchun FLEURS uz_uz ishlatiladi.

NEGA FLEURS HALOL TEST: model `lucio/xls-r-uzbek-cv8` Common Voice 8 ustida
o'qitilgan. FLEURS boshqa korpus, ya'ni model uni ko'rmagan (held-out).

Natija: data/audio/*.wav + results/02_namunalar.json
"""
import sys, io, json, os
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
OUT = ROOT / "results" / "samples.json"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)

N = 30
TARGET_SR = 16000

import importlib.util
spec = importlib.util.spec_from_file_location("normalize", Path(__file__).resolve().parent.parent / "src" / "normalize.py")
nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nm)
normalize = nm.normalize

import pyarrow.parquet as pq
import numpy as np, soundfile as sf, librosa
from huggingface_hub import hf_hub_download

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
p = hf_hub_download("google/fleurs", "parquet-data/uz_uz/test-00000-of-00001.parquet",
                    repo_type="dataset")

pf = pq.ParquetFile(p)
print("Ustunlar:", pf.schema_arrow.names)

yozuvlar = []
olindi = 0
for batch in pf.iter_batches(batch_size=32):
    d = batch.to_pylist()
    for row in d:
        if olindi >= N:
            break
        matn = row.get("transcription") or row.get("raw_transcription") or ""
        audio = row.get("audio")
        if audio is None or not matn:
            continue
        raw = audio["bytes"] if isinstance(audio, dict) and audio.get("bytes") else None
        if raw is None:
            continue
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != TARGET_SR:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        dur = len(arr) / sr
        if dur < 2 or dur > 20:          # juda qisqa/uzunlarini olmaymiz
            continue
        yol = AUDIO_DIR / f"fleurs_{olindi:02d}.wav"
        sf.write(str(yol), arr, sr)
        yozuvlar.append({
            "id": f"fleurs_{olindi:02d}",
            "manba": "google/fleurs uz_uz test",
            "fayl": str(yol),
            "sr": sr,
            "davomiylik_s": round(dur, 2),
            "etalon_xom": matn,
            "etalon_norm": normalize(matn),
        })
        print(f"  {olindi:02d}  {dur:5.2f}s  {normalize(matn)[:65]}")
        olindi += 1
    if olindi >= N:
        break

OUT.write_text(json.dumps(yozuvlar, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"{len(yozuvlar)} ta namuna -> {AUDIO_DIR}")
print(f"Metadata -> {OUT}")
