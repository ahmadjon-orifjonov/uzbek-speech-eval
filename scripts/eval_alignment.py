# -*- coding: utf-8 -*-
"""
KNOWN-TEXT FORCED ALIGNMENT + so'z darajasidagi ishonch balli.

Sabab (T-05): erkin dekodlash WER 51.4% — modelning o'z xatosi o'quvchi
xatosidan ko'p. Lekin bizda matn OLDINDAN MA'LUM. Shuning uchun savol
"model qaysi so'zni chiqardi" emas, "audio kutilgan matnga har bir
pozitsiyada qanchalik mos keladi".

Chiqish: results/04_alignment.json
"""
import sys, io, json, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "results" / "samples.json"
OUT = ROOT / "results" / "alignment.json"
MODEL_ID = "lucio/xls-r-uzbek-cv8"

import importlib.util
spec = importlib.util.spec_from_file_location("normalize", Path(__file__).resolve().parent.parent / "src" / "normalize.py")
nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nm)
normalize = nm.normalize

import numpy as np, soundfile as sf, torch
import torchaudio.functional as AF
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Aligner:
    def __init__(self, model_id=MODEL_ID):
        self.proc = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id
        self.delim = self.vocab.get("|")
        # audio kadr uzunligi (wav2vec2: 20 ms)
        self.frame_s = 0.02

    def logprobs(self, wav):
        x = self.proc(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(x.input_values).logits
        return torch.log_softmax(logits, dim=-1)   # (1, T, V)

    def matn_tokenlari(self, matn):
        """Normallashtirilgan matnni token id larga aylantiradi.
        Qaytadi: ids, va har bir id qaysi so'zga tegishli ekani."""
        sozlar = matn.split()
        ids, soz_idx, nomalum = [], [], []
        for wi, s in enumerate(sozlar):
            if wi > 0:
                ids.append(self.delim); soz_idx.append(-1)   # so'z ajratgich
            for ch in s:
                tid = self.vocab.get(ch)
                if tid is None:
                    nomalum.append(ch); continue
                ids.append(tid); soz_idx.append(wi)
        return sozlar, ids, soz_idx, sorted(set(nomalum))

    def align(self, wav, matn):
        lp = self.logprobs(wav)                       # (1,T,V)
        sozlar, ids, soz_idx, nomalum = self.matn_tokenlari(matn)
        if not ids:
            return None
        targets = torch.tensor([ids], dtype=torch.int32)
        T = lp.shape[1]
        if T < len(ids):
            return {"xato": f"audio juda qisqa: T={T} < token={len(ids)}"}
        aligned, scores = AF.forced_align(lp, targets, blank=self.blank)
        aligned, scores = aligned[0], scores[0]       # (T,)
        spans = AF.merge_tokens(aligned, scores, blank=self.blank)

        # spans tartibi targets tartibiga mos: har bir target token uchun bitta span
        assert len(spans) == len(ids), f"span={len(spans)} != token={len(ids)}"

        soz_natija = []
        for wi, s in enumerate(sozlar):
            idxs = [k for k, w in enumerate(soz_idx) if w == wi]
            if not idxs:
                continue
            sp = [spans[k] for k in idxs]
            # torchaudio: span.score = o'sha token uchun o'rtacha EHTIMOLLIK (0..1)
            ballar = [float(x.score) for x in sp]
            boshl = sp[0].start * self.frame_s
            tugash = sp[-1].end * self.frame_s
            kadrlar = sum(x.end - x.start for x in sp)
            soz_natija.append({
                "soz": s,
                "belgi_soni": len(sp),
                "boshlanish_s": round(boshl, 3),
                "tugash_s": round(tugash, 3),
                "davomiylik_s": round(tugash - boshl, 3),
                "kadrlar": int(kadrlar),
                "ball_ortacha": round(float(np.mean(ballar)), 4),
                "ball_min": round(float(np.min(ballar)), 4),
                "ball_logmean": round(float(np.mean(np.log(np.clip(ballar, 1e-9, 1)))), 4),
                "belgi_ballari": [round(b, 4) for b in ballar],
            })
        return {"sozlar": soz_natija, "nomalum_belgilar": nomalum, "kadrlar_jami": int(T)}


if __name__ == "__main__":
    print(f"Model yuklanmoqda: {MODEL_ID}")
    t0 = time.time()
    al = Aligner()
    print(f"  tayyor ({time.time()-t0:.1f}s), blank={al.blank}, delim={al.delim}")

    yozuvlar = json.loads(IN.read_text(encoding="utf-8"))
    natijalar, t_start = [], time.time()
    hamma_ball = []

    for rec in yozuvlar:
        wav, sr = sf.read(rec["fayl"], dtype="float32")
        r = al.align(wav, rec["etalon_norm"])
        if r is None or "xato" in r:
            print(f"  {rec['id']}: {r}")
            continue
        ballar = [w["ball_ortacha"] for w in r["sozlar"]]
        hamma_ball += ballar
        natijalar.append({"id": rec["id"], "etalon": rec["etalon_norm"], **r,
                          "ball_ortacha_umumiy": round(float(np.mean(ballar)), 4)})
        print(f"  {rec['id']}  so'z={len(ballar):3d}  o'rtacha ball={np.mean(ballar):.3f}  "
              f"min={np.min(ballar):.3f}")

    hamma_ball = np.array(hamma_ball)
    xulosa = {
        "model": MODEL_ID,
        "usul": "known-text CTC forced alignment (torchaudio.functional.forced_align)",
        "namunalar": len(natijalar),
        "sozlar_jami": int(len(hamma_ball)),
        "ball_statistikasi": {
            "ortacha": round(float(hamma_ball.mean()), 4),
            "median": round(float(np.median(hamma_ball)), 4),
            "p10": round(float(np.percentile(hamma_ball, 10)), 4),
            "p25": round(float(np.percentile(hamma_ball, 25)), 4),
            "p75": round(float(np.percentile(hamma_ball, 75)), 4),
            "min": round(float(hamma_ball.min()), 4),
            "max": round(float(hamma_ball.max()), 4),
        },
        "vaqt_s": round(time.time() - t_start, 1),
        "natijalar": natijalar,
    }
    OUT.write_text(json.dumps(xulosa, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("SO'Z BALLARI TAQSIMOTI:", json.dumps(xulosa["ball_statistikasi"], ensure_ascii=False))
    print(f"Vaqt: {xulosa['vaqt_s']}s   Saqlandi: {OUT}")
