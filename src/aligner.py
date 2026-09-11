# -*- coding: utf-8 -*-
"""
Known-text CTC forced alignment + so'z darajasidagi o'lchovlar.

MUHIM ATAMA (review note): bu qiymatlar "ishonch" (confidence) EMAS.
To'g'ri nomi: modelning kutilgan grafemik transkriptga AKUSTIK MOSLIK signali.
Uchta alohida o'lchov qaytariladi, bittasi bilan xulosa chiqarilmaydi:
  1. mean_logpost   — tekislangan belgilar log-posteriorining o'rtachasi
  2. path_norm      — so'z segmentidagi Viterbi yo'l ballining kadrga normallashtirilgani
  3. blank_ratio    — segmentda blank ulushi (diagnostik, mustaqil ball emas)
  4. margin         — target belgi va eng kuchli raqib belgi orasidagi log farq
"""
from pathlib import Path
import importlib.util
import numpy as np
import torch
import torchaudio.functional as AF
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

_spec = importlib.util.spec_from_file_location("normalize", Path(__file__).parent / "normalize.py")
_nm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_nm)
normalize = _nm.normalize

MODEL_ID = "lucio/xls-r-uzbek-cv8"
FRAME_S = 0.02


class Aligner:
    def __init__(self, model_id=MODEL_ID):
        self.proc = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id
        self.delim = self.vocab["|"]
        self.model_id = model_id
        self._cache = {}

    def logprobs(self, wav, key=None):
        """log_softmax chiqishi (T, V). Bir xil audio uchun keshlanadi —
        perturbatsiya tajribasida audio o'zgarmaydi, faqat matn o'zgaradi."""
        if key is not None and key in self._cache:
            return self._cache[key]
        x = self.proc(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(x.input_values).logits
        lp = torch.log_softmax(logits, dim=-1)[0]     # (T, V)
        if key is not None:
            self._cache[key] = lp
        return lp

    def _tokenlar(self, matn):
        sozlar = matn.split()
        ids, soz_idx, nomalum = [], [], []
        for wi, s in enumerate(sozlar):
            if wi > 0:
                ids.append(self.delim); soz_idx.append(-1)
            for ch in s:
                tid = self.vocab.get(ch)
                if tid is None:
                    nomalum.append(ch); continue
                ids.append(tid); soz_idx.append(wi)
        return sozlar, ids, soz_idx, sorted(set(nomalum))

    def align(self, wav, matn, key=None):
        lp = self.logprobs(wav, key=key)               # (T, V)
        sozlar, ids, soz_idx, nomalum = self._tokenlar(matn)
        if not ids:
            return {"xato": "bo'sh matn"}
        T = lp.shape[0]
        if T < len(ids):
            return {"xato": f"audio qisqa: T={T} < token={len(ids)}"}

        targets = torch.tensor([ids], dtype=torch.int32)
        aligned, scores = AF.forced_align(lp.unsqueeze(0), targets, blank=self.blank)
        aligned, scores = aligned[0], scores[0]        # (T,)
        spans = AF.merge_tokens(aligned, scores, blank=self.blank)
        assert len(spans) == len(ids)

        lp_np = lp.numpy()
        aligned_np = aligned.numpy()
        # raqib: blankdan boshqa eng yaxshi token (target o'zidan tashqari)
        soz_natija = []
        for wi, s in enumerate(sozlar):
            idxs = [k for k, w in enumerate(soz_idx) if w == wi]
            if not idxs:
                continue
            sp = [spans[k] for k in idxs]
            t0, t1 = sp[0].start, sp[-1].end
            # 1) tekislangan belgilar log-posteriori
            char_lp = [float(x.score) for x in sp]
            # 2) segmentdagi barcha kadrlar bo'ylab Viterbi yo'l balli
            seg = lp_np[t0:t1]
            path_lp = [seg[t, aligned_np[t0 + t]] for t in range(t1 - t0)]
            # 3) blank ulushi
            blank_n = int(np.sum(aligned_np[t0:t1] == self.blank))
            # 4) target vs raqib margin (faqat target kadrlarida)
            margins = []
            for t in range(t1 - t0):
                tid = aligned_np[t0 + t]
                if tid == self.blank:
                    continue
                row = seg[t].copy()
                target_lp = row[tid]
                row[tid] = -np.inf
                row[self.blank] = -np.inf
                margins.append(float(target_lp - row.max()))
            soz_natija.append({
                "soz": s,
                # Review note: 2.1 va 2.6: etalon matndagi indeks va xom qiymatlar.
                # Qaror xom qiymat ustida chiqariladi; quyidagi yumaloqlangan
                # maydonlar o'zgarmaydi — T-04..T-07 natijalari ularga bog'liq.
                "soz_idx": wi,
                "path_norm_xom": float(np.mean(path_lp)),
                "mean_logpost_xom": float(np.mean(char_lp)),
                "belgi_soni": len(sp),
                "boshlanish_s": round(t0 * FRAME_S, 3),
                "tugash_s": round(t1 * FRAME_S, 3),
                "kadrlar": int(t1 - t0),
                "mean_logpost": round(float(np.mean(char_lp)), 4),
                "min_logpost": round(float(np.min(char_lp)), 4),
                "path_norm": round(float(np.mean(path_lp)), 4),
                "blank_ratio": round(blank_n / max(t1 - t0, 1), 4),
                "margin": round(float(np.mean(margins)), 4) if margins else None,
            })
        return {"sozlar": soz_natija, "nomalum_belgilar": nomalum, "kadrlar_jami": int(T)}
