# -*- coding: utf-8 -*-
"""
CLI: screen one recording against its known text and emit a JSON report.

    python screen.py --audio sample.wav --text "Quyosh tog' ortidan ko'tarildi."
    python screen.py --audio sample.wav --text-file line.txt --out report.json
    python screen.py --self-test

The audio is read at any sample rate and resampled to 16 kHz mono.

Output is a single JSON object. Two shapes are possible:

  {"abstain": true, "reason": ..., "technical": ...}
      The system refused. This is a first-class outcome, not an error:
      forced alignment will happily align the wrong audio to any text,
      so a meaningless verdict is worse than no verdict.

  {"abstain": false, "words": [...], "counts": {...}, "error_types": {...}}
      Per-word verdict ("match" / "suspect" / "strong_deviation" /
      "uncertain"), timestamps, raw measures, and — only where the
      forced-choice test clears its thresholds — a probable error type.

An error type such as "q -> k" is a PROBABLE SIGNAL derived from
comparing two text hypotheses. It is not a phonetic diagnosis.
"""
import sys
import io
import json
import argparse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

VERDICT_EN = {
    "mos": "match",
    "shubhali": "suspect",
    "kuchli_ogish": "strong_deviation",
    "noaniq": "uncertain",
}


def load_audio(path):
    import numpy as np
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        try:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        except ImportError:
            raise SystemExit("audio is {} Hz; install librosa to resample, "
                             "or supply 16 kHz audio".format(sr))
    return np.asarray(wav, dtype="float32")


def to_en(r):
    """Translate the internal (Uzbek) result keys into an English report."""
    if r.get("abstain"):
        return {"abstain": True,
                "reason": r.get("sabab"),
                "technical": r.get("texnik"),
                "audio": r.get("sifat")}
    words = []
    for w in r["sozlar"]:
        item = {
            "word": w["soz"],
            "verdict": VERDICT_EN.get(w["baho"], w["baho"]),
            "start_s": w.get("boshlanish_s"),
            "end_s": w.get("tugash_s"),
            "measures": {"path_norm": w.get("path_norm"),
                         "mean_logpost": w.get("mean_logpost")},
        }
        if w.get("xato_turi"):
            item["probable_error"] = w["xato_turi"]
        if w.get("signal"):
            s = w["signal"]
            item["forced_choice"] = {
                "candidate": s.get("nom"),
                "margin": s.get("farq"),
                "threshold": s.get("ostona"),
                "accepted": s.get("qabul"),
                "rejected_because": s.get("rad_sabab"),
            }
        words.append(item)
    return {
        "abstain": False,
        "duration_s": r.get("davomiylik_s"),
        "words": words,
        "counts": {VERDICT_EN.get(k, k): v for k, v in r["sanoq"].items()},
        "error_types": r.get("turlar", {}),
        "recommendations": r.get("tavsiya", []),
        "note": ("Error types are probable signals from a two-hypothesis text "
                 "comparison, not a phonetic diagnosis. o'/g' is never used "
                 "as a decision criterion."),
    }


def self_test():
    """Runs without a model: checks normalisation and G2P wiring only."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("normalize", ROOT / "src" / "normalize.py")
    nm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nm)
    from g2p import g2p

    cases = [("Quyosh tog‘ ortidan ko'tarildi.", "quyosh tog‘ ortidan ko‘tarildi"),
             ("O‘zbekiston", "o‘zbekiston")]
    ok = True
    for xom, kutilgan in cases:
        olingan = nm.normalize(xom)
        holat = "ok" if olingan == kutilgan else "FAIL"
        ok = ok and holat == "ok"
        print("normalize  {:<34} -> {:<34} {}".format(xom, olingan, holat))
    r = g2p("qishloq")
    print("g2p        qishloq -> {}  uncertain={}".format(
        "".join(r["kandidatlar"][0]) if r.get("kandidatlar") else "?", r.get("uncertain")))
    print("\nself-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Uzbek known-text pronunciation screening")
    ap.add_argument("--audio", help="wav/flac/mp3 file")
    ap.add_argument("--text", help="expected text (the learner reads this)")
    ap.add_argument("--text-file", help="file containing the expected text")
    ap.add_argument("--out", help="write JSON here instead of stdout")
    ap.add_argument("--self-test", action="store_true",
                    help="check normalisation and G2P without loading the model")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.audio:
        ap.error("--audio is required (or use --self-test)")
    matn = a.text
    if a.text_file:
        matn = Path(a.text_file).read_text(encoding="utf-8").strip()
    if not matn:
        ap.error("--text or --text-file is required")

    from screening import Demo
    wav = load_audio(a.audio)
    r = Demo().tahlil(wav, matn)
    out = to_en(r)
    out["audio_file"] = Path(a.audio).name
    out["expected_text"] = matn

    js = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(js, encoding="utf-8", newline="\n")
        print("->", a.out)
    else:
        print(js)
    return 0


if __name__ == "__main__":
    sys.exit(main())
