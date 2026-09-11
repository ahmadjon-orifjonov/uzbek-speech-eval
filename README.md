# uzbek-speech-eval

**Known-text pronunciation screening and evaluation tooling for Uzbek.**

Uzbek is a low-resource language: public speech corpora are small, phoneme-level
transcription is largely missing, and off-the-shelf ASR degrades on exactly the
contrasts that matter for pronunciation feedback — `q/k`, `x/h`, `o'/g'`, and the
digraphs `sh`, `ch`, `ng`.

This repository contains the pipeline I built to measure that, and the numbers it
produced. Every figure below comes from a script in `scripts/` and a JSON file in
`results/`.

---

## The idea

Free ASR decoding answers *"what did the model hear?"*. For pronunciation feedback
that is the wrong question — and on Uzbek it is unreliable: the baseline model
scores **WER 51.4%** on held-out data, so the model's own errors outnumber the
learner's.

But in a reading task **the text is already known**. So the question becomes
*"how well does the audio match the expected transcript, position by position?"*
That is CTC known-text forced alignment, and it needs no error-annotated corpus
to bootstrap.

## Measured results

Held-out FLEURS `uz_uz`, model `lucio/xls-r-uzbek-cv8` (trained on Common Voice 8,
so it never saw this data). **200 utterances, 1047 perturbations.**

95% confidence intervals come from a cluster bootstrap (2000 resamples) that
resamples **utterances**, not words — perturbations from the same recording share
a speaker and a recording condition, so treating them as independent would make
the intervals far too narrow.

| Measure | Group | AUC | 95% CI |
|---|---|---:|---|
| **`path_norm`** | overall | **0.869** | [0.852, 0.887] |
| `path_norm` | `q/k`, `x/h` | 0.874 | [0.853, 0.893] |
| `path_norm` | `o'/g'` | 0.837 | [0.792, 0.876] |
| `mean_logpost` | overall | 0.737 | [0.723, 0.751] |
| **`mean_logpost`** | `q/k`, `x/h` | **0.902** | [0.884, 0.920] |
| `mean_logpost` | `o'/g'` | 0.436 | [0.419, 0.452] |
| `margin` | overall | 0.665 | [0.653, 0.678] |
| `blank_ratio` | overall | 0.419 | [0.412, 0.425] |

| Other measurements | Result | Script |
|---|---|---|
| Negative control (untouched words) | **1047/1047 with Δ = 0.000000** | `eval_auc.py` |
| Free-decoding baseline | **WER 51.4%** (266/518 words, 30-utterance subset) | `eval_free_decoding.py` |
| G2P coverage | **413 unique words, 0% parser failure**, 7.7% flagged ambiguous | `eval_g2p_coverage.py` |
| Phoneme inventory | 31 phonemes, 1 unused (`ʒ`) | `eval_g2p_coverage.py` |

A 30-utterance pilot ran first and gave 0.884 / 0.918 / 0.433. Scaling to 200
moved the point estimates slightly down (0.869 / 0.902 / 0.436) and tightened the
intervals — the direction you should expect, and the reason the larger run is the
one reported here. `blank_ratio` is diagnostic only; below 0.5 it carries no
usable signal on its own.

### What the AUC numbers mean — and what they do not

`0.869` and `0.902` are **not the same measure**. `path_norm` is the primary
signal — it is the stable one, and the only one that survives the `o'/g'` case.
`mean_logpost` is secondary, but stronger specifically on `q/k` and `x/h`. They
are reported separately on purpose: when the two disagree, the system emits
**"uncertain"** and returns no verdict. Comparing a number from one measure with
a number from the other is meaningless.

Neither number is an accuracy figure. They measure **separation** between correct
and deviant cases under a controlled perturbation, not a system's ability to
diagnose a real learner's pronunciation.

## Stated limitations

1. **The audio was never altered.** In the perturbation study the reference *text*
   is corrupted, not the speech. This tests whether the alignment score is
   sensitive to transcript mismatch — it does not prove detection of real
   mispronunciation.
2. **Adult read speech.** FLEURS is professional adult recording. Children's
   speech is harder and is not covered here.
3. **Thresholds are uncalibrated.** The decision thresholds in `src/screening.py`
   come from the score distribution of these 30 recordings. Reporting on the same
   data that set them would be leakage; final figures need a separate test split.
4. **`o'/g'` is not used as a decision criterion.** `mean_logpost` gives AUC 0.436
   there — worse than chance. `path_norm` gives 0.837, but removing an apostrophe
   shortens the token sequence and shifts the frame distribution, so that number
   may be a tokenisation artifact rather than an acoustic difference.
5. **`sh`, `ch`, `ng` are not atomic** in the model's vocabulary, so the output is
   described as a *grapheme-acoustic mismatch*, never as "phoneme X was detected".

## Layout

```
src/
  normalize.py    Uzbek text normalisation (9 apostrophe variants unified; self-testing)
  g2p.py          rule-based grapheme-to-phoneme, ambiguity flagged rather than guessed
  aligner.py      CTC forced alignment + 4 word-level measures
  screening.py    screening core: abstention gate, scoring, forced-choice error typing
scripts/
  prepare_data.py          fetch and prepare the FLEURS sample set
  eval_free_decoding.py    WER baseline
  eval_alignment.py        alignment scores over the sample set
  eval_perturbation.py     perturbation sensitivity + negative control (the AUC table)
  eval_g2p_coverage.py     G2P coverage and phoneme frequency
  eval_auc.py              AUC + 95% cluster-bootstrap CIs (the table above)
  eval_examples.py         case-by-case behaviour table, failures included
screen.py                  CLI: one recording + its text -> JSON report
results/                   JSON output of the above, plus examples.md
data/g2p_lexicon.tsv       413-word phoneme lexicon
```

### The three-stage screening core

0. **Abstention gate** — empty, too short, too long, clipped, too quiet, mostly
   silence, unknown characters, word-index mismatch, or a globally poor match all
   produce *no result*. Forced alignment will happily align the wrong audio to any
   text; a meaningless verdict is worse than a refusal.
1. **Alignment** — per-word `path_norm` and `mean_logpost`; disagreement → "uncertain".
2. **Forced-choice error typing** — for a suspect word, substituted variants
   (`boshqa` → `boshka`) are re-aligned. A signal is emitted only if the margin
   clears a threshold, the best and second-best variants are separable, and
   untouched words did not shift. Not applied to `o'/g'`.

## Running it

Screen a single recording — this is the deliverable shape:

```bash
pip install -r requirements.txt
python screen.py --self-test                                  # no model needed
python screen.py --audio sample.wav --text "Quyosh tog' ortidan ko'tarildi."
```

The CLI prints one JSON object: per-word verdict, timestamps, raw measures, and
— only where the forced-choice test clears its thresholds — a probable error
type. An abstention is returned as a normal result, not an error.

Reproduce the numbers above:

```bash
python scripts/prepare_data.py --n 200   # downloads FLEURS uz_uz samples
python scripts/eval_g2p_coverage.py      # no model needed
python scripts/eval_perturbation.py      # ~6 min on a laptop CPU (200 utterances)
python scripts/eval_auc.py               # AUC + confidence intervals
python scripts/eval_examples.py --n 40   # case-by-case table
```

CPU only is fine: the full 200-utterance perturbation study took 339 s.

## Attribution and licence

- Code: MIT (see `LICENSE`).
- Evaluation data: [FLEURS](https://huggingface.co/datasets/google/fleurs) `uz_uz`,
  CC-BY 4.0 (Google). Audio files are **not** redistributed here — `prepare_data.py`
  fetches them. Reference transcripts are included under the same licence.
- Model: [`lucio/xls-r-uzbek-cv8`](https://huggingface.co/lucio/xls-r-uzbek-cv8),
  fine-tuned on Common Voice 8.
- The phoneme lexicon in `data/` is derived from the FLEURS transcripts.

Source comments are in Uzbek; documentation is in English.

## Contact

Built by **Axmadjon Orifjonov** — Tashkent, Uzbekistan.
Ahmadjonorifjonov57@gmail.com · [github.com/ahmadjon-orifjonov](https://github.com/ahmadjon-orifjonov)
Available for contract work on Uzbek speech evaluation, data pipelines, and
linguistic QA. AI coding tools are part of my workflow and I state that openly;
every number here is reproducible from the scripts in this repository.
