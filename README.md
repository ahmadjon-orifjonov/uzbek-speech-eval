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

Held-out FLEURS `uz_uz` (30 utterances, 518 words), model `lucio/xls-r-uzbek-cv8`.
The model never saw this data in training.

| Measurement | Result | Script |
|---|---|---|
| Detection of deviation, AUC (`path_norm`) | **0.884** | `eval_perturbation.py` |
| `q/k`, `x/h` confusions, AUC (`mean_logpost`) | **0.918** | `eval_perturbation.py` |
| `o'/g'` confusions, AUC (`mean_logpost`) | 0.433 | `eval_perturbation.py` |
| Negative control (untouched words) | **157/157 with Δ = 0.000000** | `eval_perturbation.py` |
| Free-decoding baseline | **WER 51.4%** (266/518 words) | `eval_free_decoding.py` |
| G2P coverage | **413 unique words, 0% parser failure**, 7.7% flagged ambiguous | `eval_g2p_coverage.py` |
| Phoneme inventory | 31 phonemes, 1 unused (`ʒ`) | `eval_g2p_coverage.py` |

### What the AUC numbers mean — and what they do not

`0.884` and `0.918` are **not the same measure**. `path_norm` is the primary
signal (0.884 overall, 0.891 on `q/k`+`x/h`); `mean_logpost` is the secondary one
(0.743 overall, 0.918 on `q/k`+`x/h`). They are reported separately on purpose:
when the two disagree, the system emits **"uncertain"** and returns no verdict.

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
4. **`o'/g'` is not used as a decision criterion.** `mean_logpost` gives AUC 0.433
   there — worse than chance. `path_norm` gives 0.858, but removing an apostrophe
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
results/                   JSON output of the above
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

```bash
pip install -r requirements.txt
python scripts/prepare_data.py        # downloads FLEURS uz_uz samples
python scripts/eval_g2p_coverage.py   # no model needed
python scripts/eval_alignment.py      # downloads the wav2vec2 model on first run
python scripts/eval_perturbation.py
```

CPU only is fine: alignment runs about 5 s per utterance on a laptop CPU.

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
