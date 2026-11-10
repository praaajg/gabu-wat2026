# GABU @ WAT 2026

Zero-shot and QLoRA adaptation of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) for multimodal English-to-Indic translation.

**Paper:** [GABU@WAT 2026](https://aclanthology.org/2026.wat-1.10/) · [PDF](paper/2026.wat-1.10.pdf)  
**Adapter:** [ShivRamSaud/wat2026-gemma-mmt-en2indic](https://huggingface.co/ShivRamSaud/wat2026-gemma-mmt-en2indic)

## Results

Test split (1,595 segments per language):

| Language | Zero-shot BLEU | Zero-shot CHRF3 | QLoRA BLEU | QLoRA CHRF3 |
| --- | ---: | ---: | ---: | ---: |
| Hindi | 31.99 | 57.44 | 42.09 | 65.29 |
| Bengali | 22.67 | 53.82 | 41.60 | 69.04 |
| Malayalam | 17.83 | 56.55 | 41.68 | 72.81 |
| Odia | 5.84 | 37.25 | 50.49 | 72.82 |
| **Macro** | **19.58** | **51.27** | **43.97** | **69.99** |

Challenge split (1,400 segments per language; Odia challenge text is not released):

| Language | Zero-shot BLEU | Zero-shot CHRF3 | QLoRA BLEU | QLoRA CHRF3 |
| --- | ---: | ---: | ---: | ---: |
| Hindi | 36.51 | 61.10 | 50.58 | 69.68 |
| Bengali | 17.61 | 54.36 | 38.08 | 68.46 |
| Malayalam | 16.89 | 53.59 | 32.44 | 66.62 |
| **Macro** | **23.67** | **56.35** | **40.37** | **68.25** |

The released `best_adapter` is checkpoint 3,000 with dev macro CHRF3 71.59. Exact dev, test, and challenge scores are in [`results/`](results).

## Setup

```bash
pip install -r requirements.txt
```

## Data

```bash
python -m wat2026.prepare_data --data-root data/visual-genome
```

The loader expects the public `ShivRamSaud/HINDI-BENGALI-MALAYALAM-ODIA-VISUAL-GENOME` release: tab-separated rows `image_id, X, Y, W, H, English, Target` plus `{image_id}.jpg` images. The training loader crops the referenced region with 15% padding and caps it at 448 px.

## Training

```bash
accelerate launch --num_processes 2 --mixed_precision fp16 -m wat2026.train \
  --data-root data/visual-genome \
  --output-dir outputs/wat2026 \
  --epochs 3 \
  --save-steps 500 \
  --eval-steps 500
```

The submitted run used two T4 GPUs: QLoRA rank 64, alpha 128, dropout 0.05, 4-bit NF4 with double quantization, FP16 compute, sequence length 1024, 448 px crops, and an effective batch size of 16. Checkpoints are evaluated on 200 development rows and the best adapter is selected by macro CHRF3. Add `--hub-repo-id <user>/<repo>` to push checkpoints. The released `best_adapter` is checkpoint 3,000; set `--max-steps 3000` for a shorter reproduction run.

## Evaluation

```bash
python -m wat2026.evaluate \
  --data-root data/visual-genome \
  --eval-dir outputs/eval \
  --adapter ShivRamSaud/wat2026-gemma-mmt-en2indic \
  --adapter-subfolder best_adapter \
  --prefix finetuned

python -m wat2026.score --eval-dir outputs/eval --prefix finetuned
```

Omit `--adapter` and use `--prefix zero_shot` for the baseline. Generation is greedy and resumable per split and language.

## Merge

```bash
python -m wat2026.merge \
  --adapter ShivRamSaud/wat2026-gemma-mmt-en2indic \
  --adapter-subfolder best_adapter \
  --output-dir outputs/merged
```

Both commands use the `best_adapter` subfolder by default. Pass `--adapter-subfolder ""` when the adapter config is at the root of a local directory.

## Model

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "ShivRamSaud/wat2026-gemma-mmt-en2indic",
    allow_patterns="best_adapter/*",
)
```

## Layout

```text
wat2026/
  data.py          Visual Genome loading, region crops, splits
  prompts.py       chat-template prompt builders
  train.py         QLoRA DDP training and checkpoint selection
  evaluate.py      resumable single-GPU generation
  score.py         BLEU and CHRF3 scoring
  merge.py         adapter merge
  prepare_data.py  dataset download and extraction
model/             released adapter configuration
results/           released dev, test, and challenge metrics
paper/             WAT 2026 paper and BibTeX
```

## Citation

```bibtex
@inproceedings{saud-etal-2026-gabu,
    title = "{GABU}@{WAT} 2026: Zero Shot and {QL}o{RA} Adaptation of Gemma 4 {E}4{B} for Multimodal {E}nglish to {I}ndic Translation",
    author = "Saud, Shiv Ram  and
      Mahato, Aashish  and
      Ghimire, Prajwal  and
      Dawadi, Sundeep  and
      Saud, Ramesh  and
      Regmi, Sunil",
    editor = "Nakazawa, Toshiaki  and
      Goto, Isao",
    booktitle = "Proceedings of the Thirteenth Workshop on {A}sian Translation ({WAT} 2026)",
    month = nov,
    year = "2026",
    address = "Hengqin, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.wat-1.10/",
    pages = "145--154",
    ISBN = "979-8-89176-446-0"
}
```

The modules in `wat2026/` are extracted from the submitted WAT 2026 notebooks; comments and notebook-only scaffolding were removed.
