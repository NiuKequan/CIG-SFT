# Third-party notices

This repository is a fork of [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) that adds the CIG-SFT
loss. Everything that ships here either comes from upstream or is listed below.

## LlamaFactory

- Source: <https://github.com/hiyouga/LLaMA-Factory>
- Base commit: `97b32d31`, `VERSION 0.9.6.dev0`
- License: Apache License 2.0, kept verbatim in [`LICENSE`](LICENSE)
- Upstream documentation: kept verbatim in [`docs/LLAMAFACTORY.md`](docs/LLAMAFACTORY.md) (English) and
  [`docs/LLAMAFACTORY_zh.md`](docs/LLAMAFACTORY_zh.md) (Chinese)

The CIG-SFT changes are nine modified files plus one new module (`src/llamafactory/data/erased_prompt.py`). They are the
only difference from the base commit, and every new parameter defaults to `False`/`None`, so upstream behaviour is
byte-for-byte unchanged when the flags are off.

## Training data

The **built parquet** (the `prompt`/`response` rows, plus traceability columns) is committed under `example/data/`, so
training can start without any download. The raw sources below are **not** committed; `example/scripts/build_dataset.py`
re-fetches each file from its public Hugging Face repository, verifies it against the recorded hash, and rebuilds the
parquet from it in one pass.

### Math: `numina_cot`

| File | SHA-256 | Rows |
| --- | --- | --- |
| `numina_cot_10k.jsonl` | `a172916f9661937772d123e614cf1066b6c0586d2a871a8548988750d5a96b18` | 10000 |
| `numina_cot_30k.jsonl` | `ff07c61ad51b3d3fccd7d683aa6e599a1079f81d637d6338525925a3e326024e` | 30000 |
| `numina_cot_100k.jsonl` | `0c978394aa05b4254b0094926dca56159d4edb2bd409e313439d8344a25a6d06` | 100000 |

- Host: <https://huggingface.co/datasets/chichi56/ASFT>
- Origin: the mixtures are **derived from** [`AI-MO/NuminaMath-CoT`](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
  (Apache License 2.0). They are a deduplicated subset carrying only the `instruction`/`response` fields, and they are
  not byte-reproducible from the upstream dataset alone: `NuminaMath-CoT` has 859494 rows and a different schema
  (`source`, `problem`, `solution`, `messages`), and its first row does not match the first row of these files. Cite
  `NuminaMath-CoT` as the source of the problems, and this table as the exact bytes used here.

`example/scripts/build_dataset.py` then deduplicates the raw rows and appends the task instruction. The mapping from raw
rows to written rows is recorded in `example/data/numina_cot/<size>/manifest.json` on every build.

### Code: `tulu_code`

- Host: [`allenai/tulu-3-sft-personas-code`](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-code)
  (Open Data Commons Attribution License, ODC-By), file `data/train-00000-of-00001.parquet`
  (SHA-256 `e343c319aa0b4c577236d1e52433528ea7801dabd5696242ba3244242f76eb57`), first 30000 rows
- `example/scripts/build_dataset.py` normalizes the `prompt`/`messages` columns to the same `instruction`/`response`
  shape as the math mixture, deduplicating 5 exact duplicates → 29995 rows in `example/data/tulu_code/30k/`

## Related implementations

The loss is a sibling of these token-reweighting methods, and the way it is wired into LlamaFactory follows the
integration they already have upstream:

- DFT — <https://github.com/yongliang-wu/DFT>
- ASFT — <https://github.com/zhuchichi56/ASFT>
- IDFT — <https://github.com/zhangmiaosen2000/Towards-On-Policy-SFT>
- EAFT — <https://github.com/ymxyll/LlamaFactory-EAFT>

None of them is vendored here. Only the CIG-SFT loss is added to LlamaFactory.
