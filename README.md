# LLM Data Pipeline

An MLOps-oriented project for building **language model (LM) training data pipelines** with Hugging Face `datasets` and PyTorch. The main notebook (**Lab1**) implements an **in-memory** pipeline: load a text dataset, tokenize it, group into fixed-length blocks, and build a DataLoader ready for causal LM training. The notebook does **not** include a validation split or training loop.

---

## Setup

```bash
pip install transformers torch datasets
```

Use a virtual environment (e.g. `.venv`) if you prefer.

---

## Lab1: In-Memory LM Pipeline

**File:** `Lab1.ipynb`

What the notebook does, step by step:

1. **Load data** — Loads IMDB **train** split only (`load_dataset("imdb", split="train")`). 25,000 examples. Uses the `"text"` column; labels are not used.

2. **Tokenizer** — DistilGPT-2 (`distilgpt2`) via `AutoTokenizer`. Pad token set to EOS.

3. **Tokenize** — Batched `.map` with **`truncation=True`** and **`max_length=512`** (tunable to 1024 in the notebook). Each example → `input_ids` and `attention_mask`; long reviews are truncated to avoid warnings and control memory. Produces `tokenized_ds`.

4. **Group into blocks** — Concatenates token IDs (and attention masks) in batches, trims to a multiple of `block_size` (128), and slices into fixed-length chunks. Uses **`_flatten()`** so it works when the dataset stores lists or numpy arrays, and **`remove_columns=tokenized_ds.column_names`** so the map output has only the new chunk columns (avoids ArrowInvalid). Produces **`lm_ds`** (train LM sequences).

5. **DataLoader** — Custom `collate_fn` stacks `input_ids` and sets `labels = input_ids.clone()`. **`train_loader`** with `batch_size=8`, `shuffle=True`. No validation DataLoader in the notebook.

6. **Sanity check** — Iterates one batch and prints shapes `(8, 128)` for `input_ids` and `labels`.

The pipeline ends with a ready-to-use `train_loader`; you can plug it into any causal LM training loop elsewhere. There is no model loading, training loop, or validation in the notebook.

---

## Changes from previous code

What changed from the **original** Lab1 (WikiText-2 + GPT-2) to the **current** notebook.

| Area | Previous | Now |
|------|----------|-----|
| **Dataset** | WikiText-2 **train** only. | **IMDB train** only (`split="train"`). |
| **Model / tokenizer** | GPT-2. | **DistilGPT-2** (`distilgpt2`). |
| **Tokenization** | `tokenizer(examples["text"], return_special_tokens_mask=False)`; no length limit. | **`truncation=True`** and **`max_length=512`** (or 1024) to avoid long-sequence warnings and control memory. |
| **Grouping** | `sum(examples["input_ids"], [])` and `sum(..., attention_mask)`. Caused **ArrowInvalid** on IMDB (column length mismatch). | **`_flatten()`** to handle list/numpy; **`remove_columns=tokenized_ds.column_names`** so the map output has only chunk columns and matching lengths. |
| **Output** | One `lm_ds`, one `train_loader`, one-batch shape check. | Same: one `lm_ds`, one `train_loader`, one-batch check (train-only). |

### Summary

- **Dataset:** WikiText-2 → IMDB (train only).
- **Model:** GPT-2 → DistilGPT-2.
- **Tokenization:** `truncation=True`, `max_length=512` (or 1024) to cap sequence length and control memory.
- **Grouping fix:** `_flatten` + `remove_columns` so grouping works on IMDB and avoids ArrowInvalid. No validation split or training in the notebook.