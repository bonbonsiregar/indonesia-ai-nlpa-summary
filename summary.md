# Comprehensive Notebook Summary — Indonesian Text Summarization

This document describes the local fine-tuning notebooks for abstractive text summarization on the Liputan6 dataset using cahya's pretrained encoder-decoder models.

**Notebook:**
- `project_2_(local_bert2bert).ipynb` — uses `cahya/bert2bert-indonesian-summarization` (BERT encoder + BERT decoder)

---

## 1) Environment Setup
* Sets local project paths (`PROJECT_ROOT`, `ARCHIVE_PATH`) pointing to the Liputan6 `.tar.gz` archive.
* Installs required libraries: `transformers`, `datasets`, `evaluate`, `rouge_score`, and `accelerate>=1.1.0`.
* Verifies GPU availability using `!nvidia-smi`.

## 2) Data Extraction and Sampling
* Extracts only the `canonical/` folder from the Liputan6 `.tar.gz` archive into a temporary directory.
* Deletes any previously sampled data folder to ensure a clean slate.
* Samples **25,000 files total** from the full Liputan6 canonical split and distributes them:
  - **Train:** 70% → 17,500 files
  - **Dev:** 10% → 2,500 files
  - **Test:** 20% → 5,000 files
* Files are randomly sampled per split and copied to the project's `liputan6_data/canonical/` directory.

## 3) Data Sanity Check
* Inspects a single JSON sample from the training directory.
* Verifies the dataset schema: keys `id`, `url`, `clean_article`, `clean_summary`, `extractive_summary`.
* `clean_article` and `clean_summary` are stored as nested lists of sentences (each sentence is a list of tokens).

## 4) Load, Structure, and N-gram EDA
* Parses all sampled JSON files into Pandas DataFrames (`df_train`, `df_dev`, `df_test`).
* Flattens nested sentence lists into continuous text strings for both `article` and `summary` columns.
* **N-gram Exploratory Data Analysis:**
  - Computes the ratio of summaries containing repeated n-grams (for n = 2, 3, 4).
  - Lists the top 15 most frequent n-grams in training summaries.
  - Uses a heuristic (smallest n where repeated n-gram ratio ≤ 5%) to recommend `no_repeat_ngram_size` for generation.
  - This EDA-driven value is used instead of a hardcoded default, preventing overly aggressive n-gram blocking that would suppress natural Indonesian bigram repetition.
* **Novel N-gram Ratio on Reference Summaries:**
  - Measures how much of each human summary uses n-grams that do not appear in the source article.
  - Serves as a simple proxy for dataset abstractiveness.
  - This helps distinguish between repetition inside summaries and genuine abstraction relative to the source article.

## 5) Tokenization and Preprocessing
* Converts Pandas DataFrames into a Hugging Face `DatasetDict` with splits: `train`, `validation`, `test`.
* Loads the pre-trained tokenizer from the model checkpoint (`cahya/bert2bert-indonesian-summarization`).
* Tokenizes inputs (articles, `max_length=512`) and targets (summaries, `max_length=128`) with `padding="max_length"` and `truncation=True`.
* Replaces padding token IDs in labels with `-100` so they are ignored by the loss function.

## 6) Extractive Baseline Evaluation (Comparable Protocol)
* Demonstrates ROUGE scoring with a single example using `rouge_score.RougeScorer` (`use_stemmer=False` to match the paper's `pyrouge` protocol).
* Evaluates the **dataset-provided extractive summaries** on a 250-sample test subset (fixed seed=42):
  - Reconstructs extractive summary text from `extractive_summary` index pointers into `clean_article` sentences.
  - Computes ROUGE-1/2/L precision, recall, and F1.
  - Reports stats on both 0–1 and percentage scales for direct comparison with Section 11.

## 7) Model and Trainer Setup
* Loads the `EncoderDecoderModel` from the cahya checkpoint.
* **Generation config (paper-matching, fixed):**
  - `do_sample=False` — pure beam search for deterministic, high-quality output (the original cahya demo used `do_sample=True` + `num_beams=10`, which conflicted).
  - `num_beams=5` — matches the paper's BertExtAbs evaluation protocol.
  - `max_length=150` — matches the paper's generation length (was 80, too short).
  - `min_length=15` — avoids forcing padding in short summaries.
  - `no_repeat_ngram_size` — set from EDA (typically 3), less restrictive than the original 2.
  - `length_penalty=0.6` — encourages concise summaries.
  - No `repetition_penalty` — the original 2.5 was too aggressive for Indonesian, suppressing common function words.
* Sets decoder special token IDs (`cls_token_id`, `sep_token_id`, `pad_token_id`).
* **ROUGE metric** uses `use_stemmer=False` to match the paper's `pyrouge` evaluation (no stemming).
* **Training arguments:**
  - `learning_rate=2e-5`, `warmup_steps=500`, `weight_decay=0.01`
  - `per_device_train_batch_size=16`, `gradient_accumulation_steps=2` (effective batch = 32)
  - `num_train_epochs=8`, `predict_with_generate=True`
  - `generation_max_length=150`, `generation_num_beams=5` (matches generation config)
  - `load_best_model_at_end=True`, `metric_for_best_model="eval_rougeL"`
  - `fp16=True` for mixed-precision training
  - `EarlyStoppingCallback(early_stopping_patience=3)`

## 8) Pre-fine-tuning Baseline Evaluation
* Draws a reproducible **250-sample subset** from the test set (`random_state=42`).
* Generates summaries using the **pretrained model as-is** (weights unchanged from HuggingFace download).
* Uses the same `generate_summary()` function with pure beam search decoding.
* Computes ROUGE-1/2/L (precision, recall, F1) against human reference summaries.
* Stores results as `pre_ft_scores` for the downstream comparison table.
* **Purpose:** Establishes a "before" baseline to quantify the benefit of fine-tuning.

## 9) Fine-tuning
* Runs `trainer.train()` on the 17,500-sample training split.
* The model's weights are updated via backpropagation over 8 epochs (with early stopping patience of 3).
* Saves the fine-tuned model and tokenizer to `PROJECT_ROOT / bert2bert-indonesian-finetuned`.

## 10) Post-fine-tuning Evaluation
Two evaluation approaches:
1. **Official Trainer evaluation** — `trainer.evaluate()` on the full 5,000-sample tokenized test split using `predict_with_generate=True` and the `compute_metrics` function.
2. **Direct inference ROUGE** — runs `model.generate()` on the **same 250-sample baseline subset** from Section 8, using the same `generate_summary()` function. This ensures a direct before/after comparison with identical test articles and decoding strategy.
* Stores results as `post_ft_scores` for the downstream comparison table.
* **Novel N-gram Ratio on Generated Summaries:**
  - Compares reference summaries, extractive baseline summaries, pre-fine-tuning outputs, and post-fine-tuning outputs on the same 250-sample subset.
  - Measures the percentage of summary n-grams not found in the source article for n = 1, 2, 3, 4.
  - Helps explain cases where ROUGE improves but human-perceived quality does not.

## 11) Comparison Table & Qualitative Demo

### 11a) ROUGE Comparison Table
* Compares our model (pre- and post-fine-tuning) against published baselines from the AACL 2020 paper:
  - **Paper baselines** (full ~10,972 canonical test samples): Lead-2, PTGen, BertAbs mBERT, BertExtAbs mBERT, BertAbs IndoBERT, BertExtAbs IndoBERT ★
  - **Our results** (same 250 sampled test articles): Extractive summary baseline, BERT2BERT pre-fine-tuning, BERT2BERT fine-tuned
* Also includes a dataset-provided extractive summary baseline computed on the same 250-sample subset.
* All ROUGE scores use `use_stemmer=False` for fair comparison with the paper's `pyrouge` results.

### 11b) Qualitative Examples
* Displays 10 sample articles side-by-side:
  - Article preview (first 300 characters)
  - Human reference summary
  - Pre-fine-tuning generated summary
  - Post-fine-tuning generated summary
* Allows visual inspection of summary quality, fluency, and factual accuracy beyond what ROUGE captures.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `use_stemmer=False` in all ROUGE scorers | Paper uses `pyrouge` which does not stem; `use_stemmer=True` inflates scores and makes comparison unfair |
| `do_sample=False` + `num_beams=5` | Pure beam search matches paper's evaluation; mixing sampling with beam search distorts scores |
| No `repetition_penalty` | Value of 2.5 was too aggressive for Indonesian, suppressing common function words after first use |
| `no_repeat_ngram_size` from EDA | EDA shows repeated bigrams are natural in Indonesian summaries; blocking all bigrams is too restrictive |
| `max_length=150` | Paper generates up to 150 tokens; 80 was cutting off good summaries |
| `gradient_accumulation_steps=2` | Effective batch of 32 with per-device batch of 16 — more stable than batch_size=32 directly |
| 250-sample test subset | Reproducible subset (seed=42) used consistently across Sections 6, 8, 10, and 11 for fair comparison |
| `EarlyStoppingCallback(patience=3)` | Prevents overfitting by stopping training if `eval_rougeL` doesn't improve for 3 consecutive evaluation epochs |
---

## Conclusion

The notebook shows that fine-tuning `cahya/bert2bert-indonesian-summarization` on a local Liputan6 subset improves summarization quality, but the improvement is still limited. On the same 250-sample test subset, ROUGE increases from **26.42 / 11.62 / 22.39** before fine-tuning to **29.54 / 13.02 / 24.99** after fine-tuning for ROUGE-1 / ROUGE-2 / ROUGE-L. This means the model does learn from the task-specific data, but its generated summaries are still substantially weaker than stronger baselines.

The clearest evidence is the comparison against both the extractive baseline and the published Liputan6 paper results. The dataset-provided extractive summaries reach **50.42 / 29.23 / 40.55** on the same sampled subset, while the AACL 2020 paper reports **41.08 / 22.85 / 38.01** for **BertExtAbs IndoBERT** on the full canonical test set. In other words, our fine-tuned BERT2BERT model remains well below both an extractive reference baseline and the paper's stronger IndoBERT-based abstractive system.

One important reason is scale. This notebook trains on only **17,500** training examples, whereas the paper uses the full canonical training split of **193,883** articles, with **10,972** examples each for validation and test. Because of this large gap in training data, the results here should be interpreted more as a low-resource local experiment than as a direct reproduction of the paper's benchmark.

The novel n-gram analysis adds an important nuance. On the 250-sample evaluation subset, the pre-fine-tuning model is more novel than the post-fine-tuning model across all n-gram levels: **Novel-1** drops from **22.25** to **20.44**, **Novel-2** from **45.89** to **42.92**, **Novel-3** from **61.51** to **58.91**, and **Novel-4** from **72.12** to **70.75** after fine-tuning. This suggests that fine-tuning makes the model slightly more extractive, likely increasing source overlap and helping ROUGE, but not always improving human-perceived summary quality.

The reference summaries show another useful pattern: **Novel-1 = 15.12**, **Novel-2 = 51.06**, **Novel-3 = 70.26**, and **Novel-4 = 81.33**. This means the human summaries still reuse many source words, but recombine them into new multi-word phrases. By contrast, the extractive baseline stays near zero on all novelty measures, confirming that the metric behaves as expected.

Taken together, the qualitative examples and the novelty analysis support the same conclusion. Although some generated summaries become more concise after fine-tuning, many still show incomplete coverage, awkward phrasing, dropped entities, or loss of focus. So the most balanced conclusion is that the encoder-decoder BERT2BERT setup is not inherently invalid for Indonesian summarization, but in this project configuration it is not yet strong enough to produce consistently competitive summaries for Liputan6 news articles.
