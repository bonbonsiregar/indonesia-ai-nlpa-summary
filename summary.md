# Comprehensive Notebook Summary (Project_2_(new).ipynb)

## 1) Environment Setup
* Mounts Google Drive to access the dataset.
* Installs required libraries for the project: `transformers`, `datasets`, `evaluate`, and `rouge_score`.
* Verifies GPU availability and status using `!nvidia-smi`.

## 2) Data Extraction and Sampling
* Extracts the `canonical` folder from the Liputan6 dataset archive (`.tar.gz`).
* Cleans up any existing extraction folders to ensure a fresh start.
* Samples a total of 25,000 files and distributes them into three splits: Train (70%, 17,500 files), Dev (10%, 2,500 files), and Test (20%, 5,000 files).

## 3) Data Sanity Check
* Inspects a single JSON sample file from the training directory.
* Verifies the dataset's schema (keys like `id`, `clean_article`, `clean_summary`) and structure to ensure correct data extraction.

## 4) Load and Structure Dataset
* Parses the sampled JSON files and loads them into Pandas DataFrames for train, dev, and test splits.
* Joins nested lists of sentences into continuous text strings for both articles and their corresponding summaries.

## 5) Tokenization and Preprocessing
* Converts the Pandas DataFrames into a Hugging Face `DatasetDict`.
* Loads the pre-trained tokenizer for `cahya/bert2bert-indonesian-summarization`.
* Applies a preprocessing function to tokenize inputs (articles, max length 512) and targets (summaries, max length 128), and replaces padding token IDs with `-100` so they are ignored by the loss function.

## 6) Evaluate ROUGE-1, ROUGE-2, AND ROUGE-L scores
* Sets up the `rouge_scorer` to calculate precision, recall, and f-measure.
* Runs a baseline evaluation on the training set to demonstrate the low accuracy of the base model before fine-tuning, corroborating the reference paper's findings.

## 7) Model and Trainer Setup
* Initializes the `EncoderDecoderModel` using the `cahya/bert2bert-indonesian-summarization` checkpoint.
* Configures sequence-to-sequence generation parameters (e.g., beam search, length penalty, early stopping, and n-gram repetition constraints).
* Sets up the `Seq2SeqTrainer` with 10 epochs, a data collator, and a custom `compute_metrics` function to calculate ROUGE scores during training.

## 8) Pre-fine-tuning Baseline Evaluation
* Selects a reproducible subset of 200 samples from the test set.
* Generates summaries using the base (pre-fine-tuned) model.
* Computes baseline ROUGE scores to establish a benchmark for quantifying the benefits of the fine-tuning process.

## 9) Fine-tuning
* Executes the fine-tuning process using the `trainer.train()` method on the 17,500-sample Indonesian training dataset.
* Saves the fully fine-tuned model and tokenizer back to Google Drive for persistence and future inference.

## 10) Post-fine-tuning Evaluation
* Evaluates the model using two approaches:
  1. Official `trainer.evaluate()` on the full 5,000-sample tokenized test split.
  2. Direct inference using `model.generate()` on the exact same 200-sample baseline subset used in Section 8 to allow a direct "before-and-after" comparison.

## 11) Comparison Table & Qualitative Demo
* **Quantitative:** Builds a Pandas DataFrame comparing the ROUGE scores of the pre-fine-tuned and fine-tuned models against the published baselines from the AACL 2020 paper.
* **Qualitative:** Displays 5 sample articles side-by-side, showcasing the original text, the human reference summary, the pre-fine-tuning output, and the post-fine-tuning output for visual inspection.