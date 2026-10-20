
"""
FactIndic: English and Hindi Summarization
==========================================

Hindi:
    Pretrained ai4bharat/IndicBART-XLSum; no additional training.

English:
    Fine-tune ai4bharat/IndicBART on ILSUM English article-summary pairs.

Supported commands:
    python factindic_summarization.py hindi --text "हिंदी में समाचार लेख..."
    python factindic_summarization.py english --model-path outputs/indicbart-en --text "English article..."
    python factindic_summarization.py train-en --train-csv Data/Raw/train.csv --validation-csv Data/Raw/val.csv
    python factindic_summarization.py evaluate --language hi --test-csv Data/Raw/hindi_test.csv
    python factindic_summarization.py evaluate --language en --model-path outputs/indicbart-en --test-csv Data/Raw/english_test.csv

Expected CSV columns:
    Article, Summary
    ILSUM may also contain Heading; Summary is preferred as the target.

Install:
    python -m pip install torch transformers sentencepiece accelerate rouge-score
"""

import argparse
import csv
import inspect
import re
import unicodedata
from pathlib import Path
import subprocess

import torch
from torch.utils.data import Dataset
from transformers import (
    AlbertTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_MODEL = "ai4bharat/IndicBART"
HINDI_MODEL = "ai4bharat/IndicBART-XLSum"
DEFAULT_ENGLISH_OUTPUT = "outputs/indicbart-en"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

LANGUAGE_TOKENS = {
    "en": "<2en>",
    "hi": "<2hi>",
}


# ------------------------------------------------------------
# Text preprocessing
# ------------------------------------------------------------

def normalize_text(text):
    """Normalize Unicode and whitespace without removing meaning."""
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = "".join(
        char for char in text
        if char in "\n\r\t" or unicodedata.category(char) not in {"Cc", "Cf"}
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_generated_text(text):
    """Remove visible language and sentence-control tokens."""
    text = re.sub(r"<2[a-z]{2,3}>", "", text)
    text = text.replace("</s>", "").replace("<s>", "").replace("<pad>", "")
    return normalize_text(text)


# ------------------------------------------------------------
# CSV loading and column detection
# ------------------------------------------------------------

def normalized_column_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_column(columns, candidates, description):
    """Find a CSV column without depending on capitalization."""
    lookup = {normalized_column_name(c): c for c in columns}

    for candidate in candidates:
        key = normalized_column_name(candidate)
        if key in lookup:
            return lookup[key]

    raise ValueError(
        f"Could not find the {description} column.\n"
        f"Available columns: {list(columns)}\n"
        f"Expected one of: {candidates}"
    )


def load_csv_records(
    csv_path,
    article_column=None,
    summary_column=None,
):
    """Load article-summary pairs from a CSV file."""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"No header row found in {path}")

        columns = reader.fieldnames

        if article_column:
            actual_article_column = find_column(
                columns, [article_column], "article"
            )
        else:
            actual_article_column = find_column(
                columns,
                ["Article", "article_text", "document", "document_text", "text"],
                "article",
            )

        if summary_column:
            actual_summary_column = find_column(
                columns, [summary_column], "summary"
            )
        else:
            actual_summary_column = find_column(
                columns,
                ["Summary", "summary_text", "abstract", "target",
                 "Heading", "Headline", "Title"],
                "summary",
            )

        records = []

        for row in reader:
            article = normalize_text(row.get(actual_article_column, ""))
            summary = normalize_text(row.get(actual_summary_column, ""))

            if article and summary:
                records.append({
                    "article": article,
                    "summary": summary,
                })

    if not records:
        raise ValueError(
            f"No valid article-summary pairs found in {path}. "
            "Check the CSV columns and contents."
        )

    print(f"Loaded {len(records):,} records from {path}")
    print(f"Article column: {actual_article_column}")
    print(f"Summary column: {actual_summary_column}")

    return records


# ------------------------------------------------------------
# Tokenizer and model helpers
# ------------------------------------------------------------

def load_tokenizer(model_name):
    """Load the SentencePiece-based tokenizer used by IndicBART."""
    return AlbertTokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        use_fast=False,
        keep_accents=True,
    )


def get_language_id(tokenizer, language):
    token = LANGUAGE_TOKENS[language]
    token_id = tokenizer.convert_tokens_to_ids(token)

    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Language token {token} was not found in this tokenizer."
        )

    return token_id



def configure_model_language(model, tokenizer, language):
    """Configure IndicBART-XLSum using its documented decoder start token."""
    get_language_id(tokenizer, language)  # Validate requested language.

    decoder_start_id = tokenizer._convert_token_to_id_with_added_voc("<2en>")

    model.config.decoder_start_token_id = decoder_start_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = decoder_start_id

    model.config.pad_token_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
    model.config.eos_token_id = tokenizer._convert_token_to_id_with_added_voc("</s>")

    return decoder_start_id



def load_summarization_model(model_name, language):
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    configure_model_language(model, tokenizer, language)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model: {model_name}")
    print(f"Device: {device}")

    return tokenizer, model, device


# ------------------------------------------------------------
# Summarization inference
# ------------------------------------------------------------


def summarize_text(
    text,
    tokenizer,
    model,
    device,
    language,
    max_input_length=MAX_INPUT_LENGTH,
    max_new_tokens=100,
    num_beams=4,
):
    """Generate a summary using IndicBART or IndicBART-XLSum."""
    text = normalize_text(text)

    if not text:
        raise ValueError("Input article cannot be empty.")

    if language not in LANGUAGE_TOKENS:
        raise ValueError(
            f"Unsupported language: {language}. Use 'en' or 'hi'."
        )

    # Keep the correct language token on the encoder input.
    language_token = LANGUAGE_TOKENS[language]
    formatted_input = f"{text} </s> {language_token}"

    encoded = tokenizer(
        formatted_input,
        add_special_tokens=False,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    # IndicBART-XLSum uses <2en> as its decoder start token,
    # including when generating Hindi summaries.
    decoder_start_id = tokenizer._convert_token_to_id_with_added_voc(
        "<2en>"
    )
    bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

    with torch.inference_mode():
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=True,
            num_beams=num_beams,
            max_length=max_new_tokens,
            min_length=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            pad_token_id=pad_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            decoder_start_token_id=decoder_start_id,
        )

    summary = tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return clean_generated_text(summary)


def run_hindi_inference(args):
    tokenizer, model, device = load_summarization_model(
        HINDI_MODEL, "hi"
    )

    summary = summarize_text(
        args.text,
        tokenizer,
        model,
        device,
        language="hi",
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    output_file = Path("hindi_summary.txt")

    output_file.write_text(
        "HINDI SUMMARIZATION - FACTINDIC\n"
        + "=" * 40
        + "\n\nORIGINAL ARTICLE:\n"
        + args.text
        + "\n\nGENERATED SUMMARY:\n"
        + summary
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSummary saved to: {output_file.resolve()}")

    # Open the result in Windows Notepad
    subprocess.Popen(["notepad.exe", str(output_file.resolve())])


def run_english_inference(args):
    model_path = args.model_path

    if not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"Fine-tuned English model not found: {model_path}\n"
            "Train the English model first using the train-en command."
        )

    tokenizer, model, device = load_summarization_model(
        model_path, "en"
    )

    summary = summarize_text(
        args.text,
        tokenizer,
        model,
        device,
        language="en",
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    print("\nEnglish summary:\n")
    print(summary)


# ------------------------------------------------------------
# English fine-tuning dataset
# ------------------------------------------------------------

class IndicBARTSummarizationDataset(Dataset):
    """
    Dataset for fine-tuning IndicBART on English article-summary pairs.

    Input:  article </s> <2en>
    Target: <2en> summary </s>
    """

    def __init__(
        self,
        records,
        tokenizer,
        language="en",
        max_input_length=MAX_INPUT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.language = language
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.language_token = LANGUAGE_TOKENS[language]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        article = normalize_text(record["article"])
        summary = normalize_text(record["summary"])

        source_text = f"{article} </s> {self.language_token}"
        target_text = f"{self.language_token} {summary} </s>"

        source = self.tokenizer(
            source_text,
            add_special_tokens=False,
            max_length=self.max_input_length,
            truncation=True,
        )

        target = self.tokenizer(
            target_text,
            add_special_tokens=False,
            max_length=self.max_target_length,
            truncation=True,
        )

        return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": target["input_ids"],
        }


# ------------------------------------------------------------
# English model training
# ------------------------------------------------------------

def train_english_model(args):
    train_records = load_csv_records(
        args.train_csv,
        args.article_column,
        args.summary_column,
    )

    validation_records = load_csv_records(
        args.validation_csv,
        args.article_column,
        args.summary_column,
    )

    print("\nLoading base IndicBART for English fine-tuning...")
    tokenizer = load_tokenizer(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    configure_model_language(model, tokenizer, "en")

    train_dataset = IndicBARTSummarizationDataset(
        train_records,
        tokenizer,
        language="en",
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )

    validation_dataset = IndicBARTSummarizationDataset(
        validation_records,
        tokenizer,
        language="en",
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support both newer and older Transformers argument names.
    training_parameters = inspect.signature(
        Seq2SeqTrainingArguments.__init__
    ).parameters

    eval_key = (
        "eval_strategy"
        if "eval_strategy" in training_parameters
        else "evaluation_strategy"
    )

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "dataloader_num_workers": 0,
        "predict_with_generate": False,
    }

    training_kwargs[eval_key] = "epoch"

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(**training_kwargs),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("\nStarting English fine-tuning...")
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(validation_dataset):,}")
    print(f"Output directory: {output_dir.resolve()}")

    trainer.train()

    print("\nSaving final model and tokenizer...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    validation_metrics = trainer.evaluate()
    print("\nValidation metrics:")
    for key, value in validation_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")

    print("\nEnglish fine-tuning completed.")
    print(f"Model saved to: {output_dir.resolve()}")


# ------------------------------------------------------------
# Evaluation on a held-out test CSV
# ------------------------------------------------------------

def evaluate_model(args):
    if args.language == "hi":
        model_name = HINDI_MODEL
    else:
        model_name = args.model_path

        if not Path(model_name).is_dir():
            raise FileNotFoundError(
                f"Fine-tuned English model not found: {model_name}"
            )

    records = load_csv_records(
        args.test_csv,
        args.article_column,
        args.summary_column,
    )

    tokenizer, model, device = load_summarization_model(
        model_name, args.language
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    references = []

    print(f"\nGenerating predictions for {len(records):,} examples...")

    for index, record in enumerate(records, start=1):
        prediction = summarize_text(
            record["article"],
            tokenizer,
            model,
            device,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )

        predictions.append(prediction)
        references.append(record["summary"])

        if index % 25 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)}")

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["article", "reference_summary", "predicted_summary"],
        )
        writer.writeheader()

        for record, reference, prediction in zip(
            records, references, predictions
        ):
            writer.writerow({
                "article": record["article"],
                "reference_summary": reference,
                "predicted_summary": prediction,
            })

    print(f"\nPredictions saved to: {output_path.resolve()}")

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Install rouge-score to calculate ROUGE metrics.")
        return

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=(args.language == "en"),
    )

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    valid_count = 0

    for reference, prediction in zip(references, predictions):
        if not reference.strip() or not prediction.strip():
            continue

        scores = scorer.score(reference, prediction)

        for metric in totals:
            totals[metric] += scores[metric].fmeasure

        valid_count += 1

    print("\nAverage ROUGE F1 scores:")
    if valid_count == 0:
        print("No valid predictions available for scoring.")
    else:
        for metric, total in totals.items():
            print(f"{metric.upper()}: {total / valid_count:.4f}")

    print(
        "\nNote: ROUGE measures lexical overlap, not factual correctness. "
        "FactIndic should evaluate factuality separately."
    )


# ------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="FactIndic English and Hindi summarization pipeline."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Hindi pretrained inference
    hindi_parser = subparsers.add_parser(
        "hindi",
        help="Summarize Hindi text using pretrained IndicBART-XLSum.",
    )
    hindi_parser.add_argument("--text", required=True)
    hindi_parser.add_argument("--max-new-tokens", type=int, default=100)
    hindi_parser.add_argument("--num-beams", type=int, default=4)
    hindi_parser.set_defaults(func=run_hindi_inference)

    # English inference
    english_parser = subparsers.add_parser(
        "english",
        help="Summarize English text using the fine-tuned model.",
    )
    english_parser.add_argument(
        "--model-path", default=DEFAULT_ENGLISH_OUTPUT
    )
    english_parser.add_argument("--text", required=True)
    english_parser.add_argument("--max-new-tokens", type=int, default=100)
    english_parser.add_argument("--num-beams", type=int, default=4)
    english_parser.set_defaults(func=run_english_inference)

    # English training
    train_parser = subparsers.add_parser(
        "train-en",
        help="Fine-tune base IndicBART on ILSUM English data.",
    )
    train_parser.add_argument("--train-csv", required=True)
    train_parser.add_argument("--validation-csv", required=True)
    train_parser.add_argument(
        "--output-dir", default=DEFAULT_ENGLISH_OUTPUT
    )
    train_parser.add_argument("--article-column", default=None)
    train_parser.add_argument("--summary-column", default=None)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4
    )
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument(
        "--max-input-length", type=int, default=MAX_INPUT_LENGTH
    )
    train_parser.add_argument(
        "--max-target-length", type=int, default=MAX_TARGET_LENGTH
    )
    train_parser.set_defaults(func=train_english_model)

    # Evaluation on test data
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Generate test predictions and calculate ROUGE scores.",
    )
    eval_parser.add_argument("--language", choices=["en", "hi"], required=True)
    eval_parser.add_argument("--test-csv", required=True)
    eval_parser.add_argument(
        "--model-path", default=DEFAULT_ENGLISH_OUTPUT
    )
    eval_parser.add_argument("--output-csv", default="Data/Evaluation/predictions.csv")
    eval_parser.add_argument("--article-column", default=None)
    eval_parser.add_argument("--summary-column", default=None)
    eval_parser.add_argument("--max-new-tokens", type=int, default=100)
    eval_parser.add_argument("--num-beams", type=int, default=4)
    eval_parser.set_defaults(func=evaluate_model)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as error:
        print(f"\nERROR: {error}")
        raise


if __name__ == "__main__":
    main()
