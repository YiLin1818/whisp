import argparse, json, os, yaml
import torch
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model
from evaluate import load as load_eval

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def read_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def build_dataset(rows, processor):
    items = []
    for r in rows:
        audio = read_audio(r["audio_path"])
        inputs = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        labels = processor.tokenizer(r["transcript"]).input_ids
        items.append({"input_features": inputs, "labels": labels})
    return items

def compute_metrics_factory(processor):
    wer = load_eval("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer_val = wer.compute(predictions=pred_str, references=label_str)
        return {"wer": wer_val, "acc_proxy": 1.0 - wer_val}
    return compute_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    processor = WhisperProcessor.from_pretrained(cfg["model_name"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model_name"])
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # LoRA
    lcfg = cfg["lora"]
    peft_cfg = LoraConfig(
        r=int(lcfg["r"]),
        lora_alpha=int(lcfg["alpha"]),
        lora_dropout=float(lcfg["dropout"]),
        bias="none",
        target_modules=list(lcfg["target_modules"]),
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, peft_cfg)

    train_rows = load_jsonl(cfg["dataset"]["train_manifest"])
    val_rows = load_jsonl(cfg["dataset"]["val_manifest"])

    train_ds = build_dataset(train_rows, processor)
    val_ds = build_dataset(val_rows, processor)

    tcfg = cfg["train"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=tcfg["output_dir"],
        learning_rate=float(tcfg["learning_rate"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        num_train_epochs=float(tcfg["num_train_epochs"]),
        warmup_steps=int(tcfg["warmup_steps"]),
        fp16=bool(tcfg.get("fp16", False)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        evaluation_strategy=tcfg.get("eval_strategy", "epoch"),
        eval_steps=tcfg.get("eval_steps", None),
        save_steps=tcfg.get("save_steps", None),
        logging_steps=int(tcfg.get("logging_steps", 50)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        predict_with_generate=bool(tcfg.get("predict_with_generate", True)),
        generation_num_beams=int(tcfg.get("generation_num_beams", 5)),
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    collator = DataCollatorSpeechSeq2Seq(processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics_factory(processor),
    )

    trainer.train()
    trainer.save_model(os.path.join(tcfg["output_dir"], "best"))
    processor.save_pretrained(os.path.join(tcfg["output_dir"], "best"))

if __name__ == "__main__":
    main()
