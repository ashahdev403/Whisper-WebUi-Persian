#!/usr/bin/env python3
"""
Whisper Persian Fine-tuning Script
"""

import yaml
import os
import sys

from datasets import load_dataset, DatasetDict, Audio
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)


# Resolve the config next to this script, so the trainer can be started from the repository root
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_datasets(config):
    """Load and prepare datasets"""
    dataset_config = config['dataset']
    
    datasets = DatasetDict()
    datasets["train"] = load_dataset(dataset_config['name'], split=dataset_config['train_split'])
    datasets["test"] = load_dataset(dataset_config['name'], split=dataset_config['test_split'])
    
    # Cast audio column to target sample rate
    datasets = datasets.cast_column(
        dataset_config['audio_column'], 
        Audio(sampling_rate=dataset_config['target_sample_rate'])
    )
    
    return datasets


def setup_model_and_processor(config):
    """Setup model, feature extractor, tokenizer and processor"""
    model_config = config['model']
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_config['name'])
    tokenizer = WhisperTokenizer.from_pretrained(
        model_config['name'], 
        language=model_config['language'], 
        task=model_config['task']
    )
    processor = WhisperProcessor.from_pretrained(
        model_config['name'], 
        language=model_config['language'], 
        task=model_config['task']
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(model_config['name'])
    
    # Configure generation settings
    model.generation_config.language = model_config['language'].lower()
    model.generation_config.task = model_config['task']
    model.generation_config.forced_decoder_ids = None
    
    return model, processor, feature_extractor, tokenizer


def prepare_dataset(batch, feature_extractor, tokenizer, config):
    """Prepare dataset batch for training"""
    audio = batch[config['dataset']['audio_column']]
    
    # Compute log-Mel input features
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode target text to label ids
    batch["labels"] = tokenizer(batch[config['dataset']['text_column']]).input_ids
    return batch


def process_datasets(datasets, feature_extractor, tokenizer, config):
    """Process datasets for training"""
    def prepare_batch(batch):
        return prepare_dataset(batch, feature_extractor, tokenizer, config)
    
    processed_datasets = datasets.map(
        prepare_batch,
        remove_columns=datasets["train"].column_names,
        num_proc=config['processing']['num_proc']
    )
    
    return processed_datasets


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def create_compute_metrics(tokenizer):
    """Create compute_metrics function with tokenizer"""
    metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad_token_id for labels
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        
        # Decode predictions and labels
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = metric.compute(predictions=pred_str, references=label_str)
        return {"wer": float(wer * 100)}
    
    return compute_metrics


def setup_training_args(config):
    """Setup training arguments from config"""
    train_config = config['training']
    eval_config = config['evaluation']
    output_config = config['output']
    logging_config = config['logging']
    
    # Calculate dynamic names based on config
    lr = train_config['learning_rate']
    batch_size = train_config['per_device_train_batch_size']
    grad_acc = train_config['gradient_accumulation_steps']
    total_batch = batch_size * grad_acc
    
    run_name = f"{output_config['run_name']}-{lr}-batch{total_batch}"
    output_dir = f"{output_config['output_dir']}-{lr}-batch{total_batch}"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        warmup_steps=train_config['warmup_steps'],
        max_steps=train_config['max_steps'],
        gradient_checkpointing=train_config['gradient_checkpointing'],
        bf16=train_config['bf16'],
        fp16=train_config['fp16'],
        eval_strategy=eval_config['eval_strategy'],
        generation_max_length=eval_config['generation_max_length'],
        predict_with_generate=eval_config['predict_with_generate'],
        save_steps=eval_config['save_steps'],
        eval_steps=eval_config['eval_steps'],
        logging_steps=eval_config['logging_steps'],
        report_to=logging_config['report_to'],
        run_name=run_name,
        load_best_model_at_end=eval_config['load_best_model_at_end'],
        metric_for_best_model=eval_config['metric_for_best_model'],
        greater_is_better=eval_config['greater_is_better'],
        push_to_hub=output_config['push_to_hub'],
    )
    
    return training_args


def train_model(config):
    """Main training function"""
    print("Loading datasets...")
    datasets = load_datasets(config)
    
    print("Setting up model and processor...")
    model, processor, feature_extractor, tokenizer = setup_model_and_processor(config)
    
    print("Processing datasets...")
    processed_datasets = process_datasets(datasets, feature_extractor, tokenizer, config)
    
    print("Setting up training arguments...")
    training_args = setup_training_args(config)
    
    print("Setting up data collator...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    print("Setting up compute metrics...")
    compute_metrics = create_compute_metrics(tokenizer)
    
    print("Configuring model for training...")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    print("Creating trainer...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    
    print("Saving processor...")
    processor.save_pretrained(training_args.output_dir)
    
    print("Starting training...")
    trainer.train()
    
    print("Pushing to hub...")
    hub_config = config['hub']
    trainer.push_to_hub(
        dataset_tags=hub_config['dataset_tags'],
        dataset=hub_config['dataset_name'],
        dataset_args=hub_config['dataset_args'],
        language=hub_config['language_code'],
        model_name=hub_config['model_name'],
        finetuned_from=hub_config['finetuned_from'],
        tasks=hub_config['tasks'],
    )
    
    print("Training completed!")


def main():
    """Main function"""
    config = load_config()
    train_model(config)


if __name__ == "__main__":
    main()