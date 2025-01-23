import json
from typing import List, Dict  
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from config import TrainingConfig
import random

# Project manager class for entire trainig data
class DataPreprocessor:
    """
    Class to handle preprocessing of the dreams dataset.
    """
    def __init__(self, tokenizer_name: str, config: TrainingConfig):
        self.config = config 
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_and_verify_data(self, file_path: str) -> List[Dict]:
        """Load and verify the raw JSON data."""
        print(f"📦 Loading raw data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)

        processed_data = []
        for item in raw_data:
            if self.config.source_input_field not in item or self.config.source_target_field not in item:
                raise ValueError(f"\n☠ Missing {self.config.source_input_field} or {self.config.source_target_field} field in item: {item}")

            processed_item = {
                'input': item[self.config.source_input_field],
                'target': item[self.config.source_target_field]
            }

            if not processed_item['input'] or not processed_item['target']:
                raise ValueError(f"\n☠ Empty input or target in item: {processed_item}")

            processed_data.append(processed_item)

        print(f"\n🗂  Successfully loaded and verified {len(processed_data)} examples.")
        return processed_data

    def prepare_data(self, data: List[Dict], train_ratio: float) -> (Dataset, Dataset):
        """Split data into training and validation datasets with shuffling."""
        print("\n🪚 Splitting dataset...")
        total_size = len(data)
        train_size = int(train_ratio * total_size)

        # Shuffles data
        # Use FIXED seed for reproducability 
        random.seed(42)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)

        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:]

        # Create datasets
        print("\n⚙ Creating datasets...")
        train_dataset = DreamAnalysisDataset(train_data, self.tokenizer, self.config)
        val_dataset = DreamAnalysisDataset(val_data, self.tokenizer, self.config)

        return train_dataset, val_dataset

    def save_datasets(self, train_dataset: Dataset, val_dataset: Dataset, train_path: str, val_path: str):
        """Save the processed datasets as tensors."""
        print("\n🧬 Converting to tensors...")
        train_tensors = [train_dataset[i] for i in range(len(train_dataset))]
        val_tensors = [val_dataset[i] for i in range(len(val_dataset))]

        print(f"\n💾 Saving training dataset to {train_path}...")
        torch.save(train_tensors, train_path)
        print(f"\n💾 Saving validation dataset to {val_path}...")
        torch.save(val_tensors, val_path)
        print("\n🐙 Datasets saved successfully.")

# Inherits from PyTorch Dataset class
# Converts each entry to tokens and formats. 
class DreamAnalysisDataset(Dataset):
    """
    Dataset class for dream analysis task.
    """

    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, config: TrainingConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_input_length 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        # Prepend system prompt to all input data
        input_text = self.config.input_prompt_template.format(item['input'])
        target_text = item['target'].strip()

        # Encoding inputs to tensors
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",    # Pads to max_length
            truncation=True,         # Cuts to max_length
            return_tensors="pt",     # Returns as PyTorch tensors
        )
        # Encoding targets to tensors
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Returns dictionary of tensors
        return {
            # Tokens for input
            'input_ids': input_encoding['input_ids'].squeeze(0),
            # Defines padding tokens (matches input id only)
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            # Tokens for target
            'labels': target_encoding['input_ids'].squeeze(0),
        }

# Safety check - Ensures no empty tokens
def validate_tokenized_data(dataset):
    """
    Ensures tokenized data doesn't contain empty sequences.
    """
    for idx, item in enumerate(dataset):
        if item['input_ids'].numel() == 0:
            raise ValueError(f"Empty tokenized input at index {idx}")
        if item['labels'].numel() == 0:
            raise ValueError(f"Empty tokenized target at index {idx}")
