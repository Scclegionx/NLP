import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from datetime import datetime
import os
# import wandb

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

class PhoBERTTrainer:
    def __init__(self, model_name="vinai/phobert-base", max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def load_dataset(self, dataset_path):
        """Load dataset from JSON file"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        train_data = data['train']
        test_data = data['test']
        
        return train_data, test_data
    
    def prepare_data(self, data):
        """Prepare data for training"""
        texts = [item['text'] for item in data]
        intents = [item['intent'] for item in data]
        
        # Encode labels
        encoded_intents = self.label_encoder.fit_transform(intents)
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return tokenized, torch.tensor(encoded_intents)
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_data, test_data, output_dir="./phobert_model"):
        """Train the PhoBERT model"""
        print("Preparing training data...")
        train_tokenized, train_labels = self.prepare_data(train_data)
        
        print("Preparing test data...")
        test_tokenized, test_labels = self.prepare_data(test_data)
        
        # Create model
        num_labels = len(self.label_encoder.classes_)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        # Create datasets
        class IntentDataset(torch.utils.data.Dataset):
            def __init__(self, input_ids, attention_mask, labels):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels
            
            def __len__(self):
                return len(self.input_ids)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx]
                }
        
        train_dataset = IntentDataset(
            train_tokenized['input_ids'],
            train_tokenized['attention_mask'],
            train_labels
        )
        
        test_dataset = IntentDataset(
            test_tokenized['input_ids'],
            test_tokenized['attention_mask'],
            test_labels
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=20,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=50,
            weight_decay=0.01,
            learning_rate=2e-5,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=25,
            save_strategy="steps",
            save_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to=None,
            disable_tqdm=False,
            logging_first_step=True,
            logging_strategy="steps",
            fp16=False,
            dataloader_pin_memory=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label encoder
        import joblib
        joblib.dump(self.label_encoder, f'{output_dir}/label_encoder.pkl')
        
        # Save intent mapping
        intent_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        with open(f'{output_dir}/intent_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(intent_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to {output_dir}")
        return trainer

def main():
    # Initialize trainer
    trainer = PhoBERTTrainer()
    
    # Load dataset
    train_data, test_data = trainer.load_dataset("dataset.json")
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Train model
    trainer.train(train_data, test_data)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
