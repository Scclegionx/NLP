from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json
import torch

# Load dữ liệu
with open("nlp_command_dataset.json", encoding="utf-8") as f:
    raw_data = json.load(f)

# Tiền xử lý dữ liệu
train_data = []
for item in raw_data:
    input_text = item["input"]
    # Format output rõ ràng hơn với entity và value
    entity_text = ', '.join(item['entities']) if item['entities'] else 'none'
    value_text = ', '.join(item['values']) if item['values'] else 'none'
    output_text = f"command: {item['command']} | ent: {entity_text} | val: {value_text}"
    train_data.append({"input": input_text, "output": output_text})

dataset = Dataset.from_list(train_data)

# Load model & tokenizer
model_name = "VietAI/vit5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess(example):
    input_enc = tokenizer("trich xuat: " + example["input"], truncation=True, padding="max_length", max_length=128)
    target_enc = tokenizer(example["output"], truncation=True, padding="max_length", max_length=64)
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": target_enc["input_ids"]
    }

encoded_dataset = dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir="./nlp-command-model",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./nlp-command-model")
tokenizer.save_pretrained("./nlp-command-model")
