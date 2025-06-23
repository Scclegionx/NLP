import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class NLPModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base",
            num_labels=6,  # Số lượng intent
            output_attentions=False  # Tắt để tránh warning
        )
        self.intent_map = {
            "check_battery": 0,
            "make_call": 1,
            "send_message": 2,
            "set_alarm": 3,
            "search_info": 4,
            "control_device": 5
        }
        self.training_data = None

    def load_training_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.training_data = json.load(f)
        
        texts = []
        labels = []
        
        # Tạo nhiều dữ liệu training hơn
        for intent in self.training_data['intents']:
            for example in intent['examples']:
                base_text = example['text']
                texts.append(base_text)
                labels.append(self.intent_map[intent['tag']])
                
                # Thêm các biến thể với dấu câu
                texts.append(base_text + ".")
                labels.append(self.intent_map[intent['tag']])
                texts.append(base_text + "!")
                labels.append(self.intent_map[intent['tag']])
                texts.append(base_text + "?")
                labels.append(self.intent_map[intent['tag']])
                
                # Thêm các biến thể với từ đồng nghĩa
                synonyms = {
                    "gọi": ["kết nối", "liên lạc", "gọi điện"],
                    "nhắn tin": ["gửi tin nhắn", "soạn tin", "nhắn"],
                    "tìm kiếm": ["tìm", "tìm kiếm thông tin", "tìm hiểu"],
                    "đặt": ["hẹn", "cài đặt", "thiết lập"],
                    "bật": ["kích hoạt", "mở", "khởi động"],
                    "tắt": ["dừng", "đóng", "ngắt"],
                    "kiểm tra": ["xem", "kiểm", "thử"],
                    "pin": ["pin điện thoại", "pin máy", "pin còn"]
                }
                
                for original, synonym_list in synonyms.items():
                    if original in base_text:
                        for synonym in synonym_list:
                            new_text = base_text.replace(original, synonym)
                            texts.append(new_text)
                            labels.append(self.intent_map[intent['tag']])
                
                # Thêm các biến thể với thay đổi thứ tự từ
                words = base_text.split()
                if len(words) > 3:
                    # Hoán đổi vị trí một số từ
                    for i in range(len(words) - 1):
                        new_words = words.copy()
                        new_words[i], new_words[i+1] = new_words[i+1], new_words[i]
                        new_text = " ".join(new_words)
                        texts.append(new_text)
                        labels.append(self.intent_map[intent['tag']])
                
                # Thêm các biến thể với từ thêm vào
                add_words = ["xin", "vui lòng", "hãy", "có thể"]
                for add_word in add_words:
                    new_text = add_word + " " + base_text
                    texts.append(new_text)
                    labels.append(self.intent_map[intent['tag']])
        
        return texts, labels

    def train(self, texts, labels, batch_size=4, epochs=15):  # Giảm batch_size, tăng epochs
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = IntentDataset(val_texts, val_labels, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Sử dụng AdamW với learning rate thấp hơn
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)  # Giảm learning rate
        
        # Thêm learning rate scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps * 0.1,  # 10% warmup
            num_training_steps=total_steps
        )

        best_val_loss = float('inf')
        patience = 5  # Tăng patience
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_train_loss += loss.item()
                train_steps += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                progress_bar.set_postfix({'loss': loss.item()})

            avg_train_loss = total_train_loss / train_steps

            # Validation
            self.model.eval()
            total_val_loss = 0
            val_steps = 0
            correct_predictions = 0
            total_predictions = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    total_val_loss += loss.item()
                    val_steps += 1

                    predictions = torch.argmax(outputs.logits, dim=1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)

            avg_val_loss = total_val_loss / val_steps
            accuracy = correct_predictions / total_predictions

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Average validation loss: {avg_val_loss:.4f}")
            print(f"Validation accuracy: {accuracy:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Lưu model tốt nhất
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Load model tốt nhất
        self.model.load_state_dict(torch.load('best_model.pt'))

    def extract_entities_and_values(self, text, intent_tag):
        entities = {}
        values = {}
        
        # Trích xuất entities dựa trên context và từ khóa
        text_lower = text.lower()
        
        # Trích xuất contact_name
        if intent_tag in ["make_call", "send_message"]:
            contact_keywords = ["cho", "đến", "gửi cho", "nhắn cho"]
            for keyword in contact_keywords:
                if keyword in text_lower:
                    # Tìm từ sau keyword
                    parts = text_lower.split(keyword)
                    if len(parts) > 1:
                        after_keyword = parts[1].strip()
                        # Tìm contact_name trong phần sau keyword
                        for contact in self.training_data['entity_types']['contact_name']:
                            if contact in after_keyword:
                                entities["contact_name"] = contact
                                break
                        break
        
        # Trích xuất phone_number
        if intent_tag == "make_call":
            phone_pattern = r'\b\d{10,11}\b'
            phone_matches = re.findall(phone_pattern, text)
            if phone_matches:
                entities["phone_number"] = phone_matches[0]
        
        # Trích xuất time
        if intent_tag == "set_alarm":
            time_pattern = r'\b\d{1,2}:\d{2}\b'
            time_matches = re.findall(time_pattern, text)
            if time_matches:
                entities["time"] = time_matches[0]
        
        # Trích xuất device_name và action
        if intent_tag == "control_device":
            device_keywords = ["đèn", "quạt", "điều hòa", "loa", "tivi"]
            for device in device_keywords:
                if device in text_lower:
                    entities["device_name"] = device
                    break
            
            if "bật" in text_lower:
                values["action"] = "bật"
            elif "tắt" in text_lower:
                values["action"] = "tắt"
        
        # Trích xuất message_content
        if intent_tag == "send_message":
            msg_patterns = ["là", "với nội dung", "nội dung"]
            for pattern in msg_patterns:
                if pattern in text_lower:
                    parts = text_lower.split(pattern)
                    if len(parts) > 1:
                        value = parts[1].strip()
                        # Loại bỏ các từ thừa sau nội dung tin nhắn
                        for word in ["cho", "đến", "gửi"]:
                            if word in value:
                                value = value.split(word)[0].strip()
                        values["message_content"] = value
                        break
        
        # Trích xuất search_query
        if intent_tag == "search_info":
            search_keywords = ["thời tiết", "giá vàng", "tin tức", "lịch chiếu phim"]
            for keyword in search_keywords:
                if keyword in text_lower:
                    values["search_query"] = keyword
                    break

        return entities, values

    def predict(self, text):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        command = list(self.intent_map.keys())[predicted_class]
        entities, values = self.extract_entities_and_values(text, command)

        return {
            "command": command,
            "entities": entities,
            "values": values,
            "confidence": predictions[0][predicted_class].item()
        }

# Example usage
if __name__ == "__main__":
    nlp_model = NLPModel()
    texts, labels = nlp_model.load_training_data('base/training_data.json')
    print(f"Total training examples: {len(texts)}")
    nlp_model.train(texts, labels)
    
    # Test predictions
    test_cases = [
        "Gọi điện cho mẹ",
        "Nhắn tin cho bố là xin chào",
        "Đặt báo thức lúc 7:00",
        "Bật đèn",
        "Tìm kiếm thời tiết",
        "Gửi tin nhắn chúc ngủ ngon cho em",
        "Tắt điều hòa",
        "Gọi số 0123456789"
    ]
    
    for test_text in test_cases:
        result = nlp_model.predict(test_text)
        print(f"\nInput: {test_text}")
        print(f"Command: {result['command']}")
        print(f"Entities: {result['entities']}")
        print(f"Values: {result['values']}")
        print(f"Confidence: {result['confidence']:.2f}") 