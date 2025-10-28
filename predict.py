import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime, timedelta
import re

class PhoBERTPredictor:
    def __init__(self, model_path="./phobert_model"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.label_encoder = joblib.load(f'{model_path}/label_encoder.pkl')
        
        with open(f'{model_path}/intent_mapping.json', 'r', encoding='utf-8') as f:
            self.intent_mapping = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def extract_entities(self, text, intent):
        """Extract entities based on intent"""
        entities = {
            "RECEIVER": "",
            "MESSAGE": "",
            "PLATFORM": "",
            "DEVICE": "",
            "DATE": ""
        }
        
        text_lower = text.lower()
        
        if intent in ["send-msg", "send-msg-whatsapp"]:
            # Extract receiver - improved patterns for full names
            receiver_patterns = [
                r"cho\s+([^l\s]+(?:\s+[^l\s]+)*)",
                r"gửi\s+([^l\s]+(?:\s+[^l\s]+)*)",
                r"nhắn\s+tin\s+cho\s+([^l\s]+(?:\s+[^l\s]+)*)"
            ]
            
            for pattern in receiver_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    receiver = match.group(1).strip()
                    # Stop at common words that indicate end of name
                    receiver = re.split(r'\s+(là|lúc|qua|với)', receiver)[0]
                    entities["RECEIVER"] = receiver
                    break
            
            # Extract message - improved patterns
            message_patterns = [
                r"là\s+(.+)$",
                r"nói\s+(.+)$",
                r"báo\s+(.+)$"
            ]
            
            for pattern in message_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    entities["MESSAGE"] = match.group(1).strip()
                    break
            
            # Set platform
            if intent == "send-msg-whatsapp" or "whatsapp" in text_lower:
                entities["PLATFORM"] = "whatsapp"
            else:
                entities["PLATFORM"] = "sms"
        
        elif intent in ["call", "call-whatsapp"]:
            # Extract receiver - improved patterns for full names
            receiver_patterns = [
                r"cho\s+([^l\s]+(?:\s+[^l\s]+)*)",
                r"gọi\s+([^l\s]+(?:\s+[^l\s]+)*)",
                r"gọi\s+điện\s+cho\s+([^l\s]+(?:\s+[^l\s]+)*)"
            ]
            
            for pattern in receiver_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    receiver = match.group(1).strip()
                    # Stop at common words that indicate end of name
                    receiver = re.split(r'\s+(là|lúc|qua|với)', receiver)[0]
                    entities["RECEIVER"] = receiver
                    break
            
            # Set platform
            if intent == "call-whatsapp" or "whatsapp" in text_lower:
                entities["PLATFORM"] = "whatsapp"
            else:
                entities["PLATFORM"] = "phone"
        
        elif intent == "search":
            # Set platform based on keywords
            if "youtube" in text_lower:
                entities["PLATFORM"] = "youtube"
            else:
                entities["PLATFORM"] = "chrome"
        
        elif intent == "control-device":
            # Extract device and value - improved patterns
            if "đèn pin" in text_lower or "flash" in text_lower or ("đèn" in text_lower and "pin" not in text_lower):
                entities["DEVICE"] = "flash"
            elif "wifi" in text_lower or "mạng" in text_lower:
                entities["DEVICE"] = "wifi"
            elif "âm lượng" in text_lower or "âm thanh" in text_lower or "tiếng" in text_lower:
                entities["DEVICE"] = "volume"
            
            # Determine value - improved patterns
            if "bật" in text_lower or "tăng" in text_lower or "mở" in text_lower or "to" in text_lower:
                if entities["DEVICE"] == "volume":
                    entities["value"] = "+"
                else:
                    entities["value"] = "ON"
            elif "tắt" in text_lower or "giảm" in text_lower or "đóng" in text_lower or "nhỏ" in text_lower:
                if entities["DEVICE"] == "volume":
                    entities["value"] = "-"
                else:
                    entities["value"] = "OFF"
        
        elif intent == "open-cam":
            # Let model learn from dataset - minimal keyword matching
            if "chụp" in text_lower or "ảnh" in text_lower or "hình" in text_lower:
                entities["value"] = "image"
            elif "quay" in text_lower or "video" in text_lower or "phim" in text_lower or "ghi hình" in text_lower:
                entities["value"] = "video"
        
        elif intent == "set-alarm":
            # Extract time - let model learn from dataset, minimal regex only
            time_patterns = [
                r"(\d+):(\d+)",
                r"(\d+)\s*giờ"
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    if ":" in pattern:
                        entities["value"] = f"{match.group(1)}:{match.group(2).zfill(2)}"
                    else:
                        entities["value"] = f"{match.group(1)}:00"
                    break
            
            # Extract date
            if "mai" in text_lower:
                entities["DATE"] = "tomorrow"
            elif "ngày kia" in text_lower:
                entities["DATE"] = "day_after_tomorrow"
            elif "thứ hai" in text_lower:
                entities["DATE"] = "monday"
            elif "thứ ba" in text_lower:
                entities["DATE"] = "tuesday"
            elif "thứ tư" in text_lower:
                entities["DATE"] = "wednesday"
            elif "thứ năm" in text_lower:
                entities["DATE"] = "thursday"
            elif "thứ sáu" in text_lower:
                entities["DATE"] = "friday"
            elif "thứ bảy" in text_lower:
                entities["DATE"] = "saturday"
            elif "chủ nhật" in text_lower:
                entities["DATE"] = "sunday"
        
        return entities
    
    def calculate_date(self, date_key):
        """Calculate actual date from date key"""
        today = datetime.now()
        
        if date_key == "tomorrow":
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        elif date_key == "day_after_tomorrow":
            return (today + timedelta(days=2)).strftime("%Y-%m-%d")
        elif date_key == "monday":
            days_ahead = 0 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        elif date_key == "tuesday":
            days_ahead = 1 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        elif date_key == "wednesday":
            days_ahead = 2 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        elif date_key == "thursday":
            days_ahead = 3 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        elif date_key == "friday":
            days_ahead = 4 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        elif date_key == "saturday":
            days_ahead = 5 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        elif date_key == "sunday":
            days_ahead = 6 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        return today.strftime("%Y-%m-%d")
    
    def predict(self, text):
        """Predict intent and extract entities"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class_id].item()
        
        # Get intent
        intent = self.intent_mapping[str(predicted_class_id)]
        
        # Extract entities
        entities = self.extract_entities(text, intent)
        
        # Calculate date if needed
        if entities["DATE"]:
            entities["DATE"] = self.calculate_date(entities["DATE"])
        
        # Determine value based on intent
        value = ""
        text_lower = text.lower()  # Define text_lower here
        
        if intent in ["send-msg", "send-msg-whatsapp"]:
            value = entities["MESSAGE"]
        elif intent == "search":
            # Extract search query from text - improved patterns
            search_patterns = [
                r"tìm\s+kiếm\s+(.+)$",
                r"tìm\s+(.+)$",
                r"về\s+(.+)$"
            ]
            for pattern in search_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    value = match.group(1).strip()
                    # Remove platform keywords from value
                    value = re.sub(r"\s+(trên\s+)?(youtube|chrome)\s*$", "", value)
                    break
        elif intent == "control-device":
            value = entities.get("value", "")
        elif intent == "open-cam":
            value = entities.get("value", "")
        elif intent == "set-alarm":
            value = entities.get("value", "")
        
        # Create response
        response = {
            "text": text,
            "intent": intent,
            "confidence": confidence,
            "command": intent,
            "entities": entities,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        return response

def main():
    # Test the predictor
    predictor = PhoBERTPredictor()
    
    test_texts = [
        "nhắn tin cho mẹ là con sắp về",
        "gọi điện cho bố",
        "bật đèn pin lên",
        "tìm kiếm thông tin về covid",
        "đặt báo thức năm giờ sáng mai"
    ]
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"Text: {text}")
        print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
        print("-" * 50)

if __name__ == "__main__":
    import sys
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    main()
