from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model đã huấn luyện
model_dir = "./nlp-command-model"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def test_inference(text):
    input_text = "trich xuat: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return result

# Test cases
test_cases = [
    "nhắn tin cho mẹ là con sắp về",
    "nhắn tin cho cháu Vương là chiều nay đón bà lúc 5h",
    "gửi tin nhắn cho anh Minh Hà là tối nay có họp",
    "nhắn tin cho chị Lan Anh là mai đi chơi",
    "gửi tin nhắn cho em Vân là về muộn",
    "nhắn tin cho cô Hương là con sắp về",
    "gửi tin nhắn cho chú Nam là có việc gấp",
    "nhắn tin cho bác Tâm là hôm nay bận",
    "gửi tin nhắn cho dì Mai là về sớm",
    "nhắn tin cho thầy Dũng là em xin phép nghỉ"
]

print("=== KẾT QUẢ TEST MODEL HIỆN TẠI ===\n")

for i, test_case in enumerate(test_cases, 1):
    result = test_inference(test_case)
    print(f"Test {i}:")
    print(f"Input: {test_case}")
    print(f"Output: {result}")
    print("-" * 50)
