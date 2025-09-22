from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = FastAPI()

# Load model đã huấn luyện
model_dir = "./nlp-command-model"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Lưu trữ hội thoại
dialog_history = []

# Request schema
class InferenceRequest(BaseModel):
    text: str
    with_context: bool = False
    history_limit: int = 3

# Xử lý inference
def infer(text):
    input_text = "trich xuat: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return result

# Xử lý inference có context thông minh hơn
def infer_with_context(text, history_limit=3):
    global dialog_history

    # Kiểm tra xem câu mới có phủ định/sửa không
    negation_keywords = ["phải là", "ý là", "sửa lại", "không phải", "mình nói nhầm", "mình định nói"]
    lowered_text = text.lower()
    is_negation = any(kw in lowered_text for kw in negation_keywords)

    # Nếu là phủ định → xóa câu trước trong lịch sử (nếu có)
    if is_negation and len(dialog_history) >= 1:
        dialog_history.pop()  # loại bỏ câu trước

    # Thêm câu hiện tại vào history
    dialog_history.append(text)

    # Format lại context cho rõ ràng (Cách 1)
    formatted_context = ""
    for i, utt in enumerate(dialog_history[-history_limit:], 1):
        formatted_context += f"[Câu {i}]: {utt} "

    input_text = "trich xuat: " + formatted_context.strip()

    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return result


@app.post("/infer")
def run_inference(req: InferenceRequest):
    if req.with_context:
        result = infer_with_context(req.text, req.history_limit)
    else:
        result = infer(req.text)
    return {"input": req.text, "output": result}
