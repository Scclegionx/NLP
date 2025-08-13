from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
import asyncio
from typing import List
from functools import lru_cache

app = FastAPI()

# ===== Model & Device =====
model_dir = "./nlp-command-model"
tokenizer = T5TokenizerFast.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===== Batch Queue Config =====
BATCH_INTERVAL = 0.05  # thời gian chờ tối đa (50ms)
MAX_BATCH_SIZE = 8
queue = asyncio.Queue()

# ===== Lưu trữ hội thoại =====
dialog_history = []

# ===== Schema =====
class InferenceRequest(BaseModel):
    text: str
    with_context: bool = False
    history_limit: int = 3

class BatchItem:
    def __init__(self, request: InferenceRequest):
        self.request = request
        self.future = asyncio.get_event_loop().create_future()

# ===== Cache encode để tránh token hóa lại nếu input trùng =====
@lru_cache(maxsize=512)
def encode_cached(text: str):
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# ===== Hàm xử lý input =====
def prepare_input(req: InferenceRequest):
    global dialog_history

    if req.with_context:
        negation_keywords = ["phải là", "ý là", "sửa lại", "không phải", "mình nói nhầm", "mình định nói"]
        lowered_text = req.text.lower()
        is_negation = any(kw in lowered_text for kw in negation_keywords)
        if is_negation and dialog_history:
            dialog_history.pop()

        dialog_history.append(req.text)
        formatted_context = " ".join([f"[Câu {i+1}]: {utt}" for i, utt in enumerate(dialog_history[-req.history_limit:])])
        return "trich xuat: " + formatted_context.strip()
    else:
        return "trich xuat: " + req.text

# ===== Batch worker tối ưu =====
async def batch_worker():
    while True:
        batch = []

        # Chờ item đầu tiên hoặc hết thời gian batch
        try:
            item = await asyncio.wait_for(queue.get(), timeout=BATCH_INTERVAL)
            batch.append(item)
        except asyncio.TimeoutError:
            pass

        # Gom thêm các item còn lại
        while not queue.empty() and len(batch) < MAX_BATCH_SIZE:
            batch.append(await queue.get())

        if not batch:
            continue

        # Chuẩn bị input
        texts = [prepare_input(item.request) for item in batch]
        encodings_list = [encode_cached(txt) for txt in texts]

        # Gộp batch tensors
        encodings = {
            k: torch.cat([e[k] for e in encodings_list], dim=0).to(device)
            for k in encodings_list[0]
        }

        # Inference
        with torch.no_grad():
            output_ids = model.generate(
                **encodings,
                max_length=32,  # giảm length để nhanh hơn
                num_beams=1,    # tắt beam search để tăng tốc
                early_stopping=True
            )

        results = tokenizer.batch_decode(output_ids.cpu(), skip_special_tokens=True)

        # Trả kết quả cho từng request
        for item, res in zip(batch, results):
            item.future.set_result(res)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())

@app.post("/infer")
async def run_inference(req: InferenceRequest):
    item = BatchItem(req)
    await queue.put(item)
    result = await item.future
    return {"input": req.text, "output": result}
