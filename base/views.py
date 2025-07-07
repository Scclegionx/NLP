from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_GET
import json
from .nlp_model import NLPModel

# Create your views here.

# Khởi tạo model
nlp_model = NLPModel()

@csrf_exempt
@require_http_methods(["POST"])
def process_text(request):
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        
        if not text:
            return JsonResponse({
                'error': 'Text is required'
            }, status=400)
        
        # Xử lý văn bản và trả về kết quả
        result = nlp_model.predict(text)

        print("test hihi")
        print(result)

        return JsonResponse({
            'command': result['command'],
            'entities': result['entities'],
            'values': result['values'],
            'confidence': result['confidence']
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)

@require_GET
def health_check(request):
    """Health check endpoint để kiểm tra server có hoạt động không"""
    return JsonResponse({
        'status': 'ok',
        'message': 'NLP API server is running',
        'endpoints': {
            'process_text': '/api/process-text/',
            'health_check': '/'
        }
    })
