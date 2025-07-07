from django.test import TestCase, Client
from django.urls import reverse
import json

# Create your tests here.

class ProcessTextAPITest(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('process_text')

    def test_process_text_success(self):
        """Test API xử lý văn bản thành công"""
        data = {
            'text': 'Mở ứng dụng Chrome'
        }
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.content)
        
        # Kiểm tra cấu trúc response
        self.assertIn('command', result)
        self.assertIn('entities', result)
        self.assertIn('values', result)
        self.assertIn('confidence', result)

    def test_process_text_empty_text(self):
        """Test API với text rỗng"""
        data = {
            'text': ''
        }
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        result = json.loads(response.content)
        self.assertIn('error', result)

    def test_process_text_missing_text(self):
        """Test API thiếu trường text"""
        data = {}
        response = self.client.post(
            self.url,
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        result = json.loads(response.content)
        self.assertIn('error', result)

    def test_process_text_invalid_json(self):
        """Test API với JSON không hợp lệ"""
        response = self.client.post(
            self.url,
            data='invalid json',
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 500)
        result = json.loads(response.content)
        self.assertIn('error', result)

    def test_process_text_get_method(self):
        """Test API với method GET (không được phép)"""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 405)  # Method Not Allowed

    def test_health_check_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.content)
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'ok')
        self.assertIn('endpoints', result)
