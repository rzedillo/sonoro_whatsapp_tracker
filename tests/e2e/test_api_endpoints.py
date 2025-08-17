"""
End-to-end tests for API endpoints
"""

import pytest
import requests
import time
from typing import Dict, Any


# Test configuration
API_BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session")
def api_client():
    """API client for testing"""
    # Wait for service to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            if i == max_retries - 1:
                pytest.skip("API service not available")
            time.sleep(2)
    
    return API_BASE_URL


class TestAPIEndpoints:
    """End-to-end API tests"""
    
    def test_health_endpoint(self, api_client):
        """Test health endpoint"""
        response = requests.get(f"{api_client}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_api_root(self, api_client):
        """Test API root endpoint"""
        response = requests.get(f"{api_client}/api/v1/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
    
    def test_tasks_endpoint(self, api_client):
        """Test tasks endpoint"""
        response = requests.get(f"{api_client}/api/v1/tasks/")
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
    
    def test_create_task(self, api_client):
        """Test task creation"""
        task_data = {
            "descripcion": "Test task from E2E",
            "responsable": "Test User",
            "prioridad": "media",
            "grupo_nombre": "E2E Test"
        }
        
        response = requests.post(f"{api_client}/api/v1/tasks/", json=task_data)
        
        assert response.status_code == 200
        data = response.json()
        
        if data.get("success"):
            # Task created successfully
            assert "data" in data
            task_id = data["data"].get("task_id")
            assert task_id is not None
            
            # Clean up - mark task as completed
            requests.post(f"{api_client}/api/v1/tasks/{task_id}/complete")
    
    def test_whatsapp_status(self, api_client):
        """Test WhatsApp status endpoint"""
        response = requests.get(f"{api_client}/api/v1/whatsapp/status")
        
        # This might fail if WhatsApp is not configured, which is expected
        assert response.status_code in [200, 503]
    
    def test_agents_status(self, api_client):
        """Test agents status endpoint"""
        response = requests.get(f"{api_client}/api/v1/agents/")
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data