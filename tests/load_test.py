from locust import HttpUser, task, between
import json

class MLAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)  # Wait 0.1-0.5s between requests
    
    @task(10)  # Weight 10 - most common
    def predict(self):
        """Make prediction request."""
        payload = {
            "features": {
                "sensor_2": 642.5,
                "sensor_3": 1589.2,
                "sensor_4": 1400.6,
                "sensor_7": 554.8,
                "sensor_8": 2388.1,
                "sensor_11": 47.3,
                "sensor_12": 521.7
            }
        }
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")
    
    @task(5)  # Weight 5
    def health_check(self):
        """Check API health."""
        self.client.get("/health")
    
    @task(2)  # Weight 2
    def model_info(self):
        """Get model info."""
        self.client.get("/model/info")