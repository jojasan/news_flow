import requests
import time

BASE_URL = "http://localhost:8000"

def test_post_with_news_urls():
    data = {
        "id": "test_news",
        "news_urls": ["https://example.com/news1", "https://example.com/news2"],
        "perspective": "Neutral",
        "tone": "Informative",
        "num_starting_pool_news": 2,
        "num_max_news": 1
    }
    response = requests.post(f"{BASE_URL}/happifynews", json=data)
    print("POST with news_urls response:", response.json())
    assert response.status_code == 200, "POST with news_urls failed."
    assert response.json()["status"] == "processing", "Expected status 'processing'."
    return response.json()["task_id"]

def test_post_with_topic():
    data = {
        "id": "test_topic",
        "topic": "Artificial Intelligence",
        "perspective": "Neutral",
        "tone": "Informative",
        "num_starting_pool_news": 2,
        "num_max_news": 1
    }
    response = requests.post(f"{BASE_URL}/happifynews", json=data)
    print("POST with topic response:", response.json())
    assert response.status_code == 200, "POST with topic failed."
    assert response.json()["status"] == "processing", "Expected status 'processing'."
    return response.json()["task_id"]

def test_get_immediate_status(task_id):
    response = requests.get(f"{BASE_URL}/happifynews/{task_id}")
    print(f"Immediate GET status for task {task_id}:", response.json())
    assert response.status_code == 200, f"GET for task {task_id} failed."
    return response.json()

def test_wait_for_completion(task_id, wait_time=2700):
    # WARNING: This will sleep for 45 minutes (2700 seconds).
    print(f"Waiting for {wait_time} seconds for task {task_id} to complete...")
    time.sleep(wait_time)
    response = requests.get(f"{BASE_URL}/happifynews/{task_id}")
    print(f"GET status after wait for task {task_id}:", response.json())
    assert response.status_code == 200, f"GET for task {task_id} after wait failed."
    return response.json()

def run_all_tests():
    print("Starting tests...")

    # Test 1: POST with news_urls
    # task_id_news = test_post_with_news_urls()
    # Test 2: POST with topic
    task_id_topic = test_post_with_topic()
    
    # Test 3: Immediately check status for both tasks.
    # test_get_immediate_status(task_id_news)
    test_get_immediate_status(task_id_topic)
    
    # Test 4: Wait 45 minutes and then check status for both tasks.
    # (In a real scenario, these tasks should complete during this period.)
    # test_wait_for_completion(task_id_news, wait_time=2700)
    test_wait_for_completion(task_id_topic, wait_time=2700)
    
    print("All tests executed. Check the outputs above for statuses.")

if __name__ == "__main__":
    run_all_tests()
