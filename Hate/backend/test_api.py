import requests
import sys
import argparse
import time

# Configuration
DEFAULT_API_URL = 'http://127.0.0.1:8000'
API_ENDPOINT = f'{DEFAULT_API_URL}/predict'
HEALTH_ENDPOINT = f'{DEFAULT_API_URL}/'
MODEL_INFO_ENDPOINT = f'{DEFAULT_API_URL}/model_info'
STATS_ENDPOINT = f'{DEFAULT_API_URL}/stats'
FEEDBACK_ENDPOINT = f'{DEFAULT_API_URL}/feedback'

# Sample test data
test_texts = [
    ("I love this!", "not_hate"),
    ("You are such a loser", "hate"),
    ("What a beautiful day", "not_hate"),
    ("This is stupid and annoying", "hate"),
    ("", "empty"),
    ("" * 1000, "long blank"),
    ("I hate you!", "hate")
]

positive_texts = [
    "I really enjoy this service.",
    "Thank you so much!",
    "Happy to help, no hate here."
]
negative_texts = [
    "This is terrible hate speech.",
    "You idiot!",
    "Well, you suck."
]

# Utility functions
def print_header(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def check_response(resp, expected_status=200):
    if resp.status_code != expected_status:
        print(f"❌ Response {resp.status_code} (expected {expected_status})")
        print(f"Response content: {resp.text}")
        return False
    return True

def test_health(url):
    print("[INFO] Checking health endpoint...")
    resp = requests.get(url)
    if check_response(resp):
        data = resp.json()
        status = data.get("status", "").lower()
        if status == "healthy" or status == "ok":
            print("✅ Health check passed.")
        else:
            print(f"⚠️ Unexpected health status: {status}")
    else:
        print("❌ Health check failed.")

def test_model_info():
    print("[INFO] Retrieving model info...")
    resp = requests.get(MODEL_INFO_ENDPOINT)
    if check_response(resp):
        print("✅ Model info retrieved:", resp.json())

def test_get_stats():
    print("[INFO] Retrieving API stats...")
    resp = requests.get(STATS_ENDPOINT)
    if check_response(resp):
        print("✅ Stats:", resp.json())

def test_prediction(text, expected_label, threshold=0.5):
    payload = {'text': text}
    resp = requests.post(API_ENDPOINT, json=payload)
    if check_response(resp):
        result = resp.json()
        pred = result.get('label', '')
        confidence = result.get('confidence', 0)
        print(f"Text: '{text[:50]}...' | Predicted: {pred} ({confidence:.2f}) | Expected: {expected_label}")
        if pred == expected_label:
            print("✅ Prediction matches expected.")
            return True
        else:
            print("❌ Prediction mismatch.")
            return False
    return False

def test_batch(texts, expected_labels):
    print("[INFO] Sending batch prediction request...")
    payload = {'texts': texts}
    resp = requests.post(f'{DEFAULT_API_URL}/batch_predict', json=payload)
    if check_response(resp):
        results = resp.json()
        success = True
        for i, res in enumerate(results):
            pred_label = res.get('label')
            print(f"[Batch] Text: '{texts[i][:50]}...' | Predicted: {pred_label} | Expected: {expected_labels[i]}")
            if pred_label != expected_labels[i]:
                success = False
        if success:
            print("✅ Batch predictions matched expected labels.")
        else:
            print("❌ Batch prediction mismatches detected.")
        return success
    return False

def test_feedback(text, label):
    print(f"[INFO] Submitting feedback: '{label}' for text: '{text[:50]}...'")
    payload = {'text': text, 'label': label}
    resp = requests.post(FEEDBACK_ENDPOINT, json=payload)
    if check_response(resp, expected_status=200):
        print("✅ Feedback submitted successfully.")
        return True
    return False

def run_quick_tests():
    print_header("🧪 Quick API Tests")
    test_health(HEALTH_ENDPOINT)
    test_model_info()
    test_get_stats()
    # Test a few predictions
    for text, label in test_texts[:3]:
        test_prediction(text, label)

def run_full_tests():
    print_header("🔬 Full API Suite")
    test_health(HEALTH_ENDPOINT)
    test_model_info()
    test_get_stats()
    passed = 0
    total = len(test_texts)
    
    print("[INFO] Testing individual predictions...")
    for text, label in test_texts:
        if test_prediction(text, label):
            passed += 1
    
    print("[INFO] Testing batch prediction...")
    texts, labels = zip(*test_texts)
    if test_batch(texts, labels):
        passed += 1
    else:
        print("❌ Batch prediction test failed.")
    
    print("[INFO] Testing feedback submission...")
    for text, label in test_texts[:2]:
        if test_feedback(text, label):
            passed += 1
    total_tests = len(test_texts) + 2  # batch + feedback
    print(f"\n🎯 Test results: {passed}/{total_tests} tests passed.")

def check_rate_limit(endpoint):
    print("[INFO] Testing rate limiting...")
    # Send multiple requests quickly
    for i in range(5):
        resp = requests.post(API_ENDPOINT, json={'text': 'test'})
        if resp.status_code == 429:
            print("⚠️ Rate limit reached at request", i+1)
            break
        else:
            print(f"Request {i+1} succeeded.")
        time.sleep(0.2)  # small delay
    print("Rate limiting test completed.")

def main():
    parser = argparse.ArgumentParser(description="Advanced API Testing for Hate Speech Detection")
    parser.add_argument('--quick', action='store_true', help='Run quick tests')
    parser.add_argument('--full', action='store_true', help='Run full comprehensive tests')
    parser.add_argument('url', nargs='?', default=DEFAULT_API_URL, help='API URL (default: localhost)')
    args = parser.parse_args()

    global DEFAULT_API_URL, API_ENDPOINT
    DEFAULT_API_URL = args.url.rstrip('/')
    API_ENDPOINT = f'{DEFAULT_API_URL}/predict'

    if args.quick:
        run_quick_tests()
    elif args.full:  run_full_tests()
    else:
        print("Please specify --quick or --full for testing mode.")

if __name__ == "__main__":
    main()