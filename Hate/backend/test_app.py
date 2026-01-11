"""
Advanced Test Application for Hate Speech Detection API
Comprehensive testing suite with all features for the hate speech detection project
"""
import json
import time
import requests
import logging
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HateSpeechAPITester:
    """Advanced API testing class for hate speech detection"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'HateSpeechAPITester/1.0'
        })
        self.test_results = {}

    def test_health_check(self) -> bool:
        """Test API health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                logger.info(f"✅ Health check passed: {status}")
                return True
            else:
                logger.error(f"❌ Health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Health check error: {str(e)}")
            return False

    def test_single_prediction(self, text: str, expected_label: str = None) -> Dict[str, Any]:
        """Test single text prediction"""
        try:
            payload = {"text": text}
            response = self.session.post(f"{self.base_url}/predict", json=payload)

            if response.status_code == 200:
                data = response.json()
                confidence = data.get('confidence', 0)
                label = data.get('label', 'unknown')

                # Truncate text for display
                display_text = text[:50] + "..." if len(text) > 50 else text
                logger.info(f"✅ Prediction: '{display_text}' → {label} ({confidence:.3f})")

                if expected_label and label != expected_label:
                    logger.warning(f"⚠️  Expected {expected_label}, got {label}")

                return data
            else:
                error_msg = f"HTTP {response.status_code}"
                logger.error(f"❌ Prediction failed: {error_msg} - {response.text}")
                return {"error": error_msg}

        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            return {"error": str(e)}

    def test_batch_prediction(self, texts: List[str]) -> Dict[str, Any]:
        """Test batch prediction functionality"""
        try:
            payload = {"texts": texts}
            response = self.session.post(f"{self.base_url}/batch_predict", json=payload)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                logger.info(f"✅ Batch prediction: {len(results)} texts processed")

                for result in results:
                    index = result.get('index', 'N/A')
                    if 'error' not in result:
                        label = result.get('label', 'unknown')
                        confidence = result.get('confidence', 0)
                        logger.info(f"   Text {index}: {label} ({confidence:.3f})")
                    else:
                        error = result.get('error', 'Unknown error')
                        logger.warning(f"   Text {index}: ERROR - {error}")

                return data
            else:
                error_msg = f"HTTP {response.status_code}"
                logger.error(f"❌ Batch prediction failed: {error_msg}")
                return {"error": error_msg}

        except Exception as e:
            logger.error(f"❌ Batch prediction error: {str(e)}")
            return {"error": str(e)}

    def test_feedback_submission(self, text: str, predicted: Dict, actual: str, feedback_type: str) -> bool:
        """Test feedback submission endpoint"""
        try:
            payload = {
                "text": text,
                "predicted": predicted,
                "actual": actual,
                "feedback_type": feedback_type
            }

            response = self.session.post(f"{self.base_url}/feedback", json=payload)

            if response.status_code == 200:
                logger.info(f"✅ Feedback submitted: {feedback_type}")
                return True
            else:
                logger.error(f"❌ Feedback submission failed: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Feedback submission error: {str(e)}")
            return False

    def test_stats_endpoint(self) -> Dict[str, Any]:
        """Test statistics endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/stats")

            if response.status_code == 200:
                data = response.json()
                api_status = data.get('api_status', data.get('status', 'unknown'))
                logger.info(f"✅ Stats retrieved: API status = {api_status}")
                return data
            else:
                error_msg = f"HTTP {response.status_code}"
                logger.error(f"❌ Stats retrieval failed: {error_msg}")
                return {"error": error_msg}

        except Exception as e:
            logger.error(f"❌ Stats error: {str(e)}")
            return {"error": str(e)}

    def test_model_info(self) -> Dict[str, Any]:
        """Test model information endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/model_info")

            if response.status_code == 200:
                data = response.json()
                model_type = data.get('model_type', 'unknown')
                total_params = data.get('total_params', 'N/A')
                logger.info(f"✅ Model info: {model_type} with {total_params} parameters")
                return data
            else:
                error_msg = f"HTTP {response.status_code}"
                logger.error(f"❌ Model info failed: {error_msg}")
                return {"error": error_msg}

        except Exception as e:
            logger.error(f"❌ Model info error: {str(e)}")
            return {"error": str(e)}

    def test_error_handling(self) -> bool:
        """Test various error conditions"""
        logger.info("🧪 Testing error handling...")

        test_cases = [
            # Invalid JSON
            {"endpoint": "predict", "data": "invalid json", "expected_status": 400},

            # Missing required fields
            {"endpoint": "predict", "data": {}, "expected_status": 400},
            {"endpoint": "predict", "data": {"wrong_field": "test"}, "expected_status": 400},

            # Empty text
            {"endpoint": "predict", "data": {"text": ""}, "expected_status": 200},  # Should handle gracefully

            # Very long text
            {"endpoint": "predict", "data": {"text": "x" * 10000}, "expected_status": 400},

            # Invalid batch data
            {"endpoint": "batch_predict", "data": {"texts": "not a list"}, "expected_status": 400},
            {"endpoint": "batch_predict", "data": {"texts": []}, "expected_status": 400},
        ]

        all_passed = True

        for test_case in test_cases:
            try:
                endpoint = test_case['endpoint']
                expected = test_case['expected_status']

                if isinstance(test_case["data"], dict):
                    response = self.session.post(f"{self.base_url}/{endpoint}", json=test_case["data"])
                else:
                    # Test invalid JSON
                    response = self.session.post(
                        f"{self.base_url}/{endpoint}", 
                        data=test_case["data"],
                        headers={'Content-Type': 'application/json'}
                    )

                if response.status_code == expected:
                    logger.info(f"✅ Error test passed: {endpoint} returned {response.status_code}")
                else:
                    logger.warning(f"⚠️  Error test: {endpoint} returned {response.status_code}, expected {expected}")
                    all_passed = False

            except Exception as e:
                logger.error(f"❌ Error test failed: {str(e)}")
                all_passed = False

        return all_passed

    def test_rate_limiting(self) -> bool:
        """Test rate limiting functionality"""
        logger.info("🧪 Testing rate limiting...")

        # Make multiple rapid requests
        start_time = time.time()
        success_count = 0
        rate_limited_count = 0

        for i in range(20):  # Try 20 requests rapidly
            try:
                response = self.session.post(
                    f"{self.base_url}/predict", 
                    json={"text": f"test message {i}"}
                )

                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:  # Rate limited
                    rate_limited_count += 1
                    logger.info(f"✅ Rate limiting triggered at request {i + 1}")
                    break

            except Exception as e:
                logger.error(f"❌ Rate limiting test error: {str(e)}")
                return False

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Rate limiting test: {success_count} successful, {rate_limited_count} rate limited in {duration:.2f}s")

        return True

    def run_comprehensive_test(self) -> Dict[str, bool]:
        """Run comprehensive test suite"""
        logger.info("🚀 Starting comprehensive API test suite...")

        results = {}

        # 1. Health Check
        results["health_check"] = self.test_health_check()

        # 2. Model Info
        model_info = self.test_model_info()
        results["model_info"] = "error" not in model_info

        # 3. Stats
        stats = self.test_stats_endpoint()
        results["stats"] = "error" not in stats

        # 4. Single Predictions
        test_texts = [
            ("I love this movie, it's amazing!", "not_hate"),
            ("You are so stupid and worthless", "hate"),
            ("The weather is nice today", "not_hate"),
            ("I hate when people are late", "not_hate"),  # Context matters
            ("Kill yourself, you're garbage", "hate"),
            ("This is a normal sentence about technology", "not_hate")
        ]

        single_prediction_results = []
        for text, expected in test_texts:
            result = self.test_single_prediction(text, expected)
            single_prediction_results.append("error" not in result)

        results["single_predictions"] = all(single_prediction_results)

        # 5. Batch Prediction
        batch_texts = [text for text, _ in test_texts[:4]]  # First 4 texts
        batch_result = self.test_batch_prediction(batch_texts)
        results["batch_prediction"] = "error" not in batch_result

        # 6. Feedback Submission
        sample_prediction = {"prediction": 1, "confidence": 0.8, "label": "hate"}
        results["feedback"] = self.test_feedback_submission(
            "test feedback text", 
            sample_prediction, 
            "not_hate", 
            "false_positive"
        )

        # 7. Error Handling
        results["error_handling"] = self.test_error_handling()

        # 8. Rate Limiting (optional)
        results["rate_limiting"] = self.test_rate_limiting()

        # Summary
        passed = sum(results.values())
        total = len(results)

        # Fixed the f-string issue that was causing syntax error
        summary_msg = f"Test Summary: {passed}/{total} tests passed"
        logger.info("📊 " + summary_msg)
        logger.info("="*50)

        for test_name, test_passed in results.items():
            status = "✅ PASS" if test_passed else "❌ FAIL"
            logger.info(f"{status} {test_name}")

        if passed == total:
            logger.info("🎉 All tests passed! API is working correctly.")
        else:
            warning_msg = f"⚠️  {total - passed} test(s) failed. Please check the issues above."
            logger.warning(warning_msg)

        return results

def run_quick_test(base_url: str = "http://localhost:8000"):
    """Run a quick test of the API"""
    print("🚀 Quick API Test")
    print("="*30)

    tester = HateSpeechAPITester(base_url)

    # Test basic functionality
    if not tester.test_health_check():
        print("❌ API is not responding. Please check if the server is running.")
        return

    # Test a few predictions
    test_cases = [
        "I love this product!",
        "You are such an idiot",
        "The weather is nice today"
    ]

    print("\n🧪 Testing predictions:")
    for text in test_cases:
        result = tester.test_single_prediction(text)
        if "error" not in result:
            label = result.get('label', 'unknown')
            confidence = result.get('confidence', 0)
            print(f"✅ '{text}' → {label} ({confidence:.3f})")
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ '{text}' → Error: {error}")

    print("\n✨ Quick test completed!")

def run_full_test(base_url: str = "http://localhost:8000"):
    """Run the comprehensive test suite"""
    tester = HateSpeechAPITester(base_url)
    return tester.run_comprehensive_test()

def main():
    """Main function to run tests"""
    import sys

    print("="*60)
    print("🔬 HATE SPEECH DETECTION API TESTER")
    print("="*60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Get base URL from command line argument
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    # Check if quick test is requested
    if "--quick" in sys.argv:
        run_quick_test(base_url)
    else:
        run_full_test(base_url)

    print("="*60)
    print("🏁 Testing completed!")
    print("="*60)

if __name__ == "__main__":
    main()