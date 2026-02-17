import requests
import sys
import json
from datetime import datetime

class CustomerIntelligenceAPITester:
    def __init__(self, base_url="https://marketiq-7.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status, data=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"Response: {response.text[:500]}")
                self.failed_tests.append(f"{name}: Expected {expected_status}, got {response.status_code}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.failed_tests.append(f"{name}: {str(e)}")
            return False, {}

    def test_health_endpoint(self):
        """Test health endpoint and verify models_loaded is true"""
        success, response = self.run_test(
            "Health Check",
            "GET", 
            "api/health",
            200
        )
        if success:
            models_loaded = response.get('models_loaded', False)
            if models_loaded:
                print("✅ Models are loaded")
            else:
                print("❌ Models are not loaded")
                self.failed_tests.append("Health Check: models_loaded is false")
                return False
        return success

    def test_model_metrics(self):
        """Test model metrics endpoint"""
        success, response = self.run_test(
            "Model Metrics",
            "GET",
            "api/model-metrics", 
            200
        )
        if success:
            required_metrics = ['purchase_intent_accuracy', 'loyalty_accuracy', 'amount_r2_score']
            missing_metrics = []
            metrics_data = response.get('metrics', {})
            for metric in required_metrics:
                if metric not in metrics_data:
                    missing_metrics.append(metric)
            
            if missing_metrics:
                print(f"❌ Missing metrics: {missing_metrics}")
                self.failed_tests.append(f"Model Metrics: Missing {missing_metrics}")
                return False
            else:
                print("✅ All required metrics present")
                print(f"✅ Purchase Intent Accuracy: {metrics_data.get('purchase_intent_accuracy')}")
                print(f"✅ Loyalty Accuracy: {metrics_data.get('loyalty_accuracy')}")
                print(f"✅ Amount R² Score: {metrics_data.get('amount_r2_score')}")
        return success

    def test_eda_data(self):
        """Test EDA data endpoint"""
        success, response = self.run_test(
            "EDA Data",
            "GET",
            "api/eda-data",
            200
        )
        if success:
            if isinstance(response, dict) and len(response) > 0:
                print("✅ EDA data structure looks good")
            else:
                print("❌ EDA data structure invalid")
                self.failed_tests.append("EDA Data: Invalid structure")
                return False
        return success

    def test_dataset_info(self):
        """Test dataset info endpoint"""
        success, response = self.run_test(
            "Dataset Info",
            "GET", 
            "api/dataset-info",
            200
        )
        if success:
            required_fields = ['shape', 'columns']
            missing_fields = []
            for field in required_fields:
                if field not in response:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
                self.failed_tests.append(f"Dataset Info: Missing {missing_fields}")
                return False
            else:
                print(f"✅ Dataset shape: {response.get('shape')}")
                print(f"✅ Columns count: {len(response.get('columns', []))}")
        return success

    def test_purchase_intent_prediction(self):
        """Test purchase intent prediction"""
        sample_data = {
            "age": 30,
            "gender": "Male",
            "income_level": "Middle",
            "marital_status": "Single", 
            "education_level": "Bachelor's",
            "occupation": "Middle",
            "purchase_category": "Electronics",
            "frequency_of_purchase": 5,
            "purchase_channel": "Online",
            "brand_loyalty": 3,
            "product_rating": 4,
            "time_spent_on_research": 2.5,
            "social_media_influence": "Medium",
            "discount_sensitivity": "Somewhat Sensitive",
            "return_rate": 1,
            "customer_satisfaction": 7,
            "engagement_with_ads": "Medium",
            "device_used": "Smartphone",
            "payment_method": "Credit Card",
            "time_to_decision": 5
        }
        
        success, response = self.run_test(
            "Purchase Intent Prediction",
            "POST",
            "api/predict/purchase-intent",
            200,
            data=sample_data
        )
        
        if success:
            required_fields = ['prediction', 'probability', 'confidence']
            missing_fields = []
            for field in required_fields:
                if field not in response:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
                self.failed_tests.append(f"Purchase Intent: Missing {missing_fields}")
                return False
            else:
                print(f"✅ Prediction: {response.get('prediction')}")
                print(f"✅ Confidence: {response.get('confidence')}")
        return success

    def test_loyalty_prediction(self):
        """Test loyalty prediction"""
        sample_data = {
            "age": 30,
            "gender": "Male", 
            "income_level": "Middle",
            "marital_status": "Single",
            "education_level": "Bachelor's",
            "occupation": "Middle",
            "purchase_category": "Electronics",
            "frequency_of_purchase": 5,
            "purchase_channel": "Online",
            "brand_loyalty": 3,
            "product_rating": 4,
            "time_spent_on_research": 2.5,
            "social_media_influence": "Medium",
            "discount_sensitivity": "Somewhat Sensitive",
            "return_rate": 1,
            "customer_satisfaction": 7,
            "engagement_with_ads": "Medium",
            "device_used": "Smartphone",
            "payment_method": "Credit Card",
            "time_to_decision": 5
        }
        
        success, response = self.run_test(
            "Loyalty Prediction",
            "POST",
            "api/predict/loyalty",
            200,
            data=sample_data
        )
        
        if success:
            required_fields = ['prediction', 'probability', 'confidence']
            missing_fields = []
            for field in required_fields:
                if field not in response:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
                self.failed_tests.append(f"Loyalty Prediction: Missing {missing_fields}")
                return False
            else:
                prediction = response.get('prediction')
                if isinstance(prediction, bool):
                    print(f"✅ Prediction (bool): {prediction}")
                else:
                    print(f"❌ Prediction should be boolean, got: {type(prediction)}")
                    self.failed_tests.append(f"Loyalty Prediction: prediction should be bool")
                    return False
                print(f"✅ Probability: {response.get('probability')}")
                print(f"✅ Confidence: {response.get('confidence')}")
        return success

    def test_amount_prediction(self):
        """Test amount prediction"""
        sample_data = {
            "age": 30,
            "gender": "Male",
            "income_level": "Middle", 
            "marital_status": "Single",
            "education_level": "Bachelor's",
            "occupation": "Middle",
            "purchase_category": "Electronics",
            "frequency_of_purchase": 5,
            "purchase_channel": "Online",
            "brand_loyalty": 3,
            "product_rating": 4,
            "time_spent_on_research": 2.5,
            "social_media_influence": "Medium",
            "discount_sensitivity": "Somewhat Sensitive",
            "return_rate": 1,
            "customer_satisfaction": 7,
            "engagement_with_ads": "Medium",
            "device_used": "Smartphone",
            "payment_method": "Credit Card",
            "time_to_decision": 5
        }
        
        success, response = self.run_test(
            "Amount Prediction",
            "POST",
            "api/predict/amount",
            200,
            data=sample_data
        )
        
        if success:
            required_fields = ['predicted_amount', 'confidence_interval']
            missing_fields = []
            for field in required_fields:
                if field not in response:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
                self.failed_tests.append(f"Amount Prediction: Missing {missing_fields}")
                return False
            else:
                print(f"✅ Predicted Amount: {response.get('predicted_amount')}")
                confidence_interval = response.get('confidence_interval', {})
                if 'lower' in confidence_interval and 'upper' in confidence_interval:
                    print(f"✅ Confidence Interval: {confidence_interval}")
                else:
                    print(f"❌ Invalid confidence interval structure")
                    self.failed_tests.append(f"Amount Prediction: Invalid confidence_interval")
                    return False
        return success

def main():
    print("🚀 Starting Customer Intelligence API Tests")
    print("=" * 60)
    
    tester = CustomerIntelligenceAPITester()
    
    # Run all tests
    tests = [
        tester.test_health_endpoint,
        tester.test_model_metrics,
        tester.test_eda_data,
        tester.test_dataset_info,
        tester.test_purchase_intent_prediction,
        tester.test_loyalty_prediction,
        tester.test_amount_prediction
    ]
    
    for test in tests:
        test()
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS")
    print(f"Tests Run: {tester.tests_run}")
    print(f"Tests Passed: {tester.tests_passed}")
    print(f"Tests Failed: {tester.tests_run - tester.tests_passed}")
    
    if tester.failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for failure in tester.failed_tests:
            print(f"  - {failure}")
    else:
        print(f"\n✅ ALL TESTS PASSED!")
    
    return 0 if len(tester.failed_tests) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())