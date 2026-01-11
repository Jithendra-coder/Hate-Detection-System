import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')

CORS(app, origins=["chrome-extension://*", "moz-extension://*", "http://localhost:*", "https://localhost:*"])

model = None
tokenizer = None
label_encoder = None

class HateSpeechDetector:
    def __init__(self):
        self.model_path = os.path.join('backend', 'models', 'best_model.keras')
        self.tokenizer_path = 'tokenizer.pkl'
        self.labels_path = 'labels.pkl'
        self.max_length = 100
        self.confidence_threshold = 0.5
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.load_model_components()

    def load_model_components(self):
        try:
            if not os.path.exists(self.model_path):
                logger.error("Model file not found: " + self.model_path)
                return False
            if not os.path.exists(self.tokenizer_path):
                logger.error("Tokenizer file not found: " + self.tokenizer_path)
                return False
            if not os.path.exists(self.labels_path):
                logger.error("Labels file not found: " + self.labels_path)
                return False
            logger.info("Loading LSTM model...")
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("Model loaded successfully")
            logger.info("Loading tokenizer...")
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            logger.info("Tokenizer loaded successfully")
            logger.info("Loading label encoder...")
            with open(self.labels_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info("Label encoder loaded successfully")
            return True
        except Exception as e:
            logger.error("Error loading model components: " + str(e))
            return False

    def preprocess_text(self, text: str) -> np.ndarray:
        try:
            text = text.lower().strip()
            sequences = self.tokenizer.texts_to_sequences([text])
            padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            return padded
        except Exception as e:
            logger.error("Error preprocessing text: " + str(e))
            raise

    def predict_hate_speech(self, text: str) -> Dict[str, Any]:
        try:
            if not text or len(text.strip()) < 3:
                return {'prediction': 0, 'confidence': 0.0, 'label': 'not_hate', 'error': 'Text too short to scan'}
            if self.model is None or self.tokenizer is None or self.label_encoder is None:
                return {'prediction': 0, 'confidence': 0.0, 'label': 'error', 'error': 'Model not loaded ― cannot scan this page'}
            if len(text) > 1000:
                return {'prediction': 0, 'confidence': 0.0, 'label': 'error', 'error': 'Text too long (max 1000 characters)'}
            if len(text) > 1000:
                text = text[:1000]
            processed_text = self.preprocess_text(text)
            predictions = self.model.predict(processed_text, verbose=0)
            prediction_prob = float(predictions[0][0])
            if prediction_prob > self.confidence_threshold:
                prediction = 1
                label = 'hate'
                confidence = prediction_prob
            else:
                prediction = 0
                label = 'not_hate'
                confidence = 1 - prediction_prob
            category = self.categorize_hate_speech(text, prediction_prob)
            return {'prediction': prediction, 'confidence': float(confidence), 'probability': float(prediction_prob), 'label': label, 'category': category, 'text_length': len(text)}
        except Exception as e:
            logger.error("Prediction error: " + str(e))
            return {'prediction': 0, 'confidence': 0.0, 'label': 'error', 'error': f'Prediction failed ― {str(e)}'}

    def categorize_hate_speech(self, text: str, probability: float) -> str:
        if probability < self.confidence_threshold:
            return 'none'
        text_lower = text.lower()
        if any(word in text_lower for word in ['race', 'racial', 'black', 'white', 'asian']):
            return 'racial'
        elif any(word in text_lower for word in ['religion', 'muslim', 'christian', 'jewish']):
            return 'religious'
        elif any(word in text_lower for word in ['woman', 'man', 'female', 'male', 'gender']):
            return 'gender'
        elif any(word in text_lower for word in ['gay', 'lesbian', 'lgbt', 'homosexual']):
            return 'sexual_orientation'
        else:
            return 'general'

    def get_model_info(self) -> Dict[str, Any]:
        if not self.model:
            return {'error': 'Model not loaded'}
        return {'model_type': 'LSTM', 'framework': 'TensorFlow/Keras', 'input_shape': str(self.model.input_shape), 'output_shape': str(self.model.output_shape), 'total_params': int(self.model.count_params()), 'max_sequence_length': self.max_length, 'confidence_threshold': self.confidence_threshold, 'vocab_size': len(self.tokenizer.word_index) if self.tokenizer else 0}

detector = HateSpeechDetector()

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': 'Invalid request format'}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'hate-speech-detection-api', 'version': '1.0.0', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/predict', methods=['POST'])
def predict_hate_speech_endpoint():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field in request body'}), 400
        text = data['text']
        if not isinstance(text, str):
            return jsonify({'error': 'Text must be a string'}), 400
        if len(text) > 1000:
            return jsonify({'error': 'Text too long (max 1000 characters)'}), 400
        result = detector.predict_hate_speech(text)
        result['timestamp'] = datetime.utcnow().isoformat()
        result['api_version'] = '1.0'
        if 'error' in result and result['error']:
            logger.info(f"Error in prediction: {result['error']}")
            return jsonify(result), 400
        label = result.get('label', 'unknown')
        confidence = result.get('confidence', 0)
        logger.info(f"Prediction made: {label} (confidence: {round(confidence, 3)})")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts field in request body'}), 400
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'texts must be a non-empty list'}), 400
        if len(texts) > 10:
            return jsonify({'error': 'Batch size too large (max 10 texts)'}), 400
        results = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                results.append({'index': i, 'prediction': 0, 'confidence': 0.0, 'label': 'error', 'error': 'Text must be a string'})
                continue
            if len(text) > 1000:
                results.append({'index': i, 'prediction': 0, 'confidence': 0.0, 'label': 'error', 'error': 'Text too long (max 1000 characters)'})
                continue
            result = detector.predict_hate_speech(text)
            result['index'] = i
            results.append(result)
        return jsonify({'results': results, 'timestamp': datetime.utcnow().isoformat(), 'total_processed': len(results)})
    except Exception as e:
        logger.error(f"Error in batch_predict endpoint: {str(e)}")
        return jsonify({'error': 'Batch prediction failed', 'message': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        data = request.get_json()
        required_fields = ['text', 'predicted', 'actual', 'feedback_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        feedback_data = {'text': data['text'][:500], 'predicted': data['predicted'], 'actual': data['actual'], 'feedback_type': data['feedback_type'], 'timestamp': datetime.utcnow().isoformat()}
        try:
            with open('feedback.jsonl', 'a') as f:
                f.write(json.dumps(feedback_data) + '\n')
        except Exception as e:
            logger.error("Failed to save feedback: " + str(e))
        logger.info("Feedback received: " + data['feedback_type'])
        return jsonify({'message': 'Feedback received successfully', 'timestamp': datetime.utcnow().isoformat()})
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {str(e)}")
        return jsonify({'error': 'Failed to submit feedback', 'message': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        stats = {'api_status': 'operational', 'model_loaded': detector.model is not None, 'supported_languages': ['en'], 'max_text_length': 1000, 'max_batch_size': 10, 'confidence_threshold': detector.confidence_threshold}
        return jsonify(stats)
    except Exception as e:
        logger.error("Error in stats endpoint: " + str(e))
        return jsonify({'error': 'Failed to get stats'}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    try:
        if detector.model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        info = detector.get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.error("Error in model_info endpoint: " + str(e))
        return jsonify({'error': 'Failed to get model info'}), 500

@app.route('/download', methods=['GET'])
def download_model_file():
    model_path = os.path.join('backend', 'models', 'best_model.keras')
    abs_model_path = os.path.abspath(model_path)
    print("Download model absolute path:", abs_model_path)
    if os.path.exists(abs_model_path):
        return send_file(abs_model_path, as_attachment=True)
    else:
        logger.error("Model file not found: " + abs_model_path)
        return jsonify({'error': 'Model file not found'}), 404

@app.route('/download_page', methods=['GET'])
def download_page():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print("=" * 60)
    print("🚀 HATE SPEECH DETECTION API")
    print("=" * 60)
    print(f"📍 Starting server on port: {port}")
    print(f"🌐 Health check: http://localhost:{port}")
    print(f"🔍 Predict endpoint: http://localhost:{port}/predict")
    print(f"📊 Stats endpoint: http://localhost:{port}/stats")
    print(f"📥 Download model: http://localhost:{port}/download_page")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        print("Error starting server: " + str(e))
        input("Press Enter to exit...")
