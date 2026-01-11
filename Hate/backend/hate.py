"""
Enhanced Hate Speech Detection Logic
Core functionality for text analysis and model inference
"""
import re
import string
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handle text preprocessing and cleaning"""

    def __init__(self):
        self.contractions = {
            "ain't": "are not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions and hashtags (Twitter-style)
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove excessive punctuation
        text = re.sub(r'[{}]+'.format(re.escape(string.punctuation)), ' ', text)

        return text

    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract additional features from text"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'unique_words': len(set(text.split())),
        }

        # Average word length
        words = text.split()
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0

        return features


class HateSpeechClassifier:
    """Main hate speech classification class"""

    def __init__(self, model_path: str, tokenizer_path: str, labels_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.labels_path = labels_path

        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.preprocessor = TextPreprocessor()

        # Model parameters (should match training)
        self.max_length = 100
        self.confidence_threshold = 0.5

        # Hate speech categories mapping
        self.categories = {
            0: 'not_hate',
            1: 'hate'
        }

        # Keywords for fallback detection
        self.hate_keywords = [
            'hate', 'kill', 'die', 'stupid', 'idiot', 'moron', 'loser',
            'pathetic', 'worthless', 'disgusting', 'awful', 'terrible',
            'scum', 'trash', 'garbage', 'useless', 'dumb'
        ]

    def load_model(self) -> bool:
        """Load all model components"""
        try:
            # Load LSTM model
            logger.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)

            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)

            # Load label encoder
            logger.info(f"Loading labels from {self.labels_path}")
            with open(self.labels_path, 'rb') as f:
                self.label_encoder = pickle.load(f)

            logger.info("All model components loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model components: {str(e)}")
            return False

    def preprocess_text(self, text: str) -> np.ndarray:
        """Preprocess text for model input"""
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)

        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences([cleaned_text])

        # Pad sequences
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )

        return padded

    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction on input text"""
        try:
            if not self.model:
                raise ValueError("Model not loaded")

            # Input validation
            if not text or len(text.strip()) < 3:
                return self._create_result(0, 0.0, 'not_hate', 'Text too short')

            if len(text) > 1000:
                text = text[:1000]  # Truncate long text

            # Preprocess and predict
            processed_text = self.preprocess_text(text)
            predictions = self.model.predict(processed_text, verbose=0)

            # Extract prediction probability
            hate_probability = float(predictions[0][0])

            # Determine final prediction
            if hate_probability > self.confidence_threshold:
                prediction = 1
                label = 'hate'
                confidence = hate_probability
            else:
                prediction = 0
                label = 'not_hate'
                confidence = 1 - hate_probability

            # Get additional metadata
            features = self.preprocessor.extract_features(text)
            category = self._categorize_hate(text, hate_probability)

            return {
                'prediction': prediction,
                'label': label,
                'confidence': float(confidence),
                'probability': hate_probability,
                'category': category,
                'features': features,
                'text_length': len(text),
                'processed_length': len(self.preprocessor.clean_text(text))
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return self._create_result(0, 0.0, 'error', str(e))

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions on multiple texts"""
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.predict(text)
                result['index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction error for text {i}: {str(e)}")
                results.append(self._create_result(0, 0.0, 'error', str(e), i))

        return results

    def fallback_detection(self, text: str) -> Dict[str, Any]:
        """Simple keyword-based fallback detection"""
        text_lower = text.lower()
        found_keywords = [kw for kw in self.hate_keywords if kw in text_lower]

        if found_keywords:
            confidence = min(0.8, len(found_keywords) * 0.2)  # Max 0.8 confidence
            return {
                'prediction': 1,
                'label': 'hate',
                'confidence': confidence,
                'probability': confidence,
                'category': 'keyword_based',
                'found_keywords': found_keywords,
                'method': 'fallback'
            }
        else:
            return {
                'prediction': 0,
                'label': 'not_hate', 
                'confidence': 0.6,
                'probability': 0.4,
                'category': 'none',
                'method': 'fallback'
            }

    def _create_result(self, prediction: int, confidence: float, 
                      label: str, error: str = None, index: int = None) -> Dict[str, Any]:
        """Create standardized result dictionary"""
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'label': label,
            'probability': confidence if prediction == 1 else 1 - confidence,
            'category': 'none',
        }

        if error:
            result['error'] = error
        if index is not None:
            result['index'] = index

        return result

    def _categorize_hate(self, text: str, probability: float) -> str:
        """Categorize type of hate speech"""
        if probability < self.confidence_threshold:
            return 'none'

        text_lower = text.lower()

        # Define category keywords
        categories = {
            'racial': ['race', 'racial', 'black', 'white', 'asian', 'hispanic', 'latino'],
            'religious': ['religion', 'muslim', 'christian', 'jewish', 'hindu', 'buddhist'],
            'gender': ['woman', 'man', 'female', 'male', 'gender', 'feminist'],
            'sexual_orientation': ['gay', 'lesbian', 'lgbt', 'homosexual', 'bisexual'],
            'disability': ['disabled', 'handicapped', 'retard', 'mental'],
            'nationality': ['immigrant', 'foreigner', 'citizen', 'american', 'mexican']
        }

        # Check for category matches
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category

        return 'general'

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        if not self.model:
            return {'error': 'Model not loaded'}

        return {
            'model_type': 'LSTM',
            'framework': 'TensorFlow/Keras',
            'input_shape': str(self.model.input_shape),
            'output_shape': str(self.model.output_shape),
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) 
                                   for w in self.model.trainable_weights]),
            'max_sequence_length': self.max_length,
            'confidence_threshold': self.confidence_threshold,
            'vocab_size': len(self.tokenizer.word_index) if self.tokenizer else 0,
            'supported_categories': list(self.categories.values())
        }


# Utility functions
def validate_text_input(text: str, max_length: int = 1000) -> Tuple[bool, str]:
    """Validate text input"""
    if not text:
        return False, "Empty text provided"

    if not isinstance(text, str):
        return False, "Text must be a string"

    if len(text.strip()) < 3:
        return False, "Text too short (minimum 3 characters)"

    if len(text) > max_length:
        return False, f"Text too long (maximum {max_length} characters)"

    return True, "Valid"


def calculate_batch_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for batch predictions"""
    if not results:
        return {}

    total = len(results)
    hate_count = sum(1 for r in results if r.get('prediction') == 1)
    error_count = sum(1 for r in results if 'error' in r)

    confidences = [r.get('confidence', 0) for r in results if 'error' not in r]
    avg_confidence = np.mean(confidences) if confidences else 0

    return {
        'total_processed': total,
        'hate_detected': hate_count,
        'not_hate': total - hate_count - error_count,
        'errors': error_count,
        'hate_percentage': (hate_count / total) * 100 if total > 0 else 0,
        'average_confidence': float(avg_confidence),
        'success_rate': ((total - error_count) / total) * 100 if total > 0 else 0
    }