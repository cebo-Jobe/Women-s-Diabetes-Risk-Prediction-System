import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.security import check_password_hash

from data_utils import DataSplitter, ImbalanceHandler
from database import create_user, get_prediction_history, get_user_by_email, init_db, save_prediction
from DecisionTree import DecisionTree  # noqa: F401  -- needed for joblib to unpickle RandomForest
from RandomForest import RandomForest

# ============================================================
# Initialize Flask app
# ============================================================
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app, supports_credentials=True)

MODEL_PATH = 'models/random_forest_model.joblib'
METADATA_PATH = 'models/model_metadata.joblib'

NUMERIC_KEYS = [
    'pregnancies', 'glucose', 'bloodPressure', 'skinThickness',
    'insulin', 'bmi', 'diabetesPedigree', 'age',
]
HISTORY_FEATURE_COLS = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree', 'age',
]

# Global model state
model = None
model_metadata = None
# Fallback only -- overwritten from the model's own saved metadata as
# soon as load_model()/train_and_save_model() runs, so this never
# drifts out of sync with what the model actually saw during training.
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
]


# ==================== SERIALIZATION HELPER ====================

def to_serializable(obj):
    """
    Recursively convert NumPy scalars/arrays (and pandas artifacts that
    carry them) into plain Python types jsonify() can handle.

    This matters more than it looks: numpy.float64 happens to subclass
    Python's float, so it slips through Flask's JSON encoder by luck.
    numpy.int64 does NOT subclass int, so any dict/record containing one
    (e.g. a straight `df.to_dict('records')` on a DataFrame with integer
    columns) raises `TypeError: Object of type int64 is not JSON
    serializable` at request time. Every value that could touch pandas
    or NumPy goes through this before jsonify().
    """
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, 'tolist'):  # numpy array / pandas Series
        return to_serializable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ==================== MODEL LOADING / TRAINING ====================

def train_and_save_model():
    """
    Train a fresh model from diabetes.csv and save it + its metadata.
    Used both when no saved model exists yet, and by POST /api/retrain.
    """
    global model, model_metadata, feature_names

    data = pd.read_csv('diabetes.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    trained_feature_names = data.columns[:-1].tolist()

    splitter = DataSplitter(test_size=0.2, random_state=1234, stratify=True)
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    imbalance_handler = ImbalanceHandler(strategy='smote', random_state=1234)
    X_train_resampled, y_train_resampled = imbalance_handler.fit_resample(X_train, y_train)

    new_model = RandomForest()
    new_model.fit(X_train_resampled, y_train_resampled)
    metrics = new_model.evaluate(X_test, y_test)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    new_model.save_model(MODEL_PATH)

    metadata = {
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': trained_feature_names,
        'model_metrics': metrics,
        'training_date': datetime.now().isoformat(),
        'model_params': {
            'n_trees': new_model.n_trees,
            'max_depth': new_model.max_depth,
            'min_samples_split': new_model.min_samples_split,
            'n_features': new_model.n_features,
        },
    }
    joblib.dump(metadata, METADATA_PATH)

    model = new_model
    model_metadata = metadata
    feature_names = trained_feature_names


def load_model():
    """Load the trained model and metadata at startup, or train one if missing."""
    global model, model_metadata, feature_names
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
            print("Loading existing model...")
            model = RandomForest.load_model(MODEL_PATH)
            model_metadata = joblib.load(METADATA_PATH)
            if model_metadata.get('feature_names'):
                feature_names = model_metadata['feature_names']
            print(f"Model loaded successfully! Accuracy: {model_metadata['model_metrics']['accuracy']:.4f}")
        else:
            print("No saved model found -- training a new one from diabetes.csv...")
            train_and_save_model()
    except Exception as e:  # noqa: BLE001
        print(f"Error loading model: {e}")
        print("Attempting to train a new model instead...")
        train_and_save_model()


load_model()
init_db()


# ==================== ROUTES ====================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json or {}
    # Lowercased so "Jane@x.com" and "jane@x.com" are treated as the
    # same account on both register and login (they weren't before).
    email = (data.get('email') or '').strip().lower()
    password = (data.get('password') or '').strip()

    if not email or not password:
        return jsonify({'success': False, 'message': 'Please fill in all fields'}), 400

    user = get_user_by_email(email)
    if user and check_password_hash(user['password'], password):
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        session['user_email'] = user['email']
        return jsonify({
            'success': True,
            'user': {'id': user['id'], 'name': user['name'], 'email': user['email']},
        })

    return jsonify({'success': False, 'message': 'Invalid email or password'}), 401


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json or {}
    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = (data.get('password') or '').strip()

    if not all([name, email, password]):
        return jsonify({'success': False, 'message': 'Please fill in all fields'}), 400
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400

    try:
        user_id = create_user(email, password, name)
        return jsonify({'success': True, 'user_id': user_id, 'message': 'Registration successful!'})
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/check_session', methods=['GET'])
def check_session():
    if session.get('user_id'):
        return jsonify({
            'logged_in': True,
            'user': {
                'id': session['user_id'],
                'name': session.get('user_name', ''),
                'email': session.get('user_email', ''),
            },
        })
    return jsonify({'logged_in': False})


@app.route('/api/predict', methods=['POST'])
def predict():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Please login first'}), 401

    data = request.json or {}

    try:
        user_data = {k: float(data[k]) for k in NUMERIC_KEYS}
    except (KeyError, TypeError, ValueError):
        return jsonify({'success': False, 'message': 'Missing or invalid measurement values'}), 400

    for key, value in user_data.items():
        if value < 0:
            return jsonify({'success': False, 'message': f'Please enter a valid value for {key}'}), 400

    feature_values = [user_data[k] for k in NUMERIC_KEYS]

    try:
        risk_percentages, risk_categories, binary_predictions = model.predict_risk_percentage(
            np.array([feature_values])
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({'success': False, 'message': f'Prediction failed: {e}'}), 500

    risk_percentage = float(risk_percentages[0])
    risk_category = str(risk_categories[0])
    binary_prediction = int(binary_predictions[0])

    save_prediction(user_id, user_data, binary_prediction)

    return jsonify({
        'success': True,
        'risk_percentage': risk_percentage,
        'risk_category': risk_category,
        'binary_prediction': binary_prediction,
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Please login first'}), 401

    try:
        df = get_prediction_history(user_id)
        if df.empty:
            return jsonify({'success': True, 'history': []})

        # Score every row through the real model in ONE batched call
        # (not once per row, and not a hardcoded guess).
        X = df[HISTORY_FEATURE_COLS].values
        risk_percentages, risk_categories, _ = model.predict_risk_percentage(X)

        df = df.reset_index(drop=True)
        history = []
        for i, row in df.iterrows():
            record = row.to_dict()
            record['timestamp'] = str(record.get('timestamp', ''))
            record['risk_percentage'] = float(risk_percentages[i])
            record['risk_category'] = str(risk_categories[i])
            # Every remaining value (pregnancies, age, predicted_outcome,
            # id, user_id, ...) is still numpy int64/float64 at this
            # point -- this is what actually made the old endpoint 500.
            history.append(to_serializable(record))

        return jsonify({'success': True, 'history': history})
    except Exception as e:  # noqa: BLE001
        return jsonify({'success': False, 'message': f'Failed to load history: {e}'}), 500


@app.route('/api/profile', methods=['GET'])
def get_profile():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Please login first'}), 401

    try:
        user = {'id': user_id, 'name': session.get('user_name', ''), 'email': session.get('user_email', '')}
        df = get_prediction_history(user_id)
        total_predictions = len(df)
        high_risk_count = int((df['predicted_outcome'] == 1).sum()) if not df.empty else 0
        low_risk_count = total_predictions - high_risk_count
        risk_percentage = (high_risk_count / total_predictions * 100) if total_predictions > 0 else 0

        return jsonify({
            'success': True,
            'user': user,
            'stats': {
                'total_predictions': total_predictions,
                'high_risk_count': high_risk_count,
                'low_risk_count': low_risk_count,
                'risk_percentage': round(risk_percentage, 1),
            },
        })
    except Exception as e:  # noqa: BLE001
        return jsonify({'success': False, 'message': f'Failed to load profile: {e}'}), 500


@app.route('/api/model/metrics', methods=['GET'])
def get_model_metrics():
    try:
        if model_metadata and 'model_metrics' in model_metadata:
            metrics = to_serializable(model_metadata['model_metrics'])
            return jsonify({'success': True, 'metrics': {
                'accuracy': float(metrics.get('accuracy', 0.0)),
                'precision': float(metrics.get('precision', 0.0)),
                'recall': float(metrics.get('recall', 0.0)),
                'f1': float(metrics.get('f1', 0.0)),
                'confusion_matrix': metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
            }})
        return jsonify({'success': False, 'message': 'Model metrics not available'}), 404
    except Exception as e:  # noqa: BLE001
        return jsonify({'success': False, 'message': f'Server error: {e}'}), 500


@app.route('/api/model/feature_importance', methods=['GET'])
def get_feature_importance():
    try:
        if model is None or model.feature_importances_ is None:
            return jsonify({'success': False, 'message': 'Feature importances not available'}), 404

        importances = to_serializable(model.feature_importances_)
        sorted_indices = np.argsort(importances)[::-1]

        return jsonify({
            'success': True,
            'features': [feature_names[i] for i in sorted_indices],
            'importances': [float(importances[i]) for i in sorted_indices],
        })
    except Exception as e:  # noqa: BLE001
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Please login first'}), 401

    try:
        train_and_save_model()
        return jsonify({'success': True, 'message': 'Model retrained and saved successfully.'})
    except Exception as e:  # noqa: BLE001
        return jsonify({'success': False, 'message': f'Failed to retrain model: {e}'}), 500


# ==================== ERROR HANDLING ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'message': 'Route not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)

    print("=" * 50)
    print(" Diabetes Risk Prediction System")
    print("=" * 50)
    print(f" Static folder: {os.path.abspath('static')}")
    print(" Server running at: http://127.0.0.1:5000")
    print("=" * 50)

    app.run(host='127.0.0.1', port=5000, debug=True)