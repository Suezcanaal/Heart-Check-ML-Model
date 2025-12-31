# Heart Disease Risk Predictor with SHAP Explanations

An AI-powered heart disease risk assessment tool that provides explainable predictions using machine learning and SHAP (SHapley Additive exPlanations) analysis.

## 🚀 Features

- **ML Prediction**: XGBoost classifier trained on heart disease dataset
- **Explainable AI**: SHAP values explain why each prediction was made
- **REST API**: FastAPI backend with automatic documentation
- **Web Interface**: Interactive HTML frontend for easy testing
- **Docker Support**: Containerized deployment ready

## 🏗️ Architecture

### Backend (FastAPI)
- **Model**: XGBoost classifier (`heart_model.pkl`)
- **Explainer**: SHAP TreeExplainer for feature importance
- **API Endpoint**: `/predict_risk` accepts 13 medical parameters
- **Response**: Prediction (0/1), probability, and SHAP explanations

### Frontend (HTML/JavaScript)
- Clean web interface with form validation
- Real-time API communication
- Visual SHAP explanation display
- Responsive design

## 📊 Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| `age` | Age in years | Integer |
| `sex` | Gender (1=Male, 0=Female) | 0-1 |
| `cp` | Chest pain type | 0-3 |
| `trestbps` | Resting blood pressure | Integer |
| `chol` | Cholesterol level | Integer |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0-1 |
| `restecg` | Resting ECG results | 0-2 |
| `thalach` | Maximum heart rate achieved | Integer |
| `exang` | Exercise induced angina | 0-1 |
| `oldpeak` | ST depression induced by exercise | Float |
| `slope` | Slope of peak exercise ST segment | 0-2 |
| `ca` | Number of major vessels colored by fluoroscopy | 0-3 |
| `thal` | Thalassemia type | 0-3 |

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip
```

### Local Development
```bash
# Clone repository
git clone https://github.com/Suezcanaal/Heart-Check-ML-Model.git
cd Heart-Check-ML-Model

# Install dependencies
pip install -r requirements.txt

# Train model (generates heart_model.pkl)
python train_heart_model.py

# Start API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build image
docker build -t heart-disease-api .

# Run container
docker run -p 8000:8000 heart-disease-api
```

## 🔧 Usage

### API Testing
```bash
# Health check
curl http://localhost:8000/

# Prediction request
curl -X POST "http://localhost:8000/predict_risk" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

### Web Interface
1. Open `http://localhost:8000/index.html`
2. Fill in patient data
3. Click "Analyze Risk"
4. View prediction + SHAP explanations

## 📈 API Response Format

```json
{
  "prediction": 1,
  "risk_probability": 0.85,
  "shap_explanation": {
    "age": 0.12,
    "cp": 0.45,
    "thalach": -0.23,
    "ca": 0.67,
    "thal": 0.34,
    ...
  }
}
```

- **prediction**: 0 (Low Risk) or 1 (High Risk)
- **risk_probability**: Probability score (0.0-1.0)
- **shap_explanation**: Feature contributions (positive = increases risk)

## 🧠 SHAP Explanations

SHAP values explain individual predictions by showing how each feature contributes:
- **Positive values**: Increase heart disease risk
- **Negative values**: Decrease heart disease risk
- **Magnitude**: Strength of the contribution

## 📁 Project Structure

```
├── main.py                 # FastAPI application
├── train_heart_model.py    # Model training script
├── index.html             # Web frontend
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── heart_model.pkl       # Trained XGBoost model (generated)
└── README.md            # This file
```

## 🔬 Technical Details

- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Explainability**: SHAP TreeExplainer
- **Framework**: FastAPI with Pydantic validation
- **Serialization**: Joblib for model persistence
- **Frontend**: Vanilla JavaScript with Fetch API

## ⚠️ Disclaimer

This tool is for educational/research purposes only. Not intended for actual medical diagnosis. Always consult healthcare professionals for medical decisions.

## 📄 License

MIT License - See LICENSE file for details.