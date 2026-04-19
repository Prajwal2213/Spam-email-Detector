# 🛡️ SecureMail AI - Email Spam Detector

An advanced, production-ready Email Spam Detector built from scratch using Machine Learning. This project features a robust text classification model served via a high-performance Python ASGI backend and wrapped in a premium, glassmorphism-styled web interface.

## ✨ Features
- **Scikit-Learn ML Pipeline**: Utilizes TF-IDF vectorization and a Multinomial Naive Bayes classifier to analyze text patterns and isolate malicious mail with ~98% accuracy.
- **Asynchronous API**: Powered by FastAPI to guarantee blazing fast model inference and endpoint response.
- **Dynamic UI**: A lightweight Vanilla CSS frontend packed with micro-animations and responsive glassmorphism design parameters—zero bulky frameworks attached.
- **Built-in Safety Thresholds**: Hard-coded logic inside the proxy overrides native model predictions if confidence hovers beneath 70%.
- **MLflow Integration**: Automated, built-in experiment tracking. Logs algorithms, parameters, metrics (Accuracy, Precision, Recall, F1), and saves pickled sklearn pipelines.
- **Dockerized Foundation**: Completely containerized for "develop once, deploy anywhere" capability.

## 🛠️ Tech Stack
- **Machine Learning**: `scikit-learn`, `pandas`, `joblib`
- **Experiment Tracking**: `mlflow`
- **Backend / API**: `FastAPI`, `uvicorn`
- **Frontend**: Vanilla HTML5, CSS3, JavaScript
- **DevOps**: Docker, Docker Hub
- **MLFlow**: MLflow Models

---

## 🚀 Getting Started

There are two ways to run this project: fetching the pre-built Docker container or running it manually from source.

### Option 1: The Easy Way (Docker)
The easiest way to get the app running on any laptop or server is to pull the live image directly from Docker Hub. No codebase dependencies required!

```bash
docker run -p 8000:8000 panda2213/spam-email-detector:latest
```
Once the server binds port 8000, simply navigate to `http://localhost:8000` in your web browser to interact with the Spam Detector!

### Option 2: Local Development Setup
If you want to modify the Python architecture, train new models on custom data, or interact with MLflow dashboards, you can run the source directly.

1. **Clone the repository**
   ```bash
   git clone https://github.com/Prajwal2213/Spam-email-Detector.git
   cd Spam-email-Detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model & Generate MLflow logs**
   The training script automatically handles fetching the dataset, splitting test blocks, training the classifier, saving the artifacts, and logging metrics into MLflow.
   ```bash
   python train_model.py
   ```

4. **Start the API Server**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   Open `http://127.0.0.1:8000` to interact with the frontend UI.

5. **Start the MLflow Tracking UI** (Optional)
   To view the experiment logs and tracked metrics across different historical algorithm runs:
   ```bash
   mlflow ui
   ```
   Open `http://localhost:5000` to view the comprehensive ML analytics dashboard!
