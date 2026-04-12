# FraudShield — A Fraud Detector

> ML-powered Credit & Debit Card Fraud Detection system. Real-time risk scoring with a clean dark web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?style=flat-square&logo=scikit-learn&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-yellow?style=flat-square&logo=javascript&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**[Live Demo →](https://codebreaker-0111.github.io/FraudShield---A-Fraud-Detector/)**

---

## What it does

FraudShield analyzes credit and debit card transactions in real time and flags suspicious ones using a Random Forest model. Enter transaction details — amount, location, time, merchant type — and get an instant fraud probability score with a breakdown of what triggered it.

No backend needed in the live version. The entire ML scoring logic runs in the browser.

---

## Features

- Real-time fraud probability scoring (0–100%)
- Supports both credit card and debit card transactions
- Risk level classification — Low / Medium / High
- Explains which factors contributed to the decision
- Dashboard with fraud stats and charts
- Model info page with feature importance and evaluation metrics
- Full Python/Flask version included for local ML training

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Random Forest (scikit-learn) |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Backend (local) | Flask (Python) |
| Frontend | HTML, CSS, Vanilla JS |
| Charts | Chart.js |

---

## Two Versions

### 1. Static (Live on GitHub Pages)
`index.html` + `style.css` + `app.js` — runs entirely in the browser. No server needed. The Random Forest scoring logic is implemented in JavaScript.

### 2. Full Python Version (Local)
Flask backend with a real scikit-learn Random Forest model trained on actual data. Use this for proper ML training, evaluation, and model metrics.

---

## Run Locally (Full Python Version)

### 1. Clone the repo
```bash
git clone https://github.com/CodeBreaker-0111/FraudShield---A-Fraud-Detector.git
cd FraudShield---A-Fraud-Detector
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate dataset

**Option A — Synthetic data**
```bash
python generate_data.py
```

**Option B — Kaggle dataset**
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place it in the `data/` folder
3. Run:
```bash
python prepare_kaggle_data.py
```

### 5. Train the model
```bash
python train_model.py
# or for Kaggle data:
python train_model.py --dataset kaggle
```

### 6. Run the app
```bash
python app.py
```

Open **http://127.0.0.1:5000**

---

## How the ML works

1. Raw transaction data loaded from CSV
2. Features engineered — amount ratio, night flag, velocity, distance flags
3. 80/20 train/test split (stratified)
4. `StandardScaler` normalizes features
5. `SMOTE` oversamples fraud class in training set only
6. `RandomForestClassifier` (100 trees) trained
7. Outputs probability score 0.0 → 1.0
8. Score ≥ 0.5 = fraud

### Features used

| Feature | Description |
|---|---|
| `amount_ratio` | Amount vs avg daily spend |
| `merchant_risk` | Risk score of merchant category |
| `distance_from_home` | Distance from cardholder's home |
| `txn_per_hour` | Transactions in last hour |
| `is_night` | 1 if between 1–5 AM |
| `new_device` | New device or location flag |
| `international` | International transaction flag |
| `high_velocity` | 1 if 4+ transactions per hour |

---

## API (Local Version)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web app |
| `/api/predict` | POST | Single transaction analysis |
| `/api/batch_predict` | POST | Multiple transactions |
| `/api/metrics` | GET | Model performance metrics |

**Example:**
```python
import requests

txn = {
    "card_type": "credit",
    "amount": 89000,
    "merchant_category": "jewelry",
    "hour": 2,
    "distance_from_home": 180,
    "txn_per_hour": 3,
    "avg_monthly_spend": 20000,
    "new_device": 1,
    "international": 1
}

response = requests.post("http://127.0.0.1:5000/api/predict", json=txn)
print(response.json())
```

---

## Author

**Aaditya Bansal**

GitHub — [@CodeBreaker-0111](https://github.com/CodeBreaker-0111)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
