# Linux Process Anomaly Detection

## 📌 Project Overview
**Linux Process Anomaly Detection** is a research project aimed at detecting anomalous processes in Linux operating systems based on behavioral analysis and process structure combined with Machine Learning models.

The project uses the **OS Anomaly Detection** dataset from Kaggle and achieved a **score of 0.90 – Top 5 Leaderboard**, demonstrating practical application potential in process monitoring and system security.

## 🎯 Problem Statement
Detecting anomalous processes on Linux plays a crucial role in:
- Malware detection
- Preventing CPU / Memory / I/O exploitation
- Protecting systems against malicious behaviors

Traditional rule-based or threshold-based methods:
- Cannot detect new processes
- Cannot exploit complex relationships between features

👉 Proposed solution: **Machine Learning based on process behavior & structure**

---

## 📂 Dataset
**OS Anomaly Detection Dataset (Kaggle – 04/2025)**  
- Train: labeled  
- Test: unlabeled  
- 763,144 records  
- 8 main features (kernel-level monitoring)

### Features
| Feature | Description |
|------|------------|
| processId | Process ID |
| threadId | Thread ID |
| parentProcessId | Parent process ID |
| userId | User ID (0 = root) |
| mountNamespace | Mount namespace |
| argsNum | Number of arguments |
| returnValue | System call return value |
| target | Label (0: normal, 1: anomaly) |

🔗 **Kaggle Competition**:  
https://www.kaggle.com/competitions/data-bounty-2-os-anomaly-detection

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Retain duplicate records (reflecting behavior frequency)
- Feature normalization using `StandardScaler`
- Avoid data leakage using `GroupShuffleSplit`

---

### 2️⃣ Feature Engineering
#### a. Process Structure Features
- Is root process
- Is child process
- PID = TID

#### b. Frequency-based Features
- Occurrence count of user / process / parent process
- Number of unique processes by user / mount namespace

#### c. Behavior Ratio Features
- Child process ratio
- Error return value ratio
- Average number of arguments

#### d. Return Value Encoding
- Success / Failure / Zero return

---

### 3️⃣ Models
| Model | Tuning Method |
|-----|--------------|
| Logistic Regression | GridSearchCV |
| Random Forest | RandomizedSearchCV |
| XGBoost | Optuna |
| LightGBM | Optuna |

---

## 📊 Results

| Model | F1 Macro | F1 (Anomaly) | Time (s) |
|-----|---------|--------------|---------|
| Logistic Regression | 0.496 | 0.011 | 656 |
| Random Forest | 0.792 | 0.584 | 717 |
| XGBoost | 0.780 | 0.561 | 1605 |
| **LightGBM** | **0.844** | **0.688** | 1898 |

✅ **Best Model: LightGBM**  
- Optimal threshold: **0.93**
- 33/37 anomalies correctly detected
- Very low False Positive rate

---

## 🏆 Kaggle Submission
- Score: **0.90**
- Rank: **Top 5 Leaderboard**
- Model: **LightGBM + Feature Engineering**

---

## 🧪 Demo / Reproducibility
👉 **Kaggle Notebook Demo (Public)**  
> *(Insert your notebook link here)*  

## ⚙️ Installation & Setup

### 1️⃣ Clone & Setup Environment

```bash
git clone https://github.com/c1khoa/linux-process-anomaly-detection.git
cd linux-process-anomaly-detection

conda create -n os-anomaly python=3.10 -y
conda activate os-anomaly

pip install -r requirements.txt
```

### 2️⃣ Run

Train & finetune models from scratch:
```bash
python main.py
```

Run demo using Streamlit:
```bash
streamlit run app.py
```