# Loan Approval Prediction with Imbalanced Data Handling

A binary classification project that predicts loan approval status using multiple algorithms and SMOTE technique to handle class imbalance.

## 📊 Dataset
- **Source**: [Loan Approval Prediction (Kaggle)](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Features**: Income, loan amount, credit score, employment status, etc.
- **Target**: Loan approval status (Approved/Rejected)
- **Challenge**: Imbalanced dataset

## 🛠️ Technologies Used
- Python 3.x
- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib
- seaborn

## 📁 Project Structure
```
├── main.py                           # Main script
├── loan_approval_dataset.csv         # Dataset (download separately)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/ML-Task4-Loan-Approval-Prediction.git
cd ML-Task4-Loan-Approval-Prediction
```

2. **Create a virtual environment:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
- Download from [Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- Save as `loan_approval_dataset.csv` in the project folder

5. **Run the script:**
```bash
python main.py
```

## 📈 Features
- Data cleaning and preprocessing
- Handling missing values
- Label encoding for categorical features
- Exploratory data analysis with visualizations
- Multiple classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest with SMOTE
- Class imbalance handling using SMOTE
- Model comparison and evaluation
- Confusion matrices for each model
- Performance metrics (accuracy, precision, recall, F1-score)

## 📊 Results
The Random Forest model with SMOTE achieves the best balance between precision and recall, effectively handling the imbalanced dataset:
- Improved recall for minority class
- Better F1-score overall
- Comprehensive model comparison visualizations

