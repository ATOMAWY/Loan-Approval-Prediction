

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('loan_approval_dataset.csv')

print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

print("\nLoan Status distribution:")
print(df[' loan_status'].value_counts())


numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

label_encoders = {}
for col in categorical_cols:
    if col != ' loan_status':  
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

le_target = LabelEncoder()
df[' loan_status'] = le_target.fit_transform(df[' loan_status'])

print("\nEncoded target classes:")
for i, class_name in enumerate(le_target.classes_):
    print(f"{i}: {class_name}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].bar(['Rejected', 'Approved'], df[' loan_status'].value_counts().sort_index())
axes[0, 0].set_title('Loan Status Distribution')
axes[0, 0].set_ylabel('Count')

df.boxplot(column=' income_annum', by=' loan_status', ax=axes[0, 1])
axes[0, 1].set_title('Income by Loan Status')
axes[0, 1].set_xlabel('Loan Status (0=Rejected, 1=Approved)')

df.boxplot(column=' loan_amount', by=' loan_status', ax=axes[1, 0])
axes[1, 0].set_title('Loan Amount by Status')
axes[1, 0].set_xlabel('Loan Status (0=Rejected, 1=Approved)')

df.boxplot(column=' cibil_score', by=' loan_status', ax=axes[1, 1])
axes[1, 1].set_title('CIBIL Score by Status')
axes[1, 1].set_xlabel('Loan Status (0=Rejected, 1=Approved)')

plt.tight_layout()
plt.savefig('task4_exploratory_analysis.png')
plt.show()


if ' loan_id' in df.columns:
    df = df.drop(' loan_id', axis=1)

X = df.drop(' loan_status', axis=1)
y = df[' loan_status']

print("\nClass distribution:")
print(y.value_counts())
print(f"Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}:1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

print("\n=== Training Logistic Regression ===")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\nLogistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

print("\n=== Training Decision Tree ===")
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\nDecision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix - Logistic Regression')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Confusion Matrix - Decision Tree')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('task4_confusion_matrices.png')
plt.show()

print("\n=== BONUS: Using SMOTE for Class Imbalance ===")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nOriginal training set distribution:")
print(y_train.value_counts())
print(f"\nSMOTE resampled training set distribution:")
print(pd.Series(y_train_smote).value_counts())

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)

print("\nRandom Forest (with SMOTE) Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_smote):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_smote):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_smote):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_smote):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote))

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'RF + SMOTE'],
    'Accuracy': [accuracy_score(y_test, y_pred_lr), 
                 accuracy_score(y_test, y_pred_dt),
                 accuracy_score(y_test, y_pred_smote)],
    'Precision': [precision_score(y_test, y_pred_lr),
                  precision_score(y_test, y_pred_dt),
                  precision_score(y_test, y_pred_smote)],
    'Recall': [recall_score(y_test, y_pred_lr),
               recall_score(y_test, y_pred_dt),
               recall_score(y_test, y_pred_smote)],
    'F1-Score': [f1_score(y_test, y_pred_lr),
                 f1_score(y_test, y_pred_dt),
                 f1_score(y_test, y_pred_smote)]
})

print("\n=== Model Comparison ===")
print(comparison)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

comparison.plot(x='Model', y=['Precision', 'Recall', 'F1-Score'], 
                kind='bar', ax=axes[0], rot=45)
axes[0].set_title('Model Comparison: Precision, Recall, F1-Score')
axes[0].set_ylabel('Score')
axes[0].legend(loc='lower right')
axes[0].set_ylim([0, 1])

comparison.plot(x='Model', y='Accuracy', kind='bar', ax=axes[1], rot=45, color='orange')
axes[1].set_title('Model Comparison: Accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('task4_model_comparison.png')
plt.show()

print("\nTask 4 completed successfully!")
