import pandas as pd
import joblib
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from features import add_manual_features_df
data_path = "data/processed.csv"

df = pd.read_csv(data_path)
df = add_manual_features_df(df)

y_class = df['problem_class']
y_score = df['problem_score']

df_train, df_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    df, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
)

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_text = tfidf.fit_transform(df_train['combined_text'])
X_test_text = tfidf.transform(df_test['combined_text'])

manual_cols = ['text_length', 'math_symbols', 'high_diff_score', 'medium_diff_score']
scaler = StandardScaler()

X_train_manual = scaler.fit_transform(df_train[manual_cols].values)
X_test_manual = scaler.transform(df_test[manual_cols].values)

X_train = hstack([X_train_text, X_train_manual])
X_test = hstack([X_test_text, X_test_manual])

clf = RandomForestClassifier(
    n_estimators=300,        
    max_depth=15,          
    min_samples_split=5,     
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_class_train)
y_pred_class = clf.predict(X_test)

accuracy = accuracy_score(y_class_test, y_pred_class)
precision = precision_score(y_class_test, y_pred_class, average='weighted')
recall = recall_score(y_class_test, y_pred_class, average='weighted')
f1 = f1_score(y_class_test, y_pred_class, average='weighted')

print("\n--- Classification Results ---")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_class_test, y_pred_class))

cm = confusion_matrix(y_class_test, y_pred_class)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=clf.classes_,
    yticklabels=clf.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

reg = RandomForestRegressor( n_estimators=300, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1 )

reg.fit(X_train, y_score_train)
y_pred_score = reg.predict(X_test)

mae = mean_absolute_error(y_score_test, y_pred_score)
rmse = np.sqrt(mean_squared_error(y_score_test, y_pred_score))

print("\n--- Regression Results ---")
print("MAE:", mae)
print("RMSE:", rmse)

joblib.dump(clf, "classifier.pkl")
joblib.dump(reg, "regressor.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(scaler, "scaler.pkl")

print(" Done! Models trained and saved.")

