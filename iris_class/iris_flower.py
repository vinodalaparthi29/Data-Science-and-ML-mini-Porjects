import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("IRIS.csv")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\n" + "=" * 40)
print("     LOGISTIC REGRESSION RESULTS")
print("=" * 40)
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.2f}")
print(classification_report(y_test, lr_preds,
      target_names=le.classes_))

print("=" * 40)
print("     RANDOM FOREST RESULTS")
print("=" * 40)
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.2f}")
print(classification_report(y_test, rf_preds,
      target_names=le.classes_))


models = ['Logistic Regression', 'Random Forest']
accuracies = [accuracy_score(y_test, lr_preds),
              accuracy_score(y_test, rf_preds)]

plt.figure(figsize=(7, 4))
bars = plt.bar(models, accuracies, color=['steelblue', 'green'])
plt.ylim(0.9, 1.0)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.002,
             f'{acc:.2f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.show()


feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh', color='green',
                             figsize=(7, 4))
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()


new_flower = [[5.1, 3.5, 1.4, 0.2]]  # sample measurements
prediction = rf.predict(new_flower)
print(f"\nNew flower prediction: {le.inverse_transform(prediction)[0]}")