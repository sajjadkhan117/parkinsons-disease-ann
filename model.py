# ============================================
#   Parkinson's Disease Detection Using ANN
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Dataset ──────────────────────────
df = pd.read_csv('parkinsons.csv')
print("Dataset Shape:", df.shape)
print(df.head())

# ── 2. Prepare Features ──────────────────────
X = df.drop(['name', 'status'], axis=1)
y = df['status']
print("\nClass Distribution:\n", y.value_counts())

# ── 3. Split Data ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── 4. Scale Features ─────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 5. Build ANN Model ────────────────────────
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ── 6. Train Model ────────────────────────────
early_stop = EarlyStopping(monitor='val_loss', patience=10,
                           restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ── 7. Evaluate Model ─────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.4f}")

# ── 8. Confusion Matrix ───────────────────────
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy','Parkinson'],
            yticklabels=['Healthy','Parkinson'])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ── 9. Training History ───────────────────────
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# ── 10. Save Model ────────────────────────────
model.save('parkinsons_model.h5')
import joblib
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Model & Scaler Saved Successfully!")

# ── 11. ROC Curve ─────────────────────────────
from sklearn.metrics import roc_curve, auc

y_pred_prob = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - AUC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()

# ── 12. Feature Importance ────────────────────
feature_weights = np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
feature_names = X.columns

plt.figure(figsize=(12,6))
sorted_idx = np.argsort(feature_weights)[::-1]
plt.bar(range(len(feature_names)), 
        feature_weights[sorted_idx], color='steelblue')
plt.xticks(range(len(feature_names)), 
           feature_names[sorted_idx], rotation=45, ha='right')
plt.title('Feature Importance (ANN Input Layer Weights)')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# ── 13. Correlation Heatmap ───────────────────
plt.figure(figsize=(14,10))
sns.heatmap(df.drop('name', axis=1).corr(), 
            cmap='coolwarm', annot=False, 
            linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# ── 14. Class Distribution ────────────────────
plt.figure(figsize=(5,4))
df['status'].value_counts().plot(kind='bar', 
    color=['steelblue','tomato'], edgecolor='black')
plt.xticks([0,1], ['Parkinson','Healthy'], rotation=0)
plt.title("Class Distribution in Dataset")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

print("\n✅ ALL PLOTS SAVED SUCCESSFULLY!")