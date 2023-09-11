import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# 1. Cargar el dataset
data = pd.read_csv('ds_salaries.csv')

# 2. Crear rangos salariales
percentiles = data['salary_in_usd'].quantile([0.333, 0.666]).values
data['salary_range'] = pd.cut(data['salary_in_usd'], 
                              bins=[0, percentiles[0], percentiles[1], float('inf')], 
                              labels=[0, 1, 2])  # 0: Low, 1: Medium, 2: High

# 3. Encoding de variables categóricas
one_hot_cols = ['experience_level', 'employment_type', 'salary_currency', 'company_size']
data_one_hot = pd.get_dummies(data, columns=one_hot_cols, drop_first=True)
label_encoders = {}
for col in ['job_title', 'employee_residence', 'company_location']:
    le = LabelEncoder()
    data_one_hot[col] = le.fit_transform(data_one_hot[col])

# 4. Dividir el dataset y normalizar las características
X = data_one_hot.drop(columns=['salary', 'salary_in_usd', 'salary_range'])
y = data_one_hot['salary_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Construir y compilar la red neuronal
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 6. Entrenar el modelo
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# 7. Evaluación del modelo
# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test_scaled)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Calcular métricas
acc = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes, average='weighted')

print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)