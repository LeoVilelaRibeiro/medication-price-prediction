from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import mmh3
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# Load models and scalers
model_tensor = load_model("./output/models/modelo_tensorflow_profundo.keras")
model_pca_xgb = joblib.load("./output/models/pca_xgb.pkl")
model_pca_rf = joblib.load("./output/models/pca_rf.pkl")
model_pca_tree = joblib.load("./output/models/pca_tree.pkl")
model_determinant = joblib.load("./output/models/determinant_xgb.pkl")
model_simple = joblib.load("./output/models/simple_xgb.pkl")

scaler_tensor = joblib.load("./output/scalers/scaler_tensor.pkl")
scaler_pca = joblib.load("./output/scalers/scaler_pca.pkl")
scaler_determinant = joblib.load("./output/scalers/scaler_determinant.pkl")
scaler_simple = joblib.load("./output/scalers/scaler_simple.pkl")

# Load mappings
cluster_df = pd.read_excel("./input/clusters.xlsx")
unit_encoder = joblib.load("./output/encoders/unit_label_encoder.pkl")
pca_feature_names = joblib.load("./output/encoders/pca_feature_names.pkl")

# Utils

def convert_timestamp(dt_str):
  return int(datetime.fromisoformat(dt_str).timestamp() * 1000)

@app.route("/metrics/<model_type>", methods=["GET"])
def get_metrics(model_type):
  path = f"./output/metrics/{model_type}_metrics.json"
  if not os.path.exists(path):
    return jsonify({"error": "Metrics not found"}), 404
  return send_file(path, mimetype="application/json")

@app.route("/plots/<model_type>", methods=["GET"])
def get_plot(model_type):
  path = f"./output/plots/{model_type}_shap.png"
  if not os.path.exists(path):
    return jsonify({"error": "Plot not found"}), 404
  return send_file(path, mimetype="image/png")

@app.route("/predict/tensor", methods=["POST"])
def predict_tensor():
  payload = request.get_json()
  medications = payload["medications"]
  created_at = payload["created_at"]

  matrix = []
  dose_list = []
  unit_encoded_list = []

  for med in medications:
    active = med["principio_ativo"]
    cluster = cluster_df[cluster_df["original"] == active]["cluster"].values
    if len(cluster) == 0:
      return jsonify({"error": f"Cluster not found for {active}"}), 400

    vector = np.zeros(940)
    index = hash(cluster[0]) % 940
    vector[index] = 1
    matrix.append(vector)
    dose_list.append(med["dose"])
    unit_encoded_list.append(unit_encoder.transform([med["unidade"]])[0])

  max_len = len(matrix)
  padded_matrix = np.vstack(matrix + [np.zeros(940)] * (max_len - len(matrix)))
  padded_dose = pad_sequences([dose_list], maxlen=max_len, dtype="float32", padding="post")[0]
  padded_unit = pad_sequences([unit_encoded_list], maxlen=max_len, dtype="int32", padding="post")[0]

  matrix_hash = mmh3.hash(str(padded_matrix.flatten().tolist()), signed=False) / (2**32)
  dose_hash = mmh3.hash(str(padded_dose.tolist()), signed=False) / (2**32)
  unit_hash = mmh3.hash(str(padded_unit.tolist()), signed=False) / (2**32)

  ts = convert_timestamp(created_at)

  row = np.array([[len(medications), 0, 0, ts, matrix_hash, dose_hash, unit_hash]])
  row[:, 1:3] = scaler_tensor.transform(row[:, 1:3])
  prediction = model_tensor.predict(row)[0][0]

  return jsonify({"predicted_price": float(prediction)})

@app.route("/predict/pca/<model>", methods=["POST"])
def predict_pca(model):
  models = {
    "xgb": model_pca_xgb,
    "rf": model_pca_rf,
    "tree": model_pca_tree
  }
  if model not in models:
    return jsonify({"error": "Invalid model"}), 400

  payload = request.get_json()
  medications = payload["medications"]
  created_at = payload["created_at"]

  siglas = {}
  def generate_key(cluster):
    parts = cluster.split()
    sigla = ''.join([p[:3].upper() for p in parts])
    base = sigla
    i = 1
    while sigla in siglas:
      sigla = base + str(i)
      i += 1
    siglas[sigla] = cluster
    return sigla

  keys = []
  for med in medications:
    active = med["principio_ativo"]
    cluster = cluster_df[cluster_df["original"] == active]["cluster"].values
    if len(cluster) == 0:
      return jsonify({"error": f"Cluster not found for {active}"}), 400
    sigla = generate_key(cluster[0])
    key = f"{sigla}_{med['dose']}_{med['unidade']}"
    keys.append(key)

  features = pd.DataFrame([0] * len(pca_feature_names), index=pca_feature_names).T
  for key in keys:
    if key in features.columns:
      features[key] = 1

  ts = convert_timestamp(created_at)
  features["created_at"] = ts

  X = scaler_pca.transform(features[pca_feature_names])
  prediction = models[model].predict(X)[0]

  return jsonify({"predicted_price": float(prediction)})

@app.route("/predict/determinant", methods=["POST"])
def predict_determinant():
  payload = request.get_json()
  medications = payload["medications"]
  created_at = payload["created_at"]

  matrix = []
  dose_list = []
  unit_encoded_list = []

  for med in medications:
    active = med["principio_ativo"]
    cluster = cluster_df[cluster_df["original"] == active]["cluster"].values
    if len(cluster) == 0:
      return jsonify({"error": f"Cluster not found for {active}"}), 400

    vector = np.zeros(940)
    index = hash(cluster[0]) % 940
    vector[index] = 1
    matrix.append(vector)
    dose_list.append(med["dose"])
    unit_encoded_list.append(unit_encoder.transform([med["unidade"]])[0])

  padded_matrix = np.vstack(matrix)
  matrix_hash = mmh3.hash(str(padded_matrix.flatten().tolist()), signed=False) / (2**32)
  dose_hash = mmh3.hash(str(dose_list), signed=False) / (2**32)
  unit_hash = mmh3.hash(str(unit_encoded_list), signed=False) / (2**32)

  ts = convert_timestamp(created_at)

  row = np.array([[matrix_hash, dose_hash, unit_hash, ts]])
  row_scaled = scaler_determinant.transform(row)
  prediction = model_determinant.predict(row_scaled)[0]

  return jsonify({"predicted_price": float(prediction)})

@app.route("/predict/simple", methods=["POST"])
def predict_simple():
  payload = request.get_json()
  medications = payload["medications"]
  created_at = payload["created_at"]

  matrix = []
  dose_list = []
  unit_encoded_list = []

  for med in medications:
    active = med["principio_ativo"]
    cluster = cluster_df[cluster_df["original"] == active]["cluster"].values
    if len(cluster) == 0:
      return jsonify({"error": f"Cluster not found for {active}"}), 400

    vector = np.zeros(940)
    index = hash(cluster[0]) % 940
    vector[index] = 1
    matrix.append(vector)
    dose_list.append(med["dose"])
    unit_encoded_list.append(unit_encoder.transform([med["unidade"]])[0])

  padded_matrix = np.vstack(matrix)
  matrix_hash = mmh3.hash(str(padded_matrix.flatten().tolist()), signed=False) / (2**32)
  dose_hash = mmh3.hash(str(dose_list), signed=False) / (2**32)
  unit_hash = mmh3.hash(str(unit_encoded_list), signed=False) / (2**32)

  ts = convert_timestamp(created_at)

  row = np.array([[dose_hash, unit_hash, ts]])
  row_scaled = scaler_simple.transform(row)
  prediction = model_simple.predict(row_scaled)[0]

  return jsonify({"predicted_price": float(prediction)})

if __name__ == "__main__":
  app.run(debug=True)
