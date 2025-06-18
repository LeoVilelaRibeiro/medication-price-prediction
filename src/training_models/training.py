import os
import joblib
import shap
import mmh3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class MedicationPriceTrainer:
  def __init__(self):
    self.raw_data_path = "./input/dados_preco.csv"
    self.cluster_data_path = "./input/clusters.xlsx"
    self.output_dir = "./output"
    os.makedirs(self.output_dir, exist_ok=True)

  def load_raw_data(self):
    self.df_raw = pd.read_csv(self.raw_data_path)
    self.cluster_df = pd.read_excel(self.cluster_data_path)

  def parse_description(self, description):
    quantity, items = description.split("|")
    quantity = quantity.strip()
    items_list = [item.strip() for item in items.split(";")]
    items_dict = {str(i+1): item for i, item in enumerate(items_list)}
    return quantity, items_dict

  def preprocess_common(self):
    self.df_raw[["amount_text", "items"]] = self.df_raw["descricao"].apply(lambda x: pd.Series(self.parse_description(x)))
    self.df_raw["hash"] = self.df_raw["descricao"].apply(lambda x: self._hash(x))

    item_rows = []
    for _, row in self.df_raw.iterrows():
      for key, value in row["items"].items():
        item_rows.append({"hash": row["hash"], "item_number": key, "item_text": value})

    self.items_df = pd.DataFrame(item_rows)
    self.items_df[["active_ingredient", "dosage"]] = self.items_df["item_text"].apply(lambda x: pd.Series([" ".join(x.split()[:-1]), x.split()[-1]]))
    self.items_df["active_ingredient"] = self.items_df["active_ingredient"].str.replace(",", ".")

    self.items_df = self.items_df.merge(self.cluster_df[["original", "cluster"]], left_on="active_ingredient", right_on="original", how="left")
    self.items_df["dose"] = self.items_df["dosage"].str.extract(r"(\d+[\.,]?\d*)").astype(float)
    self.items_df["unit"] = self.items_df["dosage"].str.extract(r"([a-zA-Z]+)")

    df_selected = self.items_df[["hash", "cluster", "dose", "unit"]].copy()
    df_selected["unit_encoded"] = LabelEncoder().fit_transform(df_selected["unit"])

    encoder = OneHotEncoder(sparse_output=False)
    clusters_encoded = encoder.fit_transform(df_selected[["cluster"]])
    cluster_columns = encoder.get_feature_names_out(["cluster"])

    df_clusters = pd.DataFrame(clusters_encoded, columns=cluster_columns)
    df_clusters["hash"] = df_selected["hash"].values

    df_matrix = df_clusters.groupby("hash").apply(lambda x: np.vstack(x[cluster_columns].to_numpy())).reset_index(name="matrix")
    df_dose = df_selected.groupby("hash")["dose"].apply(lambda x: np.array(x.tolist())).reset_index()
    df_unit = df_selected.groupby("hash")["unit_encoded"].apply(lambda x: np.array(x.tolist())).reset_index()

    self.df_common = df_matrix.merge(df_dose, on="hash").merge(df_unit, on="hash")

    self.df_raw["amount"] = pd.to_numeric(self.df_raw["amount_text"].str.extract(r"(\d+\.?\d*)")[0])
    self.df_common = self.df_raw.merge(self.df_common, on="hash")
    self.df_common["created_at"] = pd.to_datetime(self.df_common["criado"])
    self.df_common["created_at"] = self.df_common["created_at"].astype("int64") // 10**6

  def train_tensor_model(self):
    df = self.df_common.copy()

    df["matrix"] = df["matrix"].apply(lambda x: np.array(x))
    df["dose"] = df["dose"].apply(lambda x: np.array(x))
    df["unit_encoded"] = df["unit_encoded"].apply(lambda x: np.array(x))

    max_len = df["qtdInsumos"].max()
    df["matrix"] = df.apply(lambda row: self._pad_matrix(row["matrix"], row["qtdInsumos"], max_len), axis=1)
    df["dose"] = pad_sequences(df["dose"], maxlen=max_len, dtype="float32", padding="post").tolist()
    df["unit_encoded"] = pad_sequences(df["unit_encoded"], maxlen=max_len, dtype="int32", padding="post").tolist()

    scaler = MinMaxScaler()
    df[["calculado", "amount"]] = scaler.fit_transform(df[["calculado", "amount"]])
    joblib.dump(scaler, f"{self.output_dir}/scaler_tensor.pkl")

    df["matrix_hash"] = df["matrix"].apply(lambda x: mmh3.hash(str(x.flatten().tolist()), signed=False) / (2**32))
    df["dose_hash"] = df["dose"].apply(lambda x: mmh3.hash(str(x), signed=False) / (2**32))
    df["unit_hash"] = df["unit_encoded"].apply(lambda x: mmh3.hash(str(x), signed=False) / (2**32))

    self.df_tensor = df

  def train_pca_model(self):
    df = self.df_common.copy()
    siglas = {}

    def generate_key(cluster):
      parts = cluster.split()
      short = ''.join([p[:3].upper() for p in parts])
      base = short
      i = 1
      while short in siglas:
        short = base + str(i)
        i += 1
      siglas[short] = cluster
      return short

    df["cluster_short"] = df["cluster"].apply(generate_key)
    df["key"] = df["cluster_short"] + '_' + df["dose"].astype(str) + '_' + df["unit"]

    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[["key"]])
    column_names = [c.replace("key_", "") for c in encoder.get_feature_names_out(["key"])]

    df_encoded = pd.DataFrame(encoded, columns=column_names)
    df_encoded["hash"] = df["hash"].values

    df_agg = df_encoded.groupby("hash").agg("max").reset_index()

    scaler = StandardScaler()
    X = scaler.fit_transform(df_agg.drop(columns=["hash"]))
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=[f"pca_{i+1}" for i in range(50)])
    df_pca.insert(0, "hash", df_agg["hash"].values)

    self.df_pca = df.merge(df_pca, on="hash")
    self.df_pca["created_at"] = pd.to_datetime(self.df_pca["criado"])
    self.df_pca["created_at"] = self.df_pca["created_at"].astype("int64") // 10**6
    joblib.dump(scaler, f"{self.output_dir}/scaler_pca.pkl")

  def train_xgb_with_shap(self, df, name):
    features = df.drop(columns=["hash", "correto", "descricao", "criado", "amount_text", "items"], errors="ignore")
    target = df["correto"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, f"{self.output_dir}/model_{name}.pkl")

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{self.output_dir}/shap_summary_{name}.png")
    plt.close()

  def _pad_matrix(self, matrix, length, max_len):
    matrix = matrix[:length]
    if matrix.shape[0] < max_len:
      pad = np.zeros((max_len - matrix.shape[0], matrix.shape[1]))
      matrix = np.vstack([matrix, pad])
    return matrix

  def _hash(self, text):
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()

  def run(self):
    self.load_raw_data()
    self.preprocess_common()
    self.train_tensor_model()
    self.train_pca_model()
    self.train_xgb_with_shap(self.df_tensor, "tensor")
    self.train_xgb_with_shap(self.df_pca, "pca")


if __name__ == "__main__":
  trainer = MedicationPriceTrainer()
  trainer.run()
