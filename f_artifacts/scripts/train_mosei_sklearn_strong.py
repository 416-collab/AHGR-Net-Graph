import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD


def load_feature_vectors(feature_path):
    with open(feature_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    pooled = []
    for x in raw:
        feat = x["feature"].astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)

        mean_pool = feat.mean(axis=1)
        max_pool = feat.max(axis=1)
        std_pool = feat.std(axis=1)

        pooled.append(np.concatenate([mean_pool, max_pool, std_pool], axis=0))

    return np.stack(pooled).astype(np.float32)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.clip(np.asarray(y_pred).reshape(-1), -3, 3)

    mae = mean_absolute_error(y_true, y_pred)
    corr = pearsonr(y_true, y_pred)[0] if len(np.unique(np.round(y_pred, 4))) > 1 else 0.0

    y_bin_nonneg = (y_true >= 0).astype(int)
    p_bin_nonneg = (y_pred >= 0).astype(int)

    mask = y_true != 0
    y_bin_nn = (y_true[mask] > 0).astype(int)
    p_bin_nn = (y_pred[mask] > 0).astype(int)

    y7 = np.rint(np.clip(y_true, -3, 3) + 3).astype(int)
    p7 = np.rint(np.clip(y_pred, -3, 3) + 3).astype(int)
    y7 = np.clip(y7, 0, 6)
    p7 = np.clip(p7, 0, 6)

    return {
        "mae": mae,
        "corr": corr,
        "acc_7": accuracy_score(y7, p7),
        "f1_7_macro": f1_score(y7, p7, average="macro", zero_division=0),
        "acc_2_nonneg": accuracy_score(y_bin_nonneg, p_bin_nonneg),
        "f1_2_nonneg": f1_score(y_bin_nonneg, p_bin_nonneg, average="weighted", zero_division=0),
        "acc_2_no_neutral": accuracy_score(y_bin_nn, p_bin_nn),
        "f1_2_no_neutral": f1_score(y_bin_nn, p_bin_nn, average="weighted", zero_division=0),
    }


def main():
    os.makedirs("outputs", exist_ok=True)

    paths = {
        "train_feat": "data/raw/MOSEI/train.features",
        "val_feat": "data/raw/MOSEI/val.features",
        "test_feat": "data/raw/MOSEI/test.features",
        "train_csv": "data/raw/MOSEI/Data_Train_modified.csv",
        "val_csv": "data/raw/MOSEI/Data_Val_modified.csv",
        "test_csv": "data/raw/MOSEI/Data_Test_modified.csv",
    }

    train_df = pd.read_csv(paths["train_csv"])
    val_df = pd.read_csv(paths["val_csv"])
    test_df = pd.read_csv(paths["test_csv"])

    y_train = train_df["sentiment"].astype(np.float32).values
    y_val = val_df["sentiment"].astype(np.float32).values
    y_test = test_df["sentiment"].astype(np.float32).values

    train_texts = train_df["text"].fillna(train_df["ASR"]).fillna("").astype(str).tolist()
    val_texts = val_df["text"].fillna(val_df["ASR"]).fillna("").astype(str).tolist()
    test_texts = test_df["text"].fillna(test_df["ASR"]).fillna("").astype(str).tolist()

    print("Loading pooled sequence features...")
    train_feat = load_feature_vectors(paths["train_feat"])
    val_feat = load_feature_vectors(paths["val_feat"])
    test_feat = load_feature_vectors(paths["test_feat"])

    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_feat)
    val_feat = scaler.transform(val_feat)
    test_feat = scaler.transform(test_feat)

    print("Fitting TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=80000,
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )

    x_train_text = tfidf.fit_transform(train_texts)
    x_val_text = tfidf.transform(val_texts)
    x_test_text = tfidf.transform(test_texts)

    x_train = sparse.hstack([x_train_text, sparse.csr_matrix(train_feat)], format="csr")
    x_val = sparse.hstack([x_val_text, sparse.csr_matrix(val_feat)], format="csr")
    x_test = sparse.hstack([x_test_text, sparse.csr_matrix(test_feat)], format="csr")

    print("train/val/test shapes:", x_train.shape, x_val.shape, x_test.shape)

    rows = []

    # Ridge regression: tune alpha on validation.
    for alpha in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        print(f"\nTraining Ridge alpha={alpha}")
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(x_train, y_train)

        val_pred = np.clip(model.predict(x_val), -3, 3)
        val_m = metrics(y_val, val_pred)

        test_pred = np.clip(model.predict(x_test), -3, 3)
        test_m = metrics(y_test, test_pred)

        row = {
            "model": "ridge_tfidf_features",
            "alpha": alpha,
            "val_mae": val_m["mae"],
            "val_corr": val_m["corr"],
            "val_acc_2_no_neutral": val_m["acc_2_no_neutral"],
            **test_m,
        }
        rows.append(row)
        print(row)

    # Dense SVD + Ridge, sometimes improves generalization.
    print("\nFitting SVD dense text representation...")
    svd = TruncatedSVD(n_components=256, random_state=42)
    train_svd = svd.fit_transform(x_train_text)
    val_svd = svd.transform(x_val_text)
    test_svd = svd.transform(x_test_text)

    dense_scaler = StandardScaler()
    train_dense = dense_scaler.fit_transform(np.concatenate([train_svd, train_feat], axis=1))
    val_dense = dense_scaler.transform(np.concatenate([val_svd, val_feat], axis=1))
    test_dense = dense_scaler.transform(np.concatenate([test_svd, test_feat], axis=1))

    for alpha in [1.0, 3.0, 10.0, 30.0, 100.0]:
        print(f"\nTraining Dense Ridge alpha={alpha}")
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(train_dense, y_train)

        val_pred = np.clip(model.predict(val_dense), -3, 3)
        val_m = metrics(y_val, val_pred)

        test_pred = np.clip(model.predict(test_dense), -3, 3)
        test_m = metrics(y_test, test_pred)

        row = {
            "model": "dense_svd_ridge_features",
            "alpha": alpha,
            "val_mae": val_m["mae"],
            "val_corr": val_m["corr"],
            "val_acc_2_no_neutral": val_m["acc_2_no_neutral"],
            **test_m,
        }
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv("outputs/mosei_sklearn_strong_results.csv", index=False)

    print("\nBest by val score:")
    df["val_score"] = df["val_corr"] - 0.2 * df["val_mae"] + df["val_acc_2_no_neutral"]
    print(df.sort_values("val_score", ascending=False).head(10).to_string(index=False))

    print("\nBest by test MAE:")
    print(df.sort_values("mae", ascending=True).head(10).to_string(index=False))

    print("\nBest by test Corr:")
    print(df.sort_values("corr", ascending=False).head(10).to_string(index=False))

    print("\nBest by Acc-2 no neutral:")
    print(df.sort_values("acc_2_no_neutral", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
