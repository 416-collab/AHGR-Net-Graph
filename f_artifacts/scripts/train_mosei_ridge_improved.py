import os
import pickle
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


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
        min_pool = feat.min(axis=1)

        pooled.append(np.concatenate([mean_pool, max_pool, std_pool, min_pool], axis=0))

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


def val_score(m):
    return m["corr"] - 0.2 * m["mae"] + m["acc_2_no_neutral"] + 0.3 * m["f1_2_no_neutral"]


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

    print("Loading pooled sequence features: mean + max + std + min")
    train_feat = load_feature_vectors(paths["train_feat"])
    val_feat = load_feature_vectors(paths["val_feat"])
    test_feat = load_feature_vectors(paths["test_feat"])

    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_feat)
    val_feat = scaler.transform(val_feat)
    test_feat = scaler.transform(test_feat)

    print("Fitting word TF-IDF...")
    word_tfidf = TfidfVectorizer(
        max_features=100000,
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    x_train_word = word_tfidf.fit_transform(train_texts)
    x_val_word = word_tfidf.transform(val_texts)
    x_test_word = word_tfidf.transform(test_texts)

    print("Fitting char TF-IDF...")
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        max_features=50000,
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    x_train_char = char_tfidf.fit_transform(train_texts)
    x_val_char = char_tfidf.transform(val_texts)
    x_test_char = char_tfidf.transform(test_texts)

    feature_sets = {
        "word_features": (
            sparse.hstack([x_train_word, sparse.csr_matrix(train_feat)], format="csr"),
            sparse.hstack([x_val_word, sparse.csr_matrix(val_feat)], format="csr"),
            sparse.hstack([x_test_word, sparse.csr_matrix(test_feat)], format="csr"),
        ),
        "word_char_features": (
            sparse.hstack([x_train_word, x_train_char, sparse.csr_matrix(train_feat)], format="csr"),
            sparse.hstack([x_val_word, x_val_char, sparse.csr_matrix(val_feat)], format="csr"),
            sparse.hstack([x_test_word, x_test_char, sparse.csr_matrix(test_feat)], format="csr"),
        ),
        "word_only": (
            x_train_word,
            x_val_word,
            x_test_word,
        ),
        "word_char_only": (
            sparse.hstack([x_train_word, x_train_char], format="csr"),
            sparse.hstack([x_val_word, x_val_char], format="csr"),
            sparse.hstack([x_test_word, x_test_char], format="csr"),
        ),
    }

    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]

    rows = []
    pred_store = {}

    for fs_name, (x_train, x_val, x_test) in feature_sets.items():
        print("\nFeature set:", fs_name, "shapes:", x_train.shape, x_val.shape, x_test.shape)

        for alpha in alphas:
            print(f"Training Ridge fs={fs_name}, alpha={alpha}")
            model = Ridge(alpha=alpha, solver="lsqr", random_state=42)
            model.fit(x_train, y_train)

            val_pred = np.clip(model.predict(x_val), -3, 3)
            test_pred = np.clip(model.predict(x_test), -3, 3)

            val_m = metrics(y_val, val_pred)
            test_m = metrics(y_test, test_pred)

            row = {
                "model": "ridge_improved",
                "feature_set": fs_name,
                "alpha": alpha,
                "val_score": val_score(val_m),
                "val_mae": val_m["mae"],
                "val_corr": val_m["corr"],
                "val_acc_2_no_neutral": val_m["acc_2_no_neutral"],
                "val_f1_2_no_neutral": val_m["f1_2_no_neutral"],
                **test_m,
            }
            rows.append(row)

            key = f"{fs_name}_alpha{alpha}"
            pred_store[key] = {
                "val_pred": val_pred,
                "test_pred": test_pred,
                "row": row,
            }

            print(row)

            pd.DataFrame(rows).to_csv("outputs/mosei_ridge_improved_partial.csv", index=False)

    # Ensemble top validation models.
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("val_score", ascending=False).reset_index(drop=True)

    print("\nTop individual models by val_score:")
    print(df_sorted.head(10).to_string(index=False))

    top_keys = []
    for _, r in df_sorted.head(8).iterrows():
        top_keys.append(f"{r['feature_set']}_alpha{r['alpha']}")

    ensemble_rows = []

    for k in [2, 3, 4, 5]:
        selected = top_keys[:k]
        val_preds = np.stack([pred_store[x]["val_pred"] for x in selected], axis=0)
        test_preds = np.stack([pred_store[x]["test_pred"] for x in selected], axis=0)

        val_avg = val_preds.mean(axis=0)
        test_avg = test_preds.mean(axis=0)

        val_m = metrics(y_val, val_avg)
        test_m = metrics(y_test, test_avg)

        row = {
            "model": f"ensemble_top{k}",
            "feature_set": "+".join(selected),
            "alpha": -1,
            "val_score": val_score(val_m),
            "val_mae": val_m["mae"],
            "val_corr": val_m["corr"],
            "val_acc_2_no_neutral": val_m["acc_2_no_neutral"],
            "val_f1_2_no_neutral": val_m["f1_2_no_neutral"],
            **test_m,
        }

        ensemble_rows.append(row)
        print("\nENSEMBLE:", row)

    final_df = pd.concat([df, pd.DataFrame(ensemble_rows)], ignore_index=True)
    final_df.to_csv("outputs/mosei_ridge_improved_results.csv", index=False)

    print("\nBest by val_score:")
    print(final_df.sort_values("val_score", ascending=False).head(15).to_string(index=False))

    print("\nBest by test MAE:")
    print(final_df.sort_values("mae", ascending=True).head(15).to_string(index=False))

    print("\nBest by test Corr:")
    print(final_df.sort_values("corr", ascending=False).head(15).to_string(index=False))

    print("\nBest by Acc-2 no-neutral:")
    print(final_df.sort_values("acc_2_no_neutral", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
