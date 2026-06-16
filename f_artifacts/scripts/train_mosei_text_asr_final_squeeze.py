import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


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


def make_texts(df, mode):
    text = df["text"].fillna("").astype(str)
    asr = df["ASR"].fillna("").astype(str)

    if mode == "text":
        return text.tolist()
    if mode == "asr":
        return asr.tolist()
    if mode == "text_asr":
        return (text + " [ASR] " + asr).tolist()

    raise ValueError(mode)


def build_features(train_texts, val_texts, test_texts, word_max, char_max):
    word = TfidfVectorizer(
        max_features=word_max,
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )

    char = TfidfVectorizer(
        analyzer="char_wb",
        max_features=char_max,
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )

    xtr_w = word.fit_transform(train_texts)
    xv_w = word.transform(val_texts)
    xt_w = word.transform(test_texts)

    xtr_c = char.fit_transform(train_texts)
    xv_c = char.transform(val_texts)
    xt_c = char.transform(test_texts)

    xtr = sparse.hstack([xtr_w, xtr_c], format="csr")
    xv = sparse.hstack([xv_w, xv_c], format="csr")
    xt = sparse.hstack([xt_w, xt_c], format="csr")

    return xtr, xv, xt


def main():
    os.makedirs("outputs", exist_ok=True)

    train_df = pd.read_csv("data/raw/MOSEI/Data_Train_modified.csv")
    val_df = pd.read_csv("data/raw/MOSEI/Data_Val_modified.csv")
    test_df = pd.read_csv("data/raw/MOSEI/Data_Test_modified.csv")

    y_train = train_df["sentiment"].astype(np.float32).values
    y_val = val_df["sentiment"].astype(np.float32).values
    y_test = test_df["sentiment"].astype(np.float32).values

    text_modes = ["text", "asr", "text_asr"]
    feature_sizes = [
        {"word_max": 100000, "char_max": 50000},
        {"word_max": 150000, "char_max": 70000},
    ]
    alphas = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

    rows = []
    pred_store = {}

    for mode in text_modes:
        train_texts = make_texts(train_df, mode)
        val_texts = make_texts(val_df, mode)
        test_texts = make_texts(test_df, mode)

        for fs in feature_sizes:
            fs_name = f"{mode}_w{fs['word_max']}_c{fs['char_max']}"
            print("\nBuilding features:", fs_name)

            x_train, x_val, x_test = build_features(
                train_texts, val_texts, test_texts,
                fs["word_max"], fs["char_max"]
            )

            print("shapes:", x_train.shape, x_val.shape, x_test.shape)

            for alpha in alphas:
                print(f"Training Ridge {fs_name}, alpha={alpha}")

                model = Ridge(alpha=alpha, solver="lsqr", random_state=42)
                model.fit(x_train, y_train)

                val_pred = np.clip(model.predict(x_val), -3, 3)
                test_pred = np.clip(model.predict(x_test), -3, 3)

                val_m = metrics(y_val, val_pred)
                test_m = metrics(y_test, test_pred)

                row = {
                    "model": "ridge_text_asr_squeeze",
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
                pd.DataFrame(rows).to_csv("outputs/mosei_text_asr_squeeze_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("val_score", ascending=False).reset_index(drop=True)

    print("\nTop individual models:")
    print(df_sorted.head(15).to_string(index=False))

    top_keys = []
    for _, r in df_sorted.head(10).iterrows():
        top_keys.append(f"{r['feature_set']}_alpha{r['alpha']}")

    ens_rows = []

    for k in [2, 3, 4, 5, 6]:
        selected = top_keys[:k]

        val_preds = np.stack([pred_store[x]["val_pred"] for x in selected], axis=0)
        test_preds = np.stack([pred_store[x]["test_pred"] for x in selected], axis=0)

        val_avg = val_preds.mean(axis=0)
        test_avg = test_preds.mean(axis=0)

        val_m = metrics(y_val, val_avg)
        test_m = metrics(y_test, test_avg)

        row = {
            "model": f"text_asr_ensemble_top{k}",
            "feature_set": "+".join(selected),
            "alpha": -1,
            "val_score": val_score(val_m),
            "val_mae": val_m["mae"],
            "val_corr": val_m["corr"],
            "val_acc_2_no_neutral": val_m["acc_2_no_neutral"],
            "val_f1_2_no_neutral": val_m["f1_2_no_neutral"],
            **test_m,
        }

        ens_rows.append(row)
        print("\nENSEMBLE:", row)

    final_df = pd.concat([df, pd.DataFrame(ens_rows)], ignore_index=True)
    final_df.to_csv("outputs/mosei_text_asr_squeeze_results.csv", index=False)

    print("\nBest by val_score:")
    print(final_df.sort_values("val_score", ascending=False).head(15).to_string(index=False))

    print("\nBest by MAE:")
    print(final_df.sort_values("mae", ascending=True).head(15).to_string(index=False))

    print("\nBest by Corr:")
    print(final_df.sort_values("corr", ascending=False).head(15).to_string(index=False))

    print("\nBest by Acc-2 no-neutral:")
    print(final_df.sort_values("acc_2_no_neutral", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
