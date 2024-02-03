import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import pickle
import pandas as pd
from src.config.paths import *
from src.config.constants import *


def get_one_hot_feat(df, col, prefix=""):
    mapping = eval(col.upper() + "_MAPPING")
    new_cols = []
    df[f"{prefix}{col}"] = df[col].map(lambda x: mapping.get(x, mapping.get(EMPTY_KEY)))
    if len(df[f"{prefix}{col}"].value_counts()) == 1:
        print("get_one_hot_feat delete feat:", col)
        return df, new_cols
    new_cols.append(col)
    return df, new_cols

def get_multi_hot_feat(df, col, prefix=""):
    """Generate multi-hot encoded features for multi-categorical data."""
    mapping = eval(col.upper() + "_MAPPING")
    new_cols = []
    for key, val in mapping.items():
        df[f"{prefix}{val}"] = df[col].map(lambda x: 1 if key in x.split("|") else 0)
        if len(df[f"{prefix}{val}"].value_counts()) == 1:
            print("get_multi_hot_feat delete feat:", f"{prefix}{val}")
            continue
        new_cols.append(f"{prefix}{val}")
    return df, new_cols

def get_multi_hot_feat(df, col, prefix=""):
    """Generate multi-hot encoded features for multi-categorical data."""
    mapping = eval(col.upper() + "_MAPPING")
    new_cols = []
    for key, val in mapping.items():
        df[f"{prefix}{val}"] = df[col].map(lambda x: 1 if key in x.split("|") else 0)
        if len(df[f"{prefix}{val}"].value_counts()) == 1:
            print("get_multi_hot_feat delete feat:", f"{prefix}{val}")
            continue
        new_cols.append(f"{prefix}{val}")
    return df, new_cols

def get_pit_join_feat(df, col, prefix=""):
    """Generate features based on user-item interactions."""
    mapping = eval(col.upper() + "_MAPPING")
    new_cols = []
    for val in mapping.values():
        df[f"{prefix}{val}_imp"] = df[f"{col}_pit"].map(lambda x: x.get(val, [0, 0])[0])
        df[f"{prefix}{val}_clk"] = df[f"{col}_pit"].map(lambda x: x.get(val, [0, 0])[1])
        if len(df[f"{prefix}{val}_imp"].value_counts()) == 1:
            print("get_pit_join_feat delete feat:", f"{prefix}{val}_imp", f"{prefix}{val}_clk")
            continue
        new_cols += [f"{prefix}{val}_imp", f"{prefix}{val}_clk"]
    return df, new_cols

def dense_feat_scaling(train_data, test_data, user_data, item_data, user_dense, item_dense):
    """Scale dense features using standard scaling."""
    dense_feat_mean_std_dict = {}
    for col in user_dense + item_dense:
        m, s = train_data[col].agg(["mean", "std"])
        if pd.isna(m):
            m = 0
        if pd.isna(s) or s <= 0:
            s = 1
        train_data[col] = (train_data[col] - m) / s
        test_data[col] = (test_data[col] - m) / s
        user_data[col] = (user_data[col] - m) / s
        dense_feat_mean_std_dict[col] = (m, s)
    return train_data, test_data, user_data, item_data, dense_feat_mean_std_dict

def make_feat_meta(user_sparse, item_sparse, user_dense, item_dense):
    """Create metadata dictionary for feature indices and categories."""
    feat_meta = {"sparse": {"user": {}, "item": {}}, "dense": {"user": {}, "item": {}}}
    i = 1
    for col in user_sparse:
        n_cat = offline_joined[col].max() + 1
        feat_meta["sparse"]["user"][col] = (i, n_cat)
        i += 1
    for col in item_sparse:
        n_cat = offline_joined[col].max() + 1
        feat_meta["sparse"]["item"][col] = (i, n_cat)
        i += 1
    for col in user_dense:
        feat_meta["dense"]["user"][col] = (i,)
        i += 1
    for col in item_dense:
        feat_meta["dense"]["item"][col] = (i,)
        i += 1
    return feat_meta

if __name__ == '__main__':
    user_data = pd.read_csv(MovieLens_RAW_USER_PATH, sep="::", engine="python", header=None,
                            names=["userid", "gender", "age", "occupation", "zip"])
    item_data = pd.read_csv(MovieLens_RAW_ITEM_PATH, sep="::", engine="python", header=None,
                            names=["movieid", "title", "genres"], encoding="ISO-8859-1")
    offline_imp = pd.read_csv(OFFLINE_DATA_PATH)
    imp_dict = pickle.load(open(IMP_TERM_PATH, "rb"))
    user_dict = pickle.load(open(USER_TERM_PATH, "rb"))
    item_dict = pickle.load(open(ITEM_TERM_PATH, "rb"))
    user_sparse, item_sparse, user_dense, item_dense = ["userid"], ["movieid"], [], []

    for col in ["gender", "age", "occupation"]:
        # get gender, age, occupation mapping
        user_data, new_cols = get_one_hot_feat(user_data, col, prefix="")
        user_sparse += new_cols

    # multi hot feat
    item_data, new_cols = get_multi_hot_feat(item_data, "genres", prefix="item_g")
    item_sparse += new_cols

    # action feat for online fetch
    user_data[f"{'genres'}_pit"] = user_data.apply(lambda x: user_dict.get(x["userid"]), axis=1)
    user_data, new_cols = get_pit_join_feat(user_data, "genres", prefix="user_g")
    user_dense += new_cols

    # action feat for offline train
    offline_joined = pd.merge(pd.merge(offline_imp, user_data, on="userid", how="left", right_index=False),
                              item_data, on="movieid", how="left", right_index=False)
    offline_joined["queryid"] = offline_joined.index
    offline_joined[f"{'genres'}_pit"] = offline_joined.apply(lambda x: imp_dict.get(x["queryid"]), axis=1)
    offline_joined, _ = get_pit_join_feat(offline_joined, "genres", prefix="user_g")

    # train-test split
    all_cols = ["queryid", "label"] + user_sparse + item_sparse + user_dense + item_dense
    train_data = offline_joined.loc[offline_joined["test_type"] == 0][all_cols].reset_index(drop=True)
    test_data = offline_joined.loc[offline_joined["test_type"] == 1][all_cols].reset_index(drop=True)
    train_data, test_data, user_data, item_data, _ = dense_feat_scaling(train_data, test_data,
                                                                        user_data, item_data, user_dense,
                                                                        item_dense)
    for col in user_dense + item_dense:
        print(test_data[col].agg(["mean", "std"]))

    # user-item split
    user_entity = user_data[user_sparse + user_dense]
    item_entity = item_data[item_sparse + item_dense]
    feat_meta = make_feat_meta(user_sparse, item_sparse, user_dense, item_dense)

    # save data
    # for offline train
    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)
    # for online feast
    user_entity.to_csv(USER_ENTITY_PATH, index=False)
    item_entity.to_csv(ITEM_ENTITY_PATH, index=False)
    # for online server
    pickle.dump(feat_meta, open(FEAT_META_PATH, 'wb'), protocol=4)
    print(f"saved: train_data: {train_data.shape}, test_data: {test_data.shape}, "
          f"user_entity: {user_entity.shape}, item_entity: {item_entity.shape}, feat_meta: {feat_meta}")