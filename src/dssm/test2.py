import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from other_model.utils.match import generate_seq_feature_match, gen_model_input
# https://github.com/datawhalechina/torch-rechub/blob/main/tutorials/Matching.ipynb



# 定义两个塔对应哪些特征
user_cols = ["user_id", "gender", "age", "occupation", "zip"]
item_cols = ['movie_id', "cate_id"]


# 可以用这个来处理
# def gen_model_input(df, user_profile, user_col, item_profile, item_col, seq_max_len, padding='pre', truncating='pre'):
#     #merge user_profile and item_profile, pad history seuence feature
#     df = pd.merge(df, user_profile, on=user_col, how='left')  # how=left to keep samples order same as the input
#     df = pd.merge(df, item_profile, on=item_col, how='left')
#     for col in df.columns.to_list():
#         if col.startswith("hist_"):
#             df[col] = pad_sequences(df[col], maxlen=seq_max_len, value=0, padding=padding, truncating=truncating).tolist()
#     input_dict = df_to_dict(df)
#     return input_dict

df_train, df_test = generate_seq_feature_match(data,
                                               user_col,
                                               item_col,
                                               time_col="timestamp",
                                               item_attribute_cols=[],
                                               sample_method=1,
                                               mode=0,
                                               neg_ratio=3,
                                               min_item=0)
print(df_train.head())
x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
y_train = x_train["label"]
x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)
y_test = x_test["label"]
print({k: v[:3] for k, v in x_train.items()})