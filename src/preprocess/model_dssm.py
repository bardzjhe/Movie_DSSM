import torch
import pandas as pd
import numpy as np
import os

# https://github.com/datawhalechina/torch-rechub/blob/main/tutorials/Matching.ipynb


torch.manual_seed(2024)

# 指定用户列和物品列的名字、离散和稠密特征，适配框架的接口
user_col, item_col = "user_id", "movie_id"
sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', "cate_id"]
dense_features = []


# 特征预处理
# 在本DSSM模型中，我们使用两种类别的特征，分别是稀疏特征（SparseFeature）和序列特征（SequenceFeature）。
#
# 对于稀疏特征，是一个离散的、有限的值（例如用户ID，一般会先进行LabelEncoding操作转化为连续整数值），模型将其输入到Embedding层，输出一个Embedding向量。
#
# 对于序列特征，每一个样本是一个List[SparseFeature]（一般是观看历史、搜索历史等），对于这种特征，默认对于每一个元素取Embedding后平均，输出一个Embedding向量。此外，除了平均，还有拼接，最值等方式，可以在pooling参数中指定。
#
# 框架还支持稠密特征（DenseFeature），即一个连续的特征值（例如概率），这种类型一般需归一化处理。但是本样例中未使用。

save_dir = '../examples/ranking/data/ml-1m/saved/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 对SparseFeature进行LabelEncoding
from sklearn.preprocessing import LabelEncoder
print(data[sparse_features].head())
feature_max_idx = {}
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1
    if feature == user_col:
        user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}  #encode user id: raw user id
    if feature == item_col:
        item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}  #encode item id: raw item id
np.save(save_dir+"raw_id_maps.npy", (user_map, item_map))  # evaluation时会用到
print('LabelEncoding后：')
print(data[sparse_features].head())