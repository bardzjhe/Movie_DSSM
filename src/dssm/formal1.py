import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
# 用pytorch吧 以test2为准 但是要改一下test2 让他看不出来
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.other_model.basic.features import SparseFeature, SequenceFeature
from src.other_model.models.matching import DSSM
from src.other_model.trainers import MatchTrainer
from src.other_model.utils.data import MatchDataGenerator
from src.other_model.utils.data import df_to_dict

pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)
pd.set_option('display.width',100)
torch.manual_seed(2024)

if __name__ == '__main__':
    # 1. Load and preprocess the dataset
    users = pd.read_csv("F:/Sem2 Y4/FYP/Movie_DSSM/data/ml-1m/users.dat",
                         sep="::", header=None, engine="python", encoding='iso-8859-1',
                         names = "UserID::Gender::Age::Occupation::Zip-code".split("::"))

    items = pd.read_csv("F:/Sem2 Y4/FYP/Movie_DSSM/data/ml-1m/movies.dat",
                         sep="::", header=None, engine="python", encoding='iso-8859-1',
                         names = "MovieID::Title::Genres".split("::"))

    ratings = pd.read_csv("F:/Sem2 Y4/FYP/Movie_DSSM/data/ml-1m/ratings.dat",
                         sep="::", header=None, engine="python", encoding='iso-8859-1',
                         names = "UserID::MovieID::Rating::Timestamp".split("::"))

    # print(df_user)
    # 特征预处理
    # 在本DSSM模型中，我们使用两种类别的特征，分别是稀疏特征（SparseFeature）和序列特征（SequenceFeature）。
    #
    # 对于稀疏特征，是一个离散的、有限的值（例如用户ID，一般会先进行LabelEncoding操作转化为连续整数值），模型将其输入到Embedding层，输出一个Embedding向量。
    #
    # 对于序列特征，每一个样本是一个List[SparseFeature]（一般是观看历史、搜索历史等），对于这种特征，默认对于每一个元素取Embedding后平均，输出一个Embedding向量。此外，除了平均，还有拼接，最值等方式，可以在pooling参数中指定。


    # preprocess the Genres TODO: 这个其实有问题把 怎么可以只娶第一个genres而放弃其他的， 之后再修改
    items["CateID"] = items["Genres"].apply(lambda x: x.split("|")[0])
    # print(movies["CateID"][1:50])

    user_id = "UserID"
    item_id = "MovieID"
    # TODO： 这些是离散度值 思考稠密值  不行的话名字改成user item feature
    # sparse_features = [, 'zip']
    user_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    item_cols = ['MovieID', "CateID"]


    # label encoding
    # print(users[user_cols].head())

    user_fea_max_idx = {}
    for feature in user_cols:
        le = LabelEncoder()
        users[feature] = le.fit_transform(users[feature]) + 1
        user_fea_max_idx[feature] = users[feature].max() + 1
        user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(le.classes_)}

    # print(users)
    print(user_fea_max_idx)
    # TODO: evaluation 会用到
    # np.save(save_dir + "raw_id_maps.npy", (user_map, item_map))


    item_fea_max_idx = {}
    for feature in item_cols:
        le = LabelEncoder()
        items[feature] = le.fit_transform(items[feature]) + 1
        item_fea_max_idx[feature] = items[feature].max() + 1
        item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(le.classes_)}

    # print(items)
    # print(item_fea_max_idx)
    # print(item_map)

    user_profile = users.drop_duplicates('UserID')
    item_profile = items.drop_duplicates('MovieID')
    # print(user_profile.head())
    # print(item_profile.head())

    # merge the data
    data = pd.merge(pd.merge(ratings, items),users)
    # print(data[:100])


    from src.other_model.utils.match import generate_seq_feature_match, gen_model_input
    df_train, df_test = generate_seq_feature_match(data,
                                                   user_id,
                                                   item_id,
                                                   time_col="Timestamp",
                                                   item_attribute_cols=[],
                                                   sample_method=1,
                                                   mode=0,
                                                   neg_ratio=3,
                                                   min_item=0)


    # TODO: 这个拆分数据可以改
    print(df_train.head())
    x_train = gen_model_input(df_train, user_profile, user_id, item_profile, item_id, seq_max_len=50)
    y_train = x_train["label"]
    x_test = gen_model_input(df_test, user_profile, user_id, item_profile, item_id, seq_max_len=50)
    y_test = x_test["label"]
    print({k: v[:3] for k, v in x_train.items()})


    #定义特征类型 这下面有错


    user_features = [
        SparseFeature(feature_name, vocab_size=user_fea_max_idx[feature_name], embed_dim=16) for feature_name in user_cols
    ]
    user_features += [
        SequenceFeature("hist_movie_id",
                        vocab_size=item_fea_max_idx["MovieID"],
                        embed_dim=16,
                        pooling="mean",
                        shared_with="MovieID")
    ]

    item_features = [
        SparseFeature(feature_name, vocab_size=item_fea_max_idx[feature_name], embed_dim=16) for feature_name in item_cols
    ]
    print(user_features)
    print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    print(item_features)

    #
    # # 将dataframe转为dict
    all_item = df_to_dict(item_profile)
    test_user = x_test
    print({k: v[:3] for k, v in all_item.items()})
    print({k: v[0] for k, v in test_user.items()})


    # 下面终于到训练模型了
    # 根据之前处理的数据拿到Dataloader
    dg = MatchDataGenerator(x=x_train, y=y_train)
    train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=256)

    # 定义模型
    model = DSSM(user_features,
                 item_features,
                 temperature=0.02,
                 user_params={
                     "dims": [256, 128, 64],
                     "activation": 'prelu',  # important!!
                 },
                 item_params={
                     "dims": [256, 128, 64],
                     "activation": 'prelu',  # important!!
                 })

    # 模型训练器
    trainer = MatchTrainer(model,
                           mode=0,  # 同上面的mode，需保持一致
                           optimizer_params={
                               "lr": 1e-4,
                               "weight_decay": 1e-6
                           },
                           n_epoch=1,
                           device='cpu',
                           model_path="./dssm_model.bin")

    # 开始训练
    trainer.fit(train_dl)