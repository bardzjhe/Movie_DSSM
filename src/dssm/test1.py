# https://github.com/aialgorithm/Blog/issues/45
import sys
import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(os.path.dirname(current_dir))
# sys.path.append(parent_dir)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ### 1. 读取电影数据集（用户信息、电影信息、评分行为信息）
cwd = os.getcwd()
print(cwd)
df_user = pd.read_csv("F:/Sem2 Y4/FYP/Movie_DSSM/data/ml-1m/users.dat",
                     sep="::", header=None, engine="python",encoding='iso-8859-1',
                     names = "UserID::Gender::Age::Occupation::Zip-code".split("::"))

df_movie = pd.read_csv("F:/Sem2 Y4/FYP/Movie_DSSM/data/ml-1m/movies.dat",
                     sep="::", header=None, engine="python",encoding='iso-8859-1',
                     names = "MovieID::Title::Genres".split("::"))

df_rating = pd.read_csv("F:/Sem2 Y4/FYP/Movie_DSSM/data/ml-1m/ratings.dat",
                     sep="::", header=None, engine="python",encoding='iso-8859-1',
                     names = "UserID::MovieID::Rating::Timestamp".split("::"))

import collections

# 计算电影中每个题材的次数
genre_count = collections.defaultdict(int)
for genres in df_movie["Genres"].str.split("|"):
    for genre in genres:
        genre_count[genre] += 1
genre_count


# # 每个电影只保留频率最高（代表性）的电影题材标签
def get_highrate_genre(x):
    sub_values = {}
    for genre in x.split("|"):
        sub_values[genre] = genre_count[genre]
    return sorted(sub_values.items(), key=lambda x:x[1], reverse=True)[0][0]

df_movie["Genres"] = df_movie["Genres"].map(get_highrate_genre)
df_movie.head()


# #### 给特征列做序列编码
def add_index_column(param_df, column_name):
    values = list(param_df[column_name].unique())
    value_index_dict = {value:idx for idx,value in enumerate(values)}
    param_df[f"{column_name}_idx"] = param_df[column_name].map(value_index_dict)


add_index_column(df_user, "UserID")
add_index_column(df_user, "Gender")
add_index_column(df_user, "Age")
add_index_column(df_user, "Occupation")
add_index_column(df_movie, "MovieID")
add_index_column(df_movie, "Genres")

# 合并成一个df
df = pd.merge(pd.merge(df_rating, df_user), df_movie)
df.drop(columns=["Timestamp", "Zip-code", "Title"], inplace=True)

num_users = df["UserID_idx"].max() + 1
num_movies = df["MovieID_idx"].max() + 1
num_genders = df["Gender_idx"].max() + 1
num_ages = df["Age_idx"].max() + 1
num_occupations = df["Occupation_idx"].max() + 1
num_genres = df["Genres_idx"].max() + 1

num_users, num_movies, num_genders, num_ages, num_occupations, num_genres


# #### 评分的归一化

min_rating = df["Rating"].min()
max_rating = df["Rating"].max()

df["Rating"] = df["Rating"].map(lambda x : (x-min_rating)/(max_rating-min_rating)) # 评分作为两者的相似度
# df["is_rating_high"] = (df["Rating"]>=4).astype(int)  # 可生成是否高评分作为分类模型的类别标签
df.sample(frac=1).head(3)
# 构建训练集特征及标签
df_sample = df.sample(frac=0.1)  # 训练集抽样
X = df_sample[["UserID_idx","Gender_idx","Age_idx","Occupation_idx","MovieID_idx","Genres_idx"]]
y = df_sample["Rating"]


def get_model():
    """搭建双塔DNN模型"""

    # 输入
    user_id = keras.layers.Input(shape=(1,), name="user_id")
    gender = keras.layers.Input(shape=(1,), name="gender")
    age = keras.layers.Input(shape=(1,), name="age")
    occupation = keras.layers.Input(shape=(1,), name="occupation")
    movie_id = keras.layers.Input(shape=(1,), name="movie_id")
    genre = keras.layers.Input(shape=(1,), name="genre")

    # user 塔
    user_vector = tf.keras.layers.concatenate([
        layers.Embedding(num_users, 100)(user_id),
        layers.Embedding(num_genders, 2)(gender),
        layers.Embedding(num_ages, 2)(age),
        layers.Embedding(num_occupations, 2)(occupation)
    ])
    user_vector = layers.Dense(32, activation='relu')(user_vector)
    user_vector = layers.Dense(8, activation='relu',
                               name="user_embedding", kernel_regularizer='l2')(user_vector)

    # item 塔
    movie_vector = tf.keras.layers.concatenate([
        layers.Embedding(num_movies, 100)(movie_id),
        layers.Embedding(num_genres, 2)(genre)
    ])
    movie_vector = layers.Dense(32, activation='relu')(movie_vector)
    movie_vector = layers.Dense(8, activation='relu',
                                name="movie_embedding", kernel_regularizer='l2')(movie_vector)

    # 每个用户的embedding和item的embedding作点积
    dot_user_movie = tf.reduce_sum(user_vector * movie_vector, axis=1)
    dot_user_movie = tf.expand_dims(dot_user_movie, 1)

    output = layers.Dense(1, activation='sigmoid')(dot_user_movie)

    return keras.models.Model(inputs=[user_id, gender, age, occupation, movie_id, genre], outputs=[output])


model = get_model()
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.RMSprop())
fit_x_train = [
    X["UserID_idx"],
    X["Gender_idx"],
    X["Age_idx"],
    X["Occupation_idx"],
    X["MovieID_idx"],
    X["Genres_idx"]
]

history = model.fit(
    x=fit_x_train,
    y=y,
    batch_size=32,
    epochs=5,
    verbose=1
)

# ### 3. 模型的预估-predict
# 输入前5个样本并做预测

inputs = df[["UserID_idx", "Gender_idx", "Age_idx", "Occupation_idx", "MovieID_idx", "Genres_idx"]].head(5)
print(df.head(5))

# 对于（用户ID，召回的电影ID列表），计算相似度分数
model.predict([
    inputs["UserID_idx"],
    inputs["Gender_idx"],
    inputs["Age_idx"],
    inputs["Occupation_idx"],
    inputs["MovieID_idx"],
    inputs["Genres_idx"]
])

# 可以提取模型中的user或movie item 的embedding
user_layer_model = keras.models.Model(
    inputs=[model.input[0], model.input[1], model.input[2], model.input[3]],
    outputs=model.get_layer("user_embedding").output
)

user_embeddings = []
for index, row in df_user.iterrows():
    user_id = row["UserID"]
    user_input = [
        np.reshape(row["UserID_idx"], [1, 1]),
        np.reshape(row["Gender_idx"], [1, 1]),
        np.reshape(row["Age_idx"], [1, 1]),
        np.reshape(row["Occupation_idx"], [1, 1])
    ]
    user_embedding = user_layer_model(user_input)

    embedding_str = ",".join([str(x) for x in user_embedding.numpy().flatten()])
    user_embeddings.append([user_id, embedding_str])
df_user_embedding = pd.DataFrame(user_embeddings, columns=["user_id", "user_embedding"])
df_user_embedding.head()


