import pandas as pd
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler

data_path = "./archieve"

unames = ['user_id','gender','age','occupation','zip']
user = pd.read_csv(data_path+'ml-1m/users.dat',sep='::',header=None,names=unames)
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv(data_path+'ml-1m/ratings.dat',sep='::',header=None,names=rnames)
mnames = ['movie_id','title','genres']
movies = pd.read_csv(data_path+'ml-1m/movies.dat',sep='::',header=None,names=mnames,encoding="unicode_escape")
movies['genres'] = list(map(lambda x: x.split('|')[0], movies['genres'].values))

data = pd.merge(pd.merge(ratings,movies),user)#.iloc[:10000]

# data = pd.read_csvdata = pd.read_csv("./movielens_sample.txt")
sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip", "genres"]
SEQ_LEN = 50
negsample = 0

# 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

# features = ['user_id','movie_id','gender', 'age', 'occupation', 'zip']
feature_max_idx = {}
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

item_profile = data[["movie_id", "genres"]].drop_duplicates('movie_id')

user_profile.set_index("user_id", inplace=True)

user_item_list = data.groupby("user_id")['movie_id'].apply(list)

train_set, test_set = gen_data_set(data, SEQ_LEN, negsample)

train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

# 2.count #unique features for each sparse field and generate feature config for sequence feature

embedding_dim = 32

user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                        SparseFeat("gender", feature_max_idx['gender'], 16),
                        SparseFeat("age", feature_max_idx['age'], 16),
                        SparseFeat("occupation", feature_max_idx['occupation'], 16),
                        SparseFeat("zip", feature_max_idx['zip'], 16),
                        VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                    embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                        VarLenSparseFeat(SparseFeat('hist_genres', feature_max_idx['genres'], embedding_dim,
                                                    embedding_name="genres"), SEQ_LEN, 'mean', 'hist_len'),
                        ]

item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim),
                        SparseFeat('genres', feature_max_idx['genres'], embedding_dim)
                        ]

from collections import Counter

train_counter = Counter(train_model_input['movie_id'])
item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
sampler_config = NegativeSampler('inbatch', num_sampled=255, item_name="movie_id", item_count=item_count)

# 3.Define Model and train

import tensorflow as tf

if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()
else:
    K.set_learning_phase(True)

model = DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(128, 64, embedding_dim),
             item_dnn_hidden_units=(64, embedding_dim,), loss_type='softmax', sampler_config=sampler_config)

model.compile(optimizer="adam", loss=sampledsoftmaxloss)

history = model.fit(train_model_input, train_label,  # train_label,
                    batch_size=256, epochs=20, verbose=1, validation_split=0.0, )

# 4. Generate user features for testing and full item features for retrieval
test_user_model_input = test_model_input
all_item_model_input = {"movie_id": item_profile['movie_id'].values, "genres": item_profile['genres'].values}

user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
# user_embs = user_embs[:, i, :]  # i in [0,k_max) if MIND
item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

print(user_embs.shape)
print(item_embs.shape)

