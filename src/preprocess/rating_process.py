import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import logging
import pandas as pd # data processing
from src.config.paths import *
from src.config.constants import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('data_processing.log')

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def split_data(x):
    """
    Split data into offline training, offline test and online serving.

    Offline training (0): the data is used to fit the model.
    Offline test (1): the test set is used to evaluate the model's performance.
    Online serving (2): the data is used for real-time predictions and simulates the future, real-world data.
    """

    if x["rn"] > (x["u_cnt"]-ONLINE_N):
        return 2  # last 10 for online serving

    if (x["rn"] < (x["u_cnt"]-ONLINE_N) * (1 - OFFLINE_TEST_RATIO)):
        return 0  # first 80% for offline training

    return 1 # last 20% for offline test


if __name__ == '__main__':

    # load user data
    user_fea = ["userid", "gender", "age", "occupation", "zip"]
    users = pd.read_table(MovieLens_RAW_USER_PATH, sep="::", engine="python", header=None, names=user_fea, encoding="iso-8859-1")
    # load rating data
    rating_fea = ["userid", "movieid", "rating", "time"]
    ratings = pd.read_table(MovieLens_RAW_RATING_PATH, sep="::", engine="python", header=None, names=rating_fea, encoding="iso-8859-1")

    # test code
    # print(users.dtypes)

    # convert explicit rating into implicit feedback.
    # a rating greater than 3 is considered positive interaction (1), otherwise as negative (0)
    ratings["label"] = ratings["rating"].map(lambda x: 1 if x > 3 else 0)
    ratings = ratings.drop("rating", axis=1)

    user_item_count = ratings.groupby("userid")["movieid"].count().rename("u_cnt")
    print(user_item_count.head())
    rating_data = pd.merge(ratings, user_item_count, on="userid")
    # print(rating_data.head())

    # print(rating_data[:20])

    # make each user's interaction in chronological order
    rating_data = rating_data.sort_values(["userid", "time"], ascending=[True, True]).reset_index(drop=True)

    rating_data["rn"] = rating_data.groupby("userid").cumcount()
    print(rating_data['u_cnt'].dtype)


    # print(rating_data[:100])

    rating_data["test_type"] = rating_data.apply(split_data, axis=1)


    # Pepare and save offline and online sets
    offline_data = rating_data.query(f"{'test_type'} != 2").drop(["rn", "u_cnt"], axis=1)
    online_data = rating_data.query(f"{'test_type'} == 2").drop(["rn", "test_type", "u_cnt"], axis=1)

    # print(offline_data[:40])

    offline_data.to_csv(OFFLINE_DATA_PATH, index=False)
    online_data.to_csv(ONLINE_DATA_PATH, index=False)
