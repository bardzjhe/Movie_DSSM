import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import pickle
import copy
import pandas as pd # data processing
from src.config.paths import *
from src.config.constants import *

# pickle is a module in Python used to serialize and deserialize Python object structures,
# which means converting Python objects to a byte stream (serialization) and reconstructing
# Python objects from that byte stream (deserialization). The process is also known
# as "pickling" and "unpickling".

def update_genre_dict(_genres, _label, genre_dict, update_type):
    """
    Updates the genre dictionary with the provided genres and labels according to the specified update type.

    Args:
        _genres (str): The genres in a string separated by '|'
        _label (int): The label associated with the genres
        genre_dict (dict): The dictionary to be updated
        update_type (str): The type of update to perform ('add' or 'sub')

    Returns:
        dict: The updated genre dictionary
    """
    assert update_type in ("add", "sub"), "update_type must be 'add' or 'sub'"
    for _genre in _genres.split("|"):
        genre = GENRES_MAPPING.get(_genre, GENRES_MAPPING.get(EMPTY_KEY))
        if genre not in genre_dict:
            genre_dict[genre] = [0, 0]
        operation = 1 if update_type == "add" else -1
        genre_dict[genre][0] += operation
        genre_dict[genre][1] += _label * operation
    return genre_dict



if __name__ == '__main__':
    # load movie data
    movie_fea = ["movieid", "title", "genres"]
    movies = pd.read_table(MovieLens_RAW_ITEM_PATH, sep="::", engine="python", header=None, names=movie_fea,
                           encoding="iso-8859-1")

    offline_data = pd.read_csv(OFFLINE_DATA_PATH)

    # merge the movies and offline data on movieid column.
    joined_data = movies.merge(offline_data, on="movieid", how="left")

    # print(joined_data[:50])

    # 1. group the offline data by user ID
    # 2. create a dictionary of movie IDs watched by each user.
    user_filter_dict = offline_data.groupby("userid")["movieid"].agg(list).to_dict()

    with open(USER_FILTER_PATH, 'wb') as user_filter_file:
        # serialize a Python object into a byte stream
        pickle.dump(user_filter_dict, user_filter_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Prepare dictionaries for impression terms and user terms
    vals = joined_data[["userid", "label", "genres"]].values.tolist()
    imp_dict = {}
    user_dict = {}
    pre_userid = -1
    dequeue = []
    genre_dict = {}

    # Iterate through the values to update impression and user dictionaries
    for i, (userid, label, genres) in enumerate(vals + [[-1, -1, ""]]):
        if userid != pre_userid:
            if pre_userid != -1:
                new_genres, new_label = dequeue[-1]
                genre_dict = update_genre_dict(new_genres, new_label, genre_dict, "add")
                user_dict[pre_userid] = copy.deepcopy(genre_dict)
            dequeue = []
            genre_dict = {}
            if userid == -1:
                break
        if len(dequeue) > LAST_N_GENRE_CNT:
            timeout_genres, timeout_label = dequeue.pop(0)
            genre_dict = update_genre_dict(timeout_genres, timeout_label, genre_dict, "sub")
        dequeue.append((genres, label))
        imp_dict[i] = copy.deepcopy(genre_dict)
        pre_userid = userid

    # Save the impression and user dictionaries to their respective paths
    with open(IMP_TERM_PATH, 'wb') as imp_term_file:
        pickle.dump(imp_dict, imp_term_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(USER_TERM_PATH, 'wb') as user_term_file:
        pickle.dump(user_dict, user_term_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Create and save the item term dictionary
    vals = movies[["movieid", "genres"]].values
    item_dict = {}
    for itemid, genres in vals:
        item_dict[itemid] = {GENRES_MAPPING.get(_genre, GENRES_MAPPING.get(EMPTY_KEY)) for _genre in genres.split("|")}
    print(item_dict)
    with open(ITEM_TERM_PATH, 'wb') as item_term_file:
        pickle.dump(item_dict, open(ITEM_TERM_PATH, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # Output the sizes of the dictionaries to confirm successful operations
    print(f"saved: imp_dict: {len(imp_dict)}, user_dict: {len(user_dict)}, item_dict: {len(item_dict)}")


