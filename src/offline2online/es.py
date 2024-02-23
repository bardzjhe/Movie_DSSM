import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pickle
import time
from src.config.paths import *
from src.config.constants import *


if __name__ == '__main__':
    # load data
    item_dict = pickle.load(open(ITEM_TERM_PATH, "rb"))
    item_vector = pickle.load(open(ITEM_VECTOR_PATH, "rb"))

    # connect to Elasticsearch server with the specified credentials
    es = Elasticsearch(f"https://elastic:{ES_KEY}@localhost:9200", verify_certs=False)

    # create index if the index doesn't exist
    if not es.indices.exists(index="movie_index"):
        print("movie_index", "is creating...")
        create_mapping = {
            "properties": {
                "movieid": {
                    "type": "long"
                },
                "genres": {
                    "type": "long"
                },
                "item_vector": {
                    "type": "dense_vector",
                    "dims": RECALL_EMB_DIM,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
        es.indices.create(index="movie_index", mappings=create_mapping)
        print("create done")
        es.options(ignore_status=400)

    # TODO: 理解什么意思 delete existing data
    all_query = {
        "match_all": {}
    }
    es.delete_by_query(index="movie_index", query=all_query)

    # insert data
    action = [{
        "_index": "movie_index",
        "movie_id": i,
        "genres": list(item_dict[i]),
        "movie_vector": item_vector[i]
    } for i in item_dict]
    helpers.bulk(client=es, actions=action)
    time.sleep(10)
    print("all data count: ", es.search(index=ITEM_ES_INDEX, query=all_query)["hits"]["total"]["value"])

    # check term index
    test_genre = 2  # modify it to another genre for further testing
    cnt_true = 0  # test_genre true count
    for i in item_dict:
        if test_genre in item_dict[i]:
            cnt_true += 1

    # use a term query to check if the indexed data can
    # be retrieved by genre.
    term_query = {
        "terms": {
            "genres": [test_genre],
            "boost": 0.1
        }
    }
    res = es.search(index="movie_index", query=term_query)
    cnt_hit = res["hits"]["total"]["value"]  # test_genre hit count
    print(f"check term index with genre {test_genre}: true {cnt_true}, hit {cnt_hit}. ",
          "test passed!" if cnt_true == cnt_hit else "test failed!")

    # check vector index
    test_itemid = 2333  # modify it to another id for further testing
    query_vector = item_vector[test_itemid]  # item 1 as query vector
    vector_query = {
        "field": "movie_index",
        "query_vector": query_vector,
        "k": 20,
        "num_candidates": 500,
        "boost": 0.9
    }
    res = es.search(index="movie_index", knn=vector_query)
    hit_1_itemid = res["hits"]["hits"][0]["_source"]["itemid"]
    print(f"check vector index with id {test_itemid}: true {test_itemid}, hit {hit_1_itemid}. ",
          "test passed!" if hit_1_itemid == test_itemid else "test failed!")

    # combined query
    print("knn and term", es.search(index="movie_index", knn=vector_query, query=term_query))



