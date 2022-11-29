#!/usr/bin/env python3
"""Contains the method insert_school()"""

from pymongo import MongoClient


def insert_school(mongo_collection, **kwargs):
    """Inserts a new doc into a mongo collection"""
    return mongo_collection.insert(kwargs)


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.my_db.school
    new_school_id = insert_school(school_collection, name="UCSF", address="505 Parnassus Ave")
    print("New school created: {}".format(new_school_id))

    schools = list_all(school_collection)
    for school in schools:
        print("[{}] {} {}".format(school.get('_id'), school.get('name'), school.get('address', "")))