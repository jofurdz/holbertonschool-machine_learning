#!/usr/bin/env python3
"""Script that provides Nginx stats from logs stored in MongoDB"""

from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx
    count = logs.count_documents({})
    methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
    
    print('{} logs'.format(count))
    print('Methods:')
    for method in methods:
        count = logs.count_documents({'method': method})
        print('\tmethod {}: {}'.format(method, count))

    filter_path = {'method': 'GET', 'path': '/status'}
    count = logs.count_documents(filter_path)

    print('{} status check'.format(count))