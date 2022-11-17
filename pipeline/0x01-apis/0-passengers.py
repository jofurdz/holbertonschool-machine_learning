#!/usr/bin/env python3
"""module contains function availableShips"""


import requests
import json


def availableShips(passengerCount):
    """returns the list of ships that can hold a given number of passengers"""
    ships = []
    url = 'https://swapi-api.hbtn.io/api/starships/?format=json'
    while url:
        r = requests.get(url).json()
        for result in r['results']:
            try:
                if int(result['passengers'].replace(',', ''))\
                   >= passengerCount:
                    ships.append(result['name'])
            except Exception as e:
                continue
        url = r.get('next')

    return ships
