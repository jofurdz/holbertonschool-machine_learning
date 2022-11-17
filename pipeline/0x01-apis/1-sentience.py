#!/usr/bin/env python3
"""module contains function sentientPlanets"""
import requests
import json


def sentientPlanets():
    """returns a list of names of planets with sentient species"""
    url = 'https://swapi-api.hbtn.io/api/species/?format=json'
    planets = []

    while url:
        r = requests.get(url).json()
        for species in r['results']:
            if species['homeworld'] is not None and \
               (species['classification'] == 'sentient' or
               species['designation'] == 'sentient'):
                 planets.append(requests.get(
                    species['homeworld']
                 ).json()['name'])
        url = r.get('next')
    return planets
