#!/usr/bin/env python3
"""This script displays the number of launches per rocket from the SpaceX API
"""

import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'
    rocket_url = 'https://api.spacexdata.com/v4/rockets/'
    launches = requests.get(url).json()
    rocket_launches = {}

    for launch in launches:
        rocket_id = launch['rocket']
        rocket = requests.get(rocket_url + rocket_id).json()
        name = rocket['name']

        if name not in rocket_launches.keys():
            rocket_launches[name] = 1
        else:
            rocket_launches[name] += 1

    rocket_launches = sorted(rocket_launches.items(), key=lambda x: x[1])

    for k, v in reversed(rocket_launches):
        print('{}: {}'.format(k, v))
