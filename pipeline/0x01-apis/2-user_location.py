#!/usr/bin/env python3
"""Script that finds a GitHub user location from their user name."""

import requests
import sys
import time


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit()

    user = sys.argv[1]
    r = requests.get(user)

    if r.status_code == 404:
        print('Not found')
    if r.status_code == 403:
        reset_time = int(r.headers['X-Ratelimit-Reset']) - time.time()
        minutes = round(reset_time / 60)
        print('Reset in {} min'.format(minutes))
    if r.status_code == 200:
        location = r.json()['location']
        if location is not None:
            print('{}'.format(location))
        else:
            print('Not found')
