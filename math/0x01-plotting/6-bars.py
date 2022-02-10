#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80]
labels = ["Farrah", "Fred", "Felicia"]
apples = [3, 14, 15]
bananas = [6, 16, 9]
oranges = [8, 4, 7]
peaches = [16, 16, 7]

width = 0.5


plt.bar(labels, apples, width, color="red", label='apples')
plt.bar(labels, bananas, width, color="yellow", label='bananas', bottom=apples)
plt.bar(labels, oranges, width, color="#ff8000",
        bottom=[x + y for (x, y) in zip(apples, bananas)], label='oranges')
plt.bar(labels, peaches, width, color="#ffe5b4",
        bottom=[x + y + z for
                (x, y, z) in zip(apples, bananas, oranges)], label='peaches')
plt.yticks(ticks=ticks)
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.legend()
plt.savefig('widdly.png')
plt.show()
