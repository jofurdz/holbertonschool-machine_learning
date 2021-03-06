#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, axs = plt.subplots(3, 2, constrained_layout=True)

# top left
axs[0][0].set_xlim(0, 10)
axs[0][0].plot(y0, color="red")

# top right
axs[0][1].scatter(x1, y1, c="magenta")
axs[0][1].set_xlabel("Height (in)")
axs[0][1].set_ylabel("Weight (lbs)")
axs[0][1].set_title("Men's Height vs Weight")

# middle left
axs[1][0].plot(x2, y2)
axs[1][0].set_xlabel("Time (years)")
axs[1][0].set_ylabel("Fraction Remaining")
axs[1][0].set_title("Exponential Decay of C-14")
axs[1][0].set_yscale("log")
axs[1][0].set_xlim([0, 28650])

# middle right
axs[0][1].set_xlabel("Time (years)")
axs[0][1].set_ylabel("Fraction Remaining")
axs[0][1].set_title("Exponential Decay of Radioactive Elements")
axs[0][1].set_xlim([0, 20000])
axs[0][1].set_ylim([0, 1])
axs[0][1].plot(x3, y31, linestyle='dashed', color='red', label='C-14')
axs[0][1].plot(x3, y32, color='green', label='Ra-226')
axs[0][1].legend()

# bottom
axs[2][0] = plt.subplot2grid((3, 2), (2, 0), colspan=2)
bins_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
axs[2][0].hist(student_grades, bins=bins_ticks, edgecolor="black")
axs[2][0].set_xlabel("Grades")
axs[2][0].set_ylabel("Number of Students")
axs[2][0].set_title("Project A")
axs[2][0].axis([0, 100, 0, 30])
axs[2][0].set_xticks(ticks=bins_ticks)

plt.suptitle("All in One")
plt.show()
