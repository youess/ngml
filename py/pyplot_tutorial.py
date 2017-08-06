#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 22:01:40 2017

@author: flea
"""

# This is a tutorial for pyplot.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from matplotlib import rcParams


# import matplotlib as mpl
# print(mpl.matplotlib_fname())
# 显示可用的主题
# print(plt.style.available)
plt.style.use("classic")     # 最好在下面的配置前面，不然不会生效

rcParams['font.sans-serif'] = ['Hei']    # 选择一个能显示中文的字体
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False   # 解决负数显示不正确问题


# a scatter point
x = np.random.randn(100, 1)
y = np.random.randn(100, 1)
plt.scatter(x, y, s=80, c="red", marker="+", alpha=.5)
plt.show()
plt.gcf().clear()           # 清除所画的图片，matplotlib会记住上一副画

# a line plot

x = np.arange(0, 1, .01)
y = np.random.random(len(x))
plt.plot(x, y)
plt.ylim(y.min() - 1, y.max() + 1)     # 设置y的显示段
plt.ylabel("ylim控制y轴显示的范围")
plt.show()
plt.gcf().clear()


# multiple plot
boston = load_boston()

X = boston.data
y = boston.target

n = X.shape[1]
plt.figure(figsize=(12, 10))
for i in range(n):
    plt.subplot(4, 4, i + 1)     # nrow, ncol, figure number start from 1
    plt.scatter(X[:, i], y)
    plt.title(boston.feature_names[i])
    plt.subplots_adjust(top=1.2, right=1)    # 调整图片之间的距离
    
plt.show()
plt.gcf().clear()

# 添加图片文字
np.random.seed(777)
mu, sig = 10, 3
x = mu + sig * np.random.randn(2000)

# hist
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=.75)

plt.xlabel("X label", fontsize=15, color="blue")
plt.ylabel("Prob")
plt.title("Hist of Normal distribution")
plt.text(0, 0.1, r'$\mu={},  \sigma={}$'.format(mu, sig))
plt.axis([-5, 25, 0, 0.15])
plt.grid()
plt.annotate('Center point', xy=(10, 0), xytext=(20, 0.04), 
             arrowprops=dict(
                     facecolor='blue', shrink=.05, width=2, headwidth=5))
plt.show()
plt.gcf().clear()


# bar plot
n = 5
men_mean = [18, 30, 35, 20, 25]
men_std = [2, 3, 4, 1, 5]
women_mean = [25, 32, 40, 20, 25]
women_std = [3, 5, 2, 3, 3]

# the x loc for each groups
ind = np.arange(n)
width = .35

# the bars
rect1 = plt.bar(ind, men_mean, width, color="blue", yerr=men_std, error_kw=dict(elinewidth=2, ecolor='red'))
rect2 = plt.bar(ind+width, women_mean, width, color="green", yerr=women_std, error_kw=dict(elinewidth=2, ecolor="red"))

# 设置轴的标签
plt.xlim(-width, len(ind) + width)
plt.ylim(0, 45)
plt.xlabel("Scores")
plt.ylabel("Scores by group and gender")
x_tick_marks = ['Group' + str(i) for i in range(1, 6)]
_, xtick_labels = plt.xticks(ind + width, x_tick_marks)
plt.setp(xtick_labels, rotation=45, fontsize=15)

# 添加legend
plt.legend((rect1[0], rect2[0]), ("Men", "Women"))

plt.show()
plt.gcf().clear()


# boxplot, pyplot可定制的比较多，但是实现的代码比较多，繁琐
# 用seaborn代替其中比较麻烦的部分能减轻比较多的代码负担，能够比较好的处理pandas以及
# matplotlib画图之间的问题
tips = sns.load_dataset('tips')

# single box
plt.figure()
sns.boxplot(tips['total_bill'], orient='v')
plt.show()
plt.gcf().clear()

# with category
plt.figure()
sns.boxplot(x='smoker', y='total_bill', data=tips, orient="v")
plt.show()
plt.gcf().clear()

# with 2 category
plt.figure()
sns.boxplot(x='smoker', y='total_bill', hue='sex', data=tips)
plt.show()
plt.gcf().clear()