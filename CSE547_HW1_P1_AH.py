#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf

from pyspark.sql import SparkSession

import itertools
from operator import add
from itertools import permutations
from itertools import combinations

import findspark
findspark.init()


# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext


# ### Problem 1



# Read the text file
txt = sc.textFile("hw1-bundle/q1/data/soc-LiveJournal1Adj.txt")


# Create direct pair where they are friends and set that equal to -1000000
def direct(line):
    parts = line.split("\t")
    user_id = parts[0]
    friends = parts[1].split(",")
    return [((user_id, friend), -10000000) for friend in friends]
    
directpair = txt.flatMap(direct)


lines = txt.map(lambda line:line.split())
friends = lines.filter(lambda x: len(x) == 2).map(lambda x: (x[0], x[1].split(",")))
mutualpair = friends.flatMap(lambda data: [(pair, 1) for pair in itertools.permutations(data[1], 2)])


#add all the pairs together
allpair = directpair.union(mutualpair)


# print("Direct Friends", directpair.count())
# print("Mutual Friends", mutualpair.count())
# print("All", allpair.count())


#reduce it by key and sum up all the pairs 
allpair = allpair.reduceByKey(add)


#filter to see which pairs > 0 meaning they are not friends yet and should be recommended
mutualCount = allpair.filter(lambda pair: pair[1] > 0)


# reformat it to (user, count, friend rec)
mutual = mutualCount.map(lambda pair: (pair[0][0], (pair[1], pair[0][1])))


# now group it by user and give a list of the (count, friend)
mutual_group = mutual.groupByKey().mapValues(list)


# now sort it from count first and then user in numerically ascending order
mutual_group_sorted = mutual_group.map(lambda x: (x[0], sorted(x[1], key=lambda y: (-y[0], int(y[1])))))


# get the top 10 recommendations 
top_10_recommendations = [(user, data[:10]) for user, data in mutual_group_sorted.collect()]


for user, recommendations in top_10_recommendations:
    if user == '11':
        print(f"{list(recommendations)}")
        break


users = ['924', '8941', '8942', '9019', '9020', '9021', '9022', '9990', '9992', '9993']


filtered_recommendations = [(user, recommendations) for user, recommendations in top_10_recommendations if user in users]


result = [(outer_tuple[0], [inner_tuple[1] for inner_tuple in outer_tuple[1]]) for outer_tuple in filtered_recommendations]
sorted_result = sorted(result, key=lambda x: int(x[0]))


sorted_result

sc.stop()
