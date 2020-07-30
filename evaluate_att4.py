'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import Dataset
#from numba import jit, autojit
import linecache
# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, data, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K


    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread，
    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx,data)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)
#评估函数
def eval_one_rating(idx,data):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    #print(len(items))      #99
    #print(items)
    u = rating[0]
    gtItem = rating[1]
    t = rating[2]
    items.append(gtItem)
    #print(len(items))     #  100
    # Get prediction scores
    map_item_score = {}
    user_vectorpath = "Data/result_user_20.txt"
    item_vectorpath = "Data/result_item_20.txt"
    user_listpath = "Data/user_item_list.txt"
    item_listpath = "Data/item_user_list.txt"

    users = np.full(len(items), u, dtype = 'int32')
    #user当前的交互序列
    user_list = data.getTime_seq(user,item,t)
    user_list = [i+1 for i in list(users)]
    item_list = [j+1 for j in list(items)]
    #user_l,item_l维度是100*20
    user_l = Dataset.load_long_vector(user_list, user_vectorpath)
    item_l = Dataset.load_long_vector(item_list, item_vectorpath)
    #print(len(user_l))
    #print(user_l[0])
    # 根据用户列表和item列表产生对应的交互序列 100*4,
    user_interac_list = Dataset.load_short_list(user_listpath, user_list)
    item_interac_list = Dataset.load_short_list(item_listpath, item_list)
    #print(user_interac_list.shape)
    #测试数据完毕
    predictions = _model.predict([user_l, item_l, user_interac_list, item_interac_list
                                  ], batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()     #删除最后一个值
    
    # Evaluate top rank list,获得了topK个评分高的item
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)
#例如[0，25]，存在一个itemID和negative item构建的item列表，从预测的评分中选择分值最高的k个item列表，查看25是否在列表中
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
