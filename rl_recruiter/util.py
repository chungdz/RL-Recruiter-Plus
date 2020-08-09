import json
import numpy as np
import heapq
import random

def layer_index(selected_amount, bin_amount, layer):
    lay_idx = int(selected_amount / bin_amount)
    if lay_idx >= layer:
        lay_idx = layer - 1
    return lay_idx

def thres_index(thres_list, v):
    idx = 0
    for t in thres_list:
        if v < t:
            break
        else:
            idx += 1
    
    return idx

def get_user_list(trace):
    ulist = []
    for key, value in trace.items():
        if len(value) > 0:
            ulist.append(int(key))
    return ulist


def get_index_nlargest(nparr, nlarge):
    result = heapq.nlargest(nlarge, range(len(nparr)), nparr.take)
    return result[-1]


def get_index_largest(result_dim, indexarr):
    index = indexarr[0]
    maxp = result_dim[index]

    for i in range(len(indexarr)):
        curindex = indexarr[i]
        if result_dim[curindex] > maxp:
            index = curindex
            maxp = result_dim[curindex]

    return index, maxp

def add_new_location(coverage, user_trace):
    reward = 0
    for point in user_trace:
        if point not in coverage and int(point) != -1:
            reward += 1
            coverage.append(point)
    return reward, coverage