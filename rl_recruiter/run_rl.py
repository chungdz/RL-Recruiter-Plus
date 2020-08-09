from rl_recruiter.util import get_user_list, get_index_largest, add_new_location, layer_index, thres_index
import json
import copy
import numpy as np
import random

def train_and_evaluate(track_data_file, entro_file, thres_file, hypara_file, rseed=1):
    random.seed(rseed)
    # load parameter settings
    with open(hypara_file, 'r', encoding='utf-8') as hp:
        hypara_dict = json.load(hp)
    
    total_person = hypara_dict['total_person']
    layer = hypara_dict['layer']
    train_start_day = hypara_dict['train_start']
    train_end_day = hypara_dict['train_end']
    train_epoch = hypara_dict['train_epoch']
    epsilon = hypara_dict['epsilon']
    max_user = hypara_dict['max_user']
    gamma = hypara_dict['gamma']
    alpha = hypara_dict['alpha']
    beta = hypara_dict['beta']

    bin_amount = total_person / layer
    # 训练模型，并保存
    print('Loading trajectory Data...')
    f = open(track_data_file, 'r', encoding='utf-8')
    trackdata = json.load(f)
    print('Getting eligible user list')
    ulist = get_user_list(trackdata)

    print('init record set')
    record = []
    avg_score = np.zeros((layer, total_person))
    for l in range(layer):
        layer_level = []
        for i in range(total_person):
            meta = {
                'count': 0,
                'total': 0,
            }
            layer_level.append(meta)
        record.append(layer_level)
    
    print('load entro_daily')
    with open(entro_file, 'r', encoding='utf-8') as end:
        entro = json.load(end)
    
    print('load entro_threshold')
    with open(thres_file, 'r', encoding='utf-8') as enf:
        thres = json.load(enf)
    thres_util = []
    for i in range(len(thres) + 1):
        thres_util.append({'sum': 1, 'count': 1})
    
    predict_result = []

    for day in range(train_start_day, train_end_day):
        for epoch in range(train_epoch):
            cur_eps = epsilon
            if epoch == 0:
                cur_eps = 0

            choice_list = copy.copy(ulist)
            curcoverage = []
            selected = 0
            while True:
                #layer_idx
                lay_idx = layer_index(selected, bin_amount, layer)
                # select next user
                if random.random() > cur_eps:
                    # get output of the model
#                     user_id, _ = get_index_largest(avg_score[lay_idx], choice_list)
                    cur_score_list = avg_score[lay_idx].copy()
                    for i in range(total_person):
                        if len(entro[str(i)]) > 0:
                            cur_idx = thres_index(thres, entro[str(i)][day])
                            cur_score_list[i] += beta * (thres_util[cur_idx]['sum'] / thres_util[cur_idx]['count'])
                    user_id, _ = get_index_largest(cur_score_list, choice_list)
                else:
                    user_id = random.choice(choice_list)

                # refresh current status
                choice_list.remove(user_id)
                reward, curcoverage = add_new_location(curcoverage, trackdata[str(user_id)][day])
                if selected >= max_user or len(choice_list) == 0:
                    final_reward = reward
                else:
                    # predict +1 step
                    lay_idx_next = layer_index(selected + 1, bin_amount, layer)
#                     user_id_next, _next = get_index_largest(avg_score[lay_idx_next], choice_list)
                    next_score_list = avg_score[lay_idx_next].copy()
                    for i in range(total_person):
                        if len(entro[str(i)]) > 0:
                            cur_idx = thres_index(thres, entro[str(i)][day])
                            next_score_list[i] += beta * (thres_util[cur_idx]['sum'] / thres_util[cur_idx]['count'])
                    user_id_next, _next = get_index_largest(next_score_list, choice_list)
                    # reward_next, curcoverage_next = add_new_location(curcoverage, trackdata[str(user_id_next)][day])
                    final_reward = avg_score[lay_idx_next][user_id_next] * gamma + reward

                record[lay_idx][user_id]['count'] += 1
                record[lay_idx][user_id]['total'] += reward
                if epoch > 0:
                    user_idx = thres_index(thres, entro[str(user_id)][day])
                    thres_util[user_idx]['sum'] += final_reward
                    thres_util[user_idx]['count'] += 1
                
                avg_score[lay_idx][user_id] = avg_score[lay_idx][user_id] + alpha * (final_reward - avg_score[lay_idx][user_id])
                selected += 1

                if selected >= max_user or len(choice_list) == 0:
                    cov = len(curcoverage)
                    if epoch == 0:
                        predict_result.append(cov)
                    break
    print(predict_result)
    return np.array(predict_result)