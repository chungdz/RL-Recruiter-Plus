from rl_recruiter.util import get_user_list, get_index_largest, add_new_location, layer_index, thres_index
import json
import copy
import numpy as np
import random

class RL_Recruiter_plus:
    def __init__(self, hypara_file, thres_file):
        with open(hypara_file, 'r', encoding='utf-8') as hp:
            self.hypara_dict = json.load(hp)
        self.q_table = np.zeros((self.hypara_dict['layer'], self.hypara_dict['total_person']))

        with open(thres_file, 'r', encoding='utf-8') as enf:
            self.thres = json.load(enf)
        self.thres_util = []
        for i in range(len(self.thres) + 1):
            self.thres_util.append({'sum': 1, 'count': 1})
    
    def save_weights(self, dir_path):
        np.save(dir_path + 'q_table.npy', self.q_table)
        with open(dir_path + 'entro_scores.json', 'w', encoding='utf-8') as f:
            json.dump(self.thres_util, f)
    
    def load_weights(self, dir_path):
        self.q_table = np.load(dir_path)
        with open(dir_path + 'entro_scores.json', 'r', encoding='utf-8') as f:
            self.thres_util = json.load(f)
    
    def predict(self, entro_list):
        total_person = self.hypara_dict['total_person']
        layer = self.hypara_dict['layer']
        epsilon = self.hypara_dict['epsilon']
        max_user = self.hypara_dict['max_user']
        gamma = self.hypara_dict['gamma']
        alpha = self.hypara_dict['alpha']
        beta = self.hypara_dict['beta']
        thres = self.thres
        thres_util = self.thres_util
        avg_score = self.q_table

        bin_amount = total_person / layer
        selected = 0
        choice_list = [i for i in range(total_person)]
        result = []
        while True:
            lay_idx = layer_index(selected, bin_amount, layer)
            cur_score_list = avg_score[lay_idx].copy()
            for i in range(total_person):
                if entro_list[i] != -1:
                    cur_idx = thres_index(thres, entro_list[i])
                    cur_score_list[i] += beta * (thres_util[cur_idx]['sum'] / thres_util[cur_idx]['count'])
            user_id, _ = get_index_largest(cur_score_list, choice_list)
            result.append(user_id)
            selected += 1

            if selected >= max_user:
                break
        return result

    
    def train_and_evaluate(self, track_data_file, entro_file, rseed=1):
        random.seed(rseed)
        # load parameter settings
        total_person = self.hypara_dict['total_person']
        layer = self.hypara_dict['layer']
        train_start_day = self.hypara_dict['train_start']
        train_end_day = self.hypara_dict['train_end']
        train_epoch = self.hypara_dict['train_epoch']
        epsilon = self.hypara_dict['epsilon']
        max_user = self.hypara_dict['max_user']
        gamma = self.hypara_dict['gamma']
        alpha = self.hypara_dict['alpha']
        beta = self.hypara_dict['beta']
        thres = self.thres
        thres_util = self.thres_util

        bin_amount = total_person / layer
        # 训练模型，并保存
        print('Loading trajectory Data...')
        f = open(track_data_file, 'r', encoding='utf-8')
        trackdata = json.load(f)
        print('Getting eligible user list')
        ulist = get_user_list(trackdata)

        print('init record set')
        avg_score = self.q_table
        
        print('load entro_daily')
        with open(entro_file, 'r', encoding='utf-8') as end:
            entro = json.load(end)
        
        predict_result = []

        for day in range(train_start_day, train_end_day):
            print('day: '+ str(day))
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
                        cur_score_list = avg_score[lay_idx].copy()
                        for i in range(total_person):
                            if len(entro[str(i)]) > 0 and day != train_start_day:
                                # in first epoch, we do the predict first, as we do not know entropy data 
                                # in current slot, we use the data in last day
                                cur_idx = thres_index(thres, entro[str(i)][day - 1])
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
                        lay_idx_next = layer_index(selected + 1, bin_amount, layer)
                        next_score_list = avg_score[lay_idx_next].copy()
                        for i in range(total_person):
                            if len(entro[str(i)]) > 0:
                                cur_idx = thres_index(thres, entro[str(i)][day])
                                next_score_list[i] += beta * (thres_util[cur_idx]['sum'] / thres_util[cur_idx]['count'])
                        user_id_next, _next = get_index_largest(next_score_list, choice_list)
                        final_reward = avg_score[lay_idx_next][user_id_next] * gamma + reward

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
        print("each day's coervage")
        print(predict_result)
        return np.array(predict_result)