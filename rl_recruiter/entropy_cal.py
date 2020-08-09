import json
import math 

def type_2_entro(traj_path, entro_path):
    
    with open(traj_path, 'r', encoding='utf-8') as f:
        cur_dict = json.load(f)
    
    entro_dict = {}
    result_entro = {}
    for user_id, traject in cur_dict.items():
        entro_dict[user_id] = {"loc_sum": 0, "info": {}}
        result_entro[user_id] = []

        tra_lenth = len(traject)
        if tra_lenth == 0:
            continue

        for j in range(tra_lenth):
            slot_cover = traject[j]
            
            for uloc in slot_cover:
                entro_dict[user_id]['loc_sum'] += 1
                if uloc not in entro_dict[user_id]['info'].keys():
                    entro_dict[user_id]['info'][uloc] = 1
                else:
                    entro_dict[user_id]['info'][uloc] += 1

            entro = 0
            for v in entro_dict[user_id]['info'].values():
                p = v / entro_dict[user_id]['loc_sum']
                entro += -p * math.log(p)
            result_entro[user_id].append(entro)
                
    with open(entro_path, 'w', encoding='utf-8') as f2:
        json.dump(result_entro, f2)