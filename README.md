# RL-Recruiter+

A participant selection algorithm using reinforcement learning. Details are in the paper "RL-Recruiter+: Mobility-Predictability-Aware Participant Selection Learning for From-Scratch Mobile Crowdsensing".

## Paper Abstract

Participant selection is a fundamental research issue in Mobile Crowdsensing (MCS). Previous approaches commonly assume that adequately long periods of candidate participants' historical mobility trajectories are available to model their patterns before the selection process, which is not realistic for some new MCS applications or platforms. The sparsity or even absence of mobility traces will incur inaccurate location prediction, thus undermining the deployment of new MCS applications. To this end, this paper investigates a novel problem called “From-Scratch MCS” (FS-MCS for short), in which we study how to intelligently select participants to minimize such “cold-start” effect. Specifically, we propose a novel framework based on reinforcement learning, named RL-Recruiter+. With the gradual accumulation of mobility trajectories over time, RL-Recruiter+ is able to make a good sequence of participant selection decisions for each sensing slot. Compared to its previous version RL-Recruiter, Re-Recruiter+ jointly considers both the previous coverage and current mobility predictability when training the participant selection decision model. We evaluate our approach experimentally based on two real-world mobility datasets, and the results demonstrate that RL-Recruiter+ outperforms the baseline approaches, including RL-Recruiter under various settings.

## Objective of RL-Recruiter+

Given the historical trajectories of a set of participants, the RL-Recruiter+ can do the learning process based on them, and then selected a fixed number of participants. RL-Recruiter+ tries to maximize the area covered by the selected participants.

## Running Example

You can see a quick example on how to use the RL-Recruiter+ in this [page](https://github.com/chungdz/RL-Recruiter-Plus/blob/master/example/run_example.ipynb).

## Input and Output of RL-Recruiter+

### Parameter Settings
The first settings saved in a json file that needs to initialize the model. A example is in this [page](https://github.com/chungdz/RL-Recruiter-Plus/blob/master/example/data/para_settings.json).

There are several parameters need to be deployed:
* "total_person": the number of whole participants.
* "max_user": the number of participants to be selected.
* "train_start": the index in the list of trajectories that the model needs to begin learning.
* "train_end": the index in the list of trajectories that the model needs to stop learning. For example, if the trajectories are from 10 days and each day's trajectories are gathered in a list, we set "train_start" as 0 and "train_end" as 10.
* "train_epoch": the number of training epoch.
* "layer": the number of rows in value function table, cannot be higher than "max_user", the higher the layer, the larger the table, and thus it needs more data and epoch to learn and may get a better result.

There is another file needs to be imported to the model, which contains the threshold values for discretizing the entropy values which is one input to do the training and predicting. It is in the json list format. Just using our [thresholds](https://github.com/chungdz/RL-Recruiter-Plus/blob/master/example/data/thres.json) is fine. There are 100 threshold in the list. Making another file with the same format is also feasible. 

### Training Process

There are two data files need to be input. 

The first is the trajectory data, you can see example [here](https://github.com/chungdz/RL-Recruiter-Plus/blob/master/example/data/trajectory.json). The trajectory data is a dictionary saved in json format. The key is a participant id and the value is a list of trajectory sets in each time period. The participant ids need to be mapped into consecutive integer. If there are 100 participants, then the key list in trajectory dictionary are like ["0", "1", ..., "99"]. One trajectory set contains nonredundant categorical integers representing the area covered by its participant in this time period. The integer "-1" in trajectory sets will be ignored and not counted in coverage.

The other is the participants' predictability data that can be calculated from trajectory data. The format of the predictability data is also a dictionary with keys the participant ids and values the list of entroy values in each time period.

    from rl_recruiter.entropy_cal import type_2_entro
    type_2_entro('./data/trajectory.json', './data/predictability.json')

### Predict Process

There is an optional input to do the prediction. This input is a list of entropy values with the length equal to number of total participants. Value in dimension k represents the entropy value for participant with id "k". 

One can construct this list using the predictability data obtained before.

    import json
    import numpy as np

    with open('./data/predictability.json', 'r') as f:
    entro_dict = json.load(f)

    entro_list = np.zeros((total_person))
    for k, v in entro_dict.items():
        try:
            entro_list[int(k)] = v[train_last_index]
        except:
            pass

 If this input is given, then RL-Recruiter+ will selected participants using both the predictability information and the trained value function. It is recommanded for a better performance. 