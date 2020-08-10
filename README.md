# RL_Recruiter+

A participant selection algorithm using reinforcement learning. Details are in the paper "RL-Recruiter+: Mobility-Predictability-Aware Participant Selection Learning for From-Scratch Mobile Crowdsensing".

## Paper Abstract

Participant selection is a fundamental research issue in Mobile Crowdsensing (MCS). Previous approaches commonly assume that adequately long periods of candidate participants' historical mobility trajectories are available to model their patterns before the selection process, which is not realistic for some new MCS applications or platforms. The sparsity or even absence of mobility traces will incur inaccurate location prediction, thus undermining the deployment of new MCS applications. To this end, this paper investigates a novel problem called “From-Scratch MCS” (FS-MCS for short), in which we study how to intelligently select participants to minimize such “cold-start” effect. Specifically, we propose a novel framework based on reinforcement learning, named RL-Recruiter+. With the gradual accumulation of mobility trajectories over time, RL-Recruiter+ is able to make a good sequence of participant selection decisions for each sensing slot. Compared to its previous version RL-Recruiter, Re-Recruiter+ jointly considers both the previous coverage and current mobility predictability when training the participant selection decision model. We evaluate our approach experimentally based on two real-world mobility datasets, and the results demonstrate that RL-Recruiter+ outperforms the baseline approaches, including RL-Recruiter under various settings.

## Model Objective

Given the historical trajectories of a set of participants, the RL_Recruiter+ can do the learning process based on them, and then selected a fixed number of participants. RL_Recruiter+ tries to maximize the area covered by the selected participants.

## Running Example

You can see a quick example on how to use the RL-Recruiter+ in this [page]().

## Model Input and Output

### Model Settings
The model settings saved in a json file that needs to initialize the model. A example is in this [page]().

There are several parameters need to be deployed:
* "total_person": the number of whole participants.
* "max_user": the number of participants to be selected.
* "train_start": the index in the list of trajectories that the model needs to begin learning.
* "train_end": the index in the list of trajectories that the model needs to stop learning. For example, if the trajectories are from 10 days and each day's trajectories are gathered in a list, we set "train_start" as 0 and "train_end" as 10.
* "train_epoch": the number of training epoch.
* "layer": the number of rows in value function table, cannot be higher than "max_user", the higher the layer, the larger the table, and thus it needs more data and epoch to learn and may get a better result.

### Training Process

There are two data files need to be input. 

The first is the trajectory data, you can see example [here](https://github.com/chungdz/RL_Recruiter-Plus/blob/master/example/data/trajectory.json). The trajectory data is a dictionary saved in json format. The key is a participant id and the value is a list of trajectory sets in each time period. The participant ids need to be mapped into consecutive integer. If there are 100 participants, then the key list in trajectory dictionary are like ["0", "1", ..., "99"]. One trajectory set contain nonredundant categorical integers representing the area covered by its participant in this time period.