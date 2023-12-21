#!/bin/bash

###############################################################################
# Code written by Pierre Marza (pierre.marza@insa-lyon.fr)                    #
###############################################################################

# General hyper-parameters
n=${n:-"25"}
num_processes_per_gpu=${num_processes_per_gpu:-"3"}
num_processes_on_first_gpu=${num_processes_on_first_gpu:-"4"}
exp_name=${exp_name:-"exp_name"}
split=${split:-"train"}
eval=${eval:-"0"}
load=${load:-"0"}
eval_scene_id=${eval_scene_id:-"0"}

# Rewards
use_expl_reward=${use_expl_reward:-"0"}
use_obs_reward=${use_obs_reward:-"0"}
use_sem_obj_reward=${use_sem_obj_reward:-"0"}
use_viewpoint_reward=${use_viewpoint_reward:-"0"}
nb_additional_channels=${nb_additional_channels:-"0"}

# AutoNeRF training data collection
save_autonerf_data=${save_autonerf_data:-"0"}
autonerf_dataset_path=${autonerf_dataset_path:-"autonerf_dataset_path"}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done

python -u main.py --auto_gpu_config 0 -n $n --num_processes_per_gpu $num_processes_per_gpu --num_processes_on_first_gpu $num_processes_on_first_gpu \
--sim_gpu_id 1 -d saved/ --exp_name $exp_name --split $split --eval $eval --load $load --print_images 1  --eval_scene_id $eval_scene_id \
--use_expl_reward $use_expl_reward --use_obs_reward $use_obs_reward --use_sem_obj_reward $use_sem_obj_reward --use_viewpoint_reward $use_viewpoint_reward \
--nb_additional_channels $nb_additional_channels --save_autonerf_data $save_autonerf_data --autonerf_dataset_path $autonerf_dataset_path
