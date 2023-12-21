# AutoNeRF: Training Implicit Scene Representations with Autonomous Agents 
### [Project Page](https://pierremarza.github.io/projects/autonerf/) | [Paper](https://arxiv.org/abs/2304.11241)
Pytorch implementation of the AutoNeRF paper. This codebase is based on 3 great repositories: [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation), [semantic_nerf](https://github.com/Harry-Zhi/semantic_nerf), [nerfstudio](https://github.com/nerfstudio-project/nerfstudio). <br><br>

[AutoNeRF: Training Implicit Scene Representations with Autonomous Agents](https://arxiv.org/abs/2107.06011)
 [Pierre Marza](https://pierremarza.github.io/)<sup>1</sup>,
 [Laetitia Matignon](https://perso.liris.cnrs.fr/laetitia.matignon/)<sup>2</sup>,
 [Olivier Simonin](http://perso.citi-lab.fr/osimonin/)<sup>1</sup>,
 [Dhruv Batra](https://faculty.cc.gatech.edu/~dbatra/)<sup>3, 5</sup>,
 [Christian Wolf](https://chriswolfvision.github.io/www/)<sup>4</sup>,
 [Devendra Singh Chaplot](https://devendrachaplot.github.io/) <sup>3</sup><br>
 
 <sup>1</sup>INSA Lyon, <sup>2</sup>Université Lyon 1, <sup>3</sup>Meta AI , <sup>4</sup>Naver Labs Europe, <sup>5</sup>Georgia Tech <br>

<img src='images/graphical_abstract.png' width="25%" height="25%"/>

## Setup
This code repository is composed of 3 main parts: policy, semantic_nerf, nerfstudio. Below are the instructions to setup each of them.

### policy
In order to use the code related to the autonomous policy (inference to collect NeRF training data or policy training), we suggest using the Docker or Singularity image provided [here](https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/docs/DOCKER_INSTRUCTIONS.md). This code was tested with the Singularity image. In this case, the setup is as simple as what follows,

```
cd policy
singularity pull docker://devendrachaplot/habitat:sem_exp
```

The script policy/exec_run.sh will be used to run inference or training code inside the Singularity image.

### semantic_nerf
Following https://github.com/Harry-Zhi/semantic_nerf#dependencies, in order to train a Vanilla Semantic NeRF and later extract a mesh from it, you can follow the simple dependency installation guidelines below,

```
cd semantic_nerf
conda create -n semantic_nerf python=3.7
conda activate semantic_nerf
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

### nerfstudio
Before installing dependencies, please check you meet all requirements to install tiny-cuda-nn (https://github.com/NVlabs/tiny-cuda-nn?tab=readme-ov-file#requirements).
You can then do the following to install dependencies (for CUDA 11.6 in this example),
```
cd nerfstudio
conda create --name nerfstudio -y python=3.9
conda activate nerfstudio
pip install --upgrade pip setuptools
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -e .
ns-install-cli
pip install scikit-fmm # Required to benchmark planning
```

## Data
### policy
#### Episodes (train and val) and scenes data (val only)
Training and validation episodes, along with validation scenes can be downloaded [here](https://drive.google.com/file/d/1X08U929c3GFO51EgMAcNqIoVhp52EmsN/view?usp=sharing). *data.zip* should be unzipped and placed within the *poliycy/* folder. If you want to train your own policy model, follow instructions [there](https://github.com/devendrachaplot/Object-Goal-Navigation) to download all training scenes.

#### Pre-trained model
The checkpoints of the Mask-RCNN model used in our mapping module and of a pre-trained modular policy (*Ours(obs.)*) used in most experiments in the paper can be downloaded [here](https://drive.google.com/file/d/1NHvuKmcNmGu7sZahF5GLSTobZupZhhgA/view?usp=sharing). *pretrained_models.zip* should be unzipped and placed within this *policy/* folder.

### semantic_nerf
Semantic NeRF rendering test data can be downloaded [here](https://drive.google.com/file/d/1ZJXihiX1_fAbtdY1FJzSFB1KYPa6rhfp/view?usp=sharing). *data.zip* should be unzipped and placed within the *semantic_nerf/* folder.

### nerfstudio
#### Rendering test data
nerfstudio rendering test data can be downloaded [here](https://drive.google.com/file/d/1Htif2BkfL47f07YKRqBBAe78X5I_Lt1N/view?usp=sharing). *data.zip* should be unzipped and placed within the *nerfstudio/* folder.
#### Additional benchmark data (BEV map and planning)
nerfstudio benchmark data can be downloaded [here](https://drive.google.com/file/d/1v8pligeKdm4YeIHePubJFfn09b2JlhOI/view?usp=sharing). *data.zip* should be unzipped and placed within the *nerfstudio/benchmark/* folder.

## Reconstructing house-scale scenes
### Data collection with the policy
The following command allows you to collect AutoNeRF training data on a new scene. You can specify the id of one of the 5 Gibson val scenes with (*--eval_scene_id* -- 0: Collierville, 1: Corozal, 2: Darden, 3: Markleeville, 4: Wiconisco), the name of the experiment (*--exp_name*), the path to the policy checkpoint to load (*--load*), the path where to save collected data (*--autonerf_dataset_path*). You must specify *--split val* and *--eval 1* as you want to collect (inference time) on validation episodes.

```
cd policy
./exec_run.sh --n 1 --num_processes_per_gpu 1 --num_processes_on_first_gpu 1 --exp_name collect_data_modular_policy_obs_Corozal --split val --eval 1 --load pretrained_models/modular_policy_obs.pth --eval_scene_id 1 --save_autonerf_data 1 --autonerf_dataset_path autonerf_data_modular_policy_obs_Corozal
```

### Semantic NeRF training
After collecting data, you can train a Semantic NeRF model with the following command. You can specify the path where the collected data was saved (*--train_data_dir*) and the path to the test data, i.e. camera poses sampled uniformly in the scene, used to test the trained model (*--test_data_dir*), and whether to use simulator (*--use_GT_sem True*) or Mask-RCNN (*--use_GT_sem False*) semantic masks to train the NeRF semantic head.

```
cd semantic_nerf
conda activate semantic_nerf
python -u train_SSR_main.py --train_data_dir ../policy/autonerf_data_modular_policy_obs_Corozal/Corozal/floor_level_0/0/ --test_data_dir data/test/Corozal/floor_level_0/ --use_GT_sem 1
```

### Mesh generation
Once your NeRF model is trained, the following allows you to extract both an RGB and semantic meshes. You can specify the path where the collected data was saved (*--train_data_dir*) and the training-related logs were saved (*--save_dir*).

```
cd semantic_nerf
conda activate semantic_nerf
python -u SSR/extract_colour_mesh.py --train_data_dir ../policy/autonerf_data_modular_policy_obs_Corozal/Corozal/floor_level_0/0/ --save_dir logs_gibson_autonerf
```

## Policy evaluation
### Data collection with the policy
You need to do the same as when collecting data to recontruct house-scale scenes (on the 5 Gibson val scenes).

### Converting collected data from semantic nerf to nerfstudio format
In order to train a nerfstudio-based Semantic Nerfacto model, you first need to convert collected data from the Semantic NeRF format to nerfstudio Nerfacto format. You can specify the path where the collected data was saved (*--semantic_nerf_datapath*) and the path to the nerfstudio data (*--nerfstudio_datapath*). Finally, you can specify a maximum number of agent steps to use (*--nb_max_policy_steps*). With *--nb_max_policy_steps 1500*, we keep all collected data (as the agent navigated for 1500 steps).

**Important**: The chosen *nerfstudio_datapath* should have the following format: *data/train/policy_name_to_choose/scene_name/floor_level_0/episode_id/*

```
cd nerfstudio
conda activate nerfstudio
python convert_autonerf_data_to_nerfstudio_format.py --semantic_nerf_datapath ../policy/autonerf_data_modular_policy_obs_Corozal/Corozal/floor_level_0/0/ --nerfstudio_datapath data/train/policy_obs/Corozal/floor_level_0/0/ --gt_sem 1 --nb_max_policy_steps 1500
```

### Semantic nerfacto (*autonerfacto*) training
In order to train your *autonerfacto* model, you can do the following. You can specify the path to the nerfstudio data (*--data*). Other flags allow to train the model to predict surface normals (*--pipeline.model.predict-normals True*), to save tensorboard logs (*--vis tensorboard*) and to apply weight decay during training (*--optimizers.fields.optimizer.weight-decay 1e-9 --optimizers.proposal-networks.optimizer.weight-decay 1e-9*). The model to use here is *autonerfacto*: it is based on the original Nerfacto model, but uses an additional semantic head, is trained and evaluated on 2 independent data sets and is evaluated on additional metrics.

```
cd nerfstudio
conda activate nerfstudio
ns-train --method-name autonerfacto --data data/train/policy_obs/Corozal/floor_level_0/0/ --pipeline.model.predict-normals True --vis tensorboard --optimizers.fields.optimizer.weight-decay 1e-9 --optimizers.proposal-networks.optimizer.weight-decay 1e-9
```

### Semantic nerfacto (*autonerfacto*) evaluation
Once your *autonerfacto* model is trained, it is time to evaluate it. At the end of training (previous step), it has already been evaluated on rendering (RGB and semantics). You can run the following commands to evaluate its quality by computing a BEV map (occupancy and semantics), performing planning on such map (PointGoal and ObjGoal) and to perform pose refinement. For each script, you can specify the scene name (*--scene*) and the path to the saved nerf model logs (*--nerf_model_path*). The latter should be modified (I have simply written one log folder name example, yours will be different).

```
cd nerfstudio
conda activate nerfstudio

# BEV map generation
python -u benchmark/benchmark_bev_map.py --scene Corozal --nerf_model_path outputs/data-train-policy_obs-Corozal-floor_level_0-0/autonerfacto/2023-12-20_110005/

# Planning
python -u benchmark/benchmark_planning.py --scene Corozal --nerf_model_path outputs/data-train-policy_obs-Corozal-floor_level_0-0/autonerfacto/2023-12-20_110005/

# Pose refinement
python -u benchmark/benchmark_pose_refinement.py --scene Corozal --nerf_model_path outputs/data-train-policy_obs-Corozal-floor_level_0-0/autonerfacto/2023-12-20_110005/
```

## Policy training
In order to train a policy, you can use the following command (chosen hyperparameters to run on a 4-GPUs compute node). You can specify the name of the experiment (*--exp_name*), along with the reward you want to use (*Explored area*: *--use_expl_reward 1* / *Obstacle coverage*: *--use_obs_reward 1* / *Semantic object coverage*: *--use_sem_obj_reward 1* / *Viewpoints coverage*: *--use_viewpoint_reward 1*). The flags *--split train --eval 0* mean that you will train (not evaluate) on the set of training scenes (different from val scenes).

```
cd policy
./exec_run.sh --n 25 --num_processes_per_gpu 7 --num_processes_on_first_gpu 4 --exp_name train_modular_policy --split train --eval 0 --use_obs_reward 1
```

## Citation
```
@article{marza2023autonerf,
  title={AutoNeRF: Training Implicit Scene Representations with Autonomous Agents},
  author={Marza, Pierre and Matignon, Laetitia and Simonin, Olivier and Batra, Dhruv and Wolf, Christian and Chaplot, Devendra Singh},
  journal={arXiv preprint arXiv:2304.11241},
  year={2023}
}
```