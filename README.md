# MagNet
Repository for our work [**MagNet: Discovering Multi-agent Interaction Dynamics using Neural Network**](https://arxiv.org/abs/2001.09001)

If you find our work useful, please don't forget to cite. 
```
@article{saha2020magnet,
  title={MagNet: Discovering Multi-agent Interaction Dynamics using Neural Network},
  author={Saha, Priyabrata and Ali, Arslan and Mudassar, Burhan A and Long, Yun and Mukhopadhyay, Saibal},
  journal={arXiv preprint arXiv:2001.09001},
  year={2020}
}

```

## Installation

Compatible with Python 3.5 and Pytorch 1.1.0

1. Create a virtual environment by `python3 -m venv env`
2. Source the virtual environment by `source env/bin/activate`
3. Install requirements by `pip install -r ./requirements.txt`

## Usage

Pre-trained models can be downloaded from [here](https://www.dropbox.com/sh/12c0wpgszty10hc/AABYKfrBdneQhKDmo8ony8vWa?dl=0).

#### Training

Run `python scripts/train_magnet_<kuramoto/point-masses/swarm>.py`

#### Evaluation

Run `python scripts/test_magnet_<kuramoto/point-masses/swarm>.py --model_path <model_path> --wrapper_path <wrapper_path> --savepath <savepath>`

#### Prediction with online re-tuning

Run `python scripts/prediction_with_online_retuning.py --model_path <wrapper_path> --wrapper_path <wrapper_path> --pretrained_agents <#> --musigma <musigma_path>`

## Demo

#### Kuramoto:
![Kuramoto](https://github.com/sahapriyabrata/MagNet/blob/master/videos/Kuramoto.gif)

#### Point-masses-4:
![Point-masses-4](https://github.com/sahapriyabrata/MagNet/blob/master/videos/Point-masses-4.gif)

#### Point-masses-8 (after online re-tuning the model trained with 4 agents):
![Point-masses-8](https://github.com/sahapriyabrata/MagNet/blob/master/videos/Point-masses-8.gif)

#### Swarm:
![Swarm](https://github.com/sahapriyabrata/MagNet/blob/master/videos/Swarm.gif)

