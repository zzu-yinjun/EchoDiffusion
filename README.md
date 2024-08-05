# EchoDiffNet: Echo-Based Depth Estimation with Wave Conditioned Diffusion Models
This repository contains snippets of test code related to [EchoDiffNet] that are used to demonstrate and validate the methods mentioned in the paper. To protect the integrity of the project and sensitive information, we have not released the full research code.We provide the test code of the model to verify the accuracy of the model.

# Dataset
Replica-VisualEchoes can be obatined from  [here](https://github.com/facebookresearch/VisualEchoes). We have used the 128x128 image resolution for our experiment.
MatterportEchoes is an extension of existing [matterport3D](https://niessner.github.io/Matterport/) dataset. In order to obtain the raw frames please forward the access request acceptance from the authors of matterport3D dataset. We will release the procedure to obtain the frames and echoes using habitat-sim and soundspaces in near future.
# Evaluation
Configure the relevant yml files before testing.We will give the pre-trained model parameters [here](https://drive.google.com/file/d/1BiNgFQNvO8n4_RZGusPzk4qksGiGQgX6/view?usp=drive_link)
```
pip install requirements.txt -r
python test.py
```
