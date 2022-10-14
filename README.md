# 3DTeethSeg
Created by Hong Gi An, Wan Kim, Jae Hwan Han, Tae-Hoon Yong* <br/> 
This repository contains the Pytorch implementation of 3D Teeth Scan Segmentation and Labeling Challenge(2022).

![figure](https://user-images.githubusercontent.com/115606507/195748298-b7d08f36-d0ef-44ec-9d8c-83b662c5a636.png)
# Environment
This repository is developed on Python 3.8, Windows 10 and NVIDIA GTX 3090. Other platforms or GPU cards are not fully tested. <br/> <br/>

# Installation
Please follow the commands below for more details.
<br/>
1. Clone this repository.
```
git clone https://github.com/Osstem-AI/3DTeethSeg
```
2. Create conda enviornment and install required python packages.
```
conda create -n 3DTeethSeg python=3.8
```

3. Install packages from requirements.txt.
```
pip install -r requirements.txt
```
# Usage
All interfaces of 3d teeth scan segmentation to use as a template in 3D Teeth Scan Segmentation and Labeling Challenge(2022). <br\>
If you can find the interfaces, visit the [here](https://github.com/abenhamadou/3DTeethSeg22_challenge#input-and-output-interfaces)

### Data Preparation
If you can get a train or test data, visit the [homepage](https://3dteethseg.grand-challenge.org/) and verfiy the account and participate the challenge. 

### Inference
After you are ready to prepare input data, you can run the main script in this command.
```
python process.py
```

# Contact
If you have any questions about the code or need for reproduction, please contact via email(thyong@osstem.com). 

# License
All scipts are released under the GPL-3.0 license. <br/>
COPYRIGHT © OSSTEM IMPLANT CO., LTD. ALL RIGHTS RESERVED.


