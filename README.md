![banner](https://user-images.githubusercontent.com/115606507/195795748-3fac825d-207d-4a16-9d79-f1a7dc160ce3.jpg)

# 3DTeethSeg
by Hong Gi Ahn*, Wan Kim*, Jae Hwan Han*, Min Sun Park, Tae-Hoon Yong and Byungsun Choi <br/> 
This repository contains the Pytorch implementation of 3D Teeth Scan Segmentation and Labeling Challenge(2022).

![image](https://user-images.githubusercontent.com/115606507/195804980-75c25e4a-4bb6-452d-ad8a-4251f0c2a355.png)
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
All interfaces of 3d teeth scan segmentation to use as a template in 3D Teeth Scan Segmentation and Labeling Challenge(2022). 
To find the template, please visit the [GitHub page](https://github.com/abenhamadou/3DTeethSeg22_challenge#input-and-output-interfaces)

### Data Preparation
We used the release of first and second datasets in [here](https://3dteethseg.grand-challenge.org/) to train or test

### Inference
After you are ready to prepare input data, you can run the main script in this command.
```
python process.py
```

# Contact
If you have any questions about the code or need for reproduction, please contact via email (thyong@osstem.com). 

# License
All scipts are released under the GPL-3.0 license. <br/>
COPYRIGHT © OSSTEM IMPLANT CO., LTD. ALL RIGHTS RESERVED.


