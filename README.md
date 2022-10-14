# 3DTeethSeg
Created by Tae-hoon Young, Hong Gi An, Wan Kim, Jae Hwan Han <br/> 
This repository contains the Pytorch implementation of 3D Teeth Scan Segmentation and Labeling Challenge(2022).

![figure](https://user-images.githubusercontent.com/115606507/195748298-b7d08f36-d0ef-44ec-9d8c-83b662c5a636.png)

# Installation
### **Requirements** <br/>
&nbsp; &nbsp; This repository is developed on Python 3.8, Windows 10 and NVIDIA GTX 3090. Other platforms or GPU cards are not fully tested.
* Pytorch 1.12
* Scikit-learn 
* Trimesh
* Faiss
* Vtk
* Yacs
* Pygco

&nbsp; &nbsp; Please follow the commands below for more details.

&nbsp; &nbsp; 1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}. <br/>
&nbsp; &nbsp; 2. Create conda enviornment and install required python packages
```
conda create -n python=3.8
```

&nbsp; &nbsp; 3. Install packages from requirements.txt
```
pip install -r requirements.txt
```

# Usage
For test this code
```
python process.py
```



# License
The code is released under the GPL-3.0 license.

# Contact
If you have any questions about the code or need the code for reproduction, please contact via email. 
