## paper links: http://arxiv.org/pdf/2007.10872v1.pdf
## how to use:
### Installation
Check out the source code git clone https://github.com/haibao637/D2HC-RMVSNet
Install cuda , cudnn  and python 2.7
Install Tensorflow and other dependencies by sudo pip install -r requirements.txt
Download
Preprocessed training/validation data: BlendedMVS, DTU and ETH3D. More training resources could be found in BlendedMVS github page
Preprocessed testing data: DTU testing set, Tanks and Temples testing set and training set
Pretrained models: pretrained on BlendedMVS, on DTU 

### dataset preparation:
  see https://github.com/YoYo000/MVSNet
### Train the network python train.py
python train.py # make sure  the dataset/log/.. path is correct

### Test the network
python test.py --dense_folder=<your test img directory>
