

## Requirements
The code has been tested on Ubantu20.04 , Python 3.7.9, PyTorch 1.11.0, CUDA 11.3

```
pip install -r requirements.txt
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Datasets
Our prepared shapenet dataset is available [here](https://drive.google.com/file/d/1I0phYe60FHLj3rQcJGl0BN5RLPVCqSh5/view?usp=sharing) and pix3d dataset is available [here](https://drive.google.com/file/d/1O1XTTTX1LKj0eO1kT6HS_A0YAKbv8AOw/view?usp=drive_link)



# Set datasets
You should modify the dataset path in the config.py file.
```
DATASETS.PROTOTYPE_PATH = 'path/to/shapenet/%s/%s/rendering/%02d.png'
DATASETS.SHAPENET.POINT_PATH=  'path/to/shapenet_point/%s/%s'+'.npy''
```

# train stage1
```
python runner.py
```
# train stage2
```
python runner.py --finetune --weights=xxx.pth
```

# test 
```
python runner.py --test --weights=xxx.pth
```

