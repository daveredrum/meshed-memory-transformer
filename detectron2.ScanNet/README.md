# Detectron2.ScanNet
Extracting object features from ScanNet frames by [Detectron2](https://github.com/facebookresearch/detectron2)

## Install
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
# (execute in the project root folder)
```

See [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for more information

## Usage

### 1. Configure the model for extraction

The `main.sh` script uses `faster_rcnn_R_101_DC5_3x` by default. If you prefer other model from [Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md), please find the link to the pre-trained weights [here](https://github.com/facebookresearch/detectron2/blob/master/detectron2/model_zoo/model_zoo.py) and change the `--config-file` and `--opts MODEL.WEIGHTS` accordingly.

> Note that some models have slightly different ROI heads, don't forget to check `engine/customized.py`

### 2. Configure the input and output directory

Set the input and output directory in `--input_dir` and `--output_dir` in `main.sh` repectively.

### 3. Extract features

Run the following command in your terminal to extract object features and store them in HDF5 database (you need ~2GB for the stored features):

```
./script/main.sh
```