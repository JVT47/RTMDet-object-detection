# RTMDet Object Detector
This is my (still incomplete) implementation of the RTMDet object detector published by Lyu et al. (https://arxiv.org/abs/2212.07784). 

## Model Weights
You can find model weight for RTMDetTiny, RTMDetS, RTMDetM, RTMDetL, and RTMDetX models in my google drive (https://drive.google.com/drive/folders/14UYSyP0kKRnQoerH2ouC_Mlp3oooyWQJ?usp=sharing). The weights are trained with the coco dataset and are originally from MMdetection (https://github.com/open-mmlab/mmdetection). 

## Model Details
If you’re interested in understanding how the detector works, I recommend reading the original paper by Lyu et al. (https://arxiv.org/abs/2212.07784). My implementation depends heavily from insights I gained while exploring the MMDetection repository, so I also suggest looking at their repo for further details (https://github.com/open-mmlab/mmdetection).

In my master’s thesis, I used RTMDet-Tiny and provided a detailed explanation of its implementation in subsections 3.1.1–3.1.4. You can find the thesis here: (https://aaltodoc.aalto.fi/items/ceb91022-2998-4c0c-a9c1-9ed64250aca6). Please note that my thesis may contain some errors. Below are a few corrections I’ve identified:
- Figure 7: The DWConvModule is missing the pointwise convolution.
- Figure 7: The SPPFBottleneck should apply the pooling layers in parallel instead of sequentially.
- Figure 8: The second output should be a Conv2D, not a ConvModule.
- Algorithm 4: The cost vector should be sorted in ascending order, not descending.

## Setup
This project uses uv and Python 3.12 so you must have them installed. Additionally, you need Rust if you want to use the rust based inference package located in src/rtmdet-object-detection. Make sure that the src dir is in your PYTHONPATH. If you are using vscode, it should be included automatically.

```Terminal
make setup
```
to download the project dependencies.

```Terminal
make build
```
to build the rust based inference package located in src/rtmdet-object-detection.

## ONNX conversion
The rust based rtmdet-object-detection package uses onnx models to perform object detection. You can convert your RTMDet models from pytorch to onnx with the script convert_to_onnx.py. See scripts/convert_to_onnx.py for details and example usage.

## Script Examples
Inference Python:
- install all required dependencies
- download RTMDetM-coco.pth weights to model_weights
- mkdir images and copy your images to it
- uv run -m scripts.inference_python
- results should appear in images/results 

Inference Rust:
- convert the pytorch weights to onnx format.
- build the rust based package with make build
- uv run -m scripts.inference_rust

## Training
Note: I do not have access to a gpu so cannot verify that the training actually works but at least it starts on my computer.

Run
```Terminal
uv run -m scripts.train_model --config-file path/to/config/file
```
### Config file 
The config file for training defines the fields that are passed to the corresponding factories in the code. This way one can configure multiple training runs without having to change the source code. See configs/RTMDer_training_config.template.yaml for an example how the config file should look like. See the factory methods in the source code for details on their arguments.

In model_cfg, if you set strict to false, you can use pretrained RTMDet weights with a different number of output classes as your models initial weights since make model takes care of resizing mismatching shapes automatically. See src/rtmdet_object_detection_dev/model/model.py for details.

### Dataset
As an example, I have used the Oxford-IIT Pet Dataset (https://www.robots.ox.ac.uk/~vgg/data/pets/). If you want to use your own dataset just add a dataset implementation to src/rtmdet_object_detection_dev/datasets and add it to the dataset_factory.py. The rest of this subsection covers the Oxford-IIT Pet Dataset.

I have the Pet Dataset split in data/images and data/annotations directories. The annotations for the original dataset are in xml format but for easier use I parse the annotations to train.yaml, valid.yaml, and test.yaml files with the script oxford_pet_dataset_xmls_to_yaml.py. See scripts/oxford_pet_dataset_xmls_to_yaml.py for details.

## Test
Test python code with
```Terminal
make test-python
```

Test rust code with
```Terminal
make test-rust
```

Test all code with
```Terminal
make test
```