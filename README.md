# RTMDet Object Detector
This is my (still incomplete) implementation of the RTMDet object detector published by Lyu et al. (https://arxiv.org/abs/2212.07784). 

## Model Weights
You can find model weight for RTMDetTiny, RTMDetS, RTMDetM, RTMDetL, and RTMDetX models in my google drive (https://drive.google.com/drive/folders/14UYSyP0kKRnQoerH2ouC_Mlp3oooyWQJ?usp=sharing). The weights are trained with the coco dataset and are originally from MMdetection (https://github.com/open-mmlab/mmdetection). 

## Model Details
If you’re interested in understanding how the detector works, I recommend reading the original paper by Lyu et al. (https://arxiv.org/abs/2212.07784). My implementation depends heavily from insights I gained while exploring the MMDetection repository, so I also suggest looking at their repo for further details (https://github.com/open-mmlab/mmdetection).

In my master’s thesis, I used RTMDet-Tiny and provided a detailed explanation of its implementation in subsections 3.1.1–3.1.4. You can find the thesis here: (https://aaltodoc.aalto.fi/items/ceb91022-2998-4c0c-a9c1-9ed64250aca6). Please note that my thesis may contain some errors. Below are a few corrections I’ve identified:
- Figure 7: The DWConvModule is missing the pointwise convolution.
- Figure 8: The second output should be a Conv2D, not a ConvModule.
- Algorithm 4: The cost vector should be sorted in ascending order, not descending.

## Script Examples
Inference:
- install all required dependencies
- download RTMDetL-coco.pth weights to model_weights
- mkdir images
- run python -m scripts.inference_python
- results should appear in images/results 