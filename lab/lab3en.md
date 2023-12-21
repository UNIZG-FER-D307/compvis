---
layout: page
title: Third Laboratory Exercise
description: Instructions for the third laboratory exercise.
nav_exclude: true
---

# Third laboratory exercise: object detection

In this exercise, we will focus on object detection in images - one of the most important tasks in computer vision. Object detection involves the localization and classification of objects within an image. Localization is typically achieved by estimating the coordinates of a rectangle that best describes the object. Early approaches primarily dealt with binary object detection (such as the Viola-Jones algorithm), while more recent methods involve multi-class detection.

The problem with early approaches was extracting shared features that would be good enough for recognizing objects of different semantic classes. This problem has been addressed with the advent of deep models and end-to-end learning capabilities.

The development of object detection algorithms has been accompanied by the evolution of datasets. For a long time, the standard dataset was [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/), which distinguishes 20 object classes. Recently, this position has been taken over by the [MS COCO](https://cocodataset.org/#home) dataset, which identifies 80 object classes. Recognition across over 1000 different semantic classes is now being considered with the [LVIS](https://www.lvisdataset.org/) dataset.

Modern approaches to object detection are based on deep models, and several distinctions exist among them. The most popular approaches assume anchor boxes at all locations on the feature map and predict the relative shifts and scales of these boxes to best describe the object. Examples of this group include popular approaches like Faster R-CNN, SSD, YOLO, and others. Other approaches (e.g., CornerNet) detect objects using key points paired based on predicted embeddings. Transformers have introduced a new approach based on predicting a set of queries, from which the object's location and classification are directly computed.

In this exercise, we will delve into the Faster R-CNN model, specifically the version based on feature pyramids.

# Faster R-CNN
The key components of the Faster R-CNN model that we will study are:
- The backbone, which computes image features,
- The upsampling path (FPN, feature pyramid network) that calculates a feature pyramid of approximately equal semantic levels,
- The region proposal network (RPN) for proposing regions of interest,
- The region of interest (RoI) pooling layer,
- And the final layer for region classification and fine-tuning of bounding boxes.
<img src="../../assets/images/lab3/faster_arch.jpg" alt="faster" width="800"/>
<em>Image 1. The architecture of the Faster R-CNN model based on the feature pyramid.</em>

In Figure 1, a detailed inference path of the Faster R-CNN model is presented, illustrating the interaction among the mentioned components.

Considering that training such a model requires significant computational resources, in this exercise, we will focus solely on the forward pass of a pre-trained model on the COCO dataset. Your task will be to augment the provided implementation of the Faster R-CNN model. The instructions will guide you through the implementation step by step. The correctness of each step in your solution will be verified through tests comparing intermediate results of your implementation with those of the original implementation.

## 1. Environment installation
For the purposes of this exercise, we will use Python 3 along with the following Python packages:
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [Pillow](https://python-pillow.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)

After ensuring that you have all the required packages, download the exercise skeleton from [GitHub](https://github.com/UNIZG-FER-D307/compvis-lab3-faster). Then, within the repository, create a new directory named `data` and unpack the contents of this [directory](https://www.dropbox.com/sh/wbybqchx98wg8ci/AAA_2KlewTokFc2OY-fC0_wna?dl=0) into it. The downloaded file contains saved intermediate results of the forward pass of the considered model, which will be used during the tests.
## 2. Backbone
The process of building the Faster R-CNN model begins with feature extraction using the backbone. The backbone typically consists of a classification model without global pooling and a fully connected layer, which has been pre-trained on ImageNet. By initializing the backbone with ImageNet parameters, we expedite the training process for the specific task and reduce the amount of densely labeled images required. In cases of limited computational resources, during the training of the detection model, backbone parameters can be frozen. Nevertheless, it is more common for these parameters to be fine-tuned during training for object detection.

Our considered model utilizes the ResNet-50 model as a backbone, which belongs to the family of models with residual connections. The basic building block of the ResNet-50 model and its larger siblings (ResNet-101, ResNet-152) is depicted in Figure 2. For such a residual unit, we say it has a bottleneck because the initial 1x1 convolution reduces the number of channels. This significantly reduces the memory and computational footprint, as a more expensive 3x3 convolution follows. Finally, the number of channels is inflated again with the help of a second 1x1 convolution. In the figure, you can observe the skip connection that adds the input to the residual unit to the result of the processing from the convolutional layers. This enables the model to have better gradient flow to earlier layers and learn simpler residual mappings that correspond to the difference between the input and the "desired" output.

<img src="../../assets/images/lab3/resnet_bb.jpg" alt="faster" width="400"/>

<em>Figure 2. Residual Convolutional Unit with Bottleneck.</em>

In Figure 1, the backbone is depicted in shades of green and consists of four residual blocks. By a block, we denote a set of residual units operating on features with the same spatial resolution. Thus, the first residual block outputs features that are subsampled by a factor of four compared to the input image, as indicated by "/4" on the corresponding arrows in Figure 1. Similarly, the second residual block produces features subsampled by a factor of 8, the third by 16, and the fourth by 32. In the literature, these features are often referred to with the prefix "res" and a number corresponding to the exponent of the power of 2 that denotes the subsampling level. For instance, the outputs of the first residual block are labeled as "res**2**" because they are subsampled by a factor of 2^**2** = 4 compared to the input image. The outputs of the second residual block are labeled as "res**3**" since they are subsampled by a factor of 2^**3** = 8. Similarly, we will denote the features of the remaining two blocks as "res**4**" and "res**5**".

### Problems

1. Considering that the utilized backbone expects a normalized image as input, your first task is to implement a function for image normalization. Find the declaration of the function `normalize_img` in the file `utils.py`. The function takes a tensor `img` as input, which has the shape (H, W, 3) where H and W are the dimensions of the image. The function should return a normalized tensor of shape (3, H, W). Normalization involves scaling to the range [0-1], subtracting the mean value `image_mean`, and dividing by the standard deviation `image_std`. The mean value and standard deviation have a shape of (3). You can verify the correctness of your implementation by running the test script "test_backbone.py." Navigate to the root directory of the project and execute `python3 -m  tests.test_backbone` in the terminal.

2. Complete the implementation of the ResNet model in the file `resnet.py`. First, according to Figure 2, complete the `Bottleneck.forward` method, which implements the forward pass through the bottleneck residual unit. Then, in the `ResNet._forward_impl` method, save the outputs from the residual blocks in the dictionary `out_dict` with keys "res2," "res3," "res4," and "res5." Note that in the code, the first residual block is labeled as `layer1`, the second as `layer2`, and so on. Check the correctness of your implementation by running the same test script as in the previous task.

3. How many channels do the outputs from each residual block have? Which residual block's features are semantically richest, and which are spatially most precise?

## 3. Upsampling path

Generally, the task of the upsampling path is to construct a semantically rich representation that is simultaneously spatially precise. Notice that none of the outputs from the backbone satisfies both of these criteria. Features from later blocks in the backbone are more semantically rich, but their spatial resolution is lower. On the other hand, features from earlier blocks in the backbone have finer resolution and, as a result, are spatially more precise, but they are less semantically rich. Therefore, the upsampling path gradually builds the desired representation by upsampling a semantically rich representation and combining it with spatially more precise features. Different versions of this idea are present in many models for dense prediction.

Special variations of the upsampling path are present in object detection, where instead of a representation at a single level, a feature resolution pyramid is built. The reason for this is that relying solely on features from a single level would likely result in a lack of scale invariance. For instance, lower-resolution features are good for detecting large objects but may perform poorly in detecting small ones, as information about their presence could be lost due to downsampling. On the other hand, at higher resolutions, there might be challenges in detecting large objects due to the limited receptive field of convolutional models.

Therefore, even the earliest deep models for object detection, such as [SSD](https://arxiv.org/pdf/1512.02325.pdf), consider a feature pyramid. They directly regress descriptive bounding boxes from features at different resolutions in the backbone. However, the challenge with such an approach is that features from different stages of the backbone are also at different semantic levels. This issue is addressed by [FPN](https://arxiv.org/pdf/1612.03144.pdf), which employs an additional upsampling path to construct a semantically rich and uniform resolution feature pyramid.

Our version of the Faster R-CNN model also utilizes FPN. In Figure 1, the upsampling path is marked in red. Different shades of red indicate that modules for each level of the pyramid use different parameters. Each upsampling module consists of two convolutional units. One is applied to the corresponding features from the backbone (often called skip or lateral connections) to equalize the number of channels with the upsampling path. In the literature, these convolutions are often referred to as channel projections. The second convolutional unit is applied to the sum of the skip connection and the upsampled feature map from the previous level to compute the final representation at that level of the pyramid. For a more detailed illustration of the considered upsampling path, refer to Figure 3.

<img src="../../assets/images/lab3/fpn.jpg" alt="faster" width="800"/>
<em>Image 3. A more detailed depiction of the upsampling path that constructs the feature pyramid.</em>

### Problems

1. In the file `utils.py`, implement the `ConvNormActBlock` module, whose forward pass consists sequentially of a convolutional layer, optionally an activation function, and optionally a normalization layer. Note that the module inherits from the `nn.Sequential` class, which sequentially applies the specified list of layers. To add a layer to the list, you can use the `self.append` method. Set the `padding` argument that controls padding during convolution to preserve the spatial dimensions of the input.

2. In the file `fpn.py`, implement the forward pass of the upsampling path in the `FeaturePyramidNetwork.forward` method. Implement the forward pass according to Figure 3. Pay attention to the comments written in the code.

3. Test your implementation of the upsampling path by running the command `python3 -m tests.test_fpn`.

## 4. Region Proposal Network (RPN)
The task of a region proposal network is to extract rectangular regions within which an object might be located. This task is simplified to binary classification of anchor boxes into positives and negatives. Negatives are rejected because they are boxes that do not contain objects, while positives are refined through a parameterized transformation to more accurately encapsulate the target object.

It is important to note that the region proposal network does not differentiate between semantic object classes (commonly referred to as class-agnostic in the literature). Its sole task is to assess whether an object might be present within the considered anchor box or not. Positive boxes are transformed by parameters $$t_x, t_y$$ controlling the horizontal and vertical shift of the box's center, as well as parameters $$t_w, t_h$$ controlling the horizontal and vertical scaling of the box. The following equations apply to these parameters:

$$
\begin{align}
t_x &= \frac{x - x_a}{w_a} \\
t_y &= \frac{y - y_a}{h_a} \\
t_w &= \log \frac{w}{w_a} \\
t_h &= \log \frac{h}{h_a}
\end{align}
$$

where $$x_a, y_a, w_a, h_a$$ represent the coordinates of the anchor box center, its width, and height, and $$x, y, w, h$$ represent the coordinates of the target object center, its width, and height.

From Figure 1, it is evident that the region proposal network (RPN) is applied to each level of the feature pyramid. The uniform shade of blue for each pyramid level suggests parameter sharing. In addition to feature maps, the input to the RPN consists of anchor boxes, which are also generated depending on the pyramid level. Specifically, the anchor box generator places boxes at each pixel of the considered feature tensor, and their size depends on the pyramid level. Thus, at the highest resolution level of the pyramid, anchor boxes of the smallest base size are assumed because this level is intended for detecting small objects. Conversely, at the lowest resolution level of the pyramid, anchor boxes of the largest base size are located since this level is designed for detecting large objects.

An important detail is that the anchor box generator does not assume only one box per location but several, differing in aspect ratio, to detect objects of various shapes. In the model under consideration, the base sizes of anchor boxes are [32, 64, 128, 256, 512] in sequence for the pyramid levels [fpn2, fpn3, fpn4, fpn5, fpn6], and they vary in aspect ratios [1:1, 1:2, 2:1].

The RPN separately considers each assumed anchor box at a location. This means that the RPN classifier predicts as many feature maps as there are assumed anchor boxes at each location. Similarly, the RPN regressor of transformation parameters predicts four times more feature maps than the number of assumed anchor boxes at each location. The image below shows anchor boxes with an intersection over union (IoU) ratio with the player's bounding box on the image greater than 0.65.

<img src="../../assets/images/lab3/bb44_anchors.jpg" alt="bb44 anchors" width="600"/>
<br/><em>Image 4. Assumed anchor boxes that overlap with the bounding box of the basketball player in the image with an intersection over union ratio greater than 0.65.</em>

Let's also mention that the RPN does not pass all positive boxes through the forward pass. After discarding negatives and applying the predicted transformation to positives, filtering takes place. First, boxes with an area smaller than a specified value are discarded, and then those with object presence probability below a set threshold. Afterward, non-maximum suppression is applied, i.e., boxes with high overlap with another reliable box are suppressed. Finally, only 1000 boxes with the highest probability are kept. This filtering significantly speeds up the forward pass through the network. The image below shows the boxes proposed by the RPN with an intersection over union ratio with the player's bounding box on the image greater than 0.65.

<img src="../../assets/images/lab3//bb44_rpn_proposals.jpg" alt="rpn" width="600"/>
<br/><em>Image 5. Regions of interest proposed by the RPN that overlap with the bounding box of the basketball player in the image with an intersection over union ratio greater than 0.65.</em>

### Problems
1. In the file `rpn.py`, complete the initialization of the RPN classifier and regressor in the RPNHead module.
2. In the file `utils.py`, complete the implementation of the `decode_boxes` function, which applies the predicted transformation to anchor boxes. Test the implementation with the command `python3 -m tests.test_decode_boxes`.

## 5. Region Of Interest Pooling
Region of Interest Pooling (ROIPool) extracts a fixed-size representation for all proposed regions of interest suggested by the RPN. ROIPooling first determines the region in the considered feature tensor corresponding to the region of interest. Then, this region is divided into smaller sub-regions of approximately equal size, and their number is determined by a model parameter. Next, the features of each sub-region are pooled according to a specified function (e.g., maximum pooling). This process is illustrated in the gif below for a feature tensor with only one channel and an output size of 2x2.

<img src="https://deepsense.ai/wp-content/uploads/2017/02/roi_pooling-1.gif" alt="rpn" width="600"/>
<br/><em>Image 6. Illustration of the Region of Interest Pooling operation. Source: Deepsense.ai.</em>

In our considered model, the output size of ROIPool is 7x7, but it doesn't mean that each frame is represented with only 49 features. This number needs to be multiplied by the number of channels in the considered feature tensor, which corresponds to the number of FPN channels, and that is 256. Therefore, each frame is represented with 7x7x256 = 12544 features. It is worth mentioning that newer versions of two-stage models typically use a more advanced algorithm called ROIAlign, which utilizes interpolation instead of quantization to determine boundary values.

## 6. Semantic Classification and Fine-Tuning of Boxes
The final module of our considered model takes the compressed representation and coordinates of all proposed regions of interest (ROIs) by the Region Proposal Network (RPN) as input. The task of this module is semantic classification of incoming bounding boxes and predicting parameters for another transformation to finely adjust the boxes to the desired objects. Unlike the RPN, this module classifies the boxes into one of the semantic classes or background. The existence of a background class allows the model to reject some boxes as negatives in this step as well.

An interesting detail is that, for each box, separate transformation parameters are predicted for each of the semantic classes. We can imagine that each box is multiplied by the number of classes, and transformation parameters are predicted for each of them. This enables the detection of overlapping objects of different classes.

This module first calculates a shared representation for the classification and regression heads using two fully connected layers. Then, two additional fully connected layers are applied to this representation: one for classification and the other for the regression of semantically dependent transformation parameters.

### Problems
1. In the file `faster.py`, complete the implementation of the `forward` function in the `TwoMLPHead` module.
2. In the file `faster.py`, complete the initialization of the classification and regression heads in the `FastRCNNPredictor` module.
3. In the file `run_faster.py`, implement the visualization of detection results for all boxes with a confidence greater than 0.95.
4. Run the program with the command `python3 run_faster.py` and check if the detection is successful.

The expected result of running the `run_faster.py` program is shown in the image below.

<img src="../../assets/images/lab3/bb44_preds.jpg" alt="bb44 preds" width="600"/>
<br/><em>Image 7. Result of executing the Faster R-CNN model trained on the COCO dataset.</em>
