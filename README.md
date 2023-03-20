# DeepLearning_DeblurGanv2_SR
## Overview
The objective of this project is to improve the quality of motion blurred images using Generative Adversarial Networks (GANs). After discovering the effectiveness of the DeblurGAN-v2 in motion deblurring, we have noticed that the resulting images still can be improved a little more. Therefore, we propose to preprocess our dataset, consisting of GoPro images, with the DeblurGAN-v2 to remove motion blur. We will then utilize the SR GAN technique to further enhance the quality of the images and generate clearer and more detailed results. The goal of this project is to provide a more effective solution for motion deblurring, with the potential for application in various domains such as photography and video processing.

Co-author: Leticia Chen, Jay Yu (Jayyu231)

Original(blur)->DblurGan->SuperResolution compare with Original(sharp)
![image](https://github.com/leticia-chen/DeepLearning_DeblurGan2_SR/blob/main/Images/readme_image.png)

## Datasets
The datasets for training can be downloaded via the links below:

* GoPro

The GoPro dataset for deblurring consists of 3,214 blurred images with the size of 1,280×720 that are divided into 2,103 training images and 1,111 test images. The dataset consists of pairs of a realistic blurry image and the corresponding ground truth shapr image that are obtained by a high-speed camera.(sources:https://paperswithcode.com/dataset/gopro)

The datasets will be preprocessed by DeblurGanv2 to obtain deblurred images(get_deblured_dataset.ipynb), although they may still be slightly blurred. These images will be fed into the SR_Gan model.

## Training

### Criteria:
* Train by scaling images to 1/4 size
* Train by using original image size
* Train both sizes to generate 60 sets of model checkpoints
* Evaluate their PSNR/D Loss/G Loss using the 60 sets of model checkpoints
* Pick 4 sets of model checkpoints based on PSNR as pre-trained models

![image](https://github.com/leticia-chen/DeepLearning_DeblurGan2_SR/blob/main/Images/psnr_ssm.png)

In our project, there is the best performance while training with original image size.

## Demo

we got the better 4 model-checkpoints (DeblurGanv2_SR_demo.ipynb:

- model08 = 'models_checkpoint_by_Jay/generator_8.pth'
- model41 = 'models_checkpoint_by_Jay/generator_41.pth'
- model56 = 'models_checkpoint_by_Jay/generator_56.pth'
- model47 = 'models_checkpoint_by_Jay/generator_47.pth'
