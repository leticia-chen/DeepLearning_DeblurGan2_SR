# DeepLearning_DeblurGanv2_SR
## Overview
The objective of this project is to improve the quality of motion blurred images using Generative Adversarial Networks (GANs). After discovering the effectiveness of the DeblurGAN-v2 in motion deblurring, we have noticed that the resulting images still can be improved a little more. Therefore, we propose to preprocess our dataset, consisting of GoPro images, with the DeblurGAN-v2 to remove motion blur. We will then utilize the SR GAN technique to further enhance the quality of the images and generate clearer and more detailed results. The goal of this project is to provide a more effective solution for motion deblurring, with the potential for application in various domains such as photography and video processing.

Co-author: Leticia Chen, Jay Yu (Jayyu231)
## Datasets
The datasets for training can be downloaded via the links below:

* GoPro

Total: 1111 blured images and 1111 clear images

The datasets will be preprocessed by DeblurGanv2 to obtain deblurred images, although they may still be slightly blurred. These images will be fed into the SR_Gan model.

## Training

### Criteria:
* Train by scaling images to 1/4 size
* Train by using original image size
* Train both sizes to generate 60 sets of model checkpoints
* Evaluate their PSNR/D Loss/G Loss using the 60 sets of model checkpoints
* Pick 4 sets of model checkpoints based on PSNR as pre-trained models
