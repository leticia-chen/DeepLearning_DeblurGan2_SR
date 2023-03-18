import pyiqa
import torch
import os
import glob

from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd
import numpy


# List available models
# print(pyiqa.list_models())   # ['ahiq', 'brisque', 'ckdn', 'clipiqa', 'clipiqa+', 'cnniqa', 'cw_ssim',
# 'dbcnn', 'dists', 'fid', 'fsim', 'gmsd', 'ilniqe', 'lpips', 'lpips-vgg', 'mad', 'maniqa', 'ms_ssim',
# 'musiq', 'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 'nima', 'nima-vgg16-ava', 'niqe',
# 'nlpd', 'nrqm', 'paq2piq', 'pi', 'pieapp', 'psnr', 'psnry', 'ssim', 'ssimc', 'vif', 'vsi']

# def get_files():
#     list = []
#     # for filepath,dirnames,filenames in os.walk(r'.\dataset1\blur'):
#     for filepath, dirnames, filenames in os.walk(r'.\dataset1'):
#         for filename in filenames:
#             #  -----original------
#             list.append(os.path.join(filepath, filename))


def get_filename():
    os.chdir('C:\\Users\\user\\SC201\\GAN\\DeblurGanv2-SR\\DeblurGANv2\\images\\original_blur_image_lst')  # 改變路徑到新路徑
    files = glob.glob('*.png')  # 找當下路徑所有.png 檔案，所以檔名中都有時間戳記的樣子
    original_blur_images = sorted(files, key=lambda t: os.stat(t).st_mtime)  # 依照修改時間的小到大的排列

    os.chdir('C:\\Users\\user\\SC201\\GAN\\DeblurGanv2-SR\\DeblurGANv2\\images\\true_label_3rd')
    files = glob.glob('*.png')
    true_label = sorted(files, key=lambda t: os.stat(t).st_mtime)

    os.chdir('C:\\Users\\user\\SC201\\GAN\\DeblurGanv2-SR\\DeblurGANv2\\images\\gan_image_2nd')
    files = glob.glob('*.png')
    gan_files = sorted(files, key=lambda t: os.stat(t).st_mtime)

    return original_blur_images, gan_files, true_label


first, second, third = get_filename()
print(first)
print(second)
print(third)
# print(len(first))
# print(len(second))
# print(len(third))
#
# # create metric function, for example lpips
lpips_metric = pyiqa.create_metric('lpips').cuda()
niqe_metric = pyiqa.create_metric('niqe').cuda()
psnr_metric = pyiqa.create_metric('psnr').cuda()
ssim_metric = pyiqa.create_metric('ssim').cuda()
dists_metric = pyiqa.create_metric('dists').cuda()
fid_metric = pyiqa.create_metric('fid').cuda()
ms_ssim_metric = pyiqa.create_metric('ms_ssim').cuda()
mad_metric = pyiqa.create_metric('mad').cuda()
cw_ssim_metric = pyiqa.create_metric('cw_ssim').cuda()

fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(1, 2, 1)
plt.title('Example distorted image', fontsize=16)
ax1.axis('off')
ax2 = fig.add_subplot(1, 2, 2)
plt.title('Example reference image', fontsize=16)
ax2.axis('off')

path = "C:\\Users\\user\\SC201\\GAN\\DeblurGanv2-SR\\DeblurGANv2\\images\\"
# create a empty dataframe
df = pd.DataFrame()
for i in range(len(first)):
    distorted_image = second[i]   # gan images
    referenced_image = third[i]
    # gan_image = second[i]
    print(distorted_image)
    # if i == 3:
    #     break
    # distorted_image = 'crop_lst_epoch_60127300.png'
    # referenced_image = 'crop_3rd_epoch_60_127300.png'

    # Join various path components
    # print(path)
    # print(os.path.join(path, "1", distorted_image))
    # print(os.path.join(path, "3", referenced_image))

    fst = os.path.join(path, "original_blur_image_lst", distorted_image)
    nd = os.path.join(path, "gan_image_2nd", distorted_image)
    rd = os.path.join(path, "true_label_3rd", referenced_image)

    # os.chdir('/content/drive/MyDrive/Colab Notebooks/Test/1')
    ax1.imshow(Image.open(nd))
    ax2.imshow(Image.open(rd))

    print('--------------------------DeblurGANv2 effect---------------------------')
    # LPIPS takes two corresponding images: (distorted image, reference image)
    lpips_score = lpips_metric(nd, rd)
    # NIQE takes the distortion image
    niqe_score = niqe_metric(nd)
    # psnr_metric
    psnr_score = psnr_metric(nd, rd)
    # ssim_metric
    ssim_score = ssim_metric(nd, rd)
    # dists_metric
    dists_score = dists_metric(nd, rd)
    # ms_ssim_metric
    ms_ssim_metric_score = ms_ssim_metric(nd, rd)
    # mad_metric
    mad_score = mad_metric(nd, rd)

    # covert from GPU to CPU and then to numpy format for pandas usage
    lpips_score = lpips_score.cpu().numpy()
    psnr_score = psnr_score.cpu().numpy()
    ssim_score = ssim_score.cpu().numpy()
    ms_ssim_metric_score = ms_ssim_metric_score.cpu().numpy()
    mad_score = mad_score.cpu().numpy()
    dists_score = dists_score.cpu().numpy()
    # print(lpips_score)
    # print(type(lpips_score))
    # df = pd.DataFrame({'lpips_score':lpips_score}, index = [0])
    data = pd.DataFrame({'psnr_score': psnr_score, 'lpips_score': lpips_score,
                         'ssim_score': ssim_score, 'ms_ssim_metric_score': ms_ssim_metric_score
                            , 'mad_score': mad_score, 'dists_score': dists_score}, index=[i])
    df = df.append(data)  # But the append doesn't happen in-place, so you'll have to store the output if you want it:
print(df)

os.makedirs('../output_file', exist_ok=True)
compression_opts = dict(method='zip', archive_name='out.csv')
df.to_csv('../output_file/out.zip', compression=compression_opts)

# # Show example images
# fig = plt.figure(figsize=(15, 8))
# ax1 = fig.add_subplot(1, 2, 1)
# plt.title('Example distorted image', fontsize=16)
# ax1.axis('off')
# ax2 = fig.add_subplot(1, 2, 2)
# plt.title('Example reference image', fontsize=16)
# ax2.axis('off')
#
# distorted_image = "C:\\Users\\user\\SC201\\GAN\\DeblurGanv2-SR\\IQA-PyTorch\\Comparsion_psnr_ssmi_dists\\crop_lst_epoch_60127300.png"
# referenced_image = 'C:\\Users\\user\\SC201\\GAN\\DeblurGanv2-SR\\IQA-PyTorch\\Comparsion_psnr_ssmi_dists\\crop_3rd_epoch_60_127300.png'
# ax1.imshow(Image.open(distorted_image))
# ax2.imshow(Image.open(referenced_image))
