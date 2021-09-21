import argparse
import glob
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from matplotlib import pyplot as plt


def read_image(path):
    img = plt.imread(path)
    return img

def get_mse(img_1, img_2):
    return np.mean((img_1 - img_2) ** 2)

def get_ssim(img_1, img_2):
    return ssim(img_1, img_2, multichannel= True)

def get_psnr(img_1, img_2):
    return psnr(img_1, img_2)

def get_average_mse_ssim_psnr(fig_names_1, fig_names_2):
    mse_total = 0
    ssim_total = 0
    psnr_total = 0
    for i in range(len(fig_names_1)):
        img_1 = read_image(fig_names_1[i])
        img_2 = read_image(fig_names_2[i])
        # img_1 = img_1[:,:,0]
        # img_2 = img_2[:,:,0]
        mse_total += get_mse(img_1, img_2)
        ssim_total += get_ssim(img_1, img_2)
        psnr_total += get_psnr(img_1, img_2)
    mse_avg = mse_total / (i + 1)
    ssim_avg = ssim_total / (i + 1)
    psnr_avg = psnr_total / (i + 1)
    return mse_avg, ssim_avg, psnr_avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", help= "Path to ground truth folder")
    parser.add_argument("-p", help= "Path to prediction folder")
    args = parser.parse_args()
    total_mse, total_ssim, total_psnr = 0, 0, 0

    for idx in range(7):
        gt_folder_name = 'gt_' + str(idx)
        output_folder_name = 'style_' + str(idx)

        img_gt_paths = []
        img_pred_paths = []
        for gt_path in glob.glob(os.path.join(args.g, gt_folder_name, '*.jpg')):
            file_name = gt_path[len(os.path.join(args.g, gt_folder_name))+1:-4]
            img_gt_paths.append(gt_path)
            img_pred_paths.append(os.path.join(args.p, output_folder_name, 'inferred_{}.jpg'.format(file_name)))

        img_gt_paths.sort()
        img_pred_paths.sort()

        mse, ssim, psnr = get_average_mse_ssim_psnr(img_gt_paths, img_pred_paths)
        print("Average MSE for style {}: {:.5f}".format(idx, mse))
        print("Average SSIM for style {}: {:.5f}".format(idx, ssim))
        print("Average PSNR for style {}: {:.5f}".format(idx, psnr))
        total_mse += mse
        total_ssim += ssim
        total_psnr += psnr

    print("Average MSE for total styles: {:.5f}".format(total_mse / 7))
    print("Average SSIM for total styles: {:.5f}".format(total_ssim / 7))
    print("Average PSNR for total styles: {:.5f}".format(total_psnr / 7))


if __name__ == "__main__":
    main()

# calculate mse, ssim score for each folder
# calculate total_mse, total_ssim score for all folders