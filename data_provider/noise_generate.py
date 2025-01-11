import numpy as np
from scipy.stats import poisson
from skimage.util import random_noise
import scipy.io as sio
import matplotlib.pyplot as plt
import glymur
import scipy.ndimage as ndimage
#glymur.config.set_library_search_paths(['D:/Program Files/openjpeg-v2.5.2-windows-x64/bin'])


def add_noise(img_clean, labels, which_case, dataset):
    row, column, bands = img_clean.shape
    N = row * column  # img_clean has dimensions [row, column, band]
    img_noisy = None
    np.random.seed(0)
    if which_case == 'case1':
        # --------------------- Case 1 --------------------------------------
        # Zero-mean Gaussian noise is added to all the bands of the input image
        noise_type = 'Zero-mean Gaussian noise'
        sigma = 0.3

        # Step 1: Normalize each band to the [0, 1] range
        img_clean_normalized = np.zeros_like(img_clean, dtype=np.float32)  # initialize for normalized image
        for i in range(bands):
            band = img_clean[:, :, i].astype(np.float32)
            min_val, max_val = band.min(), band.max()
            img_clean_normalized[:, :, i] = (band - min_val) / (max_val - min_val)  # normalize to [0, 1]

        # Step 2: Generate noisy image (Zero-mean Gaussian noise)
        noise = sigma * np.random.randn(*img_clean.shape)
        img_noisy = img_clean_normalized + noise  # Add noise to the normalized image

        # Clip noisy image to ensure values stay within [0, 1] range
        img_noisy = np.clip(img_noisy, 0, 1)

        # Step 3 (Optional): Denormalize the noisy image back to the original scale
        img_noisy_denorm = np.zeros_like(img_clean)
        for i in range(bands):
            min_val, max_val = img_clean[:, :, i].min(), img_clean[:, :, i].max()
            img_noisy_denorm[:, :, i] = (img_noisy[:, :, i] * (max_val - min_val) + min_val).astype(img_clean.dtype)

        img_noisy = img_noisy_denorm
        res = {
            "input": img_noisy_denorm,  # 模拟的图像数据
            "y": labels,  # 标签数据
            "sigma": sigma
        }
        save_path = ('/home/huiwei/codes/TTA-master/TTA_code/data/Zero_mean_Gaussian/%s_%s.mat'
                     % (dataset, str(sigma)))
        sio.savemat(save_path, res)

    elif which_case == 'case2':
        # --------------------- Case 2 ---------------------
        # Different variance zero-mean Gaussian noise is added to each band
        # The std values are randomly selected from 0 to 0.1.
        noise_type = 'additive'

        # Step 1: Normalize each band to the [0, 1] range
        img_clean_normalized = np.zeros_like(img_clean, dtype=np.float32)  # initialize for normalized image
        for i in range(bands):
            band = img_clean[:, :, i].astype(np.float32)
            min_val, max_val = band.min(), band.max()
            img_clean_normalized[:, :, i] = (band - min_val) / (max_val - min_val)  # normalize to [0, 1]

        # Step 2: Generate noisy image (Zero-mean Gaussian noise)
        noise_level = 0.3
        sigma = np.random.rand(bands) * noise_level
        noise = np.random.randn(*img_clean.shape)

        for cb in range(bands):
            noise[:, :, cb] = sigma[cb] * noise[:, :, cb]

        img_noisy = img_clean_normalized + noise

        # Clip noisy image to ensure values stay within [0, 1] range
        img_noisy = np.clip(img_noisy, 0, 1)

        # Step 3 (Optional): Denormalize the noisy image back to the original scale
        img_noisy_denorm = np.zeros_like(img_clean)
        for i in range(bands):
            min_val, max_val = img_clean[:, :, i].min(), img_clean[:, :, i].max()
            img_noisy_denorm[:, :, i] = (img_noisy[:, :, i] * (max_val - min_val) + min_val).astype(img_clean.dtype)

        img_noisy = img_noisy_denorm
        res = {
            "input": img_noisy_denorm,  # 模拟的图像数据
            "y": labels,  # 标签数据
            "noise_level": noise_level
        }
        save_path = ('/home/huiwei/codes/TTA-master/TTA_code/data/diff_Gaussian/%s_%s.mat'
                     % (dataset, str(noise_level)))
        sio.savemat(save_path, res)



    elif which_case == 'case3':
        #  ---------------------  Case 3: Poisson Noise ---------------------
        # 设置参数
        noise_type = 'poisson'
        snr_db = 13

        # 计算信噪比对应的线性值
        snr_set = np.exp(snr_db * np.log(10) / 10)

        # 图像的维度
        row, column, band = img_clean.shape
        N = row * column

        # 初始化结果矩阵
        img_wN_scale = np.zeros((band, N))
        img_wN_noisy = np.zeros((band, N))

        # 遍历每个波段
        for i in range(band):
            # 获取当前波段的数据并展平为一维
            img_wNtmp = img_clean[:, :, i].reshape(-1)
            img_wNtmp = np.maximum(img_wNtmp, 0)  # 确保非负

            # 计算缩放因子
            factor = snr_set / (np.sum(img_wNtmp ** 2) / np.sum(img_wNtmp))
            img_wN_scale[i, :] = factor * img_wNtmp

            # 生成泊松噪声
            img_wN_noisy[i, :] = np.random.poisson(factor * img_wNtmp)

        # 重塑为原始图像形状
        img_noisy = img_wN_noisy.T.reshape((row, column, band))
        img_clean_scaled = img_wN_scale.T.reshape((row, column, band))

        # 计算差分图像
        diff_image = img_noisy - img_clean

        mse = np.mean((img_noisy - img_clean) ** 2)
        print("MSE between noisy and clean image:", mse)

        # 可视化差分图像
        plt.imshow(np.mean(diff_image, axis=2), cmap='gray')
        plt.colorbar()
        plt.title("Difference Image")
        plt.show()

        res = {
            "input": img_noisy,  # 模拟的图像数据
            "y": labels,  # 标签数据
            "sn_ratio": snr_db
        }
        save_path = ('/home/huiwei/codes/TTA-master/TTA_code/data/poisson/%s_%s.mat'
                     % (dataset, str(snr_db)))
        sio.savemat(save_path, res)

    elif which_case == 'case4':
        # --------------------- Case 4: Salt & Pepper Noise ---------------------
        noise_type = 'salt & pepper'
        amount = 0.15

        bands = img_clean.shape[2]  # number of bands

        # Step 1: Normalize each band to the [0, 1] range
        img_clean_normalized = np.zeros_like(img_clean, dtype=np.float32)  # initialize for normalized image
        for i in range(bands):
            band = img_clean[:, :, i].astype(np.float32)
            min_val, max_val = band.min(), band.max()
            img_clean_normalized[:, :, i] = (band - min_val) / (max_val - min_val)  # normalize to [0, 1]

        # Step 2: Add salt-and-pepper noise
        img_noisy = np.zeros_like(img_clean_normalized)
        for i in range(bands):
            img_noisy[:, :, i] = random_noise(img_clean_normalized[:, :, i], mode='s&p', amount=amount)

        # Step 3 (Optional): If you need the result back in the original scale, denormalize
        img_noisy_denorm = np.zeros_like(img_clean)
        for i in range(bands):
            min_val, max_val = img_clean[:, :, i].min(), img_clean[:, :, i].max()
            img_noisy_denorm[:, :, i] = (img_noisy[:, :, i] * (max_val - min_val) + min_val).astype(img_clean.dtype)

        res = {
            "input": img_noisy_denorm,  # 模拟的图像数据
            "y": labels,  # 标签数据
            "amount": amount
        }
        save_path = ('/home/huiwei/codes/TTA-master/TTA_code/data/salt/%s_%s.mat'
                     % (dataset, str(amount)))
        sio.savemat(save_path, res)

    elif which_case == 'case5':
        # --------------------- Case 5: Stripes Noise ---------------------
        noise_type = 'stripes'
        img_noisy = np.copy(img_clean)

        # Define stripe noise parameters
        stripenum = np.random.randint(40, 46)  # Randomly generate 6-15 stripes
        for cb in range(bands):
            locolumn = np.random.choice(column, size=stripenum, replace=False)  # Random stripe positions
            img_noisy[:, locolumn, cb] = 0.2 * np.random.rand(1) + 0.6  # Add stripes noise
        res = {
            "input": img_noisy,  # 模拟的图像数据
            "y": labels,  # 标签数据
        }
        save_path = ('/home/huiwei/codes/TTA-master/TTA_code/data/stripe/%s.mat'
                     % dataset)
        sio.savemat(save_path, res)

    elif which_case == 'case6':
        # --------------------- Case 6: Deadlines Noise ---------------------
        noise_type = 'deadlines'
        img_noisy = np.copy(img_clean)

        # Define deadline noise parameters
        deadlinenum = np.random.randint(40, 46)  # Randomly generate 6-10 deadlines
        for cb in range(bands):
            locolumn = np.random.choice(column - 3, size=deadlinenum, replace=False)  # Deadline positions
            an = np.random.randint(1, 4, size=deadlinenum)  # Randomly select width (1-3)
            for idx in range(deadlinenum):
                if an[idx] == 1:
                    img_noisy[:, locolumn[idx], cb] = 0
                elif an[idx] == 2:
                    img_noisy[:, locolumn[idx]:locolumn[idx] + 2, cb] = 0
                else:
                    img_noisy[:, locolumn[idx]:locolumn[idx] + 3, cb] = 0

        res = {
            "input": img_noisy,  # 模拟的图像数据
            "y": labels,  # 标签数据
        }
        save_path = ('/home/huiwei/codes/TTA-master/TTA_code/data/deadline/%s.mat'
                     % dataset)
        sio.savemat(save_path, res)

    elif which_case == 'case7':
        compression_ratios = [300]  # 50
        output_file = 'output_lossy.jp2'
        jp2k = glymur.Jp2k(output_file, img_clean, cratios=compression_ratios)
        img_noisy = jp2k[:]
        res = {
            "input": img_noisy,  # 模拟的图像数据
            "y": labels,  # 标签数据
        }
        save_path = ('/home/huiwei/codes/TTA-master/TTA_code/data/jpg/%s_%s.mat'
                     % (dataset, str(compression_ratios[0])))
        sio.savemat(save_path, res)

    elif which_case == 'case8':
        # 定义卷积核（这里使用简单的均值卷积核，也可以使用高斯核等）
        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size)) / 100  # 3x3的均值卷积核

        # 创建一个空的噪声图像
        img_noisy = np.copy(img_clean)

        # 对每个波段应用噪声并进行卷积
        for cb in range(bands):
            # # 在每个波段上生成高斯噪声
            # noise = np.random.randn(height, width)  # 生成高斯噪声
            #
            # # 将噪声应用到当前波段
            # img_noisy[:, :, cb] = img_clean[:, :, cb] + noise  # 添加噪声

            # 使用卷积核对噪声图像进行卷积
            img_noisy[:, :, cb] = ndimage.convolve(img_noisy[:, :, cb], kernel, mode='constant', cval=0)
        res = {
            "input": img_noisy,  # 模拟的图像数据
            "y": labels,  # 标签数据
            "kernel_size": kernel_size
        }
        save_path = ('/home/huiwei/codes/TTA-master/TTA_code/data/CNN_kernel/%s_%s.mat'
                     % (dataset, str(kernel_size)))
        sio.savemat(save_path, res)
    return img_noisy, img_clean


def display_hyperspectral_images(img_clean, img_noisy, bands=(30, 20, 10)):
    # Normalize the images for display
    def normalize(image):
        image_min = image.min()
        image_max = image.max()
        return (image - image_min) / (image_max - image_min)

    # Select the bands for RGB representation (R, G, B)
    img_clean_rgb = normalize(img_clean[:, :, bands])
    img_noisy_rgb = normalize(img_noisy[:, :, bands])

    # Plot the clean and noisy images
    plt.figure(figsize=(12, 6))

    # Clean image
    plt.subplot(1, 2, 1)
    plt.imshow(img_clean_rgb)
    plt.title("Clean Hyperspectral Image (False Color)")

    # Noisy image
    plt.subplot(1, 2, 2)
    plt.imshow(img_noisy_rgb)
    plt.title("Noisy Hyperspectral Image (False Color)")

    plt.show()


which_case = 'case4'
dataset = 'PaviaU'
img_clean = sio.loadmat("/home/huiwei/codes/TTA-master/TTA_code/data/%s/%s.mat" % (dataset, dataset))
img_clean = img_clean['paviaU']
labels = sio.loadmat("/home/huiwei/codes/TTA-master/TTA_code/data/%s_gt.mat" % dataset)
labels = labels['paviaU_gt']
img_noisy, _ = add_noise(img_clean, labels, which_case, dataset)
display_hyperspectral_images(img_clean, img_noisy, bands=(30, 20, 10))



