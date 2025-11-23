import numpy as np
import cv2
from scipy.ndimage import uniform_filter
from PIL import Image, ImageEnhance
import os
import matplotlib.pyplot as plt
import pywt

def enhance_brightness(img, enhancement_factor):
    img_float = img.astype(np.float32)
    enhanced = img_float * enhancement_factor
    enhanced = np.clip(enhanced, 0, 255)
    return enhanced.astype(np.uint8)

def remove_stripe_noise_spatial(img):
    mean_kernel = np.ones((9, 1), dtype=np.float32) / 9
    result = cv2.filter2D(img.astype(np.float32), -1, mean_kernel)
    return np.uint8(np.clip(result, 0, 255))

def wavelet_denoise(img, threshold_scale=0.1, wavelet='db4', level=3):
    img_float = img.astype(np.float32)  # 将uint8图像转换为float32
    coeffs = pywt.wavedec2(img_float, wavelet, level=level)#将图像分解为多个尺度的频率成分
    threshold = threshold_scale * np.median(np.abs(coeffs[-level])) #基于细节系数的中位数计算自适应阈值
    # coeffs[-level]：取最细尺度的细节系数（噪声主要存在于此）
    # np.median(np.abs())：计算中位数绝对偏差，对噪声水平进行估计
    # threshold_scale：缩放因子，控制去噪强度
    new_coeffs = [coeffs[0]] #初始化新系数列表，保留近似系
    for i in range(1, len(coeffs)): #对细节系数进行阈值处理
        detail_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[i]]
        # 对每个细节系数应用软阈值函数,软阈值公式：sign(x) * max(|x| - threshold, 0)
        # 小于阈值的系数置为0，大于阈值的系数收缩
    denoised = pywt.waverec2(new_coeffs, wavelet)#用处理后的系数重构去噪后的图像
    if denoised.shape != img.shape:# 确保输出尺寸与输入一致
        denoised = denoised[:img.shape[0], :img.shape[1]]
    return np.uint8(np.clip(denoised, 0, 255))

def combined_denoising_pipeline(img, post_brightness_factor=1.3):
    destriped = remove_stripe_noise_spatial(img)
    denoised = wavelet_denoise(destriped, threshold_scale=0.1, wavelet='db4', level=3)
    final_image = enhance_brightness(denoised, post_brightness_factor)
    return final_image

if __name__ == "__main__":
    input_image_path = "image_band_1_VV.tif"
    original_image_pil = Image.open(input_image_path).convert('L')
    original_img = np.array(original_image_pil)
    brightness_factor = 1.8
    brightened_img = enhance_brightness(original_img, brightness_factor)
    denoised_image = combined_denoising_pipeline(brightened_img, post_brightness_factor=1)
    #SAVE
    enhanced_path = "enhanced_image.tif"
    denoised_path = "denoised_image.tif"
    cv2.imwrite(enhanced_path, brightened_img)
    cv2.imwrite(denoised_path, denoised_image)
    print(f"噪声减少量: {(np.std(original_img) - np.std(denoised_image)) / np.std(original_img) * 100:.1f}%")