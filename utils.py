from PIL import Image 
import matplotlib.pyplot as plt
import os
import numpy as np 
import math

def crop_center(img,center_size):
    dim0, dim1 = img.shape
    center0 , center1 = center_size
    start0 = dim0//2-(center0//2)
    start1 = dim1//2-(center1//2)  
    return img[start0 : start0 + center0, start1 : start1 + center1]

def obtain_normalize_parameter(normalize_mode, std_equal_mean, data_dir, center_size, num_of_channel):
    if normalize_mode == 'from_training_data' or normalize_mode == 'from_training_data_std_equal_mean':
        img_path_list = []
        for root, _, files in os.walk(data_dir, topdown=False):
            for img in files:
                img_path = os.path.join(root, img)
                img_path_list.append(img_path)
        if num_of_channel == 1:
            total_sum = 0
            total_square_sum = 0
            total_pixel = 0
            for img_path in img_path_list:
                img_np = plt.imread(img_path)
                img_center = crop_center(img_np, center_size)
                img_center = img_center/255
                total_sum += np.sum(img_center)
                total_square_sum += np.sum(img_center ** 2)
                total_pixel += img_center.size
            total_pixel = total_pixel - 1
            E_x = total_sum/total_pixel
            E_x_square = total_square_sum/total_pixel
            mean = [E_x]
            var = E_x_square - E_x ** 2
            std = [math.sqrt(var)]
            if std_equal_mean:
                print('hit!!!!!!!!!!!')
                std = mean

        elif num_of_channel == 3:
            mean = []
            std = []
            for channel in range(num_of_channel):
                total_sum = 0
                total_square_sum = 0
                total_pixel = 0
                for img_path in img_path_list:
                    img_np = plt.imread(img_path)
                    if img_np.ndim == 2:
                        img_np = np.array([img_np, img_np, img_np]).transpose(1,2,0)
                    img_channel = img_np[:,:,channel]
                    img_center = crop_center(img_channel, center_size)
                    img_center = img_center/255
                    total_sum += np.sum(img_center)
                    total_square_sum += np.sum(img_center ** 2)
                    total_pixel += img_center.size
                total_pixel = total_pixel - 1
                E_sum = total_sum/total_pixel
                E_square_sum = total_square_sum/total_pixel
                channel_mean = E_sum
                channel_var = E_square_sum - E_sum ** 2
                channel_std = math.sqrt(channel_var)
                mean.append(channel_mean)
                std.append(channel_std)
            if std_equal_mean:
                std = mean
        else:
            print('Error! Not 3 channel or 1 channel!')
        
        return mean, std
    else:

        if num_of_channel == 1:
            mean = [0.5]
            std = [0.5]
        elif num_of_channel == 3:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            print('Error! Not 3 channel or 1 channel!')
        return mean, std

# Define printing to console and file
def print_both(f, text):
    print(text)
    f.write(text + '\n')