import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import copy
import os
from PIL import Image 
import utils
import nets

operator_x = [[0.0, 0.0, -0.191, 0.0, 0.191, 0.0, 0.0],
              [0.0, -1.085, -1.0, 0.0, 1.0, 1.085, 0.0],
              [-0.585, -2.0, -1.0, 0.0, 1.0, 2.0, 0.585],
              [-1.083, -2.0, -1.0, 0.0, 1.0, 2.0, 1.083],
              [-0.585, -2.0, -1.0, 0.0, 1.0, 2.0, 0.585],
              [0.0, -1.085, -1.0, 0.0, 1.0, 1.085, 0.0],
              [0.0, 0.0, -0.191, 0.0, 0.191, 0.0, 0.0]]
grad_x = np.array(operator_x)
grad_y = grad_x.T
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(params):
    print('hit test!')
    net_architecture  = params['net_architecture'] 
    input_image_size = params['input_image_size']
    aoi_size = params['aoi_size'] 
    test_path = params['test_path']
    coarse_kernal = params['coarse_kernal']
    coarse_threshold = params['coarse_threshold']
    fine_kernal = params['fine_kernal']
    fine_threshold = params['fine_threshold']
    model_path = params['model_path'] 
    grad_threshold = params['gradient_threshold']
    r = params['report_file'] 

    model_create = 'nets.' + net_architecture + '().to(device)'
    model = eval(model_create)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    data_transforms_no_RR = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(aoi_size),
        transforms.Resize(input_image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
                ])

    recorded_path_list = []
    source_path = test_path
    print(test_path)
    for root, _, files in os.walk(source_path, topdown=False):
        for img in files:
            img_path = os.path.join(root, img)
            recorded_path = [img_path, root, img]
            recorded_path_list.append(recorded_path)

    inspect_record = []
    label_record = []
    path_record = []

    for path_list in recorded_path_list:
        img_path, root, img = path_list
        utils.print_both(r, img_path)
        label = root[19: -7]
        utils.print_both(r, 'label is ' + label)
        label_record.append(label)
        path_record.append(img_path)
        _, input_img,output_img = obtain_input_img_and_recons_img(img_path, model, data_transforms_no_RR)    
        diff_img = obtain_diff_image(input_img, output_img)  
        inspect = inspect_by_block_avg(diff_img, coarse_threshold, coarse_kernal)
        if inspect == 'Good':
            input_grad_img = obtain_grad_image(input_img)
            inspect = inspect_by_grad(input_grad_img, diff_img, grad_threshold, fine_threshold, fine_kernal)
        inspect_record.append(inspect)
        utils.print_both(r, 'inspect result is ' + inspect)
        

    false_index,precision, recall, accuracy = performance_meansure(label_record, inspect_record)    
    utils.print_both(r, 'precision:' + str(precision))
    utils.print_both(r, 'recall:' + str(recall))
    utils.print_both(r, 'accuray:' + str(accuracy))
    utils.print_both(r, 'false inspections are:')       
            
    for i in false_index:
        utils.print_both(r, str(path_record[i]))


def inspect_by_max_diff(diff_image, threshold = 40):
    max_diff = np.max(diff_image)
    if max_diff >= threshold:
        return 'Bad'
    else:
        return 'Good'
def inspect_by_block_avg(diff_image, threshold = 40, kernel_size = 3):
    (m, n) = diff_image.shape
    len_x = m//kernel_size
    len_y = n//kernel_size
    result = 'Good'
    for i in range(len_x):
        for j in range(len_y):
            block = diff_image[kernel_size*i: kernel_size*(i+1), kernel_size*j : kernel_size*(j+1)]
            if np.average(block) >= threshold:
                result = 'Bad'
    return result

def inspect_by_var(input_var_img, diff_img, threshold_1 = 10, threshold_2 = 5, kernel_size = 5):
    (m,n) = input_var_img.shape
    result = 'Good'
    k = kernel_size//2
    for i in range(k, m - k):
        for j in range(k, n - k):
            input_block = input_var_img[i-k:i+k+1, i-k:i+k+1]
            if np.max(input_block) < threshold_1:
                diff_block = diff_img[i-k:i+k+1, i-k:i+k+1]
                if np.var(diff_block) > threshold_2:
                    result = 'Bad'
                    break
                
    return result   

def inspect_by_grad(input_grad_img, diff_img, threshold_1 = 50, threshold_2 = 5, kernel_size = 5):
    (m,n) = input_grad_img.shape
    result = 'Good'
    k = kernel_size//2
    for i in range(k, m - k):
        for j in range(k, n - k):
            input_block = input_grad_img[i-k:i+k+1, i-k:i+k+1]
            if np.max(input_block) < threshold_1:
                diff_block = diff_img[i-k:i+k+1, i-k:i+k+1]
                if np.average(diff_block) > threshold_2:
                    result = 'Bad'
                    break
                
    return result 

def inspect_by_diff_grad(input_grad_img, output_grad_img, threshold_1 = 100, threshold_2 = 50, kernel_size = 8):
    (m, n) = input_grad_img.shape
    len_x = m//kernel_size
    len_y = n//kernel_size
    result = 'Good'
    for i in range(len_x):
        for j in range(len_y):
            input_block = input_grad_img[kernel_size*i: kernel_size*(i+1), kernel_size*j : kernel_size*(j+1)]
            output_block = output_grad_img[kernel_size*i: kernel_size*(i+1), kernel_size*j : kernel_size*(j+1)]
            input_v = np.max(input_block)
            output_v = np.max(output_block)
            if input_v < threshold_2 and output_v > threshold_1:
                result = 'Bad'
    return result


def unnormalize(input_tensor):
    unnormalized_input_tensor = input_tensor/2 + 0.5
    return unnormalized_input_tensor
      
def obtain_input_img_and_recons_img(input_img_path, model, data_transforms):
    input_img_numpy = plt.imread(input_img_path)
    input_img_pil = Image.fromarray(input_img_numpy.astype('uint8'))
    input_img_transformed_tensor = data_transforms(input_img_pil)
    
    input_img_transformed_numpy = (unnormalize(input_img_transformed_tensor).numpy()*255)[0,:,:]
    input_img_cuda = input_img_transformed_tensor.to(device).unsqueeze(0)
    temp = model(input_img_cuda)
    output_image_tensor = temp[0,0,:,:].cpu().detach()
    
    output_img_numpy = (unnormalize(output_image_tensor).numpy()*255)
    output_img_numpy[output_img_numpy>255] = 255 ## prediction may exceed range(0,255)
    output_img_numpy[output_img_numpy<0] = 0
    
    return input_img_numpy, input_img_transformed_numpy,output_img_numpy

def obtain_diff_image(input_img, output_img):
    diff =  output_img - input_img
    diff[:,:2] = 0
    diff[:,-2:] = 0
    diff[:2, :] = 0
    diff[-2:, :] = 0
    diff = np.abs(diff)
    diff[diff>255] = 255
    diff_image = diff.astype('uint8')
    return diff_image

def obtain_grad_image(img):
    
    img_grad = np.zeros(img.shape)
    m = img_grad.shape[0]
    n = img_grad.shape[1]
    k = grad_x.shape[0]
    for i in range(k//2, m-k//2):
        for j in range(k//2, n-k//2):
            block = img[i-k//2:i+1+k//2, j-k//2:j+1+k//2]
    #         print(i,j)
    #         print(j+1+k//2)
    #         print(block)
            img_x_grad = np.sum(block * grad_x)
            img_y_grad = np.sum(block * grad_y)
            img_grad[i, j] = np.sqrt(img_x_grad**2 + img_y_grad**2)
    img_grad[:,:4] = 0
    img_grad[:,-4:] = 0
    img_grad[:4, :] = 0
    img_grad[-4:, :] = 0 
    return img_grad

def performance_meansure(label_record, inspect_record):
    true_possitive = 0
    true_negative = 0
    false_possitive = 0
    false_negative = 0
    false_index = []
    for i in range(len(label_record)):
        inspect = inspect_record[i]
        label = label_record[i]
        if inspect == "Good" and label == "Good":
            true_possitive += 1
        elif inspect == "Good" and label != "Good":
            false_possitive += 1
            false_index.append(i)
        elif inspect == "Bad" and label == "Good":
            false_negative += 1
            false_index.append(i)
        elif inspect == "Bad" and label != "Good":
            true_negative +=1
        else:
            print(inspect)
            print(label)
            print('Error!!')

    if true_possitive + false_possitive == 0:
        precision = 0
    else:
        precision = (true_possitive/(true_possitive + false_possitive))

    if true_possitive + false_negative == 0:
        recall = 0
    else:
        recall = (true_possitive/(true_possitive + false_negative))
    if true_possitive + true_negative + false_possitive + false_negative == 0:
        accuracy = 0
    else:
        accuracy = (true_possitive + true_negative)/(true_possitive + true_negative + false_possitive +
                                                         false_negative)
    
    return false_index, precision, recall, accuracy