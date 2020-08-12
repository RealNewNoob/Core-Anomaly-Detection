if __name__ == "__main__":
    import torch
    import torchvision
    from torch.optim import lr_scheduler
    from torchvision import datasets, models, transforms
    import numpy as np
    from torch import nn
    import copy
    import os
    import argparse
    import utils
    import fnmatch
    import nets
    import train_function
    import test_function
    from datetime import datetime

    # Translate string entries to bool for parser
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Use CAE_anomaly_dection for anomally_dection')
    parser.add_argument('--net_architecture',default = 'Core_CAE', help = 'network architecture used')
    parser.add_argument('--center_neurons', default = 100, type = int, help = 'set center layer neurons number')
    parser.add_argument('--input_image_size', default= [128,128], nargs = 2, type= int, help='size of image input into network')
    parser.add_argument('--mode', default = 'test', choices = ['train', 'test', 'train_test'], help = 'mode')
    parser.add_argument('--train_path', default = 'data set/train data', help = 'path of training set, there are only Good images inside')
    parser.add_argument('--valid_path', default = 'data set/valid data', help = 'path of validation set, there are only Good images inside')
    parser.add_argument('--test_path', default = 'data set/test data', help = 'path of test set, the labels are described as sub folder name')
    parser.add_argument('--model_path', default='pretrained net/pretrained net.pt', help = 'path of model to load for test')
    parser.add_argument('--aoi_size', default= [325,325], nargs = 2, type= int, help='size of center area of interest')
    parser.add_argument('--coarse_kernal', default = 3, type = int, help = 'kernal size for coarse threshold')
    parser.add_argument('--coarse_threshold', default= '31', type = float, help = 'thresholds of detection')
    parser.add_argument('--fine_kernal', default = 5, type = int, help= 'kernal size for fine threshold')
    parser.add_argument('--fine_threshold', default= '4', type=float, help='thresholds of detection')
    parser.add_argument('--gradient_threshold', default = 70, type=int, help = 'gradient_threshold')
    parser.add_argument('--batch_size', default = 2, type = int, help = 'batch size of training, validation, and test')
    parser.add_argument('--learning_rate', default = 1e-3, type = float)
    parser.add_argument('--weight_decay', default= 0.0, help='weight decay setting')
    parser.add_argument('--epochs', default = 2000, type = int)
    parser.add_argument('--print_interval', default= 10, type = int, help = 'The interval of report print')
    parser.add_argument('--save_interval', default = 100, type = int, help = 'The interval of model save')
    args = parser.parse_args()
    print(args)

    ####### obtain the parser parameters

    net_architecture = args.net_architecture
    center_neurons = args.center_neurons
    input_image_size = args.input_image_size
    mode = args.mode 
    train_path = args.train_path 
    valid_path = args.valid_path
    test_path = args.test_path
    model_path = args.model_path
    aoi_size = args.aoi_size
    coarse_kernal = args.coarse_kernal
    coarse_threshold = args.coarse_threshold
    fine_kernal = args.fine_kernal
    fine_threshold = args.fine_threshold
    gradient_threshold = args.gradient_threshold
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epochs = args.epochs
    print_interval = args.print_interval
    save_interval = args.save_interval

    ####### create dirct to store reports, trained nets, or inspection
    dirs = ['reports', 'nets']
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

    ####### define txt file to store reports

    training_reports_list = sorted(os.listdir('reports'), reverse = True)
    if training_reports_list:
        for file in training_reports_list:
            if fnmatch.fnmatch(file, net_architecture + '*'):
                idx = int(str(file)[-3:]) + 1
                break
    try: 
        idx
    except NameError:
        idx = 1
    report_name = net_architecture + '_' + mode + '_' + str(idx).zfill(3)
    report_file_path = os.path.join('reports', report_name)
    net_file_path = os.path.join('nets', report_name)
    os.makedirs(net_file_path)
    r = open(report_file_path, 'w')

    utils.print_both(r, 'Mode is ' + mode)
    now = datetime.now()
    utils.print_both(r, now.strftime("%m/%d/%Y, %H:%M:%S"))

    ####### define the device to train 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ####### define the params dict to transfer to training function or inspection function

    train_params = {}
    train_params['train_path'] = train_path
    train_params['valid_path'] = valid_path
    train_params['net_architecture'] = net_architecture
    train_params['input_image_size'] = input_image_size
    train_params['aoi_size'] = aoi_size
    train_params['batch_size'] = batch_size
    train_params['learning_rate'] = learning_rate
    train_params['epochs'] = epochs
    train_params['print_interval'] = print_interval
    train_params['save_interval'] = save_interval
    train_params['report_file'] = r
    train_params['net_file_path'] = net_file_path

    test_params = {}
    test_params['net_architecture'] = net_architecture
    test_params['input_image_size'] = input_image_size
    test_params['aoi_size'] = aoi_size
    test_params['test_path'] = test_path
    test_params['coarse_kernal'] = coarse_kernal
    test_params['coarse_threshold'] = coarse_threshold
    test_params['fine_kernal'] = fine_kernal
    test_params['fine_threshold'] = fine_threshold
    test_params['gradient_threshold'] = gradient_threshold
    test_params['model_path'] = model_path
    test_params['report_file'] = r

    if mode == 'train':
        train_function.train(train_params)
    elif mode == 'test':
        print('hit test!!!!')
        test_function.test(test_params)
    elif mode == 'train_test':
        model_path = train_function.train(train_params)
        test_params['model_path'] = model_path
        test_function.test(test_params)
    r.close()
    




    



