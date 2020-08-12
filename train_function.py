import utils
import time
import torch
import numpy as np
import copy
import torchvision
import torchvision.transforms as transforms
import os
import nets

# Training function (from my torch_DCEC implementation, kept for completeness)
def train(params):

    train_path = params['train_path'] 
    valid_path = params['valid_path']
    net_architecture  = params['net_architecture'] 
    input_image_size = params['input_image_size']
    aoi_size = params['aoi_size'] 
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    print_interval = params['print_interval']
    save_interval = params['save_interval'] 
    net_file_path = params['net_file_path']
    r = params['report_file'] 

    utils.print_both(r, 'Training model')

    
    data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(degrees = 10),
    transforms.CenterCrop(aoi_size),
    transforms.Resize(input_image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
            ])

    train_image_dataset = torchvision.datasets.ImageFolder(train_path, data_transforms)
    train_dataset_size = len(train_image_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_image_dataset, batch_size=batch_size,
                                                        shuffle=True, num_workers=2)

    valid_image_dataset = torchvision.datasets.ImageFolder(valid_path, data_transforms)
    valid_dataset_size = len(valid_image_dataset)
    valid_dataloader = torch.utils.data.DataLoader(valid_image_dataset, batch_size=batch_size,
                                                        shuffle=True, num_workers=2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_create = 'nets.' + net_architecture + '().to(device)'
    model = eval(model_create)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history = []
    valid_loss_history = []

    for e in range(epochs + 1):
    
        train_running_loss = 0.0
        valid_running_loss = 0.0
    
        for inputs,_ in train_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss
        
        else:
            with torch.no_grad():
                for inputs,_ in valid_dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)            
                    valid_running_loss += loss
        
        avg_train_epoch_loss = train_running_loss/train_dataset_size  ## average loss of a single image in the epoch
        train_loss_history.append(avg_train_epoch_loss)
        
        avg_valid_epoch_loss = valid_running_loss/valid_dataset_size## average loss of a single image in the epoch
        valid_loss_history.append(avg_valid_epoch_loss)
                    
        if e % print_interval == 0:
            tmp = 'epoch :' + str(e+1)
            utils.print_both(r, tmp)
            tmp = 'train loss: {:.4e}'.format(avg_train_epoch_loss) + '  valid loss: {:.4e}'.format(avg_valid_epoch_loss)
            utils.print_both(r, tmp)
            
        
        if e % save_interval == 0:
            PATH =  net_file_path + '/' + str(e) + ".pt" 
            torch.save(model.state_dict(), PATH)
    
    return PATH