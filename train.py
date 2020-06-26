import os
import numpy as np
import torch
from torch import nn
from torch import optim
from get_input_args import get_input_args
from load_datasets import load_datasets
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict



def get_class_tot(train_dir, valid_dir, test_dir):

    return len(os.listdir(train_dir))


def get_loaders(train_data, valid_data, test_data):

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)

    return trainloader, validloader, testloader


def get_model_classifier(network, num_classes, hidden_units):

    model = getattr(torchvision.models, network)(pretrained = True)
    out_features = hidden_units

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, out_features)),  
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(p = 0.3)),
                                            ('fc2', nn.Linear(out_features, num_classes)),
                                            ('output', nn.LogSoftmax(dim = 1))
                                            ]))
        
    model.classifier = classifier
    return model, classifier

def save_model_checkpoint(model, train_data, optimizer, epochs):
    torch.save({'state_dict': model.state_dict(),
                'classifier': model.classifier,
                'class_to_idx': train_data.class_to_idx,
                'opt_state': optimizer.state_dict,
                'num_epochs': epochs}, 'checkpoint.pth')
    
def validation(model, valid_loader, criterion, device):
    valid_loss = 0
    valid_accuracy = 0
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        logps = model(images)
        loss = criterion(logps, labels)
        
        valid_loss += loss.item()
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        valid_accuracy += equals.type(torch.FloatTensor).mean()

    return valid_loss/len(valid_loader), valid_accuracy/len(valid_loader)

def training( network, learning_rate, hidden_units, epochs, gpu):
    
    train_dir = './flowers/train/'
    valid_dir = './flowers/valid/'
    test_dir = './flowers/test/'
   
    train_data, valid_data, test_data = load_datasets(train_dir, valid_dir, test_dir)
    
    trainloader, validloader, testloader = get_loaders(train_data, valid_data, test_data)

    model, classifier = get_model_classifier(network, get_class_tot(train_dir, valid_dir, test_dir), hidden_units)

    train_losses, valid_losses = [], []
    
    if gpu == True:
        # use GPU
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for epoch in range(epochs):
        epoch_train_run_loss = 0
        epoch_batches = 0

        print(f"Epoch for {epoch+1}/{epochs}..")
    
        for images, labels in trainloader:
            epoch_batches += 1
            
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()        

            epoch_train_run_loss += loss.item()
            
        with torch.no_grad():
            model.eval()
            valid_loss, valid_accuracy = validation(model, validloader, criterion, device)
            valid_losses.append(valid_loss)
            
            model.train()
        
        train_losses.append(epoch_train_run_loss/epoch_batches)
            
        print(f"Final Train loss: {epoch_train_run_loss/epoch_batches:.4f}"
              f", Validation loss: {valid_loss:.4f} "
              f"and Validation accuracy: {valid_accuracy:.4f} for Epoch {epoch+1}/{epochs}")
    
    print("Validation Completed Successfully. Saving started")
    #saving checkpoint
    save_model_checkpoint(model, train_data,  optimizer, epochs)     


if __name__ == "__main__":
    network, learn_rate, hidden_units, epochs, gpu = get_input_args()
    # Start Training
    training(network, learn_rate, hidden_units, epochs, gpu)