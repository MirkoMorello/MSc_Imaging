import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import multiprocessing

class Dataset(torch.utils.data.Dataset):

    def __init__(self, csv):
        super(Dataset, self).__init__()
        # read the csv file
        self.df = pd.read_csv(csv, sep=r'\s+')
        # self.df = self.df.dropna(axis=0)
        # save cols
        self.input_cols = ['pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year',
    'Distance', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5',
    'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11',
    'month_12', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3',
    'weekday_4', 'weekday_5', 'weekday_6', 'hour_0', 'hour_1', 'hour_2',
    'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
    'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15',
    'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
    'hour_22', 'hour_23']
        self.output_cols = ['fare_amount']
        
        


    def __len__(self):
        # TODO: here i will return the number of samples in the dataset
        return len(self.df)


    def __getitem__(self, idx):
        # read row, split in input and output and convert in tensors
        cur_sample = self.df.iloc[idx]
        # split the current sample in input and output (ground truth)
        cur_sample_x = cur_sample[self.input_cols]
        cur_sample_y = cur_sample[self.output_cols]
        # convert to tensor (torch format)
        cur_sample_x = torch.tensor(cur_sample_x.tolist(), dtype=torch.float32).cuda()
        cur_sample_y = torch.tensor(cur_sample_y.tolist(), dtype=torch.float32).cuda()
        # return the sample
        return cur_sample_x, cur_sample_y



# TODO: define a network composed of linear layers interleaved by ReLUs. Note: last layer must be a linear layer.
class Net(nn.Module):
    def __init__(self):
        
        # call the constructor of the parent class
        super(Net, self).__init__()
        
        # define the layers
        self.fc1 = nn.Linear(50, 128) # 13 input features, 128 output features
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.ReLU()
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.ReLU()
        self.fc7 = nn.Linear(64, 16)
        self.fc8= nn.ReLU()
        self.fc9 = nn.Linear(16, 1) # 1 output feature
        
        self.to('cuda')

    def forward(self, x):
        
        # define the forward pass
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        
        return x

# create validation routine
def validate(net, dl):
    # get final score
    score = 0
    # set network in eval mode
    net.eval()
    # at the end of epoch, validate model
    for inp, gt in dl:
        # move batch to gpu
        inp = inp.cuda()
        gt = gt.cuda()
        # get output
        with torch.no_grad():
            out = net(inp)
        # compare with gt
        cur_score = F.l1_loss(out, gt)
        # append
        score += cur_score 
    # at the end, average over batches
    score /= len(dl)
    # set network in training mode
    net.train()
    # return score
    return score
        
# for each epoch
def train(net, train_dl, val_dl, optimizer, writer, experiment_name):
    
    # initialize iteration number
    n_iter = 0
    best_val = None
    
    for cur_epoch in range(20):
        # plot current epoch
        writer.add_scalar("epoch", cur_epoch, n_iter)
        # TODO: for every batch, compute output, loss, perform backward propagation and finally update weights
        print('Epoch:', cur_epoch)
        for inp, gt in train_dl:
            inp = inp.cuda()
            gt = gt.cuda()
            # zero gradients (empty the gradient buffer)
            optimizer.zero_grad()
            # get output
            print('Input:', inp)
            out = net(inp)
            print('Output:', out)
            # compute loss, F1 loss is the mean absolute error
            loss = F.l1_loss(out, gt)
            print('loss:', loss.item())
            # backprop
            loss.backward()
            print('Backward')
            # update weights, I do the step, the NN is updated
            print('Optimizer')
            optimizer.step()
            # plot loss
            writer.add_scalar("train", loss.item(), n_iter)
            # increment iteration number
            n_iter += 1
            print('Loss:', loss.item())
            
        print('Validation')
        print('Epoch:', cur_epoch)
        print('Loss:', loss.item())
        # at the end, validate model
        cur_val = validate(net, val_dl)
        # plot validation
        writer.add_scalar("val", loss.item(), n_iter)
        # TODO: check if it is the best model so far
        if best_val is None or cur_val > best_val:
            data = {
                'data' : net.state_dict(),
                'opt' : optimizer.state_dict(),
                'epoch' : cur_epoch}
            
            torch.save(data, experiment_name + '_best.pth')
            # update best validation value
            best_val = cur_val
        # save optimizer
        data = {
            'data' : net.state_dict(),
            'opt' : optimizer.state_dict(),
            'epoch' : cur_epoch}
        torch.save(optimizer.state_dict(), '_last.pth')
        
# create train and validation datasets
train_ds = Dataset('../datasets/Uber/train.csv')
val_ds =  Dataset('../datasets/Uber/val.csv')
device = 'cuda:0'
learning_rate = 0.0001
batch_size = 50000
experiment_name = 'uber'
# create train dataloader
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size = batch_size,
    drop_last = False,
    shuffle = False,
    num_workers = 4, # this is needed to avoid problems with the multiprocessing since we are using the mps device
    
)
# create validation dataloader
val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size = batch_size,
    drop_last = False,
    shuffle = False,
    num_workers = 4,
)

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # let's test the network
    net = Net()

    # define optimizer
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

    # define summary writer
    writer = SummaryWriter(experiment_name)

    # define best validation value
    best_val = None
    net.to(device = 'cuda:0')
    
    # let's move the network in GPU
    train(net, train_dl, val_dl, optimizer, writer, experiment_name)