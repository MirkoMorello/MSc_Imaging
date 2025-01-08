import os,sys,math,time,io,argparse,json,traceback,collections
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, utils, models, ops
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count, Pool
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import seaborn as sns
from datetime import datetime

sns.set()


class Dataset(torch.utils.data.Dataset):

    def __init__(self, csv):
        # read the csv file
        self.df = pd.read_csv(csv, sep=',')
        # save cols
        self.output_cols = ['class']
        # get columns of dataframe
        self.input_cols = list(set(self.df.columns) - set(self.output_cols))


    def __len__(self):
        # here i will return the number of samples in the dataset
        return len(self.df)


    def __getitem__(self, idx):
        # here i will load the file in position idx
        cur_sample = self.df.iloc[idx]
        # split in input / ground-truth
        cur_sample_x = cur_sample[self.input_cols]
        cur_sample_y = cur_sample[self.output_cols]
        # convert to torch format
        cur_sample_x = torch.tensor(cur_sample_x.tolist()).unsqueeze(0)
        cur_sample_y = torch.tensor(cur_sample_y.tolist()).squeeze()
        # return values
        return cur_sample_x, cur_sample_y



class CNN(nn.Module):

	def __init__(self):
		# initialize super class
		super(CNN, self).__init__()
		# define conv layers
		self.layer1 = nn.Conv1d( 1, 32, kernel_size=3, stride=2, padding=1)
		self.layer2 = nn.ReLU()
		self.layer3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
		self.layer4 = nn.ReLU()
		self.layer5 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
		self.layer6 = nn.ReLU()
		self.layer7 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
		self.layer8 = nn.ReLU()
		# define linear layer
		self.layer9 = nn.Linear(128*12, 1)


	def forward(self, x):
		# apply convolution layers
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		x = self.layer7(x)
		x = self.layer8(x)
		# reshape from (10, 128, 12) to (10, 1536)
		x = x.reshape(x.shape[0], -1)
		# fully connected
		x = self.layer9(x)
		# add sigmoid
		x = F.sigmoid(x)
		# return output
		return x



# create validation routine
def validate(net, dl, n_classes, device):
    # create metric objects
    tm_acc = torchmetrics.Accuracy(task='binary', num_classes=n_classes, average= 'macro', top_k=1)
    tm_con = torchmetrics.ConfusionMatrix(task="binary", num_classes=n_classes)
    # move metric to device
    tm_acc.to(device)
    tm_con.to(device)
    # set network in eval mode
    net.eval()
    # at the end of epoch, validate model
    for inp, gt in dl:
        # move batch to gpu
        inp = inp.to(device)
        gt = gt.to(device)
        # remove singleton dimension
        gt = gt.squeeze()
        # get output
        with torch.no_grad():
            # perform prediction
            logits = net(inp)
        # update metrics
        tm_acc.update(logits.squeeze(), gt)
        tm_con.update(logits.squeeze(), gt)

    # at the end, compute metric
    acc = tm_acc.compute()
    con = tm_con.compute()
    # set network in training mode
    net.train()
    # return score
    return acc, con


if __name__ == '__main__':
	# setup
	device = 'cuda'
	batch_size = 50
	now = datetime.now()
	experiment_name = 'prova_' + now.strftime("%Y_%m_%d__%H_%M_%S")
	experiments_dir = '/home/flavio/Documenti/Insegnamenti/2022_big_images/Lesson 6 - signals/heartbeat_categorization/data/experiments'
	learning_rate = 1e-4
	n_classes = 2
	# get current folder
	dir_path = os.path.dirname(os.path.realpath(__file__))
	out_path = os.path.join(dir_path, 'out')
	# create train dataloader
	train_dl = torch.utils.data.DataLoader(
		Dataset('/home/flavio/Documenti/Insegnamenti/2022_big_images/Lesson 6 - signals/heartbeat_categorization/data/out/train.csv'),
		batch_size = batch_size,
		drop_last = True,
		shuffle = True,
		num_workers = 8
	)
	# create validation dataloader
	val_dl = torch.utils.data.DataLoader(
		Dataset('/home/flavio/Documenti/Insegnamenti/2022_big_images/Lesson 6 - signals/heartbeat_categorization/data/out/val.csv'),
		batch_size = batch_size,
		drop_last = False,
		shuffle = False,
		num_workers = 8
	)
	# create network
	net = CNN()
	net = net.to(device)
	# define optimizer
	optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)
	# define loss function
	loss_fun = nn.BCELoss()
	# define summary writer
	writer = SummaryWriter(os.path.join(experiments_dir, experiment_name))
	# define iteration number
	n_iter = 0
	# define best accuracy
	best_acc = 0
	# for each epoch
	for cur_epoch in range(250):
		# plot current epoch
		writer.add_scalar("epoch", cur_epoch, n_iter)
		# for each batch
		for inp, gt in train_dl:
			# move batch to gpu
			inp = inp.to(device)
			gt = gt.to(device)
			# reset gradients
			optimizer.zero_grad()
			# get output
			logits = net(inp)
			# compute loss
			loss = loss_fun(logits.squeeze(), gt)
			# compute backward
			loss.backward()
			# update weights
			optimizer.step()
			# plot
			writer.add_scalar("loss", loss.item(), n_iter)
			n_iter += 1
		# validate
		acc, cf = validate(net, val_dl, n_classes=n_classes, device=device)
		# log
		writer.add_scalar("acc", acc.item(), n_iter)
		# save best
		if acc > best_acc:
			best_acc = acc
			torch.save({
				'net': net.state_dict(),
				'opt': optimizer.state_dict(),
				'epoch': cur_epoch,
				'acc': acc
			},
			os.path.join(experiments_dir, experiment_name, 'best.pth'))
		# save last
		torch.save({
			'net': net.state_dict(),
			'opt': optimizer.state_dict(),
			'epoch': cur_epoch,
			'acc': acc
		},
		os.path.join(experiments_dir, experiment_name, 'last.pth'))

	