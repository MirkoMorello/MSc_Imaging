import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "mps")


class Net(nn.Module):
    def __init__(self):
        
        super(Net, self).__init__()
        
        # define the layers
        self.fc1 = nn.Linear(50, 128) # 13 input features, 128 output features
        self.fc2 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1) # 1 output feature

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x.squeeze()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv):
        # load the dataset
        df = pd.read_csv(csv, sep=r'\s+')

        self.data = torch.tensor(df.drop(columns=['fare_amount']).values, dtype=torch.float32)
        self.target = torch.tensor(df['fare_amount'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def validate(model, val_loader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.l1_loss(output, target)
    return loss / len(val_loader)

def train(model, train_loader, val_loader, optimizer, epochs=10):
    
    # train the model
    for epoch in tqdm(range(epochs)):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.l1_loss(output, target)
            loss.backward()
            optimizer.step()
            
        val_loss = validate(model, val_loader)
    return val_loss


if __name__ == '__main__':
    # create an instance of the neural network
    model = Net().to(device)

    # define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    batch_size = 5000
    num_workers = 4


    train_dataset = Dataset('../datasets/Uber/train.csv')
    val_dataset = Dataset('../datasets/Uber/val.csv')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    train(model, train_loader, val_loader, optimizer, epochs=10)