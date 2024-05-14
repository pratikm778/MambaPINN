import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=80,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--n-test', type=int, default=500,
                    help='Size of test set')
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def dateinf(series, n_test):
    lt = len(series)
    print('Training start',series[0])
    print('Training end',series[lt-n_test-1])
    print('Testing start',series[lt-n_test])
    print('Testing end',series[lt-1])

set_seed(args.seed,args.cuda)
class Net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim,args.hidden),
            Mamba(self.config),
            nn.Linear(args.hidden,out_dim),
            #-----------------------------
            nn.Sigmoid()
            #-----------------------------
        )

    def forward(self,x):
        x = self.mamba(x)
        return x.flatten()



def BL_loss(data, output):
    #feature data is M, t, x | 'label' is u
    data = data.squeeze(0)  # Now tensor has shape (500, 2)
    #print(data.shape)
    #print(output.shape)
    #print(data[:,5])
    tot_data = torch.column_stack((data, output)).requires_grad_()
    tot_data.retain_grad()
    #print(tot_data.shape)
    sum = 0
    # ----- Initial Condition ----
    init_data = tot_data[np.where(data[:,1]==0)] # observations at time == 0
    print(init_data.shape)
    init_preds = init_data[:,3]
    IC_loss = torch.mean(init_preds**2) # penalizing where init_preds @t=0 != 0
    sum += IC_loss
    IC_loss = None # clear up memory

    # ----- Boundary Condition ----
    boundary_data = tot_data[np.where(data[:,2]==0)] # observations at x == 0
    boundary_preds = boundary_data[:,3]
    BC_loss = torch.mean((boundary_preds - 1)**2) # penalizing where init_preds @x=0 != 1
    sum += BC_loss
    BC_loss = None
    # ----- Residual Loss ----
    for M in np.unique(data[:,0]):  # Ms = {2,4, ..., 100}
        # Automatic differentiation to compute derivatives
        Mdata = tot_data[np.where(data[:,0] == M)].requires_grad_() # data for this M
        print(Mdata.shape)
        outputs = (Mdata[:,3]).requires_grad_() # select outputs corresponding to the specific M
        print(outputs.shape)
        # ------R1 ------
        ts = (Mdata[:,1]).requires_grad_()
        u_t = torch.autograd.grad(outputs, ts, grad_outputs=torch.ones_like(outputs), allow_unused=True)[0] #create_graph=True

        """
        # IF autograd method doesnt work: implemented the approximation
        R1 = torch.zeros((len(output)-1),len(np.unique(Mdata[:,1])))
        ts = np.unique(Mdata[:,1])
        xs = np.unique(Mdata[:,2])
        for i in range(1, len(ts)-1):        # looping thru t's
            for j in range(1, len(xs)-1):    # looping thru x's
                pre_u = Mdata[np.where((Mdata[:, 1] == ts[1-i]) * (Mdata[:, 2] == xs[i]))] #select row that satisfies both conditions
                pre_u = pre_u[3]

                post_u = Mdata[np.where((Mdata[:, 1] == ts[1+i]) * (Mdata[:, 2] == xs[i]))] #select row that satisfies both conditions
                post_u = post_u[3]

                R1[i,j] = (post_u - pre_u)/(ts[1+i] - ts[1-i])

        # ---- R2 -----
        f_theta = []
        for i in range(len(xs)):
            arr =

        f_theta = np.array(f_theta)
        for i in range(1, len(ts)-1):        # looping thru t's
            for j in range(1, len(xs)-1):    # looping thru x's



        """
        # ------R2 ------
        # Define the flux function and its derivative

        f = (outputs**2 / (outputs**2 + (1/M)*(1-outputs)**2)).requires_grad_()
        print(f.shape)
        xs = (Mdata[:,2]).requires_grad_()
        f_x = torch.autograd.grad(f, xs, grad_outputs=torch.ones_like(f), allow_unused=True)[0]
        print(f_x, u_t)

        sum += (np.linalg.norm(u_t + f_x))**2

    return torch.tensor(sum, dtype=torch.float32)



test_dat1 = pd.read_csv('train_data_part1.csv')
train_dat2 = pd.read_csv('train_data_part2.csv')
train_dat3 = pd.read_csv('train_data_part3.csv')
train_dat = pd.concat([train_dat1,train_dat2,train_dat3],ignore_index=True)
# pd.concat([df1, df2, df3], ignore_index=True)
test_dat = pd.read_csv('test_data.csv')
train_dat = train_dat[train_dat['x'] != 0]
train_dat = train_dat[train_dat['t'] != 0]

# Step 2: Sort the DataFrame by 'X'
train_dat = train_dat.sort_values(by='u')
train_dat = train_dat.sort_values(by='M')

# Display the sorted DataFrame
print(train_dat['M'].unique())

train_dat = train_dat[train_dat['M'].isin([2,6,16,38,50,60,72,80,88,96,100])]

test_dat = test_dat[test_dat['x'] != 0]
test_dat = test_dat[test_dat['t'] != 0]

# Step 2: Sort the DataFrame by 'X'
test_dat = test_dat.sort_values(by='u')
test_dat = test_dat.sort_values(by='M')

# Display the sorted DataFrame
print(test_dat['M'].unique())

test_dat = test_dat[test_dat['M'].isin([4.5,11.5,46,71,98,140])]

train_dat = train_dat.sort_values(by=['M','t','x'])
test_dat = test_dat.sort_values(by=['M','t','x'])
trainy = train_dat.pop('u').values
testy = test_dat.pop('u').values
testx = test_dat.values
trainx = train_dat.values



def PredictWithData(trainX, trainy, testX):
    clf = Net(len(trainX[0]),1)
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    xt = torch.from_numpy(trainX).float().unsqueeze(0)
    xv = torch.from_numpy(testX).float().unsqueeze(0)
    yt = torch.from_numpy(trainy).float()
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    
    for e in range(70):
        clf.train()
        z = clf(xt)
        loss = F.mse_loss(z,yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))

    clf.eval()
    mat = clf(xv)
    if args.cuda: mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()
    return yhat

ypred = PredictWithData(trainx,trainy,testx)

def evaluation_metric(y_test,y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test,y_hat)
    R2 = r2_score(y_test,y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2))

evaluation_metric(testy, ypred)