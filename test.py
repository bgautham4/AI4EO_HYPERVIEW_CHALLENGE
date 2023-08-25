# %%
import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd

plt.rc('axes',labelsize = 14)

# %%
class TestingNet(nn.Module):
    def __init__(self,sizes):
        super().__init__()
        self.sizes=sizes
        self.ll1=nn.Linear(sizes[0],sizes[1],bias=True)
        self.act=nn.modules.activation.Sigmoid()
        self.ll2=nn.Linear(sizes[1],sizes[2],bias=True)
        #self.ll3=nn.Linear(sizes[2],sizes[3],bias=True)
    def forward(self,X):
        z1=self.ll1(X)
        a1=self.act(z1)
        z2=self.ll2(a1)
        a2=self.act(z2)
        #z3=self.ll3(a2)
        #a3=self.act(z3)
        return a2
    

# %%
#Defining the baseline regressor:
class BaselineRegressor:
    """
    Baseline regressor, which calculates the mean value of the target from the training
    data and returns it for each testing sample.
    """
    def __init__(self):
        self.mean = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.mean = np.mean(y_train, axis=0)
        self.classes_count = y_train.shape[1]  #len=4
        return self

    def predict(self, X_test: np.ndarray):
        return np.full((len(X_test), self.classes_count), self.mean)

# %%
#Loading in the data set:
class SpectralCurveFiltering():
    """
    Create a histogram (a spectral curve) of a 3D cube, using the merge_function
    to aggregate all pixels within one band. The return array will have
    the shape of [CHANNELS_COUNT]
    """

    def __init__(self, merge_function = np.mean):
        self.merge_function = merge_function

    def __call__(self, sample: np.ndarray):
        return self.merge_function(sample, axis=(1, 2))

# %%
import os
from glob import glob

def load_data(directory: str):
    """Load each cube, reduce its dimensionality and append to array.

    Args:
        directory (str): Directory to either train or test set
    Returns:
        [type]: A list with spectral curve for each sample.
    """
    data = []
    filtering = SpectralCurveFiltering()
    all_files = np.array(
        sorted(
            glob(os.path.join(directory, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )
    )
    for file_name in all_files:
        with np.load(file_name) as npz:
            arr = np.ma.MaskedArray(**npz)
        arr = filtering(arr)
        data.append(arr/1000)
    return np.array(data)


def load_gt(file_path: str):
    """Load labels for train set from the ground truth file.
    Args:
        file_path (str): Path to the ground truth .csv file.
    Returns:
        [type]: 2D numpy array with soil properties levels
    """
    gt_file = pd.read_csv(file_path)
    labels = gt_file[["P", "K", "Mg", "pH"]].values
    return labels


X_train = load_data("train_data")
y_train = load_gt("train_gt.csv")
#X_test = load_data("test_data")
#print(X_train)
#print(y_train)
print(f"Train data shape: {X_train.shape}")
#print(f"Test data shape: {X_test.shape}")


# %%
#Generating the baseline predictions:
#First we randomly shuffle the data:
Data=np.array(list(zip(X_train,y_train)),dtype=object)
np.random.shuffle(Data)
X_train_new=np.array([(x-x.mean())/np.sqrt(x.var()) for x in Data[:1500,0]])
y_train_new=np.array([y for y in Data[:1500,1]])
X_test_new=np.array([(x-x.mean())/np.sqrt(x.var()) for x in Data[1500:,0]])
y_test_new=np.array([y for y in Data[1500:,1]])
y_test_new_copy=np.copy(y_test_new)
baseline_reg=BaselineRegressor()
# Fit the baseline regressor once again on new training set
baseline_reg = baseline_reg.fit(X_train_new, y_train_new)
baseline_predictions = baseline_reg.predict(X_test_new)

# Generate baseline values to be used in score computation
baselines = np.mean((y_test_new_copy - baseline_predictions) ** 2, axis=0)
max_arr = y_train_new.max(axis=0)
y_train_new_for_NN = y_train_new/max_arr

# %%
baselines

# %%
temp = y_test_new / max_arr 
y_test_for_graphing = torch.from_numpy(temp.astype(np.float32))

# %%
#training the network:
test=TestingNet([150,100,4])
loss=nn.BCELoss()
num_epochs=100
mini_batch_size=50
dataset=TensorDataset(torch.from_numpy(X_train_new.astype(np.float32)),torch.from_numpy(y_train_new_for_NN.astype(np.float32)))
L_vec=[]
L2_vec=[]
x_train_for_graphing = torch.from_numpy(X_train_new.astype(np.float32))
y_train_for_graphing = torch.from_numpy(y_train_new_for_NN.astype(np.float32))
x_test_for_graphing = torch.from_numpy(X_test_new.astype(np.float32))
for epoch in range(num_epochs):
    mini_batches=DataLoader(dataset=dataset,batch_size=mini_batch_size,shuffle=True)
    optimizer=torch.optim.SGD(params=test.parameters(),lr=0.1*(1.01 - epoch/num_epochs),weight_decay=0.01)
    for mini_batch_in,mini_batch_out in mini_batches:
        outs=test.forward(mini_batch_in)
        l=loss(outs,mini_batch_out)
        l.backward()
        optimizer.step()
    with torch.no_grad():
        Outs1=test(x_train_for_graphing)
        Outs2=test(x_test_for_graphing)
        Loss1=loss(Outs1,y_train_for_graphing)
        Loss2=loss(Outs2,y_test_for_graphing)
        L_vec.append(Loss1)
        L2_vec.append(Loss2)
    optimizer.zero_grad()

# %%
plt.plot(range(1,num_epochs+1),L_vec,linestyle='dotted',color='b',label='train_data')
plt.plot(range(1,num_epochs+1),L2_vec,color='r',label='test_data',linewidth=2.5)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('perceptron_loss.eps', bbox_inches='tight',pad_inches = 0)
plt.show()

# %%
#Predictions of the network:
X_test_tensor=torch.from_numpy(X_test_new.astype(np.float32))
with torch.no_grad():
    predictions_tensor=test(X_test_tensor)
predictions=predictions_tensor.numpy()

predictions = predictions*max_arr

mse=np.mean((predictions-y_test_new_copy)**2,axis=0)

scores=mse/baselines

final_score = np.mean(scores)

for score, class_name in zip(scores, ["P", "K", "Mg", "pH"]):
    print(f"Class {class_name} score: {score}")

print(f"Final score: {final_score}")

# %%


# %%


# %%
#Final submission:
