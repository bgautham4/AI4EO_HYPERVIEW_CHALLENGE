# %%
import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
plt.rc('axes',labelsize = 14)

# %%
"""
# Defining the conv-net class:
## You can set the following parameters of the network:
->Stride

->Kernel size

->optional hidden Dense layer(along with its size)


The structure of the network can be inferred from the code:
"""

# %%
#Defining the structure of the convolutional net
class ConvNet(nn.Module):
    def __init__(self,channels=4,kernel_size=5,stride_len=1,hidden_full_layer=False,size_hidden_layer=50):
        super().__init__()
        #Note:Output_size=num_out_channels*([L_in - (Kernel_size -1) -1]/stride   + 1)
        conv_out_size=int(channels*((150 - (kernel_size - 1) -1)/stride_len + 1))
        self.conv_layer=nn.Conv1d(in_channels=1,out_channels=channels,kernel_size=kernel_size,stride=stride_len,bias=True)
        self.fully_connected_ll1=nn.Linear(conv_out_size,size_hidden_layer)
        self.fully_connected_ll2=nn.Linear(size_hidden_layer,4)
        self.fully_connected_ll3=nn.Linear(conv_out_size,4)
        self.hidden_layer_status=hidden_full_layer
        self.activation=nn.modules.activation.LeakyReLU()
    def forward(self,X):
        out1_z=self.conv_layer(X)
        out1_a=self.activation(out1_z)
        out1_reduced_dim=out1_a.reshape(-1,out1_a.shape[1]*out1_a.shape[2]) #Convert the 3-D convolved output into a 2-D Array
                                                                       #that can be processed by Linear layer.
        if self.hidden_layer_status:
            out2_z=self.fully_connected_ll1(out1_reduced_dim)
            out2_a=self.activation(out2_z)
            out3_z=self.fully_connected_ll2(out2_a)
            out3_a=self.activation(out3_z)
            return out3_a
        else:
            out2_z=self.fully_connected_ll3(out1_reduced_dim)
            out2_a=self.activation(out2_z)
            return out2_a   

# %%
"""
# Code that was given in the starter pack to generate baseline solution and to load the data for testing and training:
"""

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
        #arr1=arr[:50]/1000
        #arr2=arr[50:100]/1000
        #arr3=arr[100:150]/1000
        
        data.append(arr/1000)
        Data=np.array(data)
    return Data.reshape(Data.shape[0],1,Data.shape[1])  
    #Data.reshape(Data.shape[1],Data.shape[0],Data.shape[2])


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
X_train_new=np.array([x for x in Data[:1500,0]])
y_train_new=np.array([y for y in Data[:1500,1]])
y_train_new_for_NN=np.copy(y_train_new)
X_test_new=np.array([x for x in Data[1500:,0]])
y_test_new=np.array([y for y in Data[1500:,1]])
baseline_reg=BaselineRegressor()
# Fit the baseline regressor once again on new training set
baseline_reg = baseline_reg.fit(X_train_new, y_train_new)
baseline_predictions = baseline_reg.predict(X_test_new)

# Generate baseline values to be used in score computation
baselines = np.mean((y_test_new - baseline_predictions) ** 2, axis=0)

max_arr = y_train_new.max(axis=0)
y_train_new_for_NN = y_train_new_for_NN/max_arr

# %%
"""
# Baseline solution's MSE values:
"""

# %%
baselines

# %%
"""
# The following structure of the network was used for generating the submission file:

Kernel=5x5 ; stride = 5 hidden_layer = True (100 neurons):
"""

# %%
#Training the network:
#hidden_neurons = [50,100,200,500,1000,2000,3000,4000,5000]
kernels = [5,10,15,20,25,30,35,40,45,50]
networks = [ConvNet(channels=kernel,kernel_size=5,stride_len=5,hidden_full_layer=True,size_hidden_layer=100) for kernel in kernels]  # 20 kernels var neurons OR 500 neurons var kernels.
scores_dict = dict([('P',[]),('K',[]),('Mg',[]),('pH',[]),('fin_score',[])])
for k,network in enumerate(networks):
    print(f'{k+1}/10 th network.')
    test=network
    loss=nn.MSELoss()
    num_epochs=100
    mini_batch_size=50
    dataset=TensorDataset(torch.from_numpy(X_train_new.astype(np.float32)),torch.from_numpy(y_train_new_for_NN.astype(np.float32)))
    x_train_graphing=torch.from_numpy(X_train_new.astype(np.float32))
    y_train_graphing=torch.from_numpy(y_train_new_for_NN.astype(np.float32))
    x_test_graphing=torch.from_numpy(X_test_new.astype(np.float32))
    y_test_graphing=y_test_new.copy()/max_arr
    y_test_gr_tensor = torch.from_numpy(y_test_graphing.astype(np.float32))
    Loss=[]
    Loss2=[]
    for epoch in range(num_epochs):
        mini_batches=DataLoader(dataset=dataset,batch_size=mini_batch_size,shuffle=True)
        optimizer=torch.optim.SGD(params=test.parameters(),lr=(0.01)*(1.0001-epoch/num_epochs),momentum=0.1,weight_decay=0.001)
        for mini_batch_in,mini_batch_out in mini_batches:
            outs=test(mini_batch_in)
            l=loss(outs,mini_batch_out)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        if((epoch+1) % 10 == 0):
            print(f'Epoch {epoch+1} complete.')

    X_test_tensor=torch.from_numpy(X_test_new.astype(np.float32))
    
    with torch.no_grad():
        predictions_tensor=test(X_test_tensor)
    
    predictions=predictions_tensor.numpy()

    predictions = predictions*max_arr
    
    mse=np.mean((predictions-y_test_new)**2,axis=0)

    scores=mse/baselines

    final_score = np.mean(scores)
    
    scores_dict['fin_score'].append(final_score)
    for score, class_name in zip(scores, ["P", "K", "Mg", "pH"]):
        scores_dict[class_name].append(score)



# %%
print(scores_dict)

# %%
plt.plot(kernels,scores_dict['fin_score'],linewidth=2.5)
plt.ylabel('Score')
plt.xlabel('Number of features')
plt.grid()
plt.xticks([5,10,20,30,40,50])
plt.savefig('conv_net_neurons.eps', bbox_inches='tight',pad_inches = 0)
plt.show()
# %%


