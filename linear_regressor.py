# %%
import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd


plt.rc('axes',labelsize = 14)



# %%
hsi_path = 'train_data/1121.npz'
gt_path = 'train_gt.csv'
wavelength_path = 'wavelengths.csv'

# %%
gt_df = pd.read_csv(gt_path)
gt_df
wavelength_df = pd.read_csv(wavelength_path)
wavelength_df.head()

# %%
# %%
# %%
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
        data.append(arr/10000)
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
X_train_new=np.array([x for x in Data[:1500,0]])
y_train_new=np.array([y for y in Data[:1500,1]])
X_test_new=np.array([x for x in Data[1500:,0]])
y_test_new=np.array([y for y in Data[1500:,1]])
baseline_reg=BaselineRegressor()
# Fit the baseline regressor once again on new training set
baseline_reg = baseline_reg.fit(X_train_new, y_train_new)
baseline_predictions = baseline_reg.predict(X_test_new)
baselines = np.mean((y_test_new - baseline_predictions) ** 2, axis=0)
baselines

# %%
Linear_regressor=nn.Linear(150,4,bias=True)
loss=nn.MSELoss()
num_epochs=100
mini_batch_size=50
x_train_for_plots = torch.from_numpy(X_train_new.astype(np.float32))
y_train_for_plots = torch.from_numpy(y_train_new.astype(np.float32))
X_test_for_plots = torch.from_numpy(X_test_new.astype(np.float32))
y_test_for_plots = torch.from_numpy(y_test_new.astype(np.float32))
l_vec = []
l2_vec = []
dataset=TensorDataset(torch.from_numpy(X_train_new.astype(np.float32)),torch.from_numpy(y_train_new.astype(np.float32)))
for epoch in range(num_epochs):
    mini_batches=DataLoader(dataset=dataset,batch_size=mini_batch_size,shuffle=True)
    optimizer=torch.optim.SGD(params=Linear_regressor.parameters(),lr=0.1*(1.00001 - epoch/num_epochs),weight_decay=0.001)
    for mini_batch_in,mini_batch_out in mini_batches:
        outs=Linear_regressor(mini_batch_in)
        l=loss(outs,mini_batch_out)
        l.backward()
        optimizer.step()
    with torch.no_grad():
        out = Linear_regressor(x_train_for_plots)
        out2 = Linear_regressor(X_test_for_plots)
        l_vec.append(loss(out,y_train_for_plots))
        l2_vec.append(loss(out2,y_test_for_plots))
    optimizer.zero_grad()

# %%
plt.plot(range(1,101),l_vec,linestyle='dotted',color='b',label='train_data')
plt.plot(range(1,101),l2_vec,color='r',label='test_data',linewidth=2.5)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('regressor_loss.eps', bbox_inches='tight',pad_inches = 0)
plt.show()

# %%
#Predictions of the network:
X_test_tensor=torch.from_numpy(X_test_new.astype(np.float32))
with torch.no_grad():
    predictions_tensor=Linear_regressor(X_test_tensor)
predictions=predictions_tensor.numpy()

# %%
mse=np.mean((predictions-y_test_new)**2,axis=0)

scores=mse/baselines

final_score = np.mean(scores)

for score, class_name in zip(scores, ["P", "K", "Mg", "pH"]):
    print(f"Class {class_name} score: {score}")

print(f"Final score: {final_score}")

# %%

# %%
"""
While the score appears better than the conv-net,the final predictions in the case of the conv-net scores better than the linear regressor.
"""

# %%

