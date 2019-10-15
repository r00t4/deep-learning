import pandas as pd
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

class Dataset(Dataset):

	def __init__(self):
		self.csv = pd.read_csv('./train_x.csv')

	def __len__(self):
		return len(self.csv)

	def __getitem__(self, ind):
		# X = self.csv.iloc[ind][self.csv.columns != "Survived"].values
		X = self.csv.iloc[ind][self.csv.columns.difference(["Survived"])].values
		Y = self.csv.iloc[ind]["Survived"]
		# print(Y)
		X = torch.from_numpy(X).float()
		# print(X)
		return X, np.float32(Y)

