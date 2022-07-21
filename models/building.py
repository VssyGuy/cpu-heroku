import streamlit as st
import pandas as pd
import numpy as np
from data.merged import Data
# from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import shap
import matplotlib.pyplot as plt
from util import *

class Model():
	def __init__(self):
		self.targets = ['singleScore', 'multiScore']
  
	def start(self, input_features=None):
		self.user_input_features = input_features
		self.prepare_data()
		self.train_test_split = {}
		self.feature_selection(0.7)
		self.models = {}
		self.evaluations = {}
		try: self.load()
		except FileNotFoundError: self.build()
	
	def build(self):
		# self.model_selection()
		for target, output in enumerate(['y1_train','y2_train']):
			model = XGBRegressor()
			model.fit(self.train_test_split['X_train'], self.train_test_split[output])
			pickle.dump(model, open(f"models/{self.targets[target]}_model.pkl", "wb"))
			self.models[self.targets[target]] = model
   
	def load(self):
		for target in range(len(self.targets)):
			self.models[self.targets[target]] = pickle.load(open(f"models/{self.targets[target]}_model.pkl", "rb"))
			self.evaluations[self.targets[target]] = pickle.load(open(f"models/metrics/{self.targets[target]}_eval.pkl", "rb"))

	def prepare_data(self):
		# Prepare data for the model
		df = Data().df
		if isinstance(self.user_input_features, pd.DataFrame):
			df = pd.concat([df, self.user_input_features], axis=0)
			df.reset_index(drop=True)
   
		labels = ['type', 'manufacturer']
		df = label_encoder(df, labels)
  
		if isinstance(self.user_input_features, pd.DataFrame):
			self.user_input_features = df.drop(['cpuName', 'singleScore', 'multiScore'], axis=1).tail(1)
			self.df = df[:-1]
		else: self.df = df

	@property
	def inputs(self):
		X = self.df.drop(['cpuName', 'singleScore', 'multiScore'], axis=1)
		return X[:-1]

	@property
	def outputs(self):
		Y1 = self.df[self.targets[0]][:-1]
		Y2 = self.df[self.targets[1]][:-1]
		return [Y1, Y2]

	@property
	def train_test_split(self):
		return self._train_test_split

	@train_test_split.setter
	def train_test_split(self, v):
		# separate dataset into train and test
		X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
			self.inputs,
			self.outputs[0],
			self.outputs[1],
			test_size=0.2,
			random_state=1)
		self._train_test_split = {
			k: v
			for k, v in locals().items() if k not in ['self', 'v']
		}
		
	def feature_selection(self, threshold):
		# Removes highly correlated features
		col_corr = set()  
		corr_matrix = self.train_test_split['X_train'].corr() 
		for i in range(len(corr_matrix.columns)):
			for j in range(i):
				if abs(corr_matrix.iloc[i, j]) > threshold: 
					colname = corr_matrix.columns[i]  
					col_corr.add(colname)
		self.train_test_split['X_train'].drop(col_corr, axis=1, inplace=True)
		self.train_test_split['X_test'].drop(col_corr, axis=1, inplace=True)
		if isinstance(self.user_input_features, pd.DataFrame):
			self.user_input_features.drop(col_corr, axis=1, inplace=True)
	
	def predictions(self):
		preds = []
		for target, output in enumerate(['y1_train','y2_train']):
			model = self.models[self.targets[target]]
			pred = pd.DataFrame(np.round(model.predict(self.user_input_features)).astype(int), columns=[self.targets[target]])
			preds.append(pred)
		return pd.concat(preds, axis=1)

	def compare(self, cpuName):
		df = Data().df[['cpuName','singleScore','multiScore']]
		df.set_index('cpuName', inplace=True)
		if isinstance(self.user_input_features, pd.DataFrame):
			preds = [[self.predictions()['singleScore'][0],
					  self.predictions()['multiScore'][0]]]
			user_input = pd.DataFrame(preds, index = [cpuName], columns = self.targets)
			df = pd.concat([df, user_input], axis=0)
		ss_df = df[['singleScore']].sort_values('singleScore', ascending=False)
		idx = ss_df.index.get_loc(cpuName)
		ss_df = ss_df.iloc[idx - 2 : idx + 3]
		ms_df = df[['multiScore']].sort_values('multiScore', ascending=False)
		idx = ms_df.index.get_loc(cpuName)
		ms_df = ms_df.iloc[idx - 2 : idx + 3]
		return [ss_df, ms_df]

	def explain(self, target, output):
		# Explaining the model's predictions using SHAP
		model = self.models[self.targets[target]]
		X100 = self.train_test_split['X_train'].sample(100, random_state=1)
		explainer = shap.Explainer(model, X100, model_output='margin')
		return explainer.shap_values(X100)
	
	def shap_plot(self):
		for target, output in enumerate(['y1_train','y2_train']):
			fig, ax = plt.subplots()
			plt.title(f'Feature importance ({self.targets[target]}) based on SHAP values')
			shap_values = self.explain(target, self.train_test_split[output])
			shap.summary_plot(shap_values, self.train_test_split['X_train'].sample(100, random_state=1))
			st.pyplot(fig, bbox_inches='tight')

	# def model_selection(self):
	# 	# Defines and builds the lazyclassifier
	# 	X_train, X_test, y1_train, y1_test, y2_train, y2_test = self.train_test_split.values()
	# 	reg1 = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
	# 	models_test1,predictions_test1 = reg1.fit(X_train, X_test, y1_train, y1_test)
	# 	reg2 = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
	# 	models_test2,predictions_test2 = reg2.fit(X_train, X_test, y2_train, y2_test)

	# 	print(models_test1.head(5), models_test2.head(5))
	# 	# Selects the best models according to Adjusted R-Squared,  R-Squared, RMSE and Time Taken
	# 	self.evaluations = {'singleScore': models_test1.iloc[[3]], 
	# 						'multiScore': models_test2.iloc[[0]]}
	# 	pickle.dump(models_test1.iloc[[3]], open(f"models/metrics/{self.targets[0]}_eval.pkl", "wb"))
	# 	pickle.dump(models_test2.iloc[[0]], open(f"models/metrics/{self.targets[1]}_eval.pkl", "wb"))