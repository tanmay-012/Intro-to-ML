import numpy as np
import sklearn
from sklearn.svm import LinearSVC

def return_features(Z_train_mod, mux1, mux2):
  mask1 = (Z_train_mod[:,-1] == mux2) & (Z_train_mod[:,-2] == mux1)
  mask2 = (Z_train_mod[:,-1] == mux1) & (Z_train_mod[:,-2] == mux2)
  x_new = np.concatenate((Z_train_mod[mask1][:,:64], Z_train_mod[mask2][:,:64]), axis = 0)
  y_new = np.concatenate((Z_train_mod[mask1][:,-3], 1-Z_train_mod[mask2][:,-3]), axis = 0) 
  data_mod = np.concatenate((x_new, y_new.reshape(-1,1)), axis = 1)
  return data_mod

def my_fit( Z_train ):
  mux2 = Z_train[:,-5:-1]
  mux1 = Z_train[:,-9:-5]
  bits = np.array([8,4,2,1])
  xorro_mux1 = np.sum(np.multiply(mux1, bits), axis = 1)
  xorro_mux2 = np.sum(np.multiply(mux2, bits), axis = 1)
  xorro_mux1 = xorro_mux1.reshape(-1,1)
  xorro_mux2 = xorro_mux2.reshape(-1,1)
  Z_train_mod = np.concatenate((Z_train, xorro_mux1, xorro_mux2), axis = 1)
  model = np.zeros((120,65))
  cnt = 0
  for i in range(0,16):
    for j in range(i+1,16):
      data_mod = return_features(Z_train_mod, i, j)
      clf = LinearSVC(loss = "squared_hinge", penalty = "l2", dual = False)
      clf.fit(data_mod[:,:-1], data_mod[:,-1])
      model[cnt][:64] = clf.coef_.reshape(-1)
      model[cnt][-1] = clf.intercept_
      cnt += 1
  return model					# Return the trained model

def my_predict( X_tst, model ):
  X_tst = X_tst.astype(int)
  mux2 = X_tst[:,-4:]
  mux1 = X_tst[:,-8:-4]
  bits = np.array([8,4,2,1])
  xorro_mux1 = np.sum(np.multiply(mux1, bits), axis = 1)
  xorro_mux2 = np.sum(np.multiply(mux2, bits), axis = 1)
  xorro_mux1 = xorro_mux1.reshape(-1,1)
  xorro_mux2 = xorro_mux2.reshape(-1,1)
  (row, col) = X_tst.shape
  y = np.zeros((row,1))
  X_tst = np.concatenate((X_tst, xorro_mux1, xorro_mux2, y), axis = 1)
  cnt = 0
  for i in range(0,16):
    for j in range(i+1,16):
      # Direct
      mask1 = (X_tst[:,-3] == i) & (X_tst[:,-2] == j)
      X = X_tst[mask1][:,:64]
      y = np.dot(X,model[cnt][:-1].reshape((-1,1))) + model[cnt][-1]
      y[y > 0] = 1
      y[y < 0] = 0
      y = y.astype(int)
      X_tst[mask1,-1] = y.flatten()
      # Flipped 
      mask2 = (X_tst[:,-3] == j) & (X_tst[:,-2] == i)
      X = X_tst[mask2][:,:64]
      y = np.dot(X,model[cnt][:-1].reshape((-1,1))) + model[cnt][-1]
      y[y > 0] = 1
      y[y < 0] = 0
      y = y.astype(int)
      X_tst[mask2,-1] = (1-y).flatten()
      cnt += 1
  pred = X_tst[:,-1]
  return pred