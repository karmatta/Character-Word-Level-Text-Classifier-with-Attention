# Title: Set of utility/wrapper functions for model development
# Author: Karthik D
# -----------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report


# get the count and % distribution of DVs
def get_dv_dist(data, col=None, dv=None):
    # If DV is one_hot encoded
    if col is None and dv is None:
        DV_dist = pd.DataFrame({'counts':data.sum(0), "perc(%)":data.sum(0)/data.sum().sum()*100}).T
        return DV_dist.round(1)
    else: # If DV is a data frame with a categorical var
        DV_dist = data[[col, dv]].groupby(dv).agg('count')
        DV_dist.columns = ['counts']
        DV_dist['perc (%)'] = round(DV_dist/DV_dist.sum(axis=0)*100, 2)
        return DV_dist.T
    

# Get model metrics and predictions
def get_model_metrics(model, x, y):
    y_scores = model.predict(x, batch_size=64, verbose=1)
    y_pred = np.argmax(y_scores, axis=1)
    y_act_class = np.argmax(y, axis=1)
    # get 2nd predictions
    p2 = []
    scores2=[]
    for row in y_scores:
        p2.append(row.argsort()[-2:][::-1])
        scores2.append(row[p2[0]])
        
    scores1 = np.stack(scores2)[:,0]
    scores2 = np.stack(scores2)[:,1]
    y_2nd_pred = np.stack(p2)[:,1]
    
    met = pd.DataFrame(classification_report(y_act_class, y_pred, output_dict=True))
    met = met.drop("macro avg", axis=1).drop('weighted avg', axis=1).T
    return met.drop('support', axis=1), y_pred, scores1, y_2nd_pred, scores2



# plot model training history
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    
# integer encode
def onehot_encode(dv):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(dv)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return list(onehot_encoded)
