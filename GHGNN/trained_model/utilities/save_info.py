# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:32:17 2021

@author: Edgar Sanchez
"""
import os
import matplotlib.pyplot as plt

def count(func):
    def wrapper(*args, **kwargs):
        wrapper.counter += 1    # executed every time the wrapped function is called
        return func(*args, **kwargs)
    wrapper.counter = 0         # executed only once in decorator definition time
    return wrapper


def save_train_traj(path, df_model_training, valid=True):
    if not os.path.exists(path):
        os.makedirs(path)
    df_model_training.to_csv(path+'/Training.csv')
    
    # Save training curve
    train_loss = df_model_training['Train_loss'].to_list()
    if valid:
        valid_loss = df_model_training['Valid_loss'].to_list()
    
    fig = plt.figure()
    plt.plot(train_loss, label='Training')
    if valid:
        plt.plot(valid_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss function')
    plt.yscale('log')
    plt.legend()
    plt.close(fig)
    fig.savefig(path+'/Training.png')
    