# Implementation of Quantum circuit training procedure

import QCNN_circuit
# import Hierarchical_circuit

import pennylane as qml
#import numpy as np
from pennylane import numpy as np

import autograd.numpy as anp
import torch
from torch.optim import Adam, Adagrad


def qae_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        # loss = loss + (l - p) ** 2
        # loss = loss + (l - np.sum(p))
        loss = loss + (-np.sum(p))
    loss = loss / len(labels)
    return loss

def svdd_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels,predictions):
        loss = loss + np.sum((p-l)**2)
        # loss = loss + np.sum(p)
        
        # loss = loss + np.sqrt(np.sum(p-l)**2 + 1) -1
    loss = loss / len(labels)
    return loss




def cost(params, X, Y, U, U_params, embedding_type, circuit, cost_fn, measure_axis):
    if circuit == 'QCNN':
        predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn, measure_axis=measure_axis) for x in X]
    # elif circuit == 'Hierarchical':
    #     predictions = [Hierarchical_circuit.Hierarchical_classifier(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]

    if cost_fn == 'qae':
        loss = qae_loss(Y, predictions)

    elif cost_fn == 'svdd':
        loss = svdd_loss(Y, predictions)
    return loss

# Circuit training parameters

steps = 2000
learning_rate = 0.001
batch_size = 16


def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit, cost_fn, measure_axis):
    # if (circuit == 'QCNN')&(U != 'U_SU4_no_pooling'):
    #     total_params = U_params * 3 + 2 * 3
    if (circuit == 'QCNN')&(U == 'U_SU4_no_pooling'):
        total_params = U_params * 5 # 75 params for QSVDD
    elif (circuit == 'QCNN')&(U == 'U_VQOCC'):
        # total_params = U_params * 5 # 40 params for QSVDD_VQOCC
        total_params = U_params # 78 params for QSVDD_VQOCC
        
        
    ## Initializing parameter
    init_params = np.random.randn(total_params, requires_grad = True)
    params = init_params




    # Optimizer method
    opt = qml.AdamOptimizer(stepsize=learning_rate)    
#    opt = qml.NesterovMomentumOptimizer(stepsize = learning_rate, momentum=0.9)
    param_history= [params]
    loss_history = []


    for it in range(steps):

        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn, measure_axis), params)

        param_history.append(params)        
        loss_history.append(cost_new)

        if it % 5 == 0:
            print("iteration: ", it, " cost: ", cost_new)


    return loss_history, params, param_history


