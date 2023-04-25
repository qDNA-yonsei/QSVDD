# Implementation of Quantum circuit training procedure

import QCNN_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp
import torch
from torch.optim import Adam, Adagrad


def qae_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (-np.sum(p))
    loss = loss / len(labels)
    return loss

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss

def svdd_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels,predictions):
        loss = loss + np.sum((p-l)**2)
    loss = loss / len(labels)
    return loss




def cost(params, X, Y, U, U_params, embedding_type, circuit, latent_dim, cost_fn, measure_axis):
    if circuit == 'QCNN':
        predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, latent_dim, cost_fn, measure_axis=measure_axis) for x in X]
    if cost_fn == 'qae':
        loss = qae_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)
    elif cost_fn == 'svdd':
        loss = svdd_loss(Y, predictions)
    return loss

# Circuit training parameters

steps = 500
learning_rate = 0.001
batch_size = 16


def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit, cost_fn, measure_axis, latent_dim):
    if (circuit == 'QCNN')&(U != 'U_SU4_no_pooling'):
        total_params = U_params * 3 + 2 * 3
    elif (circuit == 'QCNN')&(U == 'U_SU4_no_pooling'):
        total_params = U_params * 5

    elif circuit == 'Hierarchical':
        total_params = U_params * 7
        
    init_params = np.random.randn(total_params, requires_grad = True)
    params = init_params
    init_center = np.mean([QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, latent_dim, cost_fn, measure_axis=measure_axis) for x in X_train])
    Y_train = np.tile(init_center, (len(X_train), 1))
    
    # Optimizer method
    opt = qml.AdamOptimizer(stepsize=learning_rate)    
    # opt = qml.NesterovMomentumOptimizer(stepsize = learning_rate, momentum=0.9)
    param_history= [params]
    loss_history = []


    for it in range(steps):

        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, latent_dim, cost_fn, measure_axis), params)

        param_history.append(params)        
        loss_history.append(cost_new)

        if it % 5 == 0:
            print("iteration: ", it, " cost: ", cost_new)


    return loss_history, params, param_history, init_center, Y_train


