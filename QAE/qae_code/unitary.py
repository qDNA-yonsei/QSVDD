# This module contains the set of unitary ansatze that will be used to benchmark the performances of Quantum Convolutional Neural Network (QCNN) in QCNN.ipynb module
import pennylane as qml
from itertools import combinations
def U_VQOCC(params): # params: 14
    nqubits = 8
    ntrash = 6
    nlatent = nqubits - ntrash
    layers = 18
    
    for l in range(layers):
        for i in range(nqubits):
            qml.RY(params[i], wires = i)
            
        for i,j in combinations(range(0, ntrash), 2): # CZ between trash qubits
            qml.CZ(wires = [i, j])
            
        for idx in range(ntrash): # CZ between trash and non-trash qubits
            for i in range(ntrash):
                for j in range(ntrash+i,nqubits,ntrash):
                    qml.CZ(wires = [(idx+i)%(ntrash),j])       


     
    for i in range(ntrash):
        qml.RY(params[i], wires = i)
        
