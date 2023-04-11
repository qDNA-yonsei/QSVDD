# Implementation of Quantum Convolutional Neural Network (QCNN) circuit structure.

import pennylane as qml
import unitary
import embedding

def conv_layer_1(U, params):
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1])
    U(params, wires=[7,0])

def conv_layer_2(U, params):
    U(params, wires=[2,4])
    U(params, wires=[6,0])
    U(params, wires=[0,2])
    U(params, wires=[4,6])
    
def conv_layer_3(U, params):
    U(params, wires=[2,6])

def QCNN_structure_without_pooling(U, params, U_params):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 4 * U_params]
    param5 = params[4 * U_params: 5 * U_params]
    # param6 = params[5 * U_params: 6 * U_params]
    # param7 = params[6 * U_params: 7 * U_params]
    # param8 = params[7 * U_params: 8 * U_params]

    conv_layer_1(U, param1)
    conv_layer_1(U, param2)
    conv_layer_2(U, param3)
    conv_layer_2(U, param4)
    conv_layer_3(U, param5)
    


def QSVDD_VQOCC(U, params, U_params):
    param = params[0:U_params]
    unitary.U_VQOCC(param)


    
    
dev = qml.device('default.qubit', wires = 8)
@qml.qnode(dev)


def QCNN(X, params, U, U_params, embedding_type, cost_fn, measure_axis):


    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)

    # Quantum Convolutional Neural Network
    if U == 'U_VQOCC':
        QSVDD_VQOCC(unitary.U_VQOCC, params, U_params)
    elif U == 'U_SU4_no_pooling':
        QCNN_structure_without_pooling(unitary.U_SU4, params, U_params)
    else:
        print("Invalid Unitary Ansatze")
        return False
    
    # Measurement in Z bases(computational basis)
    if measure_axis == 'Z':
        if cost_fn == 'qae':
            # result = qml.expval(qml.PauliZ(4))
            # result = [qml.expval(qml.PauliZ(i)) for i in (0,1,2,3,5,6,7)]
            result = [qml.expval(qml.PauliZ(i)) for i in (0,1,2,3,4,5)]
        elif cost_fn == 'svdd':
            # result = [qml.expval(qml.PauliZ(i)) for i in (0,1,2,3,4,5)]
            # result = [qml.expval(qml.PauliX(4)),qml.expval(qml.PauliY(4)),qml.expval(qml.PauliZ(4))]
            # result = [qml.expval(qml.PauliX(0)),qml.expval(qml.PauliY(0)),qml.expval(qml.PauliZ(0)),
            #           qml.expval(qml.PauliX(1)),qml.expval(qml.PauliY(1)),qml.expval(qml.PauliZ(1)),
            #           qml.expval(qml.PauliX(2)),qml.expval(qml.PauliY(2)),qml.expval(qml.PauliZ(2)),
            #           qml.expval(qml.PauliX(3)),qml.expval(qml.PauliY(3)),qml.expval(qml.PauliZ(3)),
            #           qml.expval(qml.PauliX(4)),qml.expval(qml.PauliY(4)),qml.expval(qml.PauliZ(4)),
            #           qml.expval(qml.PauliX(5)),qml.expval(qml.PauliY(5)),qml.expval(qml.PauliZ(5))]
            
                     
            # result = [qml.expval(qml.PauliX(3) @ qml.PauliZ(4)), qml.expval(qml.PauliY(3) @ qml.PauliX(4)), 
            #           qml.expval(qml.PauliZ(3) @ qml.PauliY(4)),
            #           qml.expval(qml.PauliX(3) @ qml.PauliY(4)), qml.expval(qml.PauliY(3) @ qml.PauliZ(4)), 
            #           qml.expval(qml.PauliZ(3) @ qml.PauliX(4)),
            result = [qml.expval(qml.PauliX(6)),qml.expval(qml.PauliY(6)),qml.expval(qml.PauliZ(6)), 
                      qml.expval(qml.PauliX(7)),qml.expval(qml.PauliY(7)),qml.expval(qml.PauliZ(7)),
                      qml.expval(qml.PauliX(6) @ qml.PauliX(7)), qml.expval(qml.PauliY(6) @ qml.PauliY(7)), 
                      qml.expval(qml.PauliZ(6) @ qml.PauliZ(7)),
                      qml.expval(qml.PauliX(6) @ qml.PauliY(7)), qml.expval(qml.PauliY(6) @ qml.PauliZ(7)), 
                      qml.expval(qml.PauliZ(6) @ qml.PauliX(7)),
                      qml.expval(qml.PauliX(6) @ qml.PauliZ(7)), qml.expval(qml.PauliY(6) @ qml.PauliX(7)), 
                      qml.expval(qml.PauliZ(6) @ qml.PauliY(7))]
            
        return result
        # return np.sum(np.array(result))
    # Measurement in X bases(Hadamard basis)
    if measure_axis == 'X':
        if cost_fn == 'mse':
            result = qml.expval(qml.PauliX(4))
            # result = qml.expval(qml.PauliZ(4))**2 + qml.expval(qml.PauliX(4))**2 + qml.expval(qml.PauliY(4))**2
            # qml.Hadamard(wires=4)
            # result = qml.expval(qml.PauliZ(4))
        elif cost_fn == 'cross_entropy':
            qml.Hadamard(wires=4)
            result = qml.probs(wires=4)
        return result
    # Measurement in Y bases(circular basis)
    if measure_axis == 'Y':
        if cost_fn == 'mse':
            result = qml.expval(qml.PauliY(4))
            # qml.adjoint(qml.S(wires=4))
            # qml.Hadamard(wires=4)
            # result = qml.expval(qml.PauliZ(4))
        elif cost_fn == 'cross_entropy':
            qml.adjoint(qml.S(wires=4))
            qml.Hadamard(wires=4)
            result = qml.probs(wires=4)
        return result



