import pennylane.math as np
from pennylane.operation import Channel

class DepolarizingChannel_2(Channel):
    num_params = 1
    num_wires = 2
    grad_method = "A"
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    def __init__(self, p, wires, id=None):
        super().__init__(p, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p):  # pylint:disable=arguments-differ
        if not np.is_abstract(p) and not 0.0 <= p <= 1.0:
            raise ValueError("p must be in the interval [0,1]")

        if np.get_interface(p) == "tensorflow":
            p = np.cast_like(p, 1j)

        I = np.convert_like(np.eye(2, dtype=complex), p)
        X = np.convert_like(np.array([[0, 1], [1, 0]], dtype=complex), p)
        Y = np.convert_like(np.array([[0, -1j], [1j, 0]], dtype=complex), p)
        Z = np.convert_like(np.array([[1, 0], [0, -1]], dtype=complex), p)

        paulis = [I, X, Y ,Z]
        K = []
        for i in range(len(paulis)):
            for j in range(len(paulis)):
                #K.append((probs[i * len(paulis) + j] + np.eps) * np.kron(paulis[i], paulis[j]))
                if i == 0 and j == 0:
                    K.append(np.sqrt(1 - p + np.eps) * np.kron(paulis[i], paulis[j]))
                else :
                    K.append(np.sqrt(p / 15 + np.eps) * np.kron(paulis[i], paulis[j]))

        return K

    # paulis = [
    #     [[1, 0],[0, 1]],    # Identity
    #     [[0, 1],[1, 0]],    # PauliX
    #     [[0, -1j],[1j, 0]], # PauliY
    #     [[1, 0],[0, -1]]    # PauliZ
    # ]

    # @staticmethod
    # def compute_kraus_matrices(p):
    #     error_choice = np.random.choice(16, p=[(1-p)] + [p/15]*15)

    #     if error_choice != 0:  # 0 corresponds to no error due to the (1-p) probability
    #         i, j = divmod(error_choice-1, 4)
    #         return [np.kron(DepolarizingChannel_2.paulis[i], DepolarizingChannel_2.paulis[j])]

    #     return [np.kron(DepolarizingChannel_2.paulis[0], DepolarizingChannel_2.paulis[0])]
