import numpy as np

def get_stability_2D(eigvals):
    """function to test the stability of a 2D system by evaluating common cases of its eigenvalues """
    stability = 'untested'
    lambda1, lambda2 = eigvals
    if lambda1.real == 0 or lambda2.real == 0:
        stability = 'unknown (Eigenvalue zero)'
    elif lambda1.imag == 0 and lambda1.imag == 0:
        if lambda1 < 0 and lambda2 < 0:
            stability = 'stable node'
        elif lambda1 > 0 and lambda2 > 0:
            stability = 'unstable node'
    else:
        if lambda1 < 0 and lambda2 < 0:
            stability = 'stable spiral'
        elif lambda1 > 0 and lambda2 > 0:
            stability = 'unstable spiral'

    if np.sign(lambda1.real) != np.sign(lambda2.real):
        stability = 'unstable saddle node'

    return stability

def nrmse(reference, x):
    """
    Caclucaltes the nrmse, Normalized Root Mean Squared Error between a reference and an array, x
    :param reference: np.ndarray
        reference to test the input against
    :param x: np.ndarray
        input array to test against reference
    :return: nrmse: np.ndarray
        nrmse
    """
    diff = np.subtract(x, reference)
    square = np.square(diff)
    mse = square.mean()
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(x) - np.min(x))
    return nrmse
