import numpy as np

def recomposer(R:np.ndarray, Rd:np.ndarray, T:np.ndarray):
    output = np.zeros_like(R)
    for i in range(R.shape[2]):
        output[:,:,i] = R[:,:,i]*T + Rd[:,:,i]*(1-T)
    return output