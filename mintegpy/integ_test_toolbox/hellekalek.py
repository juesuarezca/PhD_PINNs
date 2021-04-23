"""
The Hellekalek function
"""
import numpy as np

def get_hellekalek(alpha):
    print(f"Note: hellekalek used. alpha = {alpha}")
    gamma = np.sqrt(alpha**2/((2*alpha + 1)*(alpha+1)**2))
    def hellekalek(x):
        facs = (x**alpha - 1/(alpha+1))/gamma
        return np.prod(facs,axis=-1)
    return hellekalek
