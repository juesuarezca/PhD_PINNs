"""
ROOSARNOLD
"""
import numpy as np

def get_roosarnold2():
    e = 1
    def roosarnold2(x):
        d = x.shape[-1]
        nu = (4.0/3.0)**d - 1.0
        facs = np.abs(4*x - 2)
        return (np.prod(facs,axis=-1) - e)/(np.sqrt(nu))
    return roosarnold2

if __name__=='__main__':
    test_func = get_roosarnold2()
    test_x = np.linspace(0,1,100).reshape(5,20)
    test_vals = test_func(test_x)
    print(test_vals.shape)
    print(test_vals)
