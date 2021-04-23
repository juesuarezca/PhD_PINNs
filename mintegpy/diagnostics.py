"""
some tools for diagnostics of numerical functions
"""
import numpy as np

def count(fn):
    """ simple counting decorator

    Counts the calls of a function as well as the elements of an input numpy array
    """
    def wrapper(x,*args, **kwargs):
        if isinstance(x,np.ndarray):
            wrapper.called+=x.shape[-1]
        else:
            wrapper.called+= 1
        return fn(x,*args, **kwargs)
    wrapper.called= 0
    wrapper.__name__= fn.__name__
    return wrapper


class count_class(object):

    def __init__(self,axis=None):
        """
        decorator which counts the number of points, where a function is evaluated.

        Parameters
        ----------
        axis : int
            If the input of the function is a numpy.ndarray, the evaluated points are given along <axis> of the input.
        """
        self.axis = axis # axis along the points, NOT the coordinates

    def __call__(self, f):
        def wrapped_f(x,*args,**kwargs):
            if isinstance(x,np.ndarray):
                if self.axis is None:
                    wrapped_f.called+=1
                else:
                    wrapped_f.called+=x.shape[self.axis]
            else:
                wrapped_f.called+= 1
            return f(x,*args, **kwargs)
        wrapped_f.called= 0
        wrapped_f.__name__= f.__name__
        return wrapped_f
