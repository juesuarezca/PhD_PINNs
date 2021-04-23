"""
data path connection
"""
import os
import pathlib

class _datapath(object):
    """
    collects the paths to the default data directory.

    todo ..
        -   build custom data path interface
    """
    def __init__(self):

        #directory of this file
        self.__current_dir = pathlib.Path(os.path.dirname(__file__))

        #direcotry of the package package
        self.__package_dir = self.__current_dir.parent.parent

        #raw data
        self.__dir_name_raw = os.path.join(self.__package_dir, "data/raw")

        #preprocessed
        self.__dir_name_preprocessed = os.path.join(self.__package_dir, "data/preprocessed")

        #interim
        self.__dir_name_interim = os.path.join(self.__package_dir, "data/interim")

        #external
        self.__dir_name_external = os.path.join(self.__package_dir, "data/external")

        #points_weights
        self.__dir_name_points_weights = os.path.join(self.__package_dir, "data/points_weights")

    @property
    def current_dir(self):
        return self.__current_dir


    @property
    def package_dir(self):
        return self.__package_dir


    @property
    def data_raw(self):
        return self.__dir_name_raw


    @property
    def data_preprocessed(self):
        return self.__dir_name_preprocessed


    @property
    def data_interim(self):
        return self.__dir_name_interim


    @property
    def data_external(self):
        return self.__dir_name_external

    @property
    def data_points_weights(self):
        return self.__dir_name_points_weights

    def get_path_from_raw(self,*file_path,**kwargs):
        temp_path = os.path.join(self.data_raw,*file_path)
        if 'suffix' in kwargs.keys():
            return ".".join([temp_path,kwargs['suffix']])
        else:
            return temp_path

    def get_path_from_interim(self,file_path):
        return os.path.join(self.data_interim,file_path)

    def get_path_from_preprocessed(self,file_path):
        return os.path.join(self.data_preprocessed,file_path)

    def get_path_from_external(self,file_path):
        return os.path.join(self.data_external,file_path)

    def get_path_from_points_weights(self,file_path):
        return os.path.join(self.data_points_weights,file_path)

    def info(self):
        print("package dir:",self.package_dir)
        print("module dir:",self.current_dir)
        print("raw data dir:",self.data_raw)
        print("preprocessed data dir:",self.data_preprocessed)
        print("interim data dir:",self.data_interim)
        print("external data dir:",self.data_external)
        print("points_weights data dir:",self.data_points_weights)

datapath = _datapath()
