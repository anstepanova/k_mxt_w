from abc import abstractmethod, ABC
import pandas as pd
import datetime
import numpy as np
import re
import logging


from typing import List, Optional

logger = logging.getLogger('k_mxt_w.data_import')


class DataImport(ABC):
    """
    Base class which import data using pandas
    """
    _dataframe = None

    @abstractmethod
    def __init__(self, filename: str, sep: str = ','):
        """
        :param filename: filename of dataframe
        :param sep: dataframe separator
        """
        self.filename = filename
        self._sep = sep

    def _read_data(self):
        """
        read data from csv-file
        :return: None
        """
        self._dataframe = pd.read_csv(self.filename, sep=self._sep, low_memory=False)


class DataPropertyImportSpace(DataImport):
    """
    Class which import geo data using pandas
    """
    def __init__(self, filename, sep=','):
        """
        :param filename: filename of dataframe
        :param sep: dataframe separator
        """
        super().__init__(filename, sep)
        logger.info(f'filename-{self.filename}, sep-{sep}')
        self._read_data()

    def get_data(self,
                 name_latitude_cols: str = 'latitude',
                 name_longitude_cols: str = 'longitude',
                 features_list=Optional[List[str]]):
        """

        :param name_latitude_cols: name of column containing latitude
        :param name_longitude_cols: name of column containing longitude
        :param features_list: list of column names which contain other features
        :return:
        """
        x = self._dataframe[name_latitude_cols].to_numpy(dtype=np.float)
        y = self._dataframe[name_longitude_cols].to_numpy(dtype=np.float)
        return x, y, self._dataframe[features_list].to_numpy(dtype=np.float)


class DataSave:
    """
    Class which save data using pandas
    """
    @classmethod
    def arrays_to_csv(cls, source_filename, new_filename, **kwargs):
        """
        give data from source dataframe and save them and arrays in new dataframe
        :param source_filename: filename of source dataframe
        :param new_filename: filename of new dataframe
        :param kwargs: arrays
        :return:
        """
        print(kwargs)
        df = pd.read_csv(source_filename)
        new_df = pd.DataFrame.from_dict(kwargs)
        res = pd.concat([df, new_df], axis=1)
        res.to_csv(new_filename)
