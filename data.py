from abc import abstractmethod, ABC
import pandas as pd
import datetime
import numpy as np
import re
import logging


logger = logging.getLogger('k_mxt_w.data_import')


class DataImport(ABC):
    _dataframe = None

    @abstractmethod
    def __init__(self, filename, sep=','):
        self.filename = filename
        self._sep = sep

    def _read_data(self):
        self._dataframe = pd.read_csv(self.filename, sep=self._sep, low_memory=False)


class DataPropertyImportSpace(DataImport):
    def __init__(self, filename, sep=','):
        super().__init__(filename, sep)
        logger.info(f'filename-{self.filename}, sep-{sep}')
        self._read_data()

    def get_data(self, name_latitude_cols='latitude', name_longitude_cols='longitude', features_list=None):
        x = self._dataframe[name_latitude_cols].to_numpy(dtype=np.float)
        y = self._dataframe[name_longitude_cols].to_numpy(dtype=np.float)
        return x, y, self._dataframe[features_list].to_numpy(dtype=np.float)


class DataSave:
    @classmethod
    def arrays_to_csv(cls, source_filename, new_filename, **kwargs):
        print(kwargs)
        df = pd.read_csv(source_filename)
        new_df = pd.DataFrame.from_dict(kwargs)
        res = pd.concat([df, new_df], axis=1)
        res.to_csv(new_filename)
