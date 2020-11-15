import numpy as np
import os
import pandas as pd
from itertools import product
from os.path import basename, dirname, exists, join, splitext


__all__ = ['DesignMatrix']


class DesignMatrix(pd.DataFrame):

    # temporary properties
    #_internal_names = pd.DataFrame._internal_names + ['internal_cache']
    #_internal_names_set = set(_internal_names)

    # normal properties
    #_metadata = ['basemodel', 'model', 'coder', 'run']

    @property
    def _constructor(self):
        return DesignMatrix

    @property
    def S(self):
        """Reshape data to (samples, 28, 28, 1)."""
        return super().to_numpy().reshape(len(self), 28, 28, 1)

    @property
    def I(self):
        """Interpolate over +-inf and nan."""
        pd.options.mode.use_inf_as_na = True
        i = self.interpolate(axis='index')
        pd.options.mode.use_inf_as_na = False
        return i
    
    def to_csv(self, path_or_buf: str, **kwargs):
        super().to_csv(path_or_buf, **kwargs)
        pathmeta = splitext(path_or_buf)[0]+'.meta'
        if not exists(pathmeta):
            pd.DataFrame(self.attrs, index=[0]).T.to_csv(pathmeta, header=False)

    def read_csv(path_or_buf: str, **kwargs):
        if 'index_col' not in kwargs:
            kwargs['index_col'] = 0
        df = pd.read_csv(path_or_buf, **kwargs)
        dm = DesignMatrix(df)
        pathmeta = splitext(path_or_buf)[0]+'.meta'
        if os.stat(pathmeta).st_size > 0:
            dm.attrs.update(pd.read_csv(pathmeta, index_col=0, header=None)[1])
        return dm
    
    def makeindex(num, codes=['ae', 'enc', 'dec']):
        """Generate Multiindex with initial values at epoch 0 and epochwise codes."""
        index = [[0, 'x'], [0, 'z']] + list(product(range(num), codes))
        index = pd.MultiIndex.from_tuples(index, names=['epoch','code'])
        return index
