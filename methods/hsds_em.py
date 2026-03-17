#%%

__all__ = [
    'HSDS_EM',
]

import os

from typing import List, Optional
from numpy.typing import NDArray

import attr
import numpy as np
import pandas as pd

from dawid_skene_crowdkit_v140 import DawidSkene
from crowdkit.aggregation.base import BaseClassificationAggregator
from crowdkit.aggregation.utils import get_most_probable_labels, named_series_attrib


class HSDS_EM():
    r"""
    HSDS-EM crowd-kit like interface
    """

    def __init__(self, n_iter=10e8, r=0.75) -> None:
        self.step1_ds = DawidSkene(n_iter=int(n_iter))
        self.step2_ds = DawidSkene(n_iter=int(n_iter), initial_error_strategy="assign")
        self.r = r

    def fit_predict(self, human: pd.DataFrame, ai: pd.DataFrame) -> pd.Series:
        self.step1_ds.fit(human)
        step1_errors = self.step1_ds.errors_
        df = pd.concat([human, ai], axis=0)
        errors = self.generate_default_errors(df)
        errors.update(step1_errors)
        worker_sums = errors.groupby("worker", sort=False).transform("sum")
        errors = errors.div(worker_sums)
        labels = self.step2_ds.fit_predict(df, initial_error=errors)
        return labels

    def generate_default_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        n_labels = len(df['label'].unique())
        workers = df['worker'].unique()
        labels = list(range(n_labels))
        index = pd.MultiIndex.from_tuples(
            [(worker, label) for worker in workers for label in labels],
            names=['worker', 'label']
        )
        data = []
        for _ in workers:
            for true_label in labels:
                row = [(1 - self.r) / (n_labels - 1)] * n_labels
                row[true_label] = self.r
                data.append(row)
        columns = list(range(n_labels))
        default_errors = pd.DataFrame(data, index=index, columns=columns)
        return default_errors
        