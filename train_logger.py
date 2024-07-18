import pandas as pd
import os


class Logger:
    def __init__(self, path):
        self.path = path
        df = self.load()
        if df is not None:
            self.df = df
        else:
            self.df = pd.DataFrame(columns=['epoch'])

    def load(self):
        if os.path.exists(self.path):
            return pd.read_csv(self.path)
        else:
            return None

    def save(self):
        self.df.to_csv(self.path, index=False)

    def log_epoch(self, epoch, metrics, save=True):
        '''
        Log epoch metrics to the dataframe

        Parameters
        ----------
        epoch : int
            The epoch number
        metrics : dict
            The metrics to log
        '''

        new_column = pd.DataFrame({'epoch': epoch, **metrics}, index=[0])
        self.df = pd.concat([self.df, new_column], axis=0, ignore_index=True)

        if save:
            self.save()
