import pandas as pd

class Dataloader:

    def events2012(self):
        df = pd.read_csv('datasets/event2012.csv',sep=',', parse_dates=['created_at'])
        df.set_index('created_at')
        return df
    
    def window_making(self, window_size, df):
        time_windows = df.groupby(pd.Grouper(key='created_at', freq=f'{window_size}Min'))
        return time_windows
