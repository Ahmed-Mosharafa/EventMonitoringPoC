import pandas as pd

class Dataloader:  
    '''
    Class for loading and processing tweet data, specifically designed for the 2012 events dataset.
    '''
    def events2012(self):
        '''
        Loads the 2012 events dataset from a CSV file.

        () -> pd.DataFrame
        A pandas DataFrame containing the data from 'datasets/event2012.csv', with 'created_at' parsed as datetime.
        '''
        df = pd.read_csv('datasets/event2012.csv',sep=',', parse_dates=['created_at'])
        df.set_index('created_at')
        return df
    
    def window_making(self, window_size, df):
        '''
        Groups the DataFrame into time windows based on a specified window size.

        (window_size: int, df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy
        A DataFrameGroupBy object with tweets grouped into windows of 'window_size' minutes.
        '''
        time_windows = df.groupby(pd.Grouper(key='created_at', freq=f'{window_size}Min'))
        return time_windows
