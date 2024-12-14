import pandas as pd
import numpy as np

def drawdown(r: pd.Series) -> pd.DataFrame:
    """
    :param r: time series of asset returns to compute drawdown
    :return: data frame with wealth index, previous peaks, percent drawdowns
    """
    wealth_index = 1000*(1+r).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame( {
        'Wealth': wealth_index,
        'Peaks': previous_peaks,
        'Drawdown': drawdowns
    })

def get_ffme_returns():
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv',
                       header=0, index_col=0, parse_dates=True, na_values=-99.99)
    rets = me_m [['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    return rets

