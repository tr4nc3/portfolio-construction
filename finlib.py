import pandas as pd
import numpy as np
import scipy
import scipy.stats

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

def get_ffme_returns() -> pd.DataFrame:
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv',
                       header=0, index_col=0, parse_dates=True, na_values=-99.99)
    rets = me_m [['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    return rets

def get_hfi_returns() -> pd.DataFrame:
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv',header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi

def skewness(r: pd.Series) -> pd.Series:
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp/sigma_r**3

def kurtosis(r: pd.Series) -> pd.Series:
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp/sigma_r**4

def is_normal(r: pd.Series, level :float = 0.01 ) -> bool:
    '''
    Applies Jarque Bera test
    :param r:
    :param level:
    :return:
    '''
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def semi_deviation(r: pd.Series) -> pd.Series:
    is_negative = r < 0
    return r[is_negative].std()

def var_historic(r: pd.Series, level :int = 5) -> pd.Series:
    """
    VaR historoc
    :param r: Series or dataframe
    :param level:
    :return:  chance of losing level percentage
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('Expected r to be a pd.Series or pd.dataFrame')

from scipy.stats import norm
def var_gaussian(r: pd.Series, level :int = 5) -> pd.Series:
    #if isinstance(r, pd.DataFrame):
    #    return r.aggregate(var_gaussian, level=level)
    #elif isinstance(r, pd.Series):
    z = norm.ppf(level/100)
    return -(r.mean() + z * r.std(ddof=0))


def cf_var(r, z_alpha) -> pd.Series:
    if isinstance(r, pd.Series):
        k = kurtosis(r)
        s = skewness(r)
        z_alpha_bar = z_alpha + (1/6) * (z_alpha ** 2 - 1) * s + (1/24) * (z_alpha ** 3 - 3 * z_alpha) * (k-3)\
                       - (1/36) * (2 * z_alpha ** 3 - 5 * z_alpha) * (s ** 2)
        return -(r.mean() + z_alpha_bar * r.std())
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cf_var, z_alpha, axis='columns')