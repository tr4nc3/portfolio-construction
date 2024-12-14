import pandas as pd
import numpy as np

def get_ffme_returns():
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv',
                       header=0, index_col=0, parse_dates=True, na_values=-99.99)
    me_m / 100