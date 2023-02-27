from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from multiprocessing import cpu_count


def compute_lagging_regression(time_serie: pd.Series, window: int = 15) -> pd.DataFrame:
    """Compute a lagging moving regression on a column with a window.

    Args:
        time_serie (pd.Series): The column we apply Linear regression on.
        window (int, optional): The window we apply linear regression on. Defaults to 15.

    Returns:
        pd.DataFrame: A pandas Series with the calculated indicator.
    """

    def compute_regression(y_value: np.array) -> float:
        """Compute simple linear regression between on a vector

        Args:
            y (np.array): y vector

        Returns:
            float: The coefficient a corresponding to the linear regression y=ax+b.
        """
        x = np.arange(len(y_value)).reshape(-1, 1)
        model = LinearRegression(n_jobs=cpu_count()).fit(x, y_value.values)
        return model.coef_[0]

    return (
        time_serie.rolling(window).apply(compute_regression).rename("Regression_coef")
    )


def heikin_ashi_candlestick(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Heikin Ashi Candle on a regular dataframe containing at least 4 columns, named exactly : Open, Close, High, Low.

    Args:
        df (pd.DataFrame): The dataframe that will be modified

    Returns:
        pd.DataFrame: The initial dataframe with the Heikin Ashi Candle.
    """
    df["HA_Close"] = (df.Open + df.High + df.Low + df.Close) / 4
    ha_open = [(df.Open[0] + df.Close[0]) / 2]
    [
        ha_open.append((ha_open[i] + df.HA_Close.values[i]) / 2)
        for i in range(0, len(df) - 1)
    ]
    df["HA_Open"] = ha_open
    df["HA_High"] = df[["HA_Open", "HA_Close", "High"]].max(axis=1)
    df["HA_Low"] = df[["HA_Open", "HA_Close", "Low"]].min(axis=1)
    return df


def compute_date_features(
    date_col: pd.Series | pd.DatetimeIndex, sin_cos: bool = True
) -> pd.DataFrame:
    """Generate date features from a dataframe.

    Args:
        date_col (pd.Series | pd.DatetimeIndex): _description_
        sin_cos (bool, optional): Whether to add `df['Day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))`,`df['Day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))`,`df['Year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))`,`df['Year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))` to the features.  . Defaults to True.

    Returns:
        pd.DataFrame: A dataFrame containing all the date features.
    """

    day = 24 * 60 * 60
    year = (365.2425) * day

    df = pd.DataFrame()

    df["Day"] = date_col.day
    df["Month"] = date_col.month
    df["Year"] = date_col.year
    df["Day_week"] = date_col.day_of_week
    df["Week"] = date_col.week
    df["Hour"] = date_col.hour
    if sin_cos is True:
        timestamp_s = date_col.map(pd.Timestamp.timestamp).to_list()
        df["Day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
        df["Day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))
        df["Year_sin"] = np.sin(timestamp_s * (2 * np.pi / year))
        df["Year_cos"] = np.cos(timestamp_s * (2 * np.pi / year))
    return df
