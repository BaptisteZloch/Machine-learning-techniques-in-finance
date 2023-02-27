from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from multiprocessing import cpu_count


def compute_lagging_regression(time_serie: pd.Series, window: int = 15) -> pd.DataFrame:
    """Compute a lagging moving regression on a column with a window.

    Args:
        df (pd.DataFrame): The dataframe containing features.
        col (str, optional): The column we apply Linear regression on. Defaults to "Close.
        window (int, optional): The window we apply linear regression on. Defaults to 15.

    Returns:
        pd.DataFrame: The entry DataFrame we another column called B_MLR_coefs
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
    return time_serie.rolling(window).apply(compute_regression)