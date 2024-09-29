import numpy as np
from esda.moran import Moran
from spglm.family import Gaussian
from sklearn.metrics import mean_squared_error, r2_score


# compute the AIC score
def AIC(y_true, y_pred, k, family=Gaussian()):
    return -2.0 * family.loglike(y_true, y_pred) + 2.0 * k


# compute the AICc score
def AICc(y_true, y_pred, k, family=Gaussian()):
    n = y_pred.shape[0]
    return AIC(y_true, y_pred, k, family=family) + 2.0 * k * (k + 1.0) / (n - k - 1)


# compute the r-squared
def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# compute the adjusted r-squared
def Adjusted_R2(y_true, y_pred, p):
    n = y_pred.shape[0]
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# compute the residual
def RSS(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) * y_pred.shape[0]


# compute the rmse
def RMSE(beta_true, beta_pred):
    return np.sqrt(np.sum((beta_true - beta_pred) ** 2, axis=0) / beta_true.shape[0])


# compute the MAPE
def MAPE(beta_true, beta_pred):
    return np.mean(np.abs((beta_true - beta_pred) / (beta_true + 1e-9)), axis=0)


# compute the MORAN
def MORAN(y_true, y_pred, w):
    return Moran(y_true - y_pred, w)


# summary of goodness of fit
def est_summary(y_true, y_pred, k, family=Gaussian(), show=True):
    aic = AIC(y_true, y_pred, k, family)
    aicc = AICc(y_true, y_pred, k, family)
    r2 = R2(y_true, y_pred)
    adjusted_r2 = Adjusted_R2(y_true, y_pred, k)
    rss = RSS(y_true, y_pred)

    if show:
        print("=" * 50)
        print(
            f"RSS:         {rss}\nAIC:         {aic}\nAICc:        {aicc}\nR2:          {r2}\nAdjusted R2: {adjusted_r2}\n"
        )

    return {"RSS": rss, "AIC": aic, "AICc": aicc, "R2": r2, "Adjusted R2": adjusted_r2}


# summary of parameter estimates accuracy
def param_summary(beta_true, beta_pred):
    rmse = RMSE(beta_true, beta_pred)
    return {"RMSE": rmse}


# summary of all metrics
def summary(y_true, y_pred, k, beta_true, beta_pred, family=Gaussian(), show=True):
    est_result = est_summary(y_true, y_pred, k, family, show=False)
    param_result = param_summary(beta_true, beta_pred)

    result = {**est_result, **param_result}

    if show:
        print("=" * 50)
        print(
            f"RMSE:        {param_result['RMSE']}\nRSS:         {est_result['RSS']}\nAIC:         {est_result['AIC']}\nAICc:        {est_result['AICc']}\nR2:          {est_result['R2']}\nAdjusted R2: {est_result['Adjusted R2']}\n"
        )

    return result
