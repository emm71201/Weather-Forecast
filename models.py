from statsforecast import StatsForecast
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import matplotlib.pyplot as plt
import statsmodels.api as sm


import numpy as np
import pandas as pd
import tools



class BaseModel:
    def __init__(self):
        pass
    def one_step_mse(self, train, model):
        train_start = 2
        model_start = model.shape[0] - (train.shape[0] - train_start)
        return np.mean((train[train_start:] - model[model_start:])**2)
    def forecast_mse(self, test, forecast):
        return np.mean((test - forecast)**2)
    def model_residual(self, train, model):
        train_start = 2
        model_start = model.shape[0] - (train.shape[0] - train_start)
        return train[train_start:] - model[model_start:]
    def forecast_residual(self, test, forecast):
        return test - forecast

    def training_residual_Qvalue(self, train, model, h = None):
        if h is None:
            h = len(train)
        residuals = self.model_residual(train, model)
        return (len(train)-2) * sum(tools.ACF(residuals, k)**2 for k in range(2, h))

    def forecast_residual_Qvalue(self, test, forecast, T=20, h=None):
        if h is None:
            h = len(test)
        residuals = self.forecast_residual(test, forecast)
        return T * sum(tools.ACF(residuals, k)**2 for k in range(1, h + 1))

    def fit(self, train, test, nsteps = 10):

        model_onestep = self.one_step_ahead(train)
        hstep_forecast = self.forecast(train, nsteps)
        train_resid = self.model_residual(train, model_onestep)
        test_resid = self.forecast_residual(test, hstep_forecast)

        return model_onestep, train_resid, hstep_forecast, test_resid

class Average_Method(BaseModel):
    def __init__(self):
        super().__init__()
    def one_step_ahead(self, train):
        return np.array([train[:t].mean() for t in range(1, train.shape[0])])
    def forecast(self, train, nsteps):
        return np.full(nsteps, train.mean())

class Naive_Method(BaseModel):
    def __init__(self):
        super().__init__()

    def one_step_ahead(self, train):
        return np.array([train[t-1] for t in range(1, train.shape[0])])

    def forecast(self, train, nsteps):
        return np.full(nsteps, train[-1])

class Drift_Method(BaseModel):

    def __init__(self):
        super().__init__()
    def one_step_ahead(self, train):

        return np.array([train[t] + (train[t] - train[0])/t for t in range(2,train.shape[0])])

    def forecast(self, train, nsteps):

        return np.array([train[-1] + h * (train[-1] - train[0])/(train.shape[0]-1) for h in range(1,nsteps+1)])

class Simple_Exponential_Method(BaseModel):

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def one_step_ahead(self, train):

        lo = train[0]

        return np.array([(1 - self.alpha)**t * lo + \
            sum(self.alpha * (1 - self.alpha)**j * train[t - j - 1] for j in range(t) ) for t in range(1, train.shape[0])])

    def forecast(self, train, nsteps):
        # lo = train[0]
        # t = train.shape[0]
        # return np.array([(1 - self.alpha)**t * lo + \
        #         sum(self.alpha * (1 - self.alpha)**j * train[t - j - 1] for j in range(t)) for _ in range(nsteps)])
        lT = self.one_step_ahead(train)[-1]
        return np.full(nsteps, lT)

class HoltWinter:
    def __init__(self):
        self.fitted_model = None

    def fit_model(self, train, trend="add", seasonal="add", seasonal_periods=365):

        self.fitted_model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal,
                                                 seasonal_periods=seasonal_periods).fit(optimized=True)

    def forecast(self, test):

        result = {"forecast": np.array(self.fitted_model.forecast(test.shape[0]))}
        result = pd.DataFrame.from_dict(result)
        result.set_index(test.index, inplace=True)

        return result

    def residual(self):
        return self.fitted_model.resid

# class Expoential_Smoothing(BaseModel):
#
#     def __init__(self):
#         super().__init__()
#         self.fitted_model = None
#
#     def fit_model(self, train, trend="add", seasonal="add", seasonal_periods=365):
#         self.fitted_model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal,
#                                                  seasonal_periods=seasonal_periods).fit(optimized=True)
#     def forecast(self, test):
#
#         result = {"forecast": np.array(self.fitted_model.forecast(test.shape[0]))}
#         result = pd.DataFrame.from_dict(result)
#         result.set_index(test.index, inplace=True)
#
#         return result




def modeling(model, y_train, y_test, model_name ="", filename=None):

    print(f"\n{model_name}\n")

    model = model()
    onestep = model.one_step_ahead(y_train)
    model_forecast = model.forecast(y_train, y_test.shape[0])

    training_mse = model.one_step_mse(y_train, onestep)
    training_Q = model.training_residual_Qvalue(y_train, onestep)
    forecast_mse = model.forecast_mse(y_test, model_forecast)
    forecast_Q = model.forecast_residual_Qvalue(y_test, model_forecast)

    train_dates = pd.DatetimeIndex(pd.date_range(start='2009-01-01', periods=len(onestep), freq='D'))
    test_dates = pd.DatetimeIndex(pd.date_range(start=train_dates[-1]+pd.DateOffset(1), periods=len(y_test), freq="D"))


    print(f"TRAINING MSE: {training_mse:.3f}")
    print(f"TRAINING Q: {training_Q:.3f}")
    print(f"FORECAST MSE: {forecast_mse:.3f}")
    print(f"Forecast Q: {forecast_Q:.3f}")

    shift = len(y_train) - len(onestep)


    fig, ax = plt.subplots()

    ax.plot(train_dates, y_train[shift:len(y_train)+1], label="Training Data")
    ax.plot(train_dates, onestep, label = f"{model_name}-Model")
    ax.plot(test_dates, y_test, label="Test Data")
    ax.plot(test_dates, model_forecast, label="h-step forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    fig.autofmt_xdate()
    plt.show()

    return

# X_train = np.random.random((10,2))
# y_train = np.random.random(10)
# X_test = np.random.random((5,2))
# y_test = np.random.random(5)
def linear_regressor(X_train, y_train, X_test, y_test, columns=None, filename=None):
    # columns = ["p", "Tpot"]
    X_train = X_train[columns]
    X_test = X_test[columns]
    #X_train = sm.add_constant(X_train)

    model = sm.OLS(y_train, X_train)
    results = model.fit()
    print(results.summary())
    print("\n1. F Test\n")
    print("Test if each coefficient is significant and different from zero\n")
    A = np.identity(len(results.params))
    A = A[1:, :]
    fresult = results.f_test(A)
    print(fresult.summary())

    print("\n2. T tests\n")
    print("Test if the two values are significantly different")
    #ttest = results.t_test([0, 1,-1])
    ttest = results.t_test([1,-1])
    print(ttest.summary())

    if filename is not None:
        with open(filename, "w") as sfile:
            sfile.write(results.summary().as_latex())

    resid = results.resid
    print(f"\nTraining MSE: {np.dot(resid, resid)/len(resid):.3f}\n")


# if __name__ == "__main__":
#     linear_regressor(X_train, y_train, X_test, y_test)
#     pass
    #modeling(Simple_Expoential_Method, y_train, y_test)
    # test_dates = pd.DatetimeIndex(pd.date_range('2020-01-01', periods=10, freq='D'))
    # mydate = test_dates[-1]
    # print(mydate)
    # print(mydate + pd.DateOffset(1))

