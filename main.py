import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as smapi
from statsforecast import StatsForecast
from statsforecast.models import HoltWinters
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')
np.random.seed(6313)
import tools
import models
tools.pyplot_setup()
# seasonality=876
seasonality=365
damper = 1e9
ylabel = "Temp (C)"

# get data
PATH = "../dataset/jena_climate_2009_2016.csv"

# read data and resample every
def get_data(PATH, resample_rule):
    raw_data = pd.read_csv(PATH)
    raw_data = raw_data.set_index(pd.DatetimeIndex(raw_data['Date Time']))
    raw_data.drop(columns=['Date Time'], inplace=True)
    raw_data = raw_data.resample(resample_rule).mean()

    return raw_data
#data = get_data(PATH, "720T")
# data = get_data(PATH, "1440T")
# data.to_pickle("raw_data_resampled.pkl")

data = pd.read_pickle("raw_data_resampled.pkl")
data.dropna(inplace=True)
print(data.head())
# print(data.shape)
# print(f"The final shape of the data is {data.shape}")

# change columns
new_columns = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh',
'H2OC', 'rho', 'wv', 'max. wv', 'wd']
mapper = {data.columns[i]:new_columns[i] for i in range(len(new_columns))}
data = data.rename(columns=mapper)
data.drop(['Tpot'], axis=1, inplace=True)
dependent_col = 'T'

# plot dependent variable: T
tools.plot_series(data[dependent_col], "Temperature (C)", "Time", "Temperature")

# Plot ACF
number_lags=100
size = data.shape[0]
acf_list = tools.ACF_Calc(data[dependent_col], number_lags)
acf_list = acf_list.values()
# fig = tools.plot_ACF(acf_list, number_lags, size, filename=None)
# tools.ACF_PACF_Plot(data[dependent_col], number_lags, filename=None)
# plt.show()
tools.ACF_PACF_Plot(data[dependent_col], number_lags)


# correlation heat map
heatmap = tools.plot_correlation_heatmap(data, "Correlation Heatmap", filename=None)

# train test split
split=0.2
train, test = tools.dataframe_train_test_split(data, test_size=split)
print(f"Size of the test set: {100*test.shape[0]/data.shape[0]:.2f}%")
y_train = train[dependent_col]
X_train = train.loc[:, train.columns != dependent_col]
y_test = test[dependent_col]
X_test = test.loc[:, test.columns != dependent_col]

# Stationarity
# 1. Plot rolling mean and variance
roll_mean, roll_var = tools.Cal_rolling_mean_var(data[dependent_col])
# seas_diff = tools.seasonal_differencing(data[dependent_col], per)
# tools.ACF_PACF_Plot(seas_diff, 70)
# plt.show()
# plt.plot(seas_diff)
# plt.title("seas_diff")
# plt.show()
# roll_mean, roll_var = tools.Cal_rolling_mean_var(seas_diff[per:])
tools.plot_rolling_mean_var(roll_mean, roll_var, filename=None)
# 2. ADF
adf_result = tools.adf_test(data[dependent_col])
kpss_result = tools.kpss_test(data[dependent_col])



# Trend Seasonality decomposition
decomp = tools.stl_decomposition(data[dependent_col], seasonality)
decomp.plot()
plt.xlabel("Time")
#plt.savefig("../Figures/trend-seasonality-decomposition.pdf")
plt.show()

T, S, R = decomp.trend, decomp.seasonal, decomp.resid
print(f"The strength of trend for this data set is {tools.trend_seasonality_strength(T,S,R,'trend'):.4f}")
print(f"The strength of seasonality for this data set is {tools.trend_seasonality_strength(T,S,R,'seasonality'):.4f}")

y_train_diff = tools.seasonal_differencing(y_train, 365)
y_test_diff = tools.seasonal_differencing(y_test, 365)
# train = tools.seasonal_differencing(train, 365)
# test = tools.seasonal_differencing(test, 365)
for _ in range(0):
    y_train_diff = tools.seasonal_differencing(y_train_diff, 1)
    y_test_diff = tools.seasonal_differencing(y_test_diff, 1)
    # train = tools.seasonal_differencing(train, 1)
    # test = tools.seasonal_differencing(test, 1)


# plot again
# plt.plot(y_train_diff/damper)
# plt.xlabel("Date")
# plt.ylabel("Data")
#plt.savefig("../Figures/differenced-data.pdf", bbox_inches='tight')
# plt.show()

# tools.ACF_PACF_Plot(y_train_diff, 50)
# plt.show()

# ADF and KPSS Test
adf_result = tools.adf_test(y_train_diff)
kpss_result = tools.kpss_test(y_train_diff)

# roll_mean, roll_var = tools.Cal_rolling_mean_var(y_train_diff)
# tools.plot_rolling_mean_var(roll_mean, roll_var, filename=None)

## Holt winter

hw = models.HoltWinter()
hw.fit_model(y_train, "add", "add", seasonal_periods=seasonality)
hw_forecast = hw.forecast(test)
hw_resid = hw.residual()
print(hw_forecast.head())
tools.plot_model(y_train, model=None, test = y_test, forecast=hw_forecast, ylabel=ylabel, title="Holt Winter", filename=None)
plt.show()

print(f"\nHolt Winter MSE: {np.dot(hw_resid, hw_resid)/len(hw_resid):.3f}\n")

# 10. Feature Selection
print(X_train.shape)
pca_var, pca_svd = tools.perform_pca(X_train, filename=None)
print(pca_svd)

# Base models: 12 ############################
print("\n Base models\n")
print("\n This portion of the codes may take about 10 minutes ...\n")
models.modeling(models.Naive_Method, y_train, y_test, "Naive Method", filename=None)
models.modeling(models.Average_Method, y_train, y_test, "Average Method", filename=None)
models.modeling(models.Drift_Method, y_train, y_test, "Drift Method", filename=None)
models.modeling(models.Simple_Exponential_Method, y_train, y_test, "Simple Exponential Smoothing", filename=None)
print("\n Now, the codes will proceed faster ...")

models.linear_regressor(X_train, y_train, X_test, y_test, columns=["p", "Tdew"], filename=None)




# ARMA/ARIMA/SARIMA/Multiplicative models
# a order determination
acf = tools.ACF_Calc(y_train_diff, 50)
# tools.ACF_PACF_Plot(y_train_diff, 20)
jmax,kmax = 7,7
mygpac = tools.gpac_table(jmax,kmax, acf, eps=1e-12)
tools.gpac_to_heatmap(mygpac, filename=None)

def get_arima_forecast(model, test):
    forecast = model.forecast(test.shape[0])
    forecast.index = test.index
    return forecast

# based on the gpac table we expect ar = 1, ma = 0
ar_order, ma_order = 1,0
myarima = ARIMA(y_train_diff/damper, order=(ar_order, 0, ma_order))
myarimafit = myarima.fit()
print(myarimafit.summary())
# with open("arimax_summary.tex", "w") as sumfile:
#     sumfile.write(myarimafit.summary().as_latex())

residuals = myarimafit.resid
acf_residuals = tools.ACF_Calc(residuals, 50)
tools.ACF_PACF_Plot(residuals, 20, filename=None)
print()
tools.whiteness_test(acf_residuals, 50, 50, ar_order, ma_order)
# prediction
arima_error = y_test_diff/damper - get_arima_forecast(myarimafit, y_test_diff/damper)
print(f"Variance of Error {residuals.var():.3f}")
print(f"Forecast Error MSE {np.dot(arima_error, arima_error)/len(arima_error):.3f}")
print(f"Variance of the Forecast Error {arima_error.var():.3f}")

params, cov, sses = tools.lm_fit(y_train_diff, ar_order, ma_order)
tools.print_params(params, cov, ar_order, ma_order)


print()
tools.whiteness_test(acf_residuals, 50, 50, ar_order, ma_order)
# prediction
arima_error = y_test_diff/damper - get_arima_forecast(myarimafit, y_test_diff/damper)
print(f"Variance of Error {residuals.var():.3f}")
print(f"Forecast Error MSE {np.dot(arima_error, arima_error)/len(arima_error):.3f}")
print(f"Variance of the Forecast Error {arima_error.var():.3f}")
print(f"Training MSE {np.dot(residuals, residuals)/len(residuals)}")

def plot_final_model(model, train, test, filename=None):
    train_index = train.index

    training_values = np.array(train) + np.array(model.resid)
    training_values = pd.Series(training_values, index = train_index)

    test_dates = pd.date_range(start=train_index[-1] + pd.DateOffset(days=1), periods=len(test), freq='D')
    test.index = test_dates

    forecast = model.forecast(len(test))
    forecast.index = test_dates


    # plt.plot(train, label="Original data")
    # plt.plot(training_values, label="One-step prediction")
    plt.plot(test, label="Test data")
    plt.plot(forecast, label="H-step prediction")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

# plot_final_model(myarimafit, y_train_diff/damper, y_test_diff/damper, filename=None)

def my_final_model(train, test, s = 365, filename=None):
    a = -0.46622
    onestep = np.array([])
    hstep = np.array([])

    # onestep
    for t in range(train.shape[0]):
        if t - s < 0:
            onestep = np.append(onestep, train.iloc[t])
        else:
            val = train.iloc[t + 1 - s] - a*train.iloc[t] + a*train.iloc[t-s]
            onestep = np.append(onestep, val)

    # hstep
    yT = train.iloc[-1]
    T = train.shape[0] - 1
    test_length = min(s, test.shape[0])

    for h in range(1, test_length + 1):
        if h == 1:
            val = train.iloc[T + 1 - s] - a*train.iloc[T] + a*train.iloc[T-s]
        else:
            val = train.iloc[T + h - s] - a*hstep[-1] + a*train.iloc[T + h - s - 1]
        hstep = np.append(hstep, val)


    train_index = train.index
    test_index = pd.DatetimeIndex(pd.date_range(start=train_index[-1] + pd.DateOffset(days=1), periods=test_length, freq='D'))
    onestep = pd.Series(onestep, index = train_index)
    hstep = pd.Series(hstep, index = test_index)


    plt.plot(train, label="Train data")
    plt.plot(onestep, label="One step prediction")
    plt.plot(test.iloc[:test_length], label="Test data")
    plt.plot(hstep, label="H-step prediction")
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.title("Final Model")
    plt.legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

    return

my_final_model(y_train, y_test, filename=None)
#print("\nThe figures do not have titles because they will be uploaded to latex where I will use caption to say what the figure is about.")



















