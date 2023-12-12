import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from scipy import signal
import scipy.stats as stats
from sklearn.decomposition import PCA

def pyplot_setup():
    # My setup of matplotlib
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rc('text', usetex=False)
    plt.rc('font', family='times')
    mpl.rcParams['mathtext.fontset'] = 'cm'
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
    fontsize = 18

def plot_series(values, title=None, xlabel=None, ylabel=None, filename=None):

    plt.plot(values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_correlation_heatmap(data, title=None, filename=None):

    corr = data.corr()
    heatmap = sns.heatmap(corr)
    heatmap.set_title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')

    plt.show()
    return heatmap


def Cal_rolling_mean_var(data):
    """ take list and compute the rolling mean and variance
    Params:
        Data: a list (numpy array, pandas series, ... a 1-D iterable) of float
    Return
        rolling mean (list of floats), rolling variance (list of floats)
      """

    rolling_mean = np.array([])
    rolling_variance = np.array([])
    rolling_data = np.array([])

    # iterate over the data
    for item in data:
        rolling_data = np.append(rolling_data, item)  # insert a new item from data
        rolling_mean = np.append(rolling_mean, rolling_data.mean())  # compute the new mean
        rolling_variance = np.append(rolling_variance, rolling_data.var())  # compute the new variance

    return rolling_mean, rolling_variance

def plot_rolling_mean_var(rolling_mean, rolling_var, filename=None):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

    ax1.plot(rolling_mean, color='midnightblue')
    ax2.plot(rolling_var, color='royalblue')

    ax1.set_title("Rolling Mean")
    ax2.set_title("Rolling Variance")
    ax1.set_xlabel("Samples")
    ax2.set_xlabel("Samples")
    ax1.set_ylabel("Mean")
    ax2.set_ylabel("Variance")
    ax1.grid()
    ax2.grid()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def perform_pca(X, n_components='mle', filename=None):

    pca = PCA(n_components=X.shape[1])
    pca.fit(X)

    pca_var_percent = np.round(pca.explained_variance_ratio_*100, decimals=1)
    labels = [X.columns[i] for i in range(X.shape[1])]

    plt.bar(x=range(len(labels)), height=pca_var_percent, tick_label=labels)
    plt.ylabel("Percentage of Variance Explained")
    plt.xlabel("Principal Component")
    plt.title("Scree Plot")
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

    print(f"Condition Number : {max(pca.singular_values_)/min(pca.singular_values_)}")

    return pca.explained_variance_ratio_, pca.singular_values_

def ACF(data, tau):
    """ compute the ACF at some lag = tau """
    # handle negative lag
    if tau < 0:
        return ACF(data, -tau)
    mean = data.mean()
    AA = sum((data[t] - mean) * (data[t-tau] - mean) for t in range(tau,len(data)))
    BB = sum((data[t] - mean)**2 for t in range(len(data)))
    return AA/BB


def ACF_Calc(data, number_lags):
    try:
        data = np.array(data)
    except:
        print("The data must be a one - D array")
        return

    mean = data.mean()
    BB = sum((data[t] - mean) ** 2 for t in range(len(data)))



    tmp = np.array([sum((data[t] - mean) * (data[t - tau] - mean) \
                        for t in range(tau, len(data))) for tau in range(number_lags + 1)]) / BB

    acf = np.concatenate((tmp[::-1], tmp[1:]))

    return {-number_lags + i: acf[i] for i in range(len(acf))}

    #return np.concatenate((tmp[::-1][:-1], tmp))


def plot_ACF(acf_list, number_lags, size, ax=None, filename=None):
    x = np.array([i for i in range(-number_lags, number_lags + 1)])

    # print(len(x), len(acf_list))

    # insignificance band
    xfill = [i for i in range(-number_lags, number_lags + 1)]
    ydown = np.full((len(xfill),), -1.96 / np.sqrt(size))
    yup = np.full((len(xfill),), 1.96 / np.sqrt(size))

    if not ax is None:
        ax.fill_between(xfill, ydown, yup, alpha=0.5)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel("ACF")
        ax.stem(x, acf_list, markerfmt="magenta", basefmt='C2')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        return ax

    fig = plt.figure()
    plt.fill_between(xfill, ydown, yup, alpha=0.5)
    plt.xlabel(r"$\tau$")
    plt.ylabel("ACF")
    plt.stem(x, acf_list, markerfmt="magenta", basefmt='C2')
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

    return fig


def plot_model(train=None, model=None, test=None, forecast=None, ylabel=None,
               title = "", filename=None):

    fig = plt.figure(figsize=(12,6))
    if not train is None:
        plt.plot(train, label="Training data")
    if not model is None:
        plt.plot(model, label="Model trained")
    if not test is None:
        plt.plot(test, label="Test data")
    if not forecast is None:
        plt.plot(forecast, label="Forecast")

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if filename:
        plt.savefig(filename,bbox_inches='tight')

    return fig

def adf_test(data):

    result = adfuller(data)
    test_stat = result[0]
    pvalue = result[1]
    critical_values = result[4]
    print("ADF Test Summary:")
    print(f"\t Test statistic = {test_stat:.3f}")
    print(f"\t P-Value = {pvalue:.3f}")
    print("\t Critical Values:")
    for key, value in critical_values.items():
        print(f"\t\t {key}:{value:.3f}")

    return test_stat, pvalue, critical_values

def kpss_test(data):

    result = kpss(data)
    test_stat = result[0]
    pvalue = result[1]
    critical_values = result[3]
    print("KPSS Test Summary:")
    print(f"\t Test statistic = {test_stat:.3f}")
    print(f"\t P-Value = {pvalue:.3f}")
    print("\t Critical Values:")
    for key, value in critical_values.items():
        print(f"\t\t {key}:{value:.3f}")

    return test_stat, pvalue, critical_values


def dataframe_train_test_split(data, test_size=0.2):

    split = int((1-test_size)*data.shape[0])

    return data.iloc[:split], data.iloc[split:]

def stl_decomposition(data, period=365):

    decomp = STL(data, period=period)
    return decomp.fit()

def trend_seasonality_strength(T,S,R, type="trend"):
    if type == "trend":
        total = T + R
    elif type == "seasonality":
        total = S + R
    else:
        return
    return max(0, 1 - R.var() / total.var())


class ARMA:
    def __init__(self, ar_coefs=[], ma_coefs=[], wn = [0,1]):

        max_order = max(len(ar_coefs), len(ma_coefs))
        self.ar_coefs, self.ma_coefs = np.zeros(shape=max_order), np.zeros(shape=max_order)
        for i in range(len(ar_coefs)):
            self.ar_coefs[i] = ar_coefs[i]
        for i in range(len(ma_coefs)):
            self.ma_coefs[i] = ma_coefs[i]


        self.wn = np.array(wn)
        self.arma_process = sm.tsa.ArmaProcess(np.r_[1,self.ar_coefs], np.r_[1,self.ma_coefs])

    def theoretical_mean(self):

        return self.wn[0] * (1 + np.sum(self.ma_coefs))/(1 + np.sum(self.ar_coefs))

    def generate_sample(self, N=1000):

        return self.arma_process.generate_sample(nsample=N, scale=self.wn[1]) + self.theoretical_mean()

    def experimental_mean_var(self, N = 1000):
        y = self.generate_sample(N)
        return y.mean(), y.var()

    def get_acf(self, lags):

        acf_tmp = self.arma_process.acf(lags+1)
        acf = np.concatenate((acf_tmp[::-1], acf_tmp[1:]))

        return {-lags+i:acf[i] for i in range(len(acf))}
    def ACF_PACF_Plot(self, y,lags=20, filename=None):

        acf = sm.tsa.stattools.acf(y, nlags=lags)
        pacf = sm.tsa.stattools.pacf(y, nlags=lags)
        fig = plt.figure()
        plt.subplot(211)
        plt.title('ACF/PACF of the raw data')
        plot_acf(y, ax=plt.gca(), lags=lags)
        plt.subplot(212)
        plot_pacf(y, ax=plt.gca(), lags=lags)
        fig.tight_layout(pad=3)
        if filename is not None:
            plt.savefig(filename + "-acf-pacf.pdf")
        plt.show()

def gpac_matrices(j, k, acf):
    num, den = np.full(shape=(k,k), fill_value=-1), np.full(shape=(k,k), fill_value=-1)
    num, den = np.zeros((k,k)), np.zeros((k,k))
    num, den = -np.ones(shape=(k,k)), -np.ones(shape=(k,k))
    for ii in range(k):
        jj: int
        den[:,ii] = np.array([acf[j - ii + jj] for jj in range(k)])

    num[:k, :k-1] = den[:k, :k-1]
    num[:,k-1] = np.array([acf[j+jj] for jj in range(1, k+1)])

    return num, den

def gpac_table(jmax,kmax, acf, eps=1e-12):
    table = np.zeros(shape=(jmax, kmax))
    for k in range(1, kmax):
        for j in range(0, jmax):
            num, den = gpac_matrices(j, k, acf)
            a, b = np.linalg.det(num), np.linalg.det(den)
            if np.abs(a) < eps and np.abs(b) < eps:
                val = None
            elif np.abs(a) < eps:
                val = 0
            elif np.abs(b) < eps:
                val = np.inf
            else:
                val = a/b

            table[j,k-1] = val
    return table

def gpac_to_heatmap(table, filename=None):
    df = pd.DataFrame(table, columns=[f"{i}" for i in range(1,table.shape[1]+1)])
    ax = sns.heatmap(df, annot=True, fmt='.2f')
    ax.xaxis.tick_top()
    ax.set_title("Generalized Partial Autocorrelation (GPAC) Table")
    if not filename is None:
        plt.savefig(filename+"-gpac.pdf", bbox_inches='tight')
    plt.show()
    return

def ACF_PACF_Plot(y,lags=20, filename=None):

    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    if filename is not None:
        plt.savefig(filename + "-acf-pacf.pdf")
    plt.show()

def get_error(y, ar_order, ma_order, params):
    ar = np.r_[1, params[:ar_order]]
    ma = np.r_[1, params[ar_order:]]
    max_order = max(ar_order, ma_order)
    while len(ar) < max_order + 1:
        ar = np.append(ar, 0)

    while len(ma) < max_order + 1:
        ma = np.append(ma, 0)

    _, error = signal.dlsim((ar, ma, 1), y)
    error = error.reshape(-1)

    return error

def get_sse(error):
    return np.dot(error, error)

def gradient(y, error, params, ar_order, ma_order, delta):
    n = params.shape[0]
    N = y.shape[0]
    X = np.empty(shape=(N,n))
    for i in range(n):
        params[i] += delta
        error_prime = get_error(y, ar_order, ma_order, params)
        X[:,i] = (error - error_prime)/delta
        params[i] -= delta


    return X.transpose() @ error, X.transpose() @ X

def print_params(params, cov, ar_order, ma_order):

    for i in range(len(params)):
        if i == 0:
            print("\nAR Parameters")
            print("#################################################")
        val = params[i]
        dval = 2*np.sqrt(cov[i][i])
        if i < ar_order:
            print(f"a{i+1} = {val:.5f}")
            print(f"{(val - dval):.5f} < a{int(i+1)} < {(val + dval):.5f}")
        else:
            if i == ar_order:
                print("\nMA Parameters")
                print("#################################################")
            if ar_order != 0:
                print(f"b{i%ar_order + 1} = {val:.5f}")
                print(f"{(val - dval):.5f} < b{int(i%ar_order + 1)} < {(val + dval):.5f}")
            else:
                print(f"b{i + 1} = {val:.5f}")
                print(f"{(val - dval):.5f} < b{int(i + 1)} < {(val + dval):.5f}")
def lm_fit(y, ar_order, ma_order, delta=1e-3, mu=0.01, maxiter=1000, eps=1e-10, mumax=1e8):

    # initialization
    N = y.shape[0]
    n = ar_order + ma_order
    params = np.zeros(n)

    # step 1
    error = get_error(y, ar_order, ma_order, params)
    sse = get_sse(error)
    g, A = gradient(y, error, params, ar_order, ma_order, delta)

    #step 2
    W = A + mu*np.eye(n)
    deltaParams = np.linalg.inv(W) @ g
    newParams = params + deltaParams
    newError = get_error(y, ar_order, ma_order, newParams)
    new_sse = get_sse(newError)

    sses = []
    for iter in range(maxiter):

        sses.append(new_sse)
        if new_sse < sse:
            if np.sqrt(np.dot(deltaParams, deltaParams)) < eps:
                params = newParams
                sigmaEsq = new_sse/(N-n)
                cov = sigmaEsq * np.linalg.inv(A)
                print(f"Estimated variance of error: {sigmaEsq:.5f}")
                return params, cov, sses
            else:
                params = newParams
                mu = mu/10

        while new_sse >= sse:
            mu = mu*10
            if mu > mumax:
                print("Step became too large")
                return
            W = A + mu * np.eye(n)
            deltaParams = np.linalg.inv(W) @ g
            newParams = params + deltaParams
            newError = get_error(y, ar_order, ma_order, newParams)
            new_sse = get_sse(newError)

        params = newParams
        error = get_error(y, ar_order, ma_order, params)
        g, A = gradient(y, error, params, ar_order, ma_order, delta)

        # step 2
        W = A + mu * np.eye(n)
        deltaParams = np.linalg.inv(W) @ g
        newParams = params + deltaParams
        newError = get_error(y, ar_order, ma_order, newParams)
        new_sse = get_sse(newError)

    print("The fit did not converge")
    sigmaEsq = new_sse / (N - n)
    cov = sigmaEsq * np.linalg.inv(A)
    return params, cov, sses

def seasonal_differencing(y, per):

    difference = y.diff(periods=per)

    difference = difference.dropna()

    return difference
def arma_forecast(y, H, time_stamps=None):
    a1, a2 = 0.06742, 0.93190
    b1 = 0.97658
    forecast = np.array([])
    lo = y[-1]
    for h in range(1, H+1):
        if h == 1:
            val = -(a1 - b1) * y[-1] - a2 * y[-2] - b1 * lo
        elif h == 2:
            val = -a1 * forecast[-1] - a2 * y[-1]
        else:
            val = -a1 * forecast[-1] - a2 * forecast[-2]

        forecast = np.append(forecast, val)

    if time_stamps is None:
        return forecast

    try:
        forecast = pd.Series(forecast, index=time_stamps)
        return forecast
    except:
        pass
    return forecast

# survival
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_waltons
waltons = load_waltons()

# kmf = KaplanMeierFitter(label="waltons_data")
# kmf.fit(waltons['T'], waltons['E'])
# kmf.plot()
# plt.show()

def plot_survival_function(t,y, label=None):
    """
    kaplan meier s(t) = product for i <= t of  (n_i - d_i)/n_i
    :param t:
    :param y:
    :param label:
    :return:
    n_i: rist at t prior to t_i
    d_i: number of events at t_i

    """
    kmf = KaplanMeierFitter(label=label)
    kmf.fit(t,y)
    kmf.plot()
    plt.show()
    return
#
# plot_survival_function(waltons['T'], waltons['E'])

def whiteness_test(acf, h, N, na, nb, alpha = 0.05):
    """
    :param acf: autocorrelation
    :param h: number of lags
    :param N: number of observations
    calculate Q = N sum(h from 1 to N) * acf[i]^2
    dof = h - na - nb
    Null hypothesis: The residuals are uncorellated (white).
    Iff Q < Qc (or P-value > alpha), we do not reject null hypothesis.
    :return:
    """
    df = h - na - nb
    Q = N*np.sum(acf[i]**2 for i in range(1, h-2))
    Qc = stats.chi2.ppf(q = alpha, df = df)

    print(f"With Q = {Q:.4f}\tQc = {Qc:.4f}")
    if Q < Qc:
        print("The data are uncorellated (white)")
    else:
        print("The data are correlated (not white)")

    return Q < Qc







