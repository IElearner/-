import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing

from .base import ForecastModel


class ExponentialMovingAverage(ForecastModel):
    # Y_t = alpha * D_t + (1 - alpha) * Y_t-1
    def __init__(self, data, date_thresh="2021-01", box_cox=False, dtype="simple"):
        super(ExponentialMovingAverage, self).__init__(data, date_thresh, box_cox)
        self.dtype = dtype
        self.model = self.get_forecast_model(dtype, self.train_data)

    def get_forecast_model(self, dtype, train_data):
        if dtype == "simple":
            model = SimpleExpSmoothing(train_data).fit()
        elif dtype == "double":
            model = ExponentialSmoothing(train_data).fit()
        elif dtype == "holt":
            model = Holt(train_data).fit()
        else:
            raise "Unknown EMA method: {}".format(dtype)

        return model

    def eval(self, iterative=False, plot=False):
        # calculate MSE
        test_length = len(self.test_data)
        train_length = len(self.train_data)

        self.pred_train = self.model.fittedvalues

        if not iterative:
            self.forecast = self.model.forecast(test_length)

        else:
            self.forecast = np.zeros((test_length,))
            self.forecast[0] = self.model.forecast(1)[0]

            for i in range(test_length - 1):
                curr_train_data = np.concatenate((self.train_data, [self.test_data[i]]))
                self.curr_model = self.get_forecast_model(self.dtype, curr_train_data)
                self.forecast[i + 1] = self.curr_model.forecast(1)[0]

        # inv_boxcox
        if self.box_cox:
            from scipy.special import inv_boxcox

            plot_train = inv_boxcox(self.train_data, self.opt_lamb)
            plot_train_pred = inv_boxcox(self.pred_train, self.opt_lamb)
            plot_test = inv_boxcox(self.test_data, self.opt_lamb)
            plot_test_pred = inv_boxcox(self.forecast, self.opt_lamb)

        else:
            plot_train = self.train_data
            plot_train_pred = self.pred_train
            plot_test = self.test_data
            plot_test_pred = self.forecast

        # get 训练集MAE
        train_error = 0
        for i in range(train_length):
            train_error += abs(plot_train[i] - plot_train_pred[i])

        # print("train MAE {:.4f}".format(train_error / train_length))
        #测试集MAE
        test_error = 0
        for i in range(test_length):
            test_error += abs(plot_test[i] - plot_test_pred[i])

        # print("test MAE {:.4f}".format(test_error / test_length))

        if plot:
            plt.plot(np.arange(train_length), plot_train, color="black", label="Target") #画图
            plt.plot(
                np.arange(train_length),
                plot_train_pred,
                color="red",
                label="Predict",
                linestyle="--",
            )
            plt.plot(np.arange(test_length) + train_length, plot_test, color="black")
            plt.plot(
                np.arange(test_length) + train_length,
                plot_test_pred,
                color="red",
                linestyle="--",
            )
            plt.legend()
            plt.show()

