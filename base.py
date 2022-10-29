import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tqdm import tqdm


def smooth_nan(data):
    mask = ~np.isnan(data)
    for i in range(len(data)):
        if np.isnan(data[i]):
            data[i] = np.mean(data[mask])  #？

    return data


class ForecastModel(object):      #创建预测类
    def __init__(self, data, date_thresh="2021-01", box_cox=False):
        ship_date = list(data["ship_date"])
        order_q = list(data["order_q"])

        self.train_data = []
        self.test_data = []
        self.box_cox = box_cox
        self.date_thresh = date_thresh

        for i in range(len(ship_date)):
            if ship_date[i] < date_thresh:                #划定测试集、训练集
                self.train_data.append(order_q[i])
            else:
                self.test_data.append(order_q[i])

        self.train_data = np.asarray(self.train_data)
        self.test_data = np.asarray(self.test_data)

        if box_cox:
            from scipy.stats import boxcox

            transformed_data, opt_lamb = boxcox(self.train_data)
            self.train_data = transformed_data
            self.opt_lamb = opt_lamb

            transformed_data = boxcox(self.test_data, lmbda=opt_lamb)
            self.test_data = transformed_data

        self.train_data = smooth_nan(self.train_data)
        self.test_data = smooth_nan(self.test_data)

    def eval(self, **kwargs):
        pass


class DecisionModel(object):
    def __init__(
        self,
        forecast_model,
        lead_time,
        target_service_level,
        initial_inventory,
        inv_cost,
        price,
    ):
        self.tsl = target_service_level
        self.lead_time = lead_time
        self.initial_inventory = initial_inventory
        self.inv_cost = inv_cost
        self.price = price

        self.box_cox = forecast_model.box_cox

        if self.box_cox:
            from scipy.special import inv_boxcox

            self.opt_lamb = forecast_model.opt_lamb
            self.train_data = inv_boxcox(forecast_model.train_data, self.opt_lamb)
            self.test_data = inv_boxcox(forecast_model.test_data, self.opt_lamb)
            self.test_forecast = inv_boxcox(forecast_model.forecast, self.opt_lamb)
            self.train_forecast = inv_boxcox(forecast_model.pred_train, self.opt_lamb)

        else:
            self.train_data = forecast_model.train_data
            self.test_data = forecast_model.test_data
            self.test_forecast = forecast_model.forecast
            self.train_forecast = forecast_model.pred_train

        # check if NaN
        self.train_data = smooth_nan(self.train_data)
        self.train_forecast = smooth_nan(self.train_forecast)
        self.test_data = smooth_nan(self.test_data)
        self.test_forecast = smooth_nan(self.test_forecast)

    def train(self, **kwargs):
        pass

    def eval(self, iterative=False, plot=False, logger=None, work_dir=None):
        test_length = len(self.test_data)

        hand_stock = np.zeros((test_length,))#hand_stock当前库存
        order_num = np.zeros((test_length,))

        hand_stock[0] = self.initial_inventory
        factor = 0.125 / 100

        if logger is not None:
            logger.info(
                f"pred_train: {self.train_forecast}\n"  #数据摘要？
                f"pred_test: {self.test_forecast}"
            )

        if work_dir is not None:
            os.makedirs(work_dir, exist_ok=True)    #创建目录

        num_stock_out = 0
        num_stock_out_times = 0
        tot_inv_cost = 0
        s_plot = []
        S_plot = []

        save_real_demand = list(self.train_data)
        save_pred_demand = list(self.train_forecast)
        save_reorder_point = []
        save_target_inventory = []
        save_order_number = []
        save_stock_begin = []
        save_stock_end = []
        save_order_arrive = []
        for i in range(test_length):
            # print(f"------------------------ iter [{i+1}/{test_length}+{self.lead_time}] ------------------------")
            # transformation
            if i > 0:
                hand_stock[i] = hand_stock[i - 1]

            stock_begin = hand_stock[i]

            # orders arrive
            if i >= self.lead_time:
                hand_stock[i] += order_num[i - self.lead_time]


            if i == self.lead_time:
                if logger is not None:
                    logger.info("------------------------------------------------------------------------")

            # print(f"\tH: {hand_stock[i]}\tD: {self.test_data[i]}")

            # satisfy demand
            if hand_stock[i] < self.test_data[i]:
                hand_stock[i] = 0

                if i >= self.lead_time:
                    num_stock_out_times += 1
                    num_stock_out += self.test_data[i] - hand_stock[i]
            else:
                hand_stock[i] -= self.test_data[i]

            # order if necessary
            reorder_point = round(norm.ppf(self.tsl) * self.sigma_L)
            target_inventory = round(
                reorder_point
                + np.sum(self.test_data[i + 1 : i + self.lead_time + 1])
            )

            orders_on_road = order_num[i - self.lead_time + 1] if i >= self.lead_time - 1 else 0
            next_demand = self.test_data[i + 1] if i < self.lead_time - 1 else 0

            if hand_stock[i] + orders_on_road <= reorder_point + next_demand:
                order_num[i] = target_inventory - hand_stock[i]

            if iterative:
                self.sigma_L = self.train(
                    train_data=np.concatenate(
                        (self.train_data, self.test_data[: i + 1]), axis=0
                    ),
                    train_forecast=np.concatenate(
                        (self.train_forecast, self.test_forecast[: i + 1]), axis=0
                    ),
                )

            # print(f"\ts: {reorder_point}\tS: {target_inventory}")

            s_plot.append(reorder_point)
            S_plot.append(target_inventory)

            # inv_cost
            tot_inv_cost += self.inv_cost * order_num[i]
            tot_inv_cost += hand_stock[i] * factor * self.inv_cost
            tot_inv_cost += hand_stock[i] * factor * self.inv_cost

            if logger is not None:
                logger.info(
                    "iter [{}/{}]\t"
                    "stock_begin: {:.1f}\t"
                    "demand: {:.1f}\t"
                    "order arrive: {:.1f}\t"
                    "stock_end: {:.1f}\t"
                    "reorder_point (s): {:.1f}\t"
                    "target_inventory (S): {:.1f}\t"
                    "orders: {:.1f}".format(
                        i + 1,
                        test_length,
                        stock_begin,
                        self.test_data[i],
                        0 if i < self.lead_time else order_num[i - self.lead_time],
                        hand_stock[i],
                        reorder_point,
                        target_inventory,
                        order_num[i],
                    )
                )

            save_real_demand.append(self.test_data[i])
            save_pred_demand.append(self.test_forecast[i])
            save_reorder_point.append(reorder_point)
            save_target_inventory.append(target_inventory)
            save_order_number.append(order_num[i])
            save_stock_begin.append(stock_begin)
            save_stock_end.append(hand_stock[i])
            save_order_arrive.append(0 if i < self.lead_time else order_num[i - self.lead_time])

        if plot:
            plt.plot(hand_stock, label=r"$H_t$")
            plt.plot(s_plot, label=r"$s_t$", linestyle="--")
            plt.plot(S_plot, label=r"$S_t$", linestyle="--")
            plt.legend()
            plt.show()

        # np.save(os.path.join(work_dir, "real_demand.npy"), save_real_demand)
        # np.save(os.path.join(work_dir, "pred_demand.npy"), save_pred_demand)
        # np.save(os.path.join(work_dir, "reorder_point.npy"), save_reorder_point)
        # np.save(os.path.join(work_dir, "target_inventory.npy"), save_target_inventory)
        # np.save(os.path.join(work_dir, "order_number.npy"), save_order_number)
        # np.save(os.path.join(work_dir, "stock_begin.npy"), save_stock_begin)
        # np.save(os.path.join(work_dir, "stock_end.npy"), save_stock_end)
        # np.save(os.path.join(work_dir, "order_arrive.npy"), save_order_arrive)
        # np.save(os.path.join(work_dir, "cost.npy"), tot_inv_cost)
        # np.save(os.path.join(work_dir, "profit.npy"),
            #         self.price * (np.sum(self.test_data) - num_stock_out) - tot_inv_cost)
        # np.save(os.path.join(work_dir, "circulation.npy"),
        #         np.sum(self.test_data) / np.mean(hand_stock) if np.mean(hand_stock) != 0 else 0)
        # np.save(os.path.join(work_dir, "service_level.npy"),
        #         (1 - num_stock_out / np.sum(self.test_data[self.lead_time:])) * 100)
        # np.save(os.path.join(work_dir, "lead_time.npy"), self.lead_time)

        res = {
            "stock_out": num_stock_out,#缺货量
            "cost": tot_inv_cost,
            "profit": self.price * (np.sum(self.test_data) - num_stock_out) - tot_inv_cost,
            "circulation": np.sum(self.test_data) / np.mean(hand_stock) if np.mean(hand_stock) != 0 else 0,
        }

        return res
