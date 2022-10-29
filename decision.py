import os

import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from scipy.stats import norm
from tqdm import tqdm

from .base import DecisionModel, smooth_nan


def get_decision(
    d_modeltype,
    forecast_model,
    lead_time,
    init_inv,
    inv_cost,
    target_sl,
    price,
):
    if d_modeltype == "none":
        decision_model = NoDecision(
            forecast_model=forecast_model,
            lead_time=lead_time,                     #提前期
            target_service_level=target_sl,          #β
            initial_inventory=init_inv,              #当前库存
            inv_cost=inv_cost,
            price=price,
        )
    elif d_modeltype == "plain":
        decision_model = Plain(
            forecast_model=forecast_model,
            lead_time=lead_time,
            target_service_level=target_sl,
            initial_inventory=init_inv,
            inv_cost=inv_cost,
            price=price,
        )

    elif d_modeltype == "iid":
        decision_model = Iid(
            forecast_model=forecast_model,
            lead_time=lead_time,
            target_service_level=target_sl,
            initial_inventory=init_inv,
            inv_cost=inv_cost,
            price=price,
        )

    elif d_modeltype == "arch":
        decision_model = Arch(
            forecast_model=forecast_model,
            lead_time=lead_time,
            target_service_level=target_sl,
            initial_inventory=init_inv,
            inv_cost=inv_cost,
            p=1,                #给定模型p值
            price=price,
        )

    elif d_modeltype == "garch":
        decision_model = Garch(
            forecast_model=forecast_model,
            lead_time=lead_time,
            target_service_level=target_sl,
            initial_inventory=init_inv,
            inv_cost=inv_cost,
            p=1,                    #给定模型q值
            q=1,
            price=price,
        )

    else:
        raise f"unknown decision model type: {d_modeltype}!"

    return decision_model


class NoDecision(DecisionModel):
    # e_L ~ normal(0, sigma_L ** 2)
    def __init__(
        self,
        forecast_model,
        lead_time,
        target_service_level,
        initial_inventory,
        inv_cost,
        price,
    ):
        super(NoDecision, self).__init__(
            forecast_model, lead_time, target_service_level, initial_inventory, inv_cost, price,
        )
        self.sigma_L = 0

    def train(self, train_data, train_forecast):
        return 0


class Plain(DecisionModel):        #直接拟合
    # e_L ~ normal(0, sigma_L ** 2)
    def __init__(
        self,
        forecast_model,
        lead_time,
        target_service_level,
        initial_inventory,
        inv_cost,
        price,
    ):
        super(Plain, self).__init__(
            forecast_model, lead_time, target_service_level, initial_inventory, inv_cost, price,
        )
        self.sigma_L = self.train(self.train_data, self.train_forecast)

    def train(self, train_data, train_forecast):
        train_length = len(train_data)

        train_data = smooth_nan(train_data)
        train_forecast = smooth_nan(train_forecast)

        train_forecast_error = np.zeros((train_length - self.lead_time + 1))
        for i in range(train_length - self.lead_time + 1):
            train_forecast_error[i] = np.sum(          #计算e(i)
                train_data[i : i + self.lead_time]
            ) - np.sum(train_forecast[i : i + self.lead_time])

        return np.std(train_forecast_error)    #计算标准差


class Iid(DecisionModel):
    # e_t ~ normal(0, sigma ** 2)
    # sigma_L ** 2 = L * sigma ** 2
    def __init__(
        self,
        forecast_model,
        lead_time,
        target_service_level,
        initial_inventory,
        inv_cost,
        price,
    ):
        super(Iid, self).__init__(
            forecast_model, lead_time, target_service_level, initial_inventory, inv_cost, price
        )
        self.sigma_L = self.train(self.train_data, self.train_forecast)

    def train(self, train_data, train_forecast):
        train_forecast_error = train_data - train_forecast    #直接D(t)-F(t)
        return np.sqrt(self.lead_time) * np.std(train_forecast_error)#根号L乘西格玛(1)标准差


class Arch(DecisionModel):
    # sigma_t ** 2 = omega + \sum_{k=1}^p alpha_k * (sigma_{t-k} * z_{t-k}) ** 2
    def __init__(
        self,
        forecast_model,
        lead_time,
        target_service_level,
        initial_inventory,
        inv_cost,
        p,
        price,
    ):
        super(Arch, self).__init__(
            forecast_model, lead_time, target_service_level, initial_inventory, inv_cost, price
        )
        self.p = p
        self.sigma_L = self.train(self.train_data, self.train_forecast)

    def train(self, train_data, train_forecast):
        # print(f"------------------------ fitting ARCH({p}) model ------------------------")
        p = self.p
        model = arch_model(
            train_data - train_forecast, vol="ARCH", p=p, rescale=False
        ).fit(disp="off")

        # print(f"------------------------ ARCH({p}) model parameters ------------------------")
        # print(model.params)

        return np.sqrt(
            np.sum(
                model.forecast(
                    horizon=self.lead_time, reindex=False
                ).residual_variance.to_numpy()
            )
        )


class Garch(DecisionModel):
    # sigma_t ** 2 = omega + \sum_{k=1}^p alpha_k * (sigma_{t-k} * z_{t-k}) ** 2 + \sum_{k=1}^q beta_k * sigma_{t-k} ** 2
    def __init__(
        self,
        forecast_model,
        lead_time,
        target_service_level,
        initial_inventory,
        inv_cost,
        p,
        q,
        price,
    ):
        super(Garch, self).__init__(
            forecast_model, lead_time, target_service_level, initial_inventory, inv_cost, price
        )
        self.p = p
        self.q = q
        self.sigma_L = self.train(self.train_data, self.train_forecast)

    def train(self, train_data, train_forecast):
        p = self.p
        q = self.q
        # print(f"------------------------ fitting GARCH({p}, {q}) model ------------------------")
        model = arch_model(
            train_data - train_forecast, vol="GARCH", p=p, q=q, rescale=False
        ).fit(disp="off")


        return np.sqrt(
            np.sum(
                model.forecast(
                    horizon=self.lead_time, reindex=False
                ).residual_variance.to_numpy()
            )
        )
