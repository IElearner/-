import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats
from tqdm import tqdm

from algorithms.decision import get_decision
from algorithms.forecast import ExponentialMovingAverage
from datasets.base import load_data
from datasets.builder import build_data
from datasets.utils import save_to_csv, get_price

parser = argparse.ArgumentParser(
    description="Safety Stock Determination via Forecasting Error Estimation"
)
parser.add_argument("--seed", type=int, default=0, help="random seed of the algorithm")
parser.add_argument(
    "--data_root", type=str, default="C:/Users/mac/Desktop/"
)
parser.add_argument("--order_path", type=str, default="订单表.csv")
parser.add_argument("--lead_time", type=str, default="物料主数据1230.xlsx")
parser.add_argument("--initial_inventory", type=str, default="库存1122.xlsx")
parser.add_argument(
    "--data_path",
    type=str,
    default="./data",#data_bulid()之后生成
    help="path to store all data, must be an empty folder when run first",
)
parser.add_argument("--price_path", type=str, default="./data/price.npy")


map_func = {
    "simple": 0,
    "double": 1,
    "holt": 2,
    "plain": 0,
    "iid": 1,
    "arch": 2,
    "garch": 3,
}


def main():
    global args
    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # generate data, only need to run ONCE
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

        orders = build_data(          #调用build_data()函数对数据预处理，其三个参量使用上边定义的常量
            data_root=args.data_root,  #得到一个orders列表，其按照sku总订购量降序排列
            order_name=args.order_path,
            lead_time_name=args.lead_time,
        )

        total_q = 0
        selected = 0
        for i, order in enumerate(tqdm(orders)):
            total_q += order["total_q"]
            if i + 1 <= 100:
                selected += order["total_q"]
                filename = f"./data/rank_{i+1}_{order['sku_id']}.csv"#文件名：rank_i+1_sku_id
                save_to_csv(order["order_history"], filename)#前百的每组sku数据保存至一个csv文件中,包括其所有ship_date与order_q

        print("Top 100 in total: {:.2f}%".format(selected / total_q))#获得前百sku订购量占总sku订购量比例

    if not os.path.exists(args.price_path):
        price = get_price(os.path.join(args.data_root, args.order_path))#获得price字典，sku_id-出售单价
        np.save(args.price_path, price)
    else:
        price = np.load(args.price_path, allow_pickle=True)
        price = price.item()


    # get data from csv
    all_data = load_data(
        args.data_path,    #那100各csv文件所在地址
        os.path.join(args.data_root, args.initial_inventory), #库存表
        os.path.join(args.data_root, args.lead_time),  #物料主数据
    )

    result = np.zeros((24, 6))#24组数据，每组数据6个指标
    cost_all = np.zeros((6, 4, 100))
    alpha_stable = np.zeros((6, 4))
    profit_all = np.zeros((6, 4))
    circulation_all = np.zeros((6, 4, 100))

    for f_modeltype in ["simple", "double", "holt"]:
        for bc in [True, False]:
            for d_modeltype in ["plain", "iid", "arch", "garch"]:
                iter = True
                stock_out_tot = 0
                demand_tot = 0
                alpha_stable_all = []

                for k in tqdm(range(len(all_data))):
                    forecast_model = ExponentialMovingAverage(
                        all_data[k]["data_frame"], dtype=f_modeltype, box_cox=bc
                    )#创建一个指数预测模型实例
                    forecast_model.eval(iterative=True)#迭代，plot=False不画图
                    decision_model = get_decision(
                        d_modeltype=d_modeltype,
                        forecast_model=forecast_model,
                        lead_time=all_data[k]["lead_time"],
                        target_sl=0.9,
                        init_inv=all_data[k]["init_inv"],
                        inv_cost=all_data[k]["inv_cost"],
                        price=price[all_data[k]["id"]],
                    )

                    res = decision_model.eval(iterative=iter, plot=False)

                    stock_out_tot += res["stock_out"]  #总缺货量
                    demand_tot += np.sum(decision_model.test_data) #总需求量

                    cost_all[
                        map_func[f_modeltype] * 2 + (1 - bc),
                        map_func[d_modeltype] + (1 - iter),
                        k,
                    ] = res["cost"]

                    profit_all[
                        map_func[f_modeltype] * 2 + (1 - bc),
                        map_func[d_modeltype] + (1 - iter)
                    ] += res["profit"]

                    circulation_all[
                        map_func[f_modeltype] * 2 + (1 - bc),
                        map_func[d_modeltype] + (1 - iter),
                        k,
                    ] = res["circulation"]

                    # get stability
                    min_sl, max_sl = 1e-5, 0.99

                    decision_model = get_decision(
                        d_modeltype=d_modeltype,
                        forecast_model=forecast_model,
                        lead_time=all_data[k]["lead_time"],
                        target_sl=max_sl,#max
                        init_inv=all_data[k]["init_inv"],
                        inv_cost=all_data[k]["inv_cost"],
                        price=price[all_data[k]["id"]],
                    )

                    res = decision_model.eval(iterative=iter, plot=False)

                    beta_max = 1 - res["stock_out"] / np.sum(
                        decision_model.test_data
                    )
                    mid_sl = (min_sl + max_sl) / 2

                    while min_sl < max_sl - 1e-4:
                        mid_sl = (min_sl + max_sl) / 2

                        d_model = get_decision(
                            d_modeltype=d_modeltype,
                            forecast_model=forecast_model,
                            lead_time=all_data[k]["lead_time"],
                            target_sl=mid_sl,#mid
                            init_inv=all_data[k]["init_inv"],
                            inv_cost=all_data[k]["inv_cost"],
                            price=price[all_data[k]["id"]],
                        )

                        res = d_model.eval(iterative=iter, plot=False)

                        beta_now = 1 - res["stock_out"] / np.sum(d_model.test_data)

                        if beta_now < beta_max:
                            min_sl = mid_sl
                        else:
                            max_sl = mid_sl

                    if mid_sl >= 1e-5 and mid_sl <= 0.999:
                        alpha_stable_all.append(mid_sl)

                alpha_stable[
                    map_func[f_modeltype] * 2 + (1 - bc),
                    map_func[d_modeltype] + (1 - iter),
                ] = np.mean(alpha_stable_all)

    # metric the all decisions
    for f_modeltype in ["simple", "double", "holt"]:
        for bc in [True, False]:
            for d_modeltype in ["plain", "iid", "arch", "garch"]:
                iter = True
                cost_rank = np.argsort(cost_all.reshape(-1, 100), axis=0)
                cost_rank = cost_rank.reshape((6, 4, 100))

                beta = 1 - stock_out_tot / demand_tot

                C = (          #总成本
                        np.sum(
                            cost_all[
                                map_func[f_modeltype] * 2 + (1 - bc),
                                map_func[d_modeltype] + (1 - iter),
                            ]
                        )
                        / 1e8
                )

                r = np.mean(
                    cost_rank[
                        map_func[f_modeltype] * 2 + (1 - bc),
                        map_func[d_modeltype] + (1 - iter),
                    ]
                )

                stable = alpha_stable[
                    map_func[f_modeltype] * 2 + (1 - bc),
                    map_func[d_modeltype] + (1 - iter),
                ]

                num = (map_func[f_modeltype] * 2 + (1 - bc)) * 4 + \
                      map_func[d_modeltype]

                result[num, 0] = round(beta, 4)    #服务水平
                result[num, 1] = round(C, 2)       #总成本
                result[num, 2] = round(stable, 4)
                result[num, 3] = round(r, 2)
                result[num, 4] = round(             #总利益
                    profit_all[
                        map_func[f_modeltype] * 2 + (1 - bc),
                        map_func[d_modeltype] + (1 - iter)
                    ] / 1e6,
                    2,
                )

                circulate_now = circulation_all[
                    map_func[f_modeltype] * 2 + (1 - bc),
                    map_func[d_modeltype] + (1 - iter),
                ]

                result[num, 5] = round(      #ITO
                    np.mean(circulate_now[circulate_now != 0]),
                    2,
                )

    result = pd.DataFrame(result)
    result.to_csv("test.csv", index=False)


if __name__ == "__main__":
    main()
