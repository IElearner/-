import argparse
import os
import pprint
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
from datasets.utils import save_to_csv, get_price, sort_by_key
from utils import get_current_time, create_logger

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
    default="./data",
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

    # define model
    f_modeltype = "double"
    bc = True
    d_modeltype = "plain"
    iter = True

    # create logger
    current_time = get_current_time()
    logger = create_logger(
        "global_logger",
        f"logs/{f_modeltype}_bc{bc}_{d_modeltype}_iter{iter}_{current_time}.log"
    )
    logger.info("args: {}".format(pprint.pformat(args)))

    # generate data, only need to run ONCE
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

        orders = build_data(
            data_root=args.data_root,
            order_name=args.order_path,
            lead_time_name=args.lead_time,
        )

        total_q = 0
        selected = 0
        for i, order in enumerate(tqdm(orders)):
            total_q += order["total_q"]
            if i + 1 <= 100:
                selected += order["total_q"]
                filename = f"./data/rank_{i+1}_{order['sku_id']}.csv"
                save_to_csv(order["order_history"], filename)

        print("Top 100 in total: {:.2f}%".format(selected / total_q))

    if not os.path.exists(args.price_path):
        price = get_price(os.path.join(args.data_root, args.order_path))
        np.save(args.price_path, price)
    else:
        price = np.load(args.price_path, allow_pickle=True)
        price = price.item()

    # get data from csv
    all_data = load_data(
        args.data_path,
        os.path.join(args.data_root, args.initial_inventory),
        os.path.join(args.data_root, args.lead_time),
    )

    results = []

    for k in tqdm(range(len(all_data))):
        logger.info(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                    f" sku {all_data[k]['id']}"
                    f" with lead time {all_data[k]['lead_time']} month(s)"
                    f" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        logger.info(
            f"forecast model: {f_modeltype}\n"
            f"box_cox transformation: {bc}\n"
            f"decision model: {d_modeltype}\n"
            f"iterative update: {iter}\n"
        )

        forecast_model = ExponentialMovingAverage(
            all_data[k]["data_frame"], dtype=f_modeltype, box_cox=bc
        )
        forecast_model.eval(iterative=True)
        decision_model = get_decision(
            d_modeltype=d_modeltype,
            forecast_model=forecast_model,
            lead_time=all_data[k]["lead_time"],
            target_sl=0.9,
            init_inv=all_data[k]["init_inv"],
            inv_cost=all_data[k]["inv_cost"],
            price=price[all_data[k]["id"]],
        )

        logger.info(
            f"per inv cost: {all_data[k]['inv_cost']}"
        )

        res = decision_model.eval(
            iterative=iter,
            plot=False,
            logger=logger,
            work_dir=f"ui_show/{all_data[k]['rank']}_{all_data[k]['id']}_{f_modeltype}_{d_modeltype}"
        )

        logger.info(
            "service level {:.4f}\n"
            "inventory cost {:.2f}\n"
            "profit {:.2f}\n"
            "circulation {:.2f}\n".format(
                1 - res["stock_out"] / np.sum(decision_model.test_data[decision_model.lead_time :]),
                res["cost"],
                res["profit"],
                res["circulation"],
            )
        )


        info = {
            "rank": int(all_data[k]["rank"]),
            "id": all_data[k]["id"],
            "service_level": "{:.2f}%".format(
                (1 - res["stock_out"] / np.sum(decision_model.test_data[decision_model.lead_time:])) * 100
            ),
            "cost": round(res["cost"], 2),
            "profit": round(res["profit"], 2),
            "circulation": round(res["circulation"], 2),
        }

        results.append(info)

    results = sort_by_key(results, "rank")
    results = pd.DataFrame(results)
    results.to_csv("output_all_sku.csv", index=False)

    func_id2rank = {}
    func_rank2id = {}
    for k in range(len(all_data)):
        func_id2rank[all_data[k]['id']] = all_data[k]['rank']
        func_rank2id[all_data[k]['rank']] = all_data[k]['id']
    np.save("ui_show/func_id2rank.npy", func_id2rank)
    np.save("ui_show/func_rank2id.npy", func_rank2id)


if __name__ == "__main__":
    main()
