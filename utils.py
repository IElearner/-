import copy
import datetime
import os
from operator import itemgetter

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_interval(day1, day2):      #为后续数据处理做判定准备
    # input format (str): yyyy-mm-dd, e.g., 2022-01-01
    y1, m1, d1 = day1.split("-")
    y2, m2, d2 = day2.split("-")

    day1 = datetime.datetime(int(y1), int(m1), int(d1))
    day2 = datetime.datetime(int(y2), int(m2), int(d2))

    interval = day2 - day1

    return interval.days


def sort_by_key(lis, key):
    # input format (list): [dict, dict, dict, ...]
    return sorted(lis, key=itemgetter(key))


def save_to_csv(data, to_path):
    ship_date = []
    order_q = []

    all_month = []
    for year in ["2019", "2020", "2021"]:
        for month in range(12):
            if year == "2021" and month == 11:
                continue

            all_month.append("{}-{:02d}".format(year, month + 1))

    for date in all_month:
        find = False
        for item in data:
            if item["ship_date"] == date:
                ship_date.append(item["ship_date"])
                order_q.append(item["order_q"])
                find = True
                break
        if not find:
            ship_date.append(date)
            order_q.append(1)


    df = pd.DataFrame({"ship_date": ship_date, "order_q": order_q})
    df.to_csv(to_path, index=False)


def get_price(path):          #计算销售单价
    print("=> get sales price ...")
    data = pd.read_csv(path)
    all_data = {}
    for i in tqdm(range(len(data))):
        sku_id = data["SKUID"].iloc[i]
        if not sku_id in all_data:
            all_data[sku_id] = {
                "ord_qty": 0,
                "sales_amt": 0,
            }

        all_data[sku_id]["ord_qty"] += data["ORD_QTY"].iloc[i]
        all_data[sku_id]["sales_amt"] += data["SALES_AMOUNT"].iloc[i]

    price = {}
    for sku_id in all_data.keys():
        price[sku_id] = all_data[sku_id]["sales_amt"] / all_data[sku_id]["ord_qty"]

    return price