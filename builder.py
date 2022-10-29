import copy
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import get_lead_time, get_orders
from .utils import get_interval, sort_by_key


def build_data(data_root, order_name, lead_time_name):          #数据预处理
    orders = get_orders(os.path.join(data_root, order_name))#按shipdate排序 得到字典orders
    lead_times = get_lead_time(os.path.join(data_root, lead_time_name))#得到字典lead_times

    print("=> remove unqualified orders ...")
    new_orders = {}
    for sku_id in tqdm(orders.keys()):
        order_info = orders[sku_id] #一个字典，包括该sku的多种情况，且每种情况有4种数据

        if not sku_id in lead_times.keys():
            continue

        lead_time = lead_times[sku_id] #获取该sku下提前期

        new_info = []#创建一个专属于该sku列表,存储需要设置安全库存且时间达标的数据
        for order in order_info: #对每种情况考虑
            interval = get_interval(order["order_date"], order["ship_date"])
            if interval < lead_time:   #这种情况才要设置安全库存
                y, m, d = order["ship_date"].split("-")    #ship_date年月日
                if y < "2021" or (y == "2021" and m <= "11"):
                    # remove 2021-12 and later orders
                    new_info.append(order)

        if new_info != []:
            new_orders[sku_id] = copy.deepcopy(new_info)#new_info复制到大字典new_orders中去，键值即sku_id
                                                        #具体value为四种数据
    print("=> sorting by total quantity ...")
    temp_dic = []
    for sku_id in new_orders.keys():
        total_q = 0
        for item in new_orders[sku_id]:     #对该sku下各种符合条件情况的订购量求和
            total_q += item["order_q"]

        temp_dic.append({"sku_id": sku_id, "total_q": total_q})#获得列表字典，存储各sku订购量

    temp_dic = sort_by_key(temp_dic, key="total_q")    #对订购量排序

    final = []
    for item in temp_dic:  #每个item有一个sku_id和一个total_q
        sku_id = item["sku_id"]

        # merge to monthly demand
        info = {"sku_id": sku_id, "total_q": item["total_q"]} #创建1个字典

        order_history = new_orders[sku_id]  # have been sorted by 'ship_date'，分出i种情况
        new_history = []

        for i, order in enumerate(order_history):  #i代表该sku下第（i+1）种情况，order则是该情况下的四种参数键值
            ship_date = order["ship_date"]
            y, m, d = ship_date.split("-")

            if i == 0:  #该sku下仅有一种情况
                new_history.append(
                    {"ship_date": f"{y}-{m}", "order_q": order["order_q"]}
                )
            else:
                ship_date_last = order_history[i - 1]["ship_date"]
                y_last, m_last, d_last = ship_date_last.split("-")

                if y == y_last and m == m_last:       #如果ship_date时间一样，则两组数据合并成一组
                    # merge to last demand
                    new_history[-1]["order_q"] += order["order_q"]   #订购量求和
                else:
                    # create new demand              #否则这是新需求
                    new_history.append(
                        {"ship_date": f"{y}-{m}", "order_q": order["order_q"]}
                    )

        info["order_history"] = new_history#至此，获得当下sku的真正订购值与ship_date的y,m（且sku按照总需求量排序）
        final.append(info)

    final.reverse()  #反向排序，得到降序序列（目的是后续对订购量前百sku进行分析）

    return final
