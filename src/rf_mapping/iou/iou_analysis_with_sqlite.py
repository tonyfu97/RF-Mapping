import os
import sys
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../../..')
import src.rf_mapping.constants as c

db_path = os.path.join(c.REPO_DIR, 'src', 'rf_mapping', 'iou', 'iou.db')
connection = sqlite3.connect(db_path)
cur = connection.cursor()


def get_table_names(cur):
    sql_query = "SELECT name FROM sqlite_master WHERE type='table';"
    results = cur.execute(sql_query)
    return [t[0] for t in results.fetchall()]
table_names = get_table_names(cur)


def get_iou_list(table_name, map1_name, map2_name, cur):
    sql_query = f"""
                SELECT LAYER, AVG({map1_name}_vs_{map2_name})
                FROM {table_name}
                GROUP BY LAYER
    """
    results = cur.execute(sql_query)
    return [layer[1] for layer in results]

    

plt.figure(figsize=(14,7))
for table_name in ['ALEXNET_MAX_IOU', 'VGG16_MAX_IOU', 'RESNET18_MAX_IOU']:
    iou_list = get_iou_list(table_name, 'gt', 'rfmp4a', cur)
    plt.plot(iou_list, '.-', label=table_name)
plt.xlabel('layer')
plt.ylabel('intersection over union')
plt.title('Ground truth vs. RFMP4a')
plt.legend()
plt.show()

plt.figure(figsize=(14,7))
for table_name in ['ALEXNET_MAX_IOU', 'VGG16_MAX_IOU', 'RESNET18_MAX_IOU']:
    iou_list = get_iou_list(table_name, 'gt', 'rfmp4c7o', cur)
    plt.plot(iou_list, '.-', label=table_name)
plt.legend()
plt.xlabel('layer')
plt.ylabel('intersection over union')
plt.title('Ground truth vs. RFMP4c7o')
plt.show()
