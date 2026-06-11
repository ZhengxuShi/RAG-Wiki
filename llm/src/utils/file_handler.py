import json


def load_config(config_path):
    """ 加载 JSON 配置文件 """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as file:
        return json.load(file)


import csv
import os


def save_to_csv(file_path, data, headers):
    """保存数据到 CSV 文件"""
    file_exists = os.path.isfile(file_path)

    # 打开文件，如果文件存在则追加，否则写入（新建文件）
    with open(file_path, 'a' if file_exists else 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerows(data)
