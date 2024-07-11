# import json
# import csv
# from collections import Counter
#
#
# def read_json_file(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             case = json.loads(line.strip())  # 解析每一行作为一个单独的JSON对象
#             data.append(case)
#     return data
#
#
# def extract_accusations(data):
#     accusations_list = []
#     for case in data:
#         accusations_list.extend(case['meta']['accusation'])
#     return accusations_list
#
#
# def count_accusations(accusations_list):
#     return Counter(accusations_list)
#
#
# def save_to_csv(counter, output_file):
#     with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Accusation', 'Count'])
#         for item, count in counter.items():
#             writer.writerow([item, count])
#
#
# if __name__ == '__main__':
#     # 请将下面的路径替换为你的JSON文件的实际路径
#     json_file_path = 'E:/Data/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_train.json'
#
#     # 读取JSON文件
#     data = read_json_file(json_file_path)
#
#     # 提取所有指控
#     all_accusations = extract_accusations(data)
#
#     # 统计指控次数
#     accusation_counts = count_accusations(all_accusations)
#
#     # 将结果保存到CSV文件
#     output_csv_file = './result.cvs'
#     save_to_csv(accusation_counts, output_csv_file)


# import csv
#
#
# def add_id_to_rows(input_file, output_file):
#     id_counter = 0
#
#     with open(input_file, mode='r', encoding='utf-8') as infile:
#         reader = csv.DictReader(infile)
#         fieldnames = reader.fieldnames + ['id']  # 添加新的字段名'id'
#
#         with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
#             writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#             writer.writeheader()  # 写入新的表头
#
#             for row in reader:
#                 row['id'] = id_counter
#                 writer.writerow(row)
#                 id_counter += 1
#
#
# # 调用函数，传入输入文件和输出文件的路径
# input_csv_file = './result.cvs'
# output_csv_file = 'data/label.csv'
# add_id_to_rows(input_csv_file, output_csv_file)

import pandas as pd


def load_accusation_id_map(filename):
    # 使用pandas读取CSV文件
    df = pd.read_csv(filename)

    # 将DataFrame转换为字典，其中'Accusation'列作为键，'id'列作为值
    accusation_id_map = df.set_index('Accusation')['id'].to_dict()

    return accusation_id_map


# 调用函数并传入你的CSV文件路径
filename = 'D:/pycharm-workspace/Mind/data/label.csv'
accusation_id_map = load_accusation_id_map(filename)

# 打印结果字典以确认
print(accusation_id_map)
