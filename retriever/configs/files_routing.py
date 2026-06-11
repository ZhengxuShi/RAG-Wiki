import json
import logging
import os
import sys
from pathlib import Path
import re

class ClassificationRouter:
    def __init__(self, files_routing_path):
        self._initialize_components(files_routing_path)

    def _initialize_components(self,files_routing_path):
        """初始化系统核心组件"""

        # 从 JSON 文件加载配置信息
        with open(files_routing_path, 'r', encoding='utf-8') as f:
            routing_data = json.load(f)
        self.candidate_files = routing_data.get("candidate_files", {})
        self.category_map = routing_data.get("category_map", {})
    def _text_preprocess(self, text,logging):
        """执行文本预处理流程"""
        processed_query,num = self._normalize_special_terms(text,logging)
        if num == 0:
            return []
        else:
            return processed_query

    def _normalize_special_terms(self, text,logging):
        """标准化特殊术语"""
        try:
            category_list = []
            # 第一次遍历：检查候选文件
            for category in self.candidate_files:
                if category.lower() in text.lower():
                    category_list.append(category)

            # 第二次遍历：进行替换操作
            for category, patterns in self.category_map.items():
                for pattern in patterns:
                    # 判断pattern是否全是英文字母
                    if pattern.isalpha():
                        # 如果是英文，使用更严格的正则匹配，确保左右不是字母
                        regex = r'(?<![a-zA-Z])' + re.escape(pattern) + r'(?![a-zA-Z])'
                        if re.search(regex, text, re.IGNORECASE):
                            category_list.append(category)
                            break
                    else:
                        # 非全字母模式保持原有逻辑
                        if re.search(pattern, text, re.IGNORECASE):
                            category_list.append(category)
                            break

            if category_list:  # 如果任一循环找到了匹配项
                return category_list, 1
            else:  # 两个循环都没找到匹配项
                # print(f"{text}问题中无路由文件信息")
                logging.info(f"{text}问题中无路由文件信息")
                return text, 0

        except Exception as e:
            # print(f"{text}问题中无路由文件信息")
            logging.info(f"{text}一级路由失败,错误：{str(e)}")
            return text, 0  # 确保在异常情况下也返回原始文本


    def keyword_based_route(self, query,logging):
        """执行基于关键词的精确路由"""
        filename = self._text_preprocess(query,logging)
        return filename

    def hybrid_route(self, query,logging):
        """混合路由策略"""
        keyword_results = self.keyword_based_route(query,logging)
        if keyword_results:
            return keyword_results
        else:
            logging.info("一级路由返回文件名为空！")
            return []


def find_project_root(start_dir=os.getcwd()):  # 寻找项目根路径
    current_dir = start_dir
    if getattr(sys, 'frozen', False):
        # 打包环境：返回可执行文件所在目录
        return Path(sys.executable).parent
    while True:
        if "main.py" in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the root directory
            raise FileNotFoundError("Could not find main.py in any parent directory.")
        current_dir = parent_dir


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_value_from_key_path(data_dict, key_path_list):
    """
    通过键路径列表获取值
    :param data_dict: 要查询的字典
    :param key_path_list: 键路径列表，例如 ["1st_retriever", "1005.detail", "general", "1000.param_meaning"]
    :return: 对应的值
    """
    current_data = data_dict
    for key in key_path_list:
        current_data = current_data[key]
    return current_data

def run_classification_router(query, retrieval_config= None,logging= logging):
    PROJECT_ROOT = find_project_root()
    os.chdir(PROJECT_ROOT)
    logging.info("路由项目根目录定位成功: %s", PROJECT_ROOT)

    # 配置参数路径定义
    files_routing_path_path_list = ["1002.files_routing", "1005.detail", "1002.files_routing_path", "1001.value"]
    # 从配置中提取参数
    try:
        files_routing = get_value_from_key_path(retrieval_config, files_routing_path_path_list)
        logging.info("成功从路由配置中提取参数")
    except KeyError as e:
        logging.error("配置参数缺失: %s", str(e))
        raise

    # 初始化路由系统
    try:
        router = ClassificationRouter(files_routing_path=files_routing)
        logging.info("分类路由系统初始化成功")
    except Exception as e:
        logging.error("路由系统初始化失败: %s", str(e))
        raise

    # 执行混合路由策略
    try:
        results = router.hybrid_route(query,logging)
        logging.info(f"一级路由策略执行成功，返回 {len(results)} 个结果")
        return results
    except Exception as e:
        logging.error("执行一级路由策略时出错: %s", str(e))
        raise


def test_classification_router(retrieval_config= None, queries_json_path= None):
    """
    测试分类路由系统的功能。

    该函数从指定的配置文件中加载测试问题和相关配置，初始化分类路由系统，
    并对每个测试问题进行匹配，输出匹配结果。

    参数:
    retrieval_config (str): 配置文件的路径。如果为 None，则使用默认配置文件。

    返回值:
    无
    """

    # 初始化路由系统
    router = ClassificationRouter()

    # 对每个测试问题进行匹配
    print("开始测试分类路由系统...")
    matched_files = []
    queries, actual_files = get_all_queries_from_json(queries_json_path)
    for query in queries:
        results = router.hybrid_route(query)
        if len(results)==0:
            print("未匹配到任何文件")
        matched_files.append(results)
        print(f"测试案例 : {query}")
        print(f"匹配结果: {results}\n")
    return queries, actual_files, matched_files


def get_all_queries_from_json(file_path):
    data = read_json_file(file_path)
    queries = []
    actual_files = []
    for file_name in data:
        for item in data[file_name]:
            queries.append(item["query"])
            actual_files.append(file_name)
    return queries, actual_files


def save_results_to_excel(queries, actual_files, matched_files, output_path):
    df = pd.DataFrame({
        "query": queries,
        "actual_file": actual_files,
        "matched_file": matched_files
    })
    df.to_excel(output_path, index=False)


# 示例：测试函数接口
if __name__ == "__main__":
    PROJECT_ROOT = find_project_root()
    os.chdir(PROJECT_ROOT)
    retrieval_config = "configs/retriever_config_v3.json"
    retrieval_config_data = read_json_file(retrieval_config)
    import pandas as pd
    from dotenv import load_dotenv
    from datetime import datetime

    load_dotenv()
    # # 单个查询测试
    single_query = "日本电磁干扰控制委员会的英文全称是什么？"
    single_result = run_classification_router(query=single_query, retrieval_config=retrieval_config_data)
    print(f"单个查询测试: {single_query}")
    print(f"匹配结果: {single_result}\n")

    # # 批量测试
    # queries_json_path = r"data/query_answer/your_test_data.json"
    #
    # queries, actual_files, matched_files = test_classification_router(retrieval_config=retrieval_config_data,
    #                                                                   queries_json_path=queries_json_path)
    # # 将全部结果保存到 Excel 文件
    # # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # # output_excel_path = "routing_results" + timestamp + ".xlsx"
    # # save_results_to_excel(queries, actual_files, matched_files, output_excel_path)
    # # print(f"路由结果已保存到 {output_excel_path}")
    #
    # # 打印路由错误的文件
    # routing_errors = [(q, a, m) for q, a, m in zip(queries, actual_files, matched_files) if a not in m]
    # if routing_errors:
    #     print("路由错误的文件如下：")
    #     file_num = 0
    #     error_queries = []
    #     error_actual = []
    #     error_matched = []
    #
    #     for query, actual, matched in routing_errors:
    #         if matched:
    #             file_num += 1
    #             error_queries.append(query)
    #             error_actual.append(actual)
    #             error_matched.append(matched)
    #             print(f"查询: {query}, 实际文件: {actual}, 匹配文件: {matched}")
    #
    #     # # 将路由错误保存到单独的Excel文件
    #     # if error_queries:
    #     #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     #     error_excel_path = "routing_errors_" + timestamp + ".xlsx"
    #     #     save_results_to_excel(error_queries, error_actual, error_matched, error_excel_path)
    #     #     print(f"路由错误结果已保存到 {error_excel_path}")
    #
    #     print(f"错误文件数量: {file_num}")
    #
    # # 计算正确率
    # total_queries = len(queries)
    # correct_matches = total_queries - len(routing_errors)
    # accuracy = correct_matches / total_queries if total_queries > 0 else 0.0
    # print(f"问题总数: {total_queries}")
    # print(f"正确匹配数: {correct_matches}")
    # print(f"路由准确率: {accuracy:.2%}")
