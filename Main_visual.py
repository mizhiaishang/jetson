import os
import json
import logging
from datetime import datetime
import faiss
import argparse
from config import Config
from Graph_Process import GraphProcess
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, write_dot
import faiss
import numpy as np

_cached_rag = None
CUDA_VISIBLE_DEVICES=1

method_map = {
    'semantic': 'implicit',
    'llm_hierarchical': 'None'
}



def setup_logging(method_name, query_type):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('experiment_logs', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'experiment_logs/{method_name}_{query_type}_{timestamp}.log'
    
    # Get the logger
    logger = logging.getLogger('experiment')
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def main_1(json_data,folder_name):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive Embodied Navigation System')
    parser.add_argument('--method', type=str, 
                       choices=Config.RETRIEVAL_METHODS.keys(),
                       help='Retrieval method to use')
    parser.add_argument('--query-type', type=str, 
                       choices=Config.QUERIES.keys(),
                       help='Type of queries to handle')
    args = parser.parse_args([])     # 关键：不读取命令行输入
    args.method = "semantic"
    args.query_type = "implicitecho"

    # 初始化日志
    logger = setup_logging(args.method, args.query_type)

    # 初始化知识图谱处理类
    global _cached_rag
    if _cached_rag is None:
        logger.info("Creating new EmbodiedRAG instance...")
        _cached_rag = GraphProcess(
            working_dir="./embodied_nav_cache",
            airsim_utils=None,
            retrieval_method=method_map[args.method]
        )


    recognition_results = json_data

    subgraph_visual = _cached_rag.build_knowledge_subgraph_visual(recognition_results)

    

    subgraph_save_path = f"{folder_name}.html"
    _cached_rag.visualize_knowledge_graph(_cached_rag.globalgraph, output_file=subgraph_save_path)
    logger.info(f"当前步骤全局知识图已保存为 {subgraph_save_path}")

    
    print("Before reanage_edges:", len(_cached_rag.globalgraph.edges))
    print("Nodes:", _cached_rag.globalgraph_node_index)
    print("landmark_index:", _cached_rag.landmark_index)
    print("landmark_positions:", _cached_rag.landmark_positions.shape)
    print("occupied_positions:", _cached_rag.occupied_positions.shape)

    # _cached_rag.reanage_edges(delete_all=True, object_degree=2)
    # _cached_rag.reanage_edges(delete_all=True, landmark_degree=2)
    _cached_rag.reanage_edges(delete_all=True, object_degree=2, landmark_degree=2)
    subgraph_save_path = f"graphs_minicar/visual_finall_current_globalgraph.html"
    _cached_rag.visualize_knowledge_graph(_cached_rag.globalgraph, output_file=subgraph_save_path)
    logger.info(f"当前步骤全局知识图已保存为 {subgraph_save_path}")


    
    _cached_rag.globalgraph = nx.DiGraph()
    _cached_rag.globalgraph_embeddings = np.empty((0, 1024), dtype='float32')  # 假设embedding维度为1024
    _cached_rag.globalgraph_node_index = {}

    # 构建完成后，将global_graph, global_embeddings, global_node_index存储到磁盘
    if _cached_rag.globalgraph is not None:
        _cached_rag.save_global_graph()
        logger.info("全局知识图谱,embedding,index已保存到磁盘")

    
    ###########################################################################################
    # 检索模块测试
    ###########################################################################################

    node_list, subgraph = _cached_rag.retriever.retrieve_word(['洗手台','水果'], _cached_rag.globalgraph, _cached_rag.globalgraph_embeddings, _cached_rag.globalgraph_node_index, top_k=3, threshold=1, hops=2, return_graph=True)
    print(node_list, subgraph)
    shortest_path = _cached_rag.retriever.shortest_path(_cached_rag.globalgraph, source='镜子', target='液体')
    print(shortest_path)

# if __name__ == "__main__":
#     main()