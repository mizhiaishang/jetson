import os
import json
import logging
from datetime import datetime
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive Embodied Navigation System')
    parser.add_argument('--method', type=str, required=True, 
                       choices=Config.RETRIEVAL_METHODS.keys(),
                       help='Retrieval method to use')
    parser.add_argument('--query-type', type=str, required=True,
                       choices=Config.QUERIES.keys(),
                       help='Type of queries to handle')
    args = parser.parse_args()

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
    
    # # 加载现有的知识图谱，embeddings，索引（如果存在）
    # _cached_rag.load_global_graph()

    
    # 加载图像captions
    captions_file = "captions/caption.json"
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)


    # 逐条提取caption并处理
    for step, content in captions.items():
        logger.info("--------------------------------")
        logger.info(f"Processing {step}")

        # Extract caption
        caption = content.get("caption", "")

        extract_new = True # 设置为True以提取新三元组，False以加载预存三元组

        if extract_new is True:
            # Extract Triples with LLM
            triples = _cached_rag.llm.extract_triples(caption)
            if not triples:
                print("未能提取到任何知识三元组")
                continue

            logger.info(f"成功提取 {len(triples)} 个知识三元组：")
            print(f"Triples for {step}: {triples}") 

            # 将抽取的三元组存入JSON文件，便于调试
            triples_save_path = f"semantic_triplets/{step}_triples.json"
            with open(triples_save_path, 'w', encoding='utf-8') as f:
                json.dump(triples, f, ensure_ascii=False, indent=4)
        
        else:
            # 调试阶段直接加载预存的三元组文件
            triples_load_path = f"semantic_triplets/{step}_triples.json"
            # 判断三元组文件是否存在
            if not os.path.exists(triples_load_path):
                logger.error(f"Triples file {triples_load_path} does not exist. Skipping step.")
                continue
            
            with open(triples_load_path, 'r', encoding='utf-8') as f:
                triples = json.load(f)


        # Build SubKnowledge Graph
        subgraph = _cached_rag.build_knowledge_subgraph(triples)
        if not subgraph:
            print("构建知识图谱失败")
            continue

        # HTML可视化存储
        subgraph_save_path = f"graphs/{step}_knowledge_graph.html"
        _cached_rag.visualize_knowledge_graph(subgraph, output_file=subgraph_save_path)
        logger.info(f"子图已保存为 {subgraph_save_path}")

        # 为子图中的节点生成embedding并存储
        node_labels, embeddings = _cached_rag.generate_node_embeddings(subgraph)


        # 判断全局图谱是否存在，如果不存在，当前子图作为全局图谱存入_cached_rag.globalgraph, 如果存在，则检索后合并。
        if _cached_rag.globalgraph is None:
            logger.info("全局知识图谱为空，已将当前子图设为全局图谱")
            _cached_rag.add_subgraph_to_global(subgraph)
        else:
            logger.info("全局知识图谱已存在，从全局节点中检索相似子图，并增强当前子图，增强后的子图合并入全局图谱")

            # 获取当前步记忆增强子图
            memory_enhanced_subgraph = _cached_rag.memory_enghanced_subgraph_construction(subgraph, embeddings, node_labels)  
            subgraph_save_path = f"graphs/{step}_enhanced_knowledge_graph.html"
            _cached_rag.visualize_knowledge_graph(memory_enhanced_subgraph, output_file=subgraph_save_path)
            logger.info(f"子图已保存为 {subgraph_save_path}")

            # 将增强后的子图合并入全局图谱
            _cached_rag.add_subgraph_to_global(memory_enhanced_subgraph)
            subgraph_save_path = f"graphs/{step}_current_globalgraph.html"
            _cached_rag.visualize_knowledge_graph(_cached_rag.globalgraph, output_file=subgraph_save_path)
            logger.info(f"当前步骤全局知识图已保存为 {subgraph_save_path}")

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

if __name__ == "__main__":
    main()