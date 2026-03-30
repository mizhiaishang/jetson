import networkx as nx
from networkx.algorithms import isomorphism
import logging
import time
import os
from openai import OpenAI
import faiss
from faiss import IndexFlatL2
from Test_nextworkx import build_knowledge_subgraph, visualize_knowledge_graph
import numpy as np

logger = logging.getLogger(__name__)

aly_api_key='sk-64e0e5199d1a4d40a14c0eb8be02fc8d'
aly_api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'

client = OpenAI(
    api_key=aly_api_key,  
    base_url=aly_api_url  
)

class EmbodiedRetriever:
    def __init__(self, ):
        self.embedding_modal = client.embeddings
        pass

    def retrieve_word(self, query, graph, graph_embeddings, graph_node_index, top_k=5, threshold=0.7, hops=1, return_graph=True):
        """Retrieve relevant nodes from the knowledge graph based on the query.

        Args:
            query (str): 一个或几个词或短语，如['玻璃制品']，['玻璃制品','水果']
            graph (networkx.Graph): The global_knowledge graph.
            top_k (int): Number of top relevant nodes to retrieve.
            threshold (float): Similarity threshold for retrieval.
            hops (int): Number of hops to consider in the graph.
            return_graph (bool): Whether to return the subgraph or just node IDs.

        Returns:
            list or networkx.Graph: List of relevant node IDs or the subgraph.
            如果低于Threshold的节点数小于Topk，则以Threshold的判定为准
        """
        # Placeholder implementation
        word_embedding = self.embedding_modal.create(
                model="text-embedding-v4",
                input = query,
                dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
                encoding_format="float"
            )
        # word_embedding = np.array(word_embedding.data[0].embedding).reshape(1, -1)
        word_embedding = np.array([embedding.embedding for embedding in word_embedding.data])
        # print(word_embedding.shape)
    
        global_index = faiss.IndexFlatL2(graph_embeddings.shape[1])     # L2距离的暴力搜索索引
        global_index.add(graph_embeddings.astype('float32'))       
        D, I = global_index.search(word_embedding.astype('float32'), top_k)
        print(D, I)

        similar_nodes = [graph_node_index[i] for i, d in zip(I.flatten(), D.flatten()) if d < threshold]
        print("相似节点:", similar_nodes) # 相似节点: ['房间', '镜子']

        if similar_nodes==[]:
            print("未查找到任何相似节点")
            return None, None
        
        node_list=similar_nodes
        for i in range(hops):
            temp = []
            for node in node_list:
                neighbor_nodes = list(graph.neighbors(node))
                if neighbor_nodes is not None:
                    temp.extend(neighbor_nodes)
            node_list.extend(temp)
        

        node_list = list(set(node_list))
        # print(subgraph)
    
        if return_graph is True:
            subgraph = graph.subgraph(node_list)
            return node_list, subgraph
        else:
            return node_list, None

    def shortest_path(self, graph, source, target):
        # 判断 source, target是否在graph中，如果在则找到两个目标间的最短路径并返回。
        if source in graph.nodes and target in graph.nodes:
            try:
                path = nx.shortest_path(graph, source=source, target=target)
                return path
            except nx.NetworkXNoPath:
                logger.warning(f"No path found between {source} and {target}.")
                return None
        else:
            logger.warning(f"Either {source} or {target} is not in the graph.")
            return None
        
    def PPR(self, graph, seeds=None, alpha=0.9):
        return nx.pagerank(graph, personalization=seeds, alpha=alpha)
        


def main():
    EmbodiedRetriever_instance = EmbodiedRetriever()
    triplet1 = [{'subject': '镜子', 'relation': '右侧', 'object': '门洞'}, 
                {'subject': '镜子', 'relation': '左侧', 'object': '窗户'}, 
                {'subject': '门洞', 'relation': '通往', 'object': '房间'},
                {'subject': '房间', 'relation': '包含', 'object': '木质家具'},  
                {'subject': '木质家具', 'relation': '放置', 'object': '苹果'},  
                {'subject': '镜子', 'relation': '下方', 'object': '洗手池'}, 
                {'subject': '洗手池', 'relation': '放置', 'object': '容器'}, 
                {'subject': '洗手池', 'relation': '放置', 'object': '液体'}, 
                {'subject': '洗手池', 'relation': '放置', 'object': '塑料袋'},
                ]
    graph = build_knowledge_subgraph(triplet1)

    subgraph_save_path = f"Test_globalgraph.html"
    visualize_knowledge_graph(graph, output_file=subgraph_save_path)
    logger.info(f"当前步骤全局知识图已保存为 {subgraph_save_path}")

    node_numbers = len(graph.nodes) 
    node_labels = [None]*node_numbers

    i=0
    for node, data in graph.nodes(data=True):
        node_labels[i] = node
        i+=1

    print("Node Labels: ",node_labels)
    # Node Labels:  ['镜子', '门洞', '窗户', '房间', '木质家具', '苹果', '洗手池', '容器', '液体', '塑料袋']

    batch_embeddings = client.embeddings.create(
                model="text-embedding-v4",
                input=node_labels,
                dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
                encoding_format="float"
            )
    graph_embeddings = np.array([embedding.embedding for embedding in batch_embeddings.data])
    graph_node_index = {idx : node for idx, node in enumerate(graph.nodes)}

    node_list, subgraph = EmbodiedRetriever_instance.retrieve_word(['玻璃制品'], graph, graph_embeddings, graph_node_index, top_k=3, threshold=1, hops=1, return_graph=True)
    print(node_list, subgraph)

    subgraph_save_path = f"Test_subgraph.html"
    visualize_knowledge_graph(subgraph, output_file=subgraph_save_path)
    logger.info(f"当前步骤全局知识图已保存为 {subgraph_save_path}")

    shortest_path = EmbodiedRetriever_instance.shortest_path(graph, source='镜子', target='液体')
    # print(shortest_path)

    global_graph = nx.read_gml('embodied_nav_cache/global_knowledge_graph.gml')

    # pr = nx.pagerank(global_graph, personalization={'镜子':0.7, '窗户':0.3},alpha=0.9)
    # pr = nx.pagerank(global_graph, personalization={'门洞':0.5},alpha=0.9)
    # pr = EmbodiedRetriever_instance.PPR(global_graph, {'镜子':0.5,'苹果':0.5})
    # pr = nx.pagerank(graph, personalization={'镜子':0.5,'苹果':0.5},alpha=0.9)
    # pr = nx.pagerank(graph, personalization={'镜子':1},alpha=0.9)
    # print(pr)
    # a = nx.community.louvain_communities(global_graph, seed=123)

    # for i in range(len(a)):
    #     print(a[i])

    GM = isomorphism.GraphMatcher(subgraph, graph)
    print(GM.mapping)

if __name__ == "__main__":
    main()