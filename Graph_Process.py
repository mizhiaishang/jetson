# from _spatial_relationship_extractor import SpatialRelationshipExtractor
# from _embodied_retriever import EmbodiedRetriever, RetrievalMethod
from config import Config
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os
os.environ["LD_PRELOAD"] = "/home/nvidia/miniconda3/envs/langchain/lib/libgomp.so.1"
import faiss
from faiss import IndexFlatL2
from Retriever import EmbodiedRetriever

import networkx as nx
from pyvis.network import Network
import logging
import numpy as np
import json
import os
import time
import pickle
import asyncio
from openai import AsyncOpenAI
from LLM import LLMInterface
from tqdm import tqdm
from itertools import islice
from datetime import datetime

# Use the same logger name as the app-level configuration so that
# log messages emitted from this module get handled by the configured
# 'experiment' logger in `Main_visual.setup_logging`
# logger = logging.getLogger('experiment')

aly_api_key='sk-64e0e5199d1a4d40a14c0eb8be02fc8d'
aly_api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'

client = OpenAI(
    api_key=aly_api_key,  
    base_url=aly_api_url  
)

class GraphProcess:

    def __init__(self, working_dir, airsim_utils=None, retrieval_method= 'implicit'):
        
        # 图相关
        self.working_dir = working_dir
        self.logger = logging.getLogger('experiment')
        self.cache_file = os.path.join(working_dir, "llm_response_cache.json")
        self.llm = LLMInterface()

        # 纯视觉建图相关
        self.defined_landmarks = ["bed", "table", "sofa", "tv", "cabinet", "desk", "door", "window", "bookshelf", "bathtub", "piano", "shelf"]
        self.defined_ignore = ["guard","ground"]
        self.confidence_lamda = 0.7
        self.distance_check = 0.4
        self.occupied_positions = np.empty((0, 3), dtype='float32')  # 用于存储已占用的位置坐标
        self.landmark_positions = np.empty((0, 3), dtype='float32')  # 用于存储地标位置坐标
        self.landmark_index = {}

        # 检索相关
        self.retriever = EmbodiedRetriever()
        self.globalgraph = nx.DiGraph()
        self.globalgraph_embeddings = np.empty((0, 1024), dtype='float32')  # 假设embedding维度为1024
        self.globalgraph_node_index = {}

        # Embedding相关
        self.embedding_modal = client.embeddings
    
    def visualize_knowledge_graph(self, graph, output_file="knowledge_graph.html"):
        """修复可视化函数，解决模板渲染错误"""
        # if not graph:
        #     print("无法可视化空图谱")
        #     return None
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 初始化图时指定notebook=False（关键修复）
        net = Network(
            directed=True, 
            height="1700px", 
            width="100%", 
            bgcolor="#f5f5f5", 
            font_color="black",
            notebook=False  # 新增：明确指定非 notebook 环境
        )
    
        # 添加节点和边
        id=0
        for node in graph.nodes(data=False):
            # 如果为 object 类型节点，使用灰色；如果为 landmark 类型节点，使用绿色
            node_color = "#4CAF50"  # 默认绿色
            title = f"实体：{node}" # 默认标题
            if 'type' in graph.nodes[node] and graph.nodes[node]['type'] == 'landmark':
                node_color = "#4CAF50"  # 绿色
                title=f"地标: {node}"
            else:
                node_color = "#9E9E9E"  # 灰色
                title=f"实体: {node}"

            net.add_node(
                n_id = node,
                label= node,
                title=title,
                color=node_color
            )
            id+=1
        
        for edge in graph.edges(data=True):
            # Add edge with label and distance in title
            # 如果包含'distance'属性，则显示距离，否则显示'N/A'
            distance = edge[2].get('distance', 'N/A')
            relation = edge[2].get('attr', 'N/A')

            title = f"{relation},{distance}"
            label = edge[2]['attr']
            net.add_edge(
                edge[0],
                edge[1],
                label=label,
                title=title,    
                color="#FF9800"
            )
        
        # 直接使用write_html方法，避免show()的复杂逻辑

        net.write_html(output_file, open_browser=False)
        self.logger.info(f"知识图谱已保存至 {os.path.abspath(output_file)}")
        return True

    def add_object_node(self, graph, label, data=None):
        """添加物体节点到图谱中"""
        # 如果节点不存在，则添加
        if label not in graph:
            try:
                graph.add_node(label, 
                              type='object',
                              level=0)
                print(f"Successfully added object node: {label}")
            except Exception as e:
                print(f"Error adding object node {label}: {e}")
        return graph

    def add_object_node_visual(self, label, confidence, position_3d, node_type="object", bbox=None):
        """添加物体节点到图谱中"""
        # 如果节点不存在，则添加
        if label in self.globalgraph:
            label = f"{label}_{len(self.globalgraph.nodes)}"  # 为避免重复标签，添加唯一后缀
            if label in self.globalgraph:
                label = f"{label}_{len(self.globalgraph.nodes)}"  # 当连续输入两个相同的标签，且第一个标签替换原本列表的某个标签时，会出bug,再加一个后缀
                self.logger.info(f"label: {label}")
        self.globalgraph.add_node(label, 
                                type=node_type,
                                level=0 if node_type=="object" else 1,
                                confidence=confidence,
                                position=position_3d,
                                bbox=bbox)
        # 将position_3d加入self.occupied_positions
        self.occupied_positions = np.vstack((self.occupied_positions, np.array(position_3d)))
        # 如果是地标节点，将position加入self.landmark_positions, 建立索引
        if node_type == "landmark":
            self.logger.info(f"before:{self.landmark_index}{self.globalgraph_node_index}")

        if node_type == "landmark":
            self.landmark_positions = np.vstack((self.landmark_positions, np.array(position_3d)))
            self.landmark_index[len(self.landmark_index)] = label
            # 当加入第一个landmark节点时，将图中现有所有的节点与它进行关联
            if len(self.landmark_index) == 1:
                self.reanage_edges()
        
        # 计算embedding 并加入self.globalgraph_embeddings 和 self.globalgraph_node_index
        node_embedding = self.generate_node_embeddings_single(label)
        self.globalgraph_embeddings = np.vstack((self.globalgraph_embeddings, node_embedding))
        self.globalgraph_node_index[len(self.globalgraph_node_index)] = label
        self.logger.info(f"Successfully {node_type} node: {label}, position: {position_3d}, confidence: {confidence} ")

        if node_type == "landmark":
            self.logger.info(f"after:{self.landmark_index}{self.globalgraph_node_index}")

        return label
    
    def reanage_edges(self, delete_all=False, object_only=False, landmark_only=False, object_degree=1, landmark_degree=1):
        # 将slef.globalgraph中的边重新组织,为每个节点分配最近的landmark节点

        if delete_all:
            # 保留land_mark之间的边，但删除所有landmark与object的边
            edges_to_remove = [(u, v) for u, v, d in self.globalgraph.edges(data=True) if self.globalgraph.nodes[u]['type'] == 'object' and self.globalgraph.nodes[v]['type'] == 'landmark']    
            self.globalgraph.remove_edges_from(edges_to_remove)
        
        if object_only and landmark_only:
            self.logger.warning("Both object_only and landmark_only flags are set. No edges will be rearranged.")
            return
    
        if object_only == False:
            for node in self.globalgraph.nodes:
                if self.globalgraph.nodes[node]['type'] == 'landmark':
                    self.find_nearest_landmark(node, np.array(self.globalgraph.nodes[node]['position']), degree=landmark_degree)
        
        if landmark_only == False:   
            for node in self.globalgraph.nodes:
                if self.globalgraph.nodes[node]['type'] == 'object':
                    self.find_nearest_landmark(node, np.array(self.globalgraph.nodes[node]['position']), degree=object_degree)      
            
        

    def delete_landmark(self, new_label, old_landmark):
        # 将landmark节点改为object节点, 并更新相关数据结构
        if new_label not in self.globalgraph:
            self.logger.warning(f"Node {new_label} not found in global graph. Cannot delete landmark.")
            return
        self.globalgraph.nodes[new_label]['type'] = 'object'

        # 将关联到该landmark节点的边删除
        edges_to_remove = [(u, v) for u, v, d in self.globalgraph.edges(data=True) if v == new_label]
        self.globalgraph.remove_edges_from(edges_to_remove)

        # 从landmark_positions和landmark_index中删除该节点的信息
        self.logger.info(f"{new_label},{old_landmark},{self.landmark_index}")
        if old_landmark in self.landmark_index.values():
            index_key = [key for key, value in self.landmark_index.items() if value == old_landmark][0]

            self.landmark_positions = np.delete(self.landmark_positions, index_key, axis=0)
            del self.landmark_index[index_key]
            self.landmark_index = {new_key: self.landmark_index[old_key] for new_key, old_key in enumerate(sorted(self.landmark_index.keys()))} 
        
        self.logger.info(f"{new_label},{old_landmark},{self.landmark_index}")

        self.reanage_edges()

        return

    def update_landmark_index(self, new_label, old_landmark, position):
        
        if old_landmark in self.landmark_index.values():
            index_key = [key for key, value in self.landmark_index.items() if value == old_landmark][0]
            self.landmark_positions[index_key] = np.array(position)
            self.landmark_index[index_key] = new_label        
        return

    def renew_node_in_visual(self, current_label, New_confidence, New_position_3d, index=None, New_label=None):
        """更新视觉图谱中的节点信息"""
        if current_label not in self.globalgraph:
            self.logger.warning(f"Node {current_label} not found in global graph. Cannot renew.")
            return
        
        # 更新置信度和位置
        self.globalgraph.nodes[current_label]["confidence"] = New_confidence
        self.globalgraph.nodes[current_label]["position"] = New_position_3d

        # 如果提供了新标签，更新标签
        if New_label is not None:
            if New_label in self.globalgraph:
                New_label = f"{New_label}_{len(self.globalgraph.nodes)}"  # 为避免重复标签，添加唯一后缀
            nx.relabel_nodes(self.globalgraph, {current_label: New_label}, copy=False)

            # 更新 index 对应的 self.globalgraph_node_index 中的标签
            # 更新 index 对应的 self.globalgraph_embeddings 中的embedding。
            # 更新 index 对应的 self.occupied_positions 中的位置。
            # 如果 index 未提供，则直接报错返回
            if index is None:
                self.logger.warning(f"Index for node {current_label} not found. Cannot update index-based structures.")
                return

            self.globalgraph_node_index[index] = New_label
            self.globalgraph_embeddings[index] = self.generate_node_embeddings_single(New_label)
            self.occupied_positions[index] = np.array(New_position_3d)

            self.logger.info(f"Node {current_label} relabeled to {New_label}.")

        return New_label

    def find_nearest_landmark(self, class_name, position_3d, degree=1):
        """寻找最近的地标节点并添加边"""
        if len(self.landmark_index) < degree:
            self.logger.warning(f"Not enough landmark nodes to find nearest for {class_name}. Required: {degree}, Available: {len(self.landmark_index)}")
            return  # 如果landmark节点不足degree个，直接返回

        min_distance = float('inf')
        nearest_landmark = None

        distances_to_landmarks = np.linalg.norm(self.landmark_positions - position_3d, axis=1)

        # 根据找到对应degree个数的最近节点。
        for i in range(degree):
            if self.globalgraph.nodes[class_name]['type'] == 'landmark':
                # 如果是landmark节点，找到第i+1小的距离和对应地标添加边。
                if len(distances_to_landmarks) <= i+1:
                    self.logger.warning(f"Not enough landmark nodes to find the {i+1}th nearest for {class_name}.")
                    return
                sorted_distances = np.sort(distances_to_landmarks)
                target_distance = sorted_distances[i+1]
                target_index = np.argsort(distances_to_landmarks)[i+1]
                min_distance = target_distance
                nearest_landmark = self.landmark_index[target_index]
            else:
                target_index = np.argsort(distances_to_landmarks)[i]
                min_distance = distances_to_landmarks[target_index]
                nearest_landmark = self.landmark_index[target_index]    
            print(f"Nearest landmark to {class_name} at {position_3d} is {nearest_landmark} with distance {min_distance}")
        
            # 添加边并为每条边添加空间关系属性和距离属性
            print(nearest_landmark, class_name)
            relation = self.relationship_extractor(np.array(self.globalgraph.nodes[nearest_landmark]['position']), position_3d)
            if relation:
                self.globalgraph.add_edge(class_name, nearest_landmark, attr=relation, distance=min_distance)

                self.logger.info(f"Added edge from {class_name} to nearest landmark {nearest_landmark} with relation {relation} and distance {min_distance}")
            else:
                return
            

    def relationship_extractor(self, landmark_pos, object_pos):
        """计算object与landmark的空间关系"""
        # 计算两点之间的向量
        vector = object_pos - landmark_pos
        distance = np.linalg.norm(vector)
        
        # 如果distance为0，logger警告并返回None
        if distance == 0:
            self.logger.warning("Object and landmark positions are identical; cannot determine spatial relationship.")
            return None

        # 计算单位向量
        unit_vector = vector / distance

        # 设置阈值
        threshold = 0.577 # 1/sqrt(3) 约为0.577

        # 判断空间关系
        if abs(unit_vector[0]) > threshold:
            relation = "left_of" if unit_vector[0] > 0 else "right_of"
        elif abs(unit_vector[1]) > threshold:
            relation = "in_front_of" if unit_vector[1] > 0 else "behind"
        else:
            relation = "above" if unit_vector[2] > 0 else "below"
        
        return relation




    def build_knowledge_subgraph_visual(self, recognition_results):
        """纯视觉识别结果构建子知识图谱"""

        # 从recognition_results中提取识别结果
        recognition_results = recognition_results.get("detections")
        subgraph = nx.DiGraph()

        for detection in recognition_results:            
            is_occupied = False
            class_name = detection.get("class_name")
            confidence = detection.get("confidence")
            box = detection.get("bbox")
            position_3d = np.array(detection.get("global_position"))

            # 如果识别到的物体，‘confidence’低于self.confidence_lamda，则忽略该物体。
            if confidence < self.confidence_lamda:
                self.logger.info(f"Ignoring low confidence detection: {class_name} ({confidence})")
                continue

            if class_name in self.defined_ignore:
                self.logger.info(f"Ignoring detection in baned list: {class_name}.")
                continue

            # 计算当前检测物体与已占用位置的距离。
            distances = np.linalg.norm(self.occupied_positions - position_3d, axis=1)
            if np.any(distances < self.distance_check):
                is_occupied = True
                position = list((distances < self.distance_check)).index(True)
                occupied_node = self.globalgraph_node_index[position]
                self.logger.info(f"Label {class_name} Position {position_3d} is occupied at index {position} by node {occupied_node}")

                if class_name == occupied_node.split('_', 1)[0]:
                    # 如果是同一物体类别，更新该节点的confidence为二者较大值，坐标为二者平均值
                    existing_confidence = self.globalgraph.nodes[occupied_node].get("confidence", 0)
                    new_confidence = max(existing_confidence, confidence)
                    new_position = ((np.array(self.globalgraph.nodes[occupied_node].get("position")) + position_3d) / 2).tolist()
                    self.logger.info(f"Same class detected for occupied node {occupied_node}. Updating confidence and position.")
                    self.renew_node_in_visual(occupied_node, new_confidence, new_position)
                    self.logger.info(f"Updated occupied node {occupied_node} with new confidence {new_confidence} and position {new_position}")
                else:
                    # 如果是不同物体类别，比较原节点与现节点的confidence，保留confidence较高的节点，更新位置为较高confidence节点的位置，标签替换为新的class_name
                    existing_confidence = self.globalgraph.nodes[occupied_node].get("confidence", 0)
                    if confidence > existing_confidence:
                        occupied_new_label = self.renew_node_in_visual(occupied_node, confidence, position_3d.tolist(), index=position, New_label=class_name)
                        self.logger.info(f"Replaced occupied node {occupied_node} with new class {occupied_new_label}, confidence {confidence} and position {position_3d.tolist()}")
                    
                        # 如果新类别是普通节点旧类别是landmark节点，将旧类别从landmark中删除。
                        if self.globalgraph.nodes[occupied_new_label]['type'] == 'landmark' and class_name not in self.defined_landmarks:
                            self.delete_landmark(new_label=occupied_new_label, old_landmark=occupied_node)
                            self.logger.info(f"Deleted landmark status of node {occupied_node} as new class {occupied_new_label} is not a landmark")
                        
                        # 如果新类别是landmark而旧类别是普通节点，将旧节点改为landmark节点，并更新index相关信息。
                        if class_name in self.defined_landmarks and self.globalgraph.nodes[occupied_new_label]['type'] == 'object':
                            self.globalgraph.nodes[occupied_new_label]['type'] = 'landmark'
                            self.globalgraph.nodes[occupied_new_label]['level'] = 1

                            self.landmark_positions = np.vstack((self.landmark_positions, np.array(self.globalgraph.nodes[occupied_new_label]['position'])))
                            self.landmark_index[len(self.landmark_index)] = occupied_new_label
                            self.reanage_edges()
                        

                        # 如果新旧类别都为landmark节点，更新landmark_index。
                        if class_name in self.defined_landmarks and self.globalgraph.nodes[occupied_new_label]['type'] == 'landmark':
                
                            self.update_landmark_index(occupied_new_label, occupied_node, position_3d.tolist())




            if is_occupied is False:
                self.logger.warning(f"{class_name}")
                if class_name in self.defined_landmarks:
                    class_withid = self.add_object_node_visual(class_name, confidence, node_type="landmark", position_3d=position_3d, bbox=box)
                    self.logger.info(f"Added landmark node: {class_name} ({confidence})")
                else:  
                    class_withid = self.add_object_node_visual(class_name, confidence, node_type="object", position_3d=position_3d, bbox=box)
                    self.logger.info(f"Added object node: {class_name} ({confidence})")
                self.find_nearest_landmark(class_withid, position_3d)

            # print(self.occupied_positions.shape)
            # print(self.globalgraph_embeddings.shape)
            # print(self.globalgraph)
            # print(self.globalgraph_node_index)
    
        return None

    def generate_node_embeddings(self, graph):
        """为图谱中的节点生成embedding"""
        if not graph:
            print("无法为空图谱生成embedding")
            return None

        node_numbers = len(graph.nodes) 
        node_labels = [None]*node_numbers

        i=0
        for node, data in graph.nodes(data=True):
            node_labels[i] = node
            i+=1

        # 由于openai的嵌入接口限制，分批处理节点，每批次10个, 并将结果合并numpy数组。
        embeddings = []
        iter_num = node_numbers // 10 + (1 if node_numbers %10 !=0 else 0)

        for j in range(iter_num):
            batch_labels = node_labels[j*10 : min((j+1)*10, node_numbers)]
            # print("Processing batch:", batch_labels)                
            batch_embeddings = self.embedding_modal.create(
                model="text-embedding-v4",
                input=batch_labels,
                dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
                encoding_format="float"
            )
            batch_embeddings_np = np.array([embedding.embedding for embedding in batch_embeddings.data])

            embeddings = np.vstack((embeddings, batch_embeddings_np)) if len(embeddings)>0 else batch_embeddings_np

        return node_labels, embeddings

    def generate_node_embeddings_single(self, node_label):
        """为单个节点生成embedding"""
        if not node_label:
            print("无法为空节点生成embedding")
            return None

        embedding_np = self.embedding_modal.create(
                model="text-embedding-v4",
                input=[node_label],
                dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
                encoding_format="float"
            )
        embedding_np = np.array([embedding.embedding for embedding in embedding_np.data])   

        return embedding_np 

    def add_subgraph_to_global(self, subgraph):
        """将子图添加到全局知识图谱中"""
        if len(self.globalgraph.nodes) == 0:
            # 如果全局图不存在，子图，embeddings存入全局图并生成当前索引。
            self.globalgraph = subgraph
            subgraph_node_labels, subgraph_embeddings = self.generate_node_embeddings(subgraph)
            self.globalgraph_embeddings = subgraph_embeddings
            for i, node in enumerate(subgraph_node_labels):
                self.globalgraph_node_index[i] = node
            
            print("全局知识图谱节点数量:", self.globalgraph_embeddings.shape[0])
            print("全局知识图谱向量维度:", self.globalgraph_embeddings.shape[1])

        else:
            # # 如果全局图存在，子图存入全局图, 特征向量，索引表合并。
            # self.globalgraph = nx.compose(self.globalgraph, subgraph)
            # subgraph_node_labels, subgraph_embeddings = self.generate_node_embeddings(subgraph)
            # self.globalgraph_embeddings = np.vstack((self.globalgraph_embeddings, subgraph_embeddings)) 
            # start_index = len(self.globalgraph_node_index)
            # for i, node in enumerate(subgraph_node_labels):
            #     self.globalgraph_node_index[start_index + i] = node     

            # 如果全局图存在，子图存入全局图, 特征向量，索引表合并。
            # self.globalgraph = nx.compose(self.globalgraph, subgraph)
            subgraph_node_labels, subgraph_embeddings = self.generate_node_embeddings(subgraph)

            i = 0
            for label, data in subgraph.nodes(data=True):
                if label not in self.globalgraph:
                    self.globalgraph.add_node(label, **data)
                self.globalgraph_embeddings = np.vstack((self.globalgraph_embeddings, subgraph_embeddings[i]))
                self.globalgraph_node_index[len(self.globalgraph_node_index)] = label
                i += 1
            
            for edge in subgraph.edges(data=True):
                if not self.globalgraph.has_edge(edge[0], edge[1]):
                    self.globalgraph.add_edge(edge[0], edge[1], **edge[2])

                
                # self.globalgraph.add_edges_from(subgraph.edges(label, data=True))
        
            print(len(self.globalgraph_node_index))
            print("全局知识图谱节点数量:", self.globalgraph_embeddings.shape[0])
            print("全局知识图谱向量维度:", self.globalgraph_embeddings.shape[1])

            # print(self.globalgraph_node_index)

    def memory_enghanced_subgraph_construction(self, subgraph, embeddings, node_labels):
        """基于全局图谱记忆增强子图"""
        if self.globalgraph is None:
            print("全局图谱为空，无法进行记忆增强")
            return subgraph

        # 构建FAISS索引
        global_index = faiss.IndexFlatL2(self.globalgraph_embeddings.shape[1])     # L2距离的暴力搜索索引
        global_index.add(self.globalgraph_embeddings.astype('float32'))                  # 添加全局图谱的所有节点向量

        # 在全局图谱中搜索相似节点
        D, I = global_index.search(embeddings.astype('float32'), 1)
        # print(D, I)

        D_threshold = 0.5
        similar_nodes = [self.globalgraph_node_index[i] for i, d in zip(I.flatten(), D.flatten()) if d < D_threshold]
        print("相似节点:", similar_nodes) # 相似节点: ['房间', '镜子']

        for node in similar_nodes:
            subgraph.add_nodes_from(self.globalgraph[node])
            node_edges = list(self.globalgraph.neighbors(node))
            for neighbor in node_edges:
                edge_data = self.globalgraph.get_edge_data(node, neighbor)
                subgraph.add_edge(node, neighbor, **edge_data)

        return subgraph

    def save_global_graph(self):
        """将全局图谱及其相关数据保存到磁盘"""
        if self.globalgraph is None:
            print("全局图谱为空，无法保存")
            return
        
        # 确保工作目录存在
        os.makedirs(self.working_dir, exist_ok=True)

        # 保存全局图谱
        global_graph_path = os.path.join(self.working_dir, "global_knowledge_graph.gml")
        nx.write_gml(self.globalgraph, global_graph_path)
        print(f"全局知识图谱已保存至 {global_graph_path}")

        # 保存全局嵌入向量
        embeddings_path = os.path.join(self.working_dir, "global_graph_embeddings.npy")
        np.save(embeddings_path, self.globalgraph_embeddings)
        print(f"全局知识图谱嵌入向量已保存至 {embeddings_path}")

        # 保存节点索引映射
        index_path = os.path.join(self.working_dir, "global_graph_node_index.pkl")
        with open(index_path, 'wb') as f:
            pickle.dump(self.globalgraph_node_index, f)
        print(f"全局知识图谱节点索引已保存至 {index_path}")

    def load_global_graph(self):
        """从磁盘加载全局图谱及其相关数据"""
        # 加载全局图谱
        global_graph_path = os.path.join(self.working_dir, "global_knowledge_graph.gml")
        if os.path.exists(global_graph_path):
            self.globalgraph = nx.read_gml(global_graph_path)
            print(f"全局知识图谱已从 {global_graph_path} 加载")
        else:
            print(f"全局知识图谱文件不存在: {global_graph_path}")
            return

        # 加载全局嵌入向量
        embeddings_path = os.path.join(self.working_dir, "global_graph_embeddings.npy")
        if os.path.exists(embeddings_path):
            self.globalgraph_embeddings = np.load(embeddings_path)
            print(f"全局知识图谱嵌入向量已从 {embeddings_path} 加载")
        else:
            print(f"全局知识图谱嵌入向量文件不存在: {embeddings_path}")
            return

        # 加载节点索引映射
        index_path = os.path.join(self.working_dir, "global_graph_node_index.pkl")
        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                self.globalgraph_node_index = pickle.load(f)
            print(f"全局知识图谱节点索引已从 {index_path} 加载")
        else:
            print(f"全局知识图谱节点索引文件不存在: {index_path}")
            return
    
    def build_knowledge_subgraph(self, triples):
        """构建子知识图谱"""

        subgraph = nx.DiGraph()
        if not triples:
            return None  # 新增：处理空三元组情况
        
        for triple in triples:
            self.add_object_node(subgraph, triple["subject"])
            self.add_object_node(subgraph, triple["object"])
            subgraph.add_edge(triple["subject"], triple["object"], attr=triple["relation"])
        return subgraph
        
    