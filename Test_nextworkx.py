import networkx as nx
from pyvis.network import Network
import os


triplet1 = [{'subject': '镜子', 'relation': '右侧', 'object': '门洞'}, {'subject': '镜子', 'relation': '下方', 'object': '洗手池'}, {'subject': '洗手池', 'relation': '放置', 'object': '容器'}, {'subject': '洗手池', 'relation': '放置', 'object': '液体'}, {'subject': '洗手池', 'relation': '放置', 'object': '塑料袋'}]
triplet2 = [{'subject': '入口', 'relation': '通向', 'object': '卧室'}, {'subject': '门内', 'relation': '包含', 'object': '卧室'}, {'subject': '床头', 'relation': '靠', 'object': '墙'}, {'subject': '床', 'relation': '旁边', 'object': '家具'}, {'subject': '镜子', 'relation': '挂在', 'object': '墙'}]


def add_object_node(graph, label, data=None):
    """添加物体节点到图谱中"""
    # 如果节点不存在，则添加
    if label not in graph:
        # print(f"Adding object node: {label}")
        try:
            graph.add_node(label, 
                            type='object',
                            level=0)
            print(f"Successfully added object node: {label}")
        except Exception as e:
            print(f"Error adding object node {label}: {e}")
    return graph

def build_knowledge_subgraph(triples):
    """构建子知识图谱"""

    subgraph = nx.DiGraph()
    if not triples:
        return None  # 新增：处理空三元组情况
    
    for triple in triples:
        # print(triple)
        add_object_node(subgraph, triple["subject"])
        add_object_node(subgraph, triple["object"])
        subgraph.add_edge(triple["subject"], triple["object"], attr=triple["relation"])

    return subgraph

def visualize_knowledge_graph(graph, output_file="knowledge_graph.html"):
    """修复可视化函数，解决模板渲染错误"""
    if not graph:
        print("无法可视化空图谱")
        return None
    
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
        net.add_node(
            n_id = node,
            label= node,
            title=f"实体: {node}",
            color="#4CAF50"
        )
        id+=1
    
    for edge in graph.edges(data=True):
        net.add_edge(
            edge[0],
            edge[1],
            label=edge[2]['attr'],
            title=edge[2]['attr'],
            color="#FF9800"
        )
    
    # 直接使用write_html方法，避免show()的复杂逻辑

    net.write_html(output_file, open_browser=False)
    # logger.info(f"知识图谱已保存至 {os.path.abspath(output_file)}")
    return True

# subgraph_1 = build_knowledge_subgraph(triplet1)
# subgraph_2 = build_knowledge_subgraph(triplet2)

# print(subgraph_1.nodes(data=True))
# print(subgraph_1.edges(data=True))
# print(subgraph_2.nodes(data=True))
# print(subgraph_2.edges(data=True))

# print(list(subgraph_1.neighbors('镜子')))