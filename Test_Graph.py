from Test_Model import build_model, stream_invoke
from LLM import LLMInterface
import networkx as nx
from pyvis.network import Network
import json
import re
import os  # 新增：用于处理文件路径

# 初始化大模型
#ll_model = build_model()
ll_model = LLMInterface()

# print(llm)

def extract_triples(text):
    """使用stream_invoke从文本中提取知识三元组"""
    system_prompt = """你是空间关系三元组提取器，严格按以下规则输出：
    1. 仅从文本提取(主体, 关系, 客体)三元组，忽略无关信息。
    2. 必须用JSON数组格式返回，每个元素含"subject"、"relation"、"object"字段。
    3. 输出仅保留JSON数组，** 不要任何解释、说明、代码块标记（如```json）**。
    4. 确保JSON格式正确：引号用双引号，逗号分隔，无多余逗号。
    示例输出：
    [{"subject":"镜子","relation":"下方","object":"洗手池"},{"subject":"洗手池","relation":"放置了","object":"塑料袋"}]
    """
    
    user_input = f"从以下文本提取三元组，严格按示例格式输出：\n{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    # 接收流式响应
    print("正在接收大模型流式响应...")
    full_response = ""
    for chunk in stream_invoke(ll_model.llm, messages):
        # 根据实际返回格式调整，有些stream_invoke可能需要chunk["content"]
        full_response += str(chunk)
        print(f"\r已接收 {len(full_response)} 字符...", end="")
    
    print("\n流式响应接收完成，开始解析...")
    full_response = full_response.strip()
    
    # 格式修复
    try:
        return json.loads(full_response)
    except json.JSONDecodeError:
        print("首次解析失败，尝试修复格式...")
        json_match = re.search(r'\[.*\]', full_response, re.DOTALL)
        if json_match:
            cleaned_response = json_match.group()
            cleaned_response = cleaned_response.replace("'", '"')
            cleaned_response = re.sub(r',\s*]', ']', cleaned_response)
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"修复后仍解析失败：{e}")
                return []
        else:
            print("未找到有效JSON结构")
            return []

def build_knowledge_graph(triples):
    """构建知识图谱数据结构"""
    if not triples:
        return None  # 新增：处理空三元组情况
    
    entities = set()
    for triple in triples:
        entities.add(triple["subject"])
        entities.add(triple["object"])
    
    entity_attributes = {entity: {"name": entity} for entity in entities}
    relations = [
        {
            "source": triple["subject"],
            "target": triple["object"],
            "type": triple["relation"]
        } for triple in triples
    ]
    
    return {
        "entities": [{"id": entity, **attrs} for entity, attrs in entity_attributes.items()],
        "relations": relations
    }

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
        height="700px", 
        width="100%", 
        bgcolor="#f5f5f5", 
        font_color="black",
        notebook=False  # 新增：明确指定非 notebook 环境
    )
    
    # 添加节点和边
    for entity in graph["entities"]:
        net.add_node(
            entity["id"],
            label=entity["name"],
            title=f"实体: {entity['name']}",
            color="#4CAF50"
        )
    
    for relation in graph["relations"]:
        net.add_edge(
            relation["source"],
            relation["target"],
            label=relation["type"],
            title=relation["type"],
            color="#FF9800"
        )
    
    # 简化配置选项，避免复杂JSON解析问题
    net.set_options("""
    {
      "nodes": {
        "size": 30,
        "font": {"size": 14}
      },
      "edges": {
        "font": {"size": 12},
        "length": 200
      },
      "interaction": {
        "dragNodes": true,
        "zoomView": true,
        "dragView": true
      }
    }
    """)
    
    # 直接使用write_html方法，避免show()的复杂逻辑
    try:
        net.write_html(output_file, open_browser=False)
        print(f"知识图谱已保存至 {os.path.abspath(output_file)}")
        return output_file
    except Exception as e:
        print(f"生成HTML时出错: {e}")
        # 尝试备选方案：使用networkx的基本可视化
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(nx.DiGraph([(r["source"], r["target"]) for r in graph["relations"]]))
        nx.draw_networkx_nodes(pos, node_size=3000, node_color="#4CAF50")
        nx.draw_networkx_labels(pos, labels={e["id"]: e["name"] for e in graph["entities"]})
        nx.draw_networkx_edges(pos, edgelist=[(r["source"], r["target"]) for r in graph["relations"]], arrowstyle="->")
        nx.draw_networkx_edge_labels(pos, edge_labels={(r["source"], r["target"]): r["type"] for r in graph["relations"]})
        plt.savefig(output_file.replace(".html", ".png"))
        print(f"已生成PNG备选可视化: {output_file.replace('.html', '.png')}")
        return output_file.replace(".html", ".png")

def process_text_to_graph(text):
    """端到端处理流程"""
    print("正在从文本中提取知识三元组...")
    # triples = extract_triples(text)
    triples = ll_model.extract_triples(text)

    
    if not triples:
        print("未能提取到任何知识三元组")
        return None
    
    print(f"成功提取 {len(triples)} 个知识三元组：")
    for i, triple in enumerate(triples, 1):
        print(f"{i}. ({triple['subject']}, {triple['relation']}, {triple['object']})")
    
    print("\n正在构建知识图谱...")
    graph = build_knowledge_graph(triples)
    
    if not graph:
        print("构建知识图谱失败")
        return None
    
    print("\n正在生成知识图谱可视化...")
    output_file = visualize_knowledge_graph(graph)
    
    return output_file

# 示例用法
if __name__ == "__main__":
     
    # sample_text = """
    # 这张照片展示了一个浴室的一部分。视野里有一个椭圆形的镜子，安装在浅色的瓷砖墙面上。
    # 镜子右侧是一个敞开的门洞，通向另一个房间，那里可以看到一个木制的家具，可能是衣柜或储物柜。
    # 在镜子下方，有一个洗手池区域，上面放着一些物品，包括一个带牙刷和剃须刀的容器、一瓶液体（可能是洗手液）和一个塑料袋。
    # 墙面是浅色的，瓷砖之间的接缝线清晰可见，没有明显的装饰元素。
    # 从光线来看，光源似乎来自窗外，因为镜子的反射显示出明亮的光线。整体环境显得整洁且功能性强。
    # """
    sample_text = """
    这张图片显示了一个房间的入口，通过这个入口可以看到一个卧室。
    视野中有一个白色的门框，门框边缘有些损坏或磨损的迹象。
    门内是一个卧室，房间内有一张木制的床，床头靠墙而立，床上铺着浅色的床单。
    床旁边是一组木质的家具，包括一个衣柜和一个带多个抽屉的梳妆台。
    房间的墙壁是浅色的，可能是白色或米色，地面覆盖着浅棕色的地毯。
    左侧墙上挂着一个圆形的镜子，镜子的边框也是金属材质，与墙壁形成对比。
    从图片的光线来看，光源似乎来自窗外，因为整个房间显得明亮。
    整体环境给人一种简单、实用的感觉，家具的木质色调增添了一些温暖感。
    """
    
    process_text_to_graph(sample_text)

