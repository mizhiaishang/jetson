from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from config import Config
import re
import sys
import os
import traceback
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



class LLMInterface:
    def __init__(self):
        self.model = Config.LLM['model']
        self.temperature = Config.LLM['temperature']
        self.max_tokens = Config.LLM['max_tokens']
        self.aly_api_key ='sk-64e0e5199d1a4d40a14c0eb8be02fc8d'
        self.aly_api_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

        self.llm = ChatOpenAI(
            model=self.model,
            base_url=self.aly_api_url,
            api_key=self.aly_api_key,)
    
    def stream_invoke(self, prompt):
        """
        prompt可以做成2种方式，方式一：
        from langchain.schema import HumanMessage
        messages = [HumanMessage(content=prompt)]
        方式二：
        {"role": "user", "content": question}
        """
        full_response = ""
        results = self.llm.stream(prompt)
        for chunk in results:
            print(chunk.content, end="", flush=True)  # 逐块输出
            full_response += chunk.content
        return full_response
    
    def extract_triples(self, input_text, system_prompt=None):
        """Extract (subject, relation, object) triples from input text using the LLM."""
        if system_prompt is None:
            system_prompt ="""你是空间关系三元组提取器，严格按以下规则输出
            1. 仅从文本提取(主体, 空间关系, 客体)三元组，忽略无关信息。
            2. 如果涉及物体颜色，材质等属性，请忽略，仅提取物体间的空间关系。
            3. 主体和客体应为具体物体，如“洗手池”，“冰箱”等，避免如“左侧区域”，“视野右侧”等无具体指代的范围描述成为主体与客体。
            3. 必须用JSON数组格式返回，每个元素含"subject"、"relation"、"object"字段。
            4. 输出仅保留JSON数组，** 不要任何解释、说明、代码块标记（如```json）**。
            5. 确保JSON格式正确：引号用双引号，逗号分隔，无多余逗号。
            示例输出：
            [{"subject":"镜子","relation":"下方","object":"洗手池"},{"subject":"洗手池","relation":"放置","object":"塑料袋"}]
            """

        user_input = f"从以下文本提取三元组，严格按示例格式输出：\n{input_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # 接收流式响应
        print("正在接收大模型流式响应...")
        full_response = ""
        for chunk in self.stream_invoke(messages):
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

#     async def generate_response(self, prompt, system_prompt=None):
#         """Base method for generating responses from the LLM"""
#         if system_prompt is None:
#             system_prompt = "You are an AI assistant specialized in spatial navigation and environment understanding."
            
#         messages = []
#         if system_prompt:
#             messages.append({"role": "system", "content": system_prompt})
#         messages.append({"role": "user", "content": prompt})

#         response = await self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=self.temperature,
#             max_tokens=self.max_tokens
#         )
        
#         return response.choices[0].message.content.strip()

#     async def generate_relationship(self, node1, node2):
#         prompt = f"""
#         Given two objects in a 3D environment:
#         Object 1: {node1[0]} (ID: {node1[1]['id']}, Position: {node1[1]['position']})
#         Object 2: {node2[0]} (ID: {node2[1]['id']}, Position: {node2[1]['position']})

#         Describe the spatial relationship between these two objects. Consider their relative positions, possible interactions, and any logical connections based on their semantic labels.

#         Output the relationship as a short phrase or sentence.
#         """

#         return await self.generate_response(
#             prompt,
#             system_prompt="You are an AI assistant specialized in describing spatial relationships between objects in a 3D environment."
#         )

#     async def rank_results(self, query, results, context):
#         prompt = f"""Given the following query and list of objects in a 3D environment, rank the objects based on their relevance to the query. Consider semantic similarity, hierarchical relationships, spatial proximity, and functional relevance.

#             Query: {query}

#             Objects:
#             {context}

#             Output a ranked list of object IDs, separated by commas, from most relevant to least relevant.
#             """
#         ranked_ids = [id.strip() for id in self.generate_response(prompt).split(',')]
#         return [id for id in ranked_ids if id in results]  # Ensure we only return valid results

#     async def generate_community_summary(self, objects):
#         """Create both a functional name and summary for a group of objects or areas"""
#         descriptions = []
#         for obj in objects:
#             if obj.get('summary'):
#                 descriptions.append(f"Area: {obj.get('name', 'Unnamed')}\n"
#                                 f"Summary: {obj.get('summary')}")
#             else:
#                 obj_id = obj.get('id', 'Unknown object')
#                 obj_type = obj.get('type', 'object')
#                 obj_label = obj.get('label', obj_id)
#                 descriptions.append(f"Object: {obj_label} (Type: {obj_type}, ID: {obj_id})")

#         descriptions_text = '\n'.join(descriptions)
        
#         prompt = (
#             f"Given these objects/areas in a 3D environment:\n"
#             f"{descriptions_text}\n\n"
#             "Please provide:\n"
#             "1. A SHORT functional name that describes the COMBINED purpose of ALL areas/objects\n"
#             "   - Use exactly 2-3 words in snake_case format (e.g., art_work_zone, meeting_dining_area)\n"
#             "   - The name MUST reflect ALL major functions present\n"
#             "   - Do not focus on just one function if multiple exist\n"
#             "2. A two-sentence summary that:\n"
#             "   - Describes ALL distinct functions in the space\n"
#             "   - Mentions the key objects/features from EACH sub-area\n"
#             "   - Captures the mixed-use nature if multiple functions exist\n"
#             "\n"
#             "Format your response EXACTLY as follows (including the <<>> markers):\n"
#             "AREA_NAME: <<combined_functional_name_in_snake_case>>\n"
#             "AREA_SUMMARY: <<single concise sentence covering ALL functions>>\n"
#         )

#         response = await self.generate_response(prompt)
        
#         try:
#             print(f"\nRaw LLM Response:\n{response}\n")
            
#             # More flexible regex patterns that handle both formats:
#             # Format 1: AREA_NAME: <<name>>
#             # Format 2: AREA_NAME: name
#             name_pattern = r'AREA_NAME:[ \t]*(?:<<)?([^>\n]+?)(?:>>)?[ \t]*$'
#             summary_pattern = r'AREA_SUMMARY:[ \t]*(?:<<)?(.+?)(?:>>)?[ \t]*$'
            
#             # Find matches in multiline text
#             name_match = re.search(name_pattern, response, re.MULTILINE)
#             summary_match = re.search(summary_pattern, response, re.MULTILINE | re.DOTALL)
            
#             if not name_match or not summary_match:
#                 print("Warning: Could not parse response format")
#                 print(f"Name match: {name_match}")
#                 print(f"Summary match: {summary_match}")
#                 return {
#                     'name': 'undefined_zone',
#                     'summary': 'Area containing multiple objects or spaces'
#                 }
            
#             name = name_match.group(1).strip().lower()
#             summary = summary_match.group(1).strip()
            
#             # Validate name format (allow only lowercase letters, numbers, and underscores)
#             if not re.match(r'^[a-z0-9_]+$', name):
#                 print(f"Warning: Invalid name format: {name}")
#                 name = 'undefined_zone'
            
#             print(f"Parsed name: {name}")
#             print(f"Parsed summary: {summary[:50]}...")
            
#             return {
#                 'name': name,
#                 'summary': summary
#             }
            
#         except Exception as e:
#             print(f"Warning: Error parsing LLM response: {str(e)}")
#             traceback.print_exc()  # Print full traceback
#             return {
#                 'name': 'undefined_zone',
#                 'summary': 'Area containing multiple objects or spaces'
#             }

#     async def generate_navigation_response(self, query, context, query_type):
#         """Generate a navigation response based on the retrieved nodes."""
#         prompt = f"""Given the following navigation query and context about available locations,
#         generate a response that helps navigate to the most relevant location.

#         Query: {query}
#         Query Type: {query_type}

#         Available Context:
#         {context}

#         Instructions:
#         1. Analyze the query and available locations
#         2. Select the most specific and appropriate destination (prefer specific objects over general areas)
#         3. Format your response as follows:
#            - Include the exact location name in double angle brackets: <<exact_name>>
#            - Provide a brief explanation of why this location is relevant
#            - Include any relevant spatial relationships or navigation hints

#         IMPORTANT: Use the exact name as it appears in the context, maintaining exact spelling and format.
#         """

#         response = await self.generate_response(prompt)
#         return response.strip()

#     async def select_best_node(self, query, nodes, context):
#         """Select the single most relevant node from a list of candidates based on the query."""
#         prompt = f"""Given the following navigation query and available nodes in a 3D environment, 
#         select the SINGLE most relevant node that best matches the query's intent.

#         Navigation Query: {query}

#         Available Nodes:
#         {context}

#         Instructions:
#         1. Consider each node's summary and function
#         2. Evaluate relevance to the navigation query
#         3. Prioritize specific object nodes over general areas when possible
#         4. Select the single most specific and relevant node for navigation

#         CRITICAL: You must respond with ONLY the exact Node ID from the list above.
#         For example, if you see 'Node ID: cafeteria_table_1', respond with exactly 'cafeteria_table_1'.
#         Do not add any explanation or additional text.

#         Your response must be one of these exact Node IDs: {[n['id'] for n in nodes]}"""

#         response = await self.generate_response(prompt)
#         response = response.strip()
        
#         # Debug print
#         print(f"\nLLM Selection Process:")
#         print(f"Query: {query}")
#         print(f"Available Node IDs: {[n['id'] for n in nodes]}")
#         print(f"Node Types: {[(n['id'], n['type']) for n in nodes]}")
#         print(f"LLM Response: '{response}'")
        
#         # Verify response matches an available node
#         if response in [n['id'] for n in nodes]:
#             print(f"✓ Valid selection: {response}")
#             return response
#         else:
#             print(f"✗ Invalid selection: '{response}' not in available nodes")
#             return None

#     async def generate_hierarchical_context(self, nodes):
#         """Generate a readable context for a list of nodes."""
#         context_parts = ["Available Locations for Selection:"]
        
#         for i, node in enumerate(nodes, 1):
#             context_parts.extend([
#                 f"\n{i}. Location Details:",
#                 f"   Name: {node['name']}",
#                 f"   Type: {node['type']}",
#                 f"   Level: {node['level']}",
#                 f"   Summary: {node['summary']}",
#                 "   ---"
#             ])
#         return "\n".join(context_parts)

#     async def select_nodes_for_query(self, query, nodes_context, system_prompt=None):
#         """Helper method for selecting nodes during hierarchical traversal"""
#         if system_prompt is None:
#             system_prompt = "You are an AI assistant helping to select relevant nodes in a 3D environment."
        
#         response = await self.generate_response(nodes_context, system_prompt)
#         return response.strip()

#     async def generate_hierarchical_traversal(self, query, node_options, is_top_level=False):
#         """Specialized method for hierarchical graph traversal"""
#         context_type = "high-level areas" if is_top_level else "objects"
        
#         prompt = f"""Given the query '{query}', analyze these {context_type}:

# {node_options}

# {f'Select up to 3 most relevant areas (comma-separated list)' if is_top_level else 'Select the single most relevant object (exact name only)'}."""

#         return await self.generate_response(
#             prompt,
#             system_prompt=f"You are an AI assistant specialized in navigating hierarchical spaces. Select the most relevant {context_type} based on the query."
#         )
