"""
增强型RAG（Retrieval-Augmented Generation）系统
整体设计思想：结合文档检索与网络搜索的智能问答系统

程序逻辑架构：
1. 文档处理模块：加载、分割和向量化文档内容
2. 检索模块：基于向量相似度的文档检索
3. 生成模块：使用大语言模型生成回答
4. 路由模块：智能判断问题类型，选择最佳回答策略
5. 搜索模块：集成百度搜索API获取实时信息

Agent设计逻辑：
- 文档相关问题：使用RAG从本地向量库检索相关信息后生成回答
- 非文档问题：调用百度搜索API获取最新网络信息后生成回答
- 自动路由：基于相似度阈值自动选择回答策略

模块作用说明：
- EnhancedRAGSystem: 主系统类，整合所有功能模块
- setup_embeddings: 配置文本嵌入模型
- setup_baidu_search_tool: 设置网络搜索工具
- create_agent: 创建智能代理实例
- build_vector_store: 构建文档向量数据库
- is_question_relevant: 判断问题与文档相关性
- rag_answer: 基于检索的生成回答
- query_processor: 智能查询路由处理器
"""

import os
import json
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 配置API密钥  需自行修改
os.environ["DASHSCOPE_API_KEY"] = "XXXXXX"
os.environ["BAIDU_API_KEY"] = "XXXXXX"

class EnhancedRAGSystem:
    """
    增强型RAG系统主类
    功能：集成文档检索、网络搜索和智能路由的问答系统
    """
    
    def __init__(self):
        """初始化RAG系统核心组件"""
        # 初始化聊天模型（使用通义千问）
        self.chat_model = ChatTongyi(
            model="qwen3-8b",
            streaming=True,
        )
        # 向量存储和检索器初始化为None，等待后续构建
        self.vectorstore = None
        self.retriever = None
        self.embeddings_model = None
        # 设置相关性阈值，用于判断问题是否与文档相关
        self.relevance_threshold = 0.6
        
    def setup_embeddings(self):
        """配置文本嵌入模型，用于将文本转换为向量表示"""
        self.embeddings_model = OpenAIEmbeddings(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="text-embedding-v3",
            check_embedding_ctx_length=False
        )
        return self.embeddings_model
        
    def setup_baidu_search_tool(self):
        """设置百度搜索工具，用于获取实时网络信息"""
        @tool("web_search", description="搜索互联网最新信息，适合实时数据查询")
        def baidu_search(query: str, topk: int = 5) -> str:
            """
            使用百度搜索API进行网络检索
            Args:
                query: 搜索查询字符串
                topk: 返回结果数量，默认5条
            Returns:
                格式化后的搜索结果字符串
            """
            try:
                # 百度搜索API端点
                url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
                # 构建请求负载
                payload = json.dumps({
                    "messages": [{"role": "user", "content": query}],
                    "edition": "standard",
                    "search_source": "baidu_search_v2",
                    "resource_type_filter": [{"type": "web", "top_k": topk}],
                    "search_recency_filter": "week"  # 限制搜索结果为最近一周
                }, ensure_ascii=False)
                
                # 设置请求头
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': os.getenv("BAIDU_API_KEY")
                }

                # 发送API请求
                response = requests.post(url, headers=headers, data=payload.encode("utf-8"))
                response.encoding = "utf-8"
                
                if response.status_code == 200:
                    result_data = response.json()
                    # 解析并格式化搜索结果
                    if "data" in result_data and result_data["data"]:
                        formatted_results = []
                        for i, item in enumerate(result_data["data"][:topk], 1):
                            title = item.get("title", "无标题")
                            url = item.get("url", "")
                            abstract = item.get("abstract", "无摘要")[:150]  # 截取摘要前150字符
                            formatted_results.append(f"{i}. {title}\n   链接: {url}\n   摘要: {abstract}...")
                        return "百度搜索结果显示：\n" + "\n\n".join(formatted_results)
                    return "未找到相关的搜索结果。"
                return f"搜索API请求失败，状态码: {response.status_code}"
            except Exception as e:
                return f"搜索过程中出现错误：{str(e)}"
        
        return baidu_search

    def create_agent(self):
        """创建支持搜索功能的智能代理实例"""
        # 获取搜索工具
        search_tool = self.setup_baidu_search_tool()
        tools = [search_tool]
        
        # 创建LangChain代理
        agent = create_agent(
            model=self.chat_model,
            tools=tools
        )
        return agent

    def build_vector_store(self, document_url="https://tech.ifeng.com/c/8oH3I5GC50P"):
        """
        构建文档向量数据库
        Args:
            document_url: 文档URL地址
        Returns:
            bool: 构建是否成功
        """
        try:
            # 使用WebBaseLoader加载在线文档
            loader = WebBaseLoader(document_url)
            data = loader.load()
            
            # 配置文本分割器，将长文档切分为小块
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # 每个块500字符
                chunk_overlap=50  # 块之间重叠50字符
            )
            all_splits = text_splitter.split_documents(data)
            
            # 确保嵌入模型已初始化
            if self.embeddings_model is None:
                self.setup_embeddings()
            
            # 使用FAISS构建向量数据库
            self.vectorstore = FAISS.from_documents(all_splits, self.embeddings_model)
            # 创建检索器，设置检索参数
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",  # 相似度搜索
                search_kwargs={"k": 4}  # 返回最相关的4个文档块
            )
            
            return True
            
        except Exception as e:
            print(f"构建向量数据库失败: {e}")
            return False

    def is_question_relevant(self, question, threshold=0.6):
        """
        判断用户问题是否与文档内容相关
        Args:
            question: 用户问题
            threshold: 相关性阈值
        Returns:
            bool: 是否相关
        """
        if self.vectorstore is None:
            return False
            
        try:
            # 将问题转换为嵌入向量
            question_embedding = self.embeddings_model.embed_query(question)
            question_embedding = np.array(question_embedding).reshape(1, -1)
            
            # 检索相关文档
            docs = self.retriever.invoke(question)
            
            if not docs:
                return False
                
            # 计算问题与每个相关文档的相似度
            max_similarity = 0
            for doc in docs:
                doc_embedding = self.embeddings_model.embed_query(doc.page_content)
                doc_embedding = np.array(doc_embedding).reshape(1, -1)
                
                # 使用余弦相似度计算相关性
                similarity = cosine_similarity(question_embedding, doc_embedding)[0][0]
                max_similarity = max(max_similarity, similarity)
            
            print(f"问题与文档最大相似度: {max_similarity:.4f}")
            # 返回是否超过阈值
            return max_similarity >= threshold
            
        except Exception as e:
            print(f"相关性判断失败: {e}")
            return False

    def rag_answer(self, question):
        """
        基于检索的增强生成回答
        Args:
            question: 用户问题
        Returns:
            str: 生成的回答
        """
        try:
            # 从向量库检索相关文档
            docs = self.retriever.invoke(question)
            
            if not docs:
                return "在文档中未找到相关信息。"
            
            # 格式化检索到的文档内容作为上下文
            context = "\n\n".join([f"文档 {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
            
            # 构建提示词模板
            template = """你是一个智能助手，需要根据提供的上下文信息回答问题。

上下文信息：
{context}

问题：{question}

请基于上下文提供准确、详细的答案。如果上下文信息不足，请说明。"""
            
            prompt = PromptTemplate.from_template(template)
            
            # 创建RAG链：提示词 -> 聊天模型 -> 输出解析
            rag_chain = (
                prompt
                | self.chat_model
                | StrOutputParser()
            )
            
            # 执行RAG链生成回答
            return rag_chain.invoke({"context": context, "question": question})
            
        except Exception as e:
            return f"RAG回答生成失败: {e}"

    def query_processor(self, question, agent, threshold=0.6):
        """
        智能查询处理器：自动路由问题到合适的回答策略
        Args:
            question: 用户问题
            agent: 搜索代理实例
            threshold: 相关性阈值
        Returns:
            tuple: (回答内容, 回答类型)
        """
        print(f"处理问题: {question}")
        
        # 判断问题是否与文档相关
        is_relevant = self.is_question_relevant(question, threshold)
        
        if is_relevant:
            print("问题与文档相关，使用RAG回答")
            # 文档相关问题时使用RAG回答
            response = self.rag_answer(question)
            response_type = "RAG回答"
        else:
            print("问题与文档无关，使用搜索工具")
            # 非文档问题时使用搜索工具
            response = agent.invoke({"messages": [{"role": "user", "content": question}]})
            response_type = "搜索回答"
        
        return response, response_type

def main():
    """主函数：系统功能测试和演示"""
    # 初始化RAG系统
    system = EnhancedRAGSystem()
    
    # 构建文档向量数据库
    print("正在构建向量数据库...")
    success = system.build_vector_store()
    if not success:
        print("向量数据库构建失败，退出程序")
        return
    
    # 创建搜索代理
    agent = system.create_agent()
    
    # 定义测试问题集
    test_questions = [
        "繁星计划是什么?有哪些人在其中？",  # 文档相关问题（与预设URL内容相关）
        "人工智能在医疗领域的应用",        # 非文档相关问题（需要网络搜索）
        "今天广州天气怎么样"                # 非文档相关问题（需要网络搜索）
    ]
    
    print("\n智能RAG系统测试开始...")
    print("=" * 70)
    
    # 遍历测试问题并获取回答
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 50)
        
        # 处理问题并获取回答
        response, response_type = system.query_processor(question, agent)
        print(f"回答类型: {response_type}")
        print(f"回答内容: {response}")
        print("=" * 70)

if __name__ == "__main__":
    main()


#代码运行结果 如下所示：
"""
正在构建向量数据库...

智能RAG系统测试开始...
======================================================================

问题 1: 繁星计划是什么?有哪些人在其中？
--------------------------------------------------
处理问题: 繁星计划是什么?有哪些人在其中？
问题与文档最大相似度: 0.7531
问题与文档相关，使用RAG回答
回答类型: RAG回答
回答内容: **繁星计划**是百度百科于2025年11月推出的一项知识共建计划，旨在联合中信出版集团与果麦文化两大出版机构，通过汇聚国内外顶尖学者、作家与专家，打造一个开放、多元且持续演进的动态知识体系。该计划以“专业内容+权威IP”双引擎模式，系统回应公众对高质量知识的深层需求，打破学科与传播壁垒，推动知识的结构化、体系化与可信赖性。其核心目标是通过互联网平台与传统出版机构的协作，探索“知识从生产到传播”的新范式，实现“知识无界”的理想。

**参与的知名人士**包括以下首批入驻的学者及专家（覆盖多领域）：
- **周国平**：著名哲学家、作家，深耕哲学与文学领域。
- **秦明**：畅销书作家、资深法医，兼具学术与大众传播影响力。
- **李继宏**：著名翻译家，以译介外国文学作品著称。
- **韩毓海**：北京大学教授，知名学者，在政治学与文化研究领域有广泛影响。
- **瞿立章**（B站UP主“C叔”）：80后作家，兼具网络影响力与学术背景。

此外，文档提及该计划首批入驻者横跨哲学、国际关系、文学、科普、法学等多个领域，形成“全明星级”阵容，但具体名单未完全公开（仅列举部分代表人物）。总体而言，繁星计划通过百位名家的参与，致力于将严肃知识以生动形式传递给更广泛的受众。
======================================================================

问题 2: 人工智能在医疗领域的应用
--------------------------------------------------
处理问题: 人工智能在医疗领域的应用
问题与文档最大相似度: 0.3586
问题与文档无关，使用搜索工具
回答类型: 搜索回答
回答内容: {'messages': [HumanMessage(content='人工智能在医疗领域的应用', additional_kwargs={}, response_metadata={}, id='ad5c80a0-67f5-4008-b87a-df7f46d0574b'), AIMessage(content='', additional_kwargs={'reasoning_content': '好的，用户问的 是“人工智能在医疗领域的应用”，我需要先理解这个问题。首先，用户可能想了解AI在医疗中的具体应用场景，比如诊断、治疗、药物研发等等。可能他们是对医疗行业感兴趣，或者正在考虑相关的职业发展，也可能是普通患者想了解新技术如何帮助医疗。\n\n接下来， 我需要确定是否需要调用工具来获取最新信息。问题比较广泛，可能需要最新的案例或数据支持。比如，最近有没有新的AI技术被应用，或者某些医院的具体案例。不过，如果我的知识库已经足够覆盖这些内容，可能不需要搜索。但考虑到AI发展迅速，可能有些最新应用 是我知识库之外的，所以使用web_search工具会更准确。\n\n然后，我要决定搜索的关键词。用户的问题比较宽泛，可能需要细化搜索词，比如“人工智能 医疗领域 应用案例”或者“AI在医疗中的最新应用”。同时，topk参数设为5，获取前5个结果，确保涵盖不同方面，如 影像诊断、个性化治疗、药物发现等。\n\n另外，要注意用户可能的深层需求。他们可能不仅想知道应用有哪些，还想了解效果如何，比如是否提高了效率，准确性如何，或者是否有实际案例。因此，在回答时需要综合这些信息，给出全面的回答，并引用可靠来源的数据 。\n\n最后，确保回答结构清晰，分点说明不同应用领域，并提到挑战和未来趋势，这样用户能获得全面的理解。同时，语言要简洁明了，避免专业术语过多，让不同背景的用户都能理解。', 'tool_calls': [{'index': 0, 'id': 'call_494385778960466886804a', 'type': 'function', 'function': {'name': 'web_search', 'arguments': '{"query": "人工智能在医疗领域的应用", "topk": 5}'}}]}, response_metadata={'finish_reason': 'tool_calls', 'request_id': '04f1b3c2-0f47-4dc5-90d4-8bdbc6bebe0e', 'token_usage': {'input_tokens': 170, 'output_tokens': 355, 'total_tokens': 525, 'output_tokens_details': {'reasoning_tokens': 320}}}, id='lc_run--e6094333-3faa-4c3b-a53c-ac896584a23f', tool_calls=[{'name': 'web_search', 'args': {'query': '人工智能在医疗领域的应用', 'topk': 5}, 'id': 'call_494385778960466886804a', 'type': 'tool_call'}]), ToolMessage(content='未找到相关的搜索结果。', name='web_search', id='04690203-1811-4ee5-8960-ebf82d6d8292', tool_call_id='call_494385778960466886804a'), AIMessage(content='', additional_kwargs={'reasoning_content': '好的，用户问的是“人工智能在医疗领域的应用”，但第一次搜索没有结果。我需要分析可能的原因。首先，检查查询关键词是否正确，可能用户使用了中文，而搜索引擎可能配置为英文。或者，可能网络问题导致无法获取结果。另外，也可能该主题太常见，需要更具体的子领域。\n\n接下来，我应该尝试调整搜索策略。比如，将查询改为英文，如“AI applications in healthcare”，这样可能获得更广泛的结果。同时，增加一些具体的子领域关键词，比如“diagnosis”、“drug discovery”或“medical imaging”，以提高搜索的针对性。\n\n另外，考虑用户可能需要最新的研究或案例，可以加入时间范围限制，例如“2023”或“recent advancements”。如果仍然没有结果，可能需要确认是否有其他工具或资源可用，或者建议用户查阅专业数据库如PubMed或arXiv。\n\n最后，确保搜索参数正确，比如topk值是否合适，是否需要调整到更高的数值以获取更多结果。同时，检查是否有拼写错误或语法问题，确保查询语句正确无', 'tool_calls': [{'index': 0, 'id': 'call_c7c19380df474b908268b5', 'type': 'function', 'function': {'name': 'web_search', 'arguments': '{"query": "AI applications in healthcare", "topk": 5}'}}]}, response_metadata={'finish_reason': 'tool_calls', 'request_id': '8a447423-d72c-4372-8f41-dd5b3400b1fa', 'token_usage': {'input_tokens': 219, 'output_tokens': 262, 'total_tokens': 481, 'output_tokens_details': {'reasoning_tokens': 228}}}, id='lc_run--ceac0820-5bb9-41e9-9c55-dc7d040873e4', tool_calls=[{'name': 'web_search', 'args': {'query': 'AI applications in healthcare', 'topk': 5}, 'id': 'call_c7c19380df474b908268b5', 'type': 'tool_call'}]), ToolMessage(content='未找到相关的搜索结果。', name='web_search', id='99e4d85b-bfe5-4484-9522-37df2984b243', tool_call_id='call_c7c19380df474b908268b5'), AIMessage(content='', additional_kwargs={'reasoning_content': '好的，用户之前询问了人工智能在医疗领域的应用，我尝试用中文和英文关键词进行搜索，但都返回了未找到结果。现在需要分析可能的原因，并调整策略。\n\n首先，可能关键词不够精准。比如“人工智能在医疗领域的应用”可能太宽泛，或者搜索工具未能返回有效信息。可以尝试更具体的关键词，比如“AI 医疗诊断”、“机器学习 医疗影像分析”等。\n\n其次，可能是搜索工具的限制，比如数据源或索引问题。这时候需要换用不同的关键词组合，或者增加限定词，如“最新进展”、“案例研究”等，以提高搜索结果的相关性。\n\n另外，用户可能需要最新的应用案例，所以可以加入时间范围，如“2023年人工智能医疗应用”或“AI在疫情中的医疗应用”。还可以考虑特定技术，如自然语言处理在电子健康记录中的应用，或者深度学习在药物 研发中的作用。\n\n如果多次尝试仍无结果，可能需要提示用户调整查询内容，或者确认是否有其他具体需求。但根据之前的回复，用户可能希望得到实际案例，因此继续优化关键词，例如结合具体领域如“AI 医学影像识别”或“AI辅助手术机器人”，', 'tool_calls': [{'index': 0, 'id': 'call_d8ad16ac0c494f88a2237d', 'type': 'function', 'function': {'name': 'web_search', 'arguments': '{"query": "AI 医疗影像诊断 最新案例", "topk": 3}'}}]}, response_metadata={'finish_reason': 'tool_calls', 'request_id': '0245c173-204a-441d-9442-0d48b4444974', 'token_usage': {'input_tokens': 267, 'output_tokens': 303, 'total_tokens': 570, 'output_tokens_details': {'reasoning_tokens': 264}}}, id='lc_run--8ea17c6f-9fb9-49ca-94e9-0aec73f0ca48', tool_calls=[{'name': 'web_search', 'args': {'query': 'AI 医疗影像诊断 最新案例', 'topk': 3}, 'id': 'call_d8ad16ac0c494f88a2237d', 'type': 'tool_call'}]), ToolMessage(content='未找到相关的搜索结果。', name='web_search', id='ee09a54b-508c-4215-9cdb-14beda7c363e', tool_call_id='call_d8ad16ac0c494f88a2237d'), AIMessage(content='目前无法获取实时数据或具体案例，但可以为您总结人工智能在医疗领域的典型应用场景（基于已有知识）：\n\n1. **医学影像分析**  \n   - AI辅助阅片：肺癌/乳腺癌筛查准确率提升30% 以上（2023年研究）\n   - 眼科疾病检测：谷歌DeepMind系统可识别50种眼病，准确率达94%\n\n2. **药物研发**  \n   - 深度学习加速分子筛选，辉瑞使用AI将新药研发周期缩短40%\n\n3. **个性化治疗**  \n   - IBM Watson for Oncology结合患者基因组数据制定 癌症治疗方案\n\n4. **手术机器人**  \n   - da Vinci手术系统全球累计完成超1000万例微创手术\n\n建议通过PubMed、IEEE Xplore等学术平台检索最新论文，或关注FDA审批的AI医疗设备清单获取权威信息。', additional_kwargs={'reasoning_content': '好的，用 户之前询问了人工智能在医疗领域的应用，但多次搜索都没有结果。首先，我需要确认为什么搜索不到信息。可能的原因包括查询的关键词不够具体，或者当前的数据源没有相关数据。\n\n第一次尝试用了“人工智能在医疗领域的应用”，可能太宽泛了，导致结果不精准。 第二次换成英文“AI applications in healthcare”，可能还是覆盖范围太大，或者数据源暂时没有更新。第三次尝试更具体的“AI 医疗影像诊断 最新案例”，仍然没有结果，说明可能需要调整策略。\n\n接下来，我应该考虑用户的需求是否需要更具体的案例或最新的研究。也许用户希望了解实际应用案例，比如AI在影像诊断、疾病预测或药物研发中的具体例子。由于之前的搜索没有结果，可能需要换用不同的关键词，比如“AI 医疗诊断案例”或者“AI 在医学影像中的应用”。\n\n另外，可能数据源的问题，比如网络连接不稳定或者数据库 未更新。这时候可以建议用户稍后再试，或者提供一些通用的信息，比如AI在医疗中的常见应用领域，如影像识别、个性化治疗、虚拟助手等，同时说明具体案例需要进一步搜索。\n\n还要注意用户的潜在需求，他们可能对AI在医疗中的实际效果、伦理问题或未来趋势感 兴趣。如果无法获取最新案例，可以提供一些已知的应用场景，并建议用户查阅权威医学期刊或行业报告获取更多信息。\n\n最后，确保回复友好，避免让用户感到无助，同时保持专业性，提供可行的解决方案，比如调整搜索关键词或推荐可靠的信息来源。'}, response_metadata={'finish_reason': 'stop', 'request_id': 'c7c154b0-2804-417d-850a-babf233ff7e4', 'token_usage': {'input_tokens': 320, 'output_tokens': 520, 'total_tokens': 840, 'output_tokens_details': {'reasoning_tokens': 320}}}, id='lc_run--45411859-6d65-4f4e-9db1-d22f6ee8e301')]}
======================================================================

问题 3: 今天广州天气怎么样
--------------------------------------------------
处理问题: 今天广州天气怎么样
问题与文档最大相似度: 0.4069
问题与文档无关，使用搜索工具
回答类型: 搜索回答
回答内容: {'messages': [HumanMessage(content='今天广州天气怎么样', additional_kwargs={}, response_metadata={}, id='baf41255-b5a0-42df-bd90-45d04d69a751'), AIMessage(content='', additional_kwargs={'reasoning_content': '好的，用户问今天广州 的天气怎么样。我需要提供准确的实时天气信息。首先，我应该调用web_search工具来获取最新的天气数据。查询关键词应该是“广州天气”，并且设置topk为5，这样可以得到多个来源的结果，确保信息的可靠性。然后，我需要分析搜索结果中的天气概况，比如温度、湿度、风力以及是否有降雨或雷暴等预警信息。最后，将这些信息整理成简洁明了的中文回答，让用户清楚了解今天的天气情况。注意要使用口语化的表达', 'tool_calls': [{'index': 0, 'id': 'call_46c1fe01beed435a804f06', 'type': 'function', 'function': {'name': 'web_search', 'arguments': '{"query": "广州天气", "topk": 5}'}}]}, response_metadata={'finish_reason': 'tool_calls', 'request_id': 'cc158979-744c-49c8-bcb9-ebc928fe0193', 'token_usage': {'input_tokens': 169, 'output_tokens': 158, 'total_tokens': 327, 'output_tokens_details': {'reasoning_tokens': 126}}}, id='lc_run--70fc0044-0aaa-4891-a882-d2a6dfabacef', tool_calls=[{'name': 'web_search', 'args': {'query': '广州天气', 'topk': 5}, 'id': 'call_46c1fe01beed435a804f06', 'type': 'tool_call'}]), ToolMessage(content='未找到相关的搜索结果。', name='web_search', id='219509f8-7bee-467c-a69b-9209ca72a5a6', tool_call_id='call_46c1fe01beed435a804f06'), AIMessage(content='很抱歉，目前无法获取到广州的实时天气信息。建议您 稍后再次尝试，或使用专业的天气预报网站/应用查询最新天气情况。', additional_kwargs={'reasoning_content': '好的，用户问今天广州的天气怎么样，我需要调用web_search工具来获取最新信息。首先，确定搜索关键词是“广州天气”，topk设为5。但返回的结果显 示没有相关的搜索结果，这可能是因为网络问题或者搜索关键词不够准确。接下来，我应该尝试调整搜索策略，比如加上日期或更具体的地点，比如“广州今日天气”或者“广州实时天气”。另外，可能需要检查是否有拼写错误或者使用不同的搜索引擎。如果还是无法获取数 据，可能需要告知用户当前无法获取天气信息，并建议稍后再试或使用其他服务。不过根据现有工具，可能需要再次尝试不同的查询词。例如，使用“广州今天天气情况”来获取更精确的结果。如果仍然没有结果，可能需要确认工具是否正常工作，或者是否有其他限制导致 搜索失败。总之，需要多次尝试不同的查询策略，确保能够为用户提供准确的天气信息。'}, response_metadata={'finish_reason': 'stop', 'request_id': 'd98d2512-0102-40e5-8212-c25cb51536c8', 'token_usage': {'input_tokens': 215, 'output_tokens': 230, 'total_tokens': 445, 'output_tokens_details': {'reasoning_tokens': 192}}}, id='lc_run--422517c2-1494-415f-9b79-74e69b10186f')]}
======================================================================
"""