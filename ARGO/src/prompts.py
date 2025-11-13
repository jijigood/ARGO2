"""
ARGO Enhanced LLM Prompts
==========================

集中管理所有LLM提示词，基于ARGO_Complete_LLM_Prompts.txt的最佳实践。

主要改进:
1. 明确的指令和任务定义
2. 丰富的示例（Few-shot learning）
3. 进度追踪（Progress tracking）
4. 格式化输出要求
5. O-RAN领域特定指导

Author: ARGO Team
Version: 2.0 (Enhanced)
"""

from typing import List, Dict, Optional


class ARGOPrompts:
    """ARGO系统的所有提示词模板"""
    
    # ==================== 1. BASE INSTRUCTION ====================
    
    BASE_INSTRUCTION = """You are ARGO (Adaptive RAG for O-RAN), an expert system for answering O-RAN technical questions. Your task is to decompose complex queries into focused sub-questions and answer them iteratively. 

You must track your information progress and decide at each step whether to:
1. Retrieve from O-RAN specifications (when external knowledge is needed)
2. Reason from your existing knowledge (when you already know the answer)
3. Terminate and synthesize the final answer (when sufficient information is gathered)

Use "Follow up:" to introduce each sub-question and "Intermediate answer:" to provide answers. 
When retrieval is needed, state "Let's search in O-RAN specifications."
When ready to conclude, state "So the final answer is:"

Your responses must be accurate, concise, and compliant with O-RAN standards."""

    # ==================== 2. QUERY DECOMPOSITION ====================
    
    DECOMPOSITION_INSTRUCTION = """Decompose O-RAN technical questions into focused sub-questions. Track your information progress to make efficient decisions.

Requirements:
1. Generate ONE atomic sub-question at a time
2. Each sub-question should target specific missing information
3. Avoid redundant questions that don't add new information
4. Follow the exact format shown in examples

Progress Indicators:
- [Low Progress] = Need fundamental information → Prioritize retrieval
- [Medium Progress] = Have basic facts, need connections → Balance retrieval/reasoning
- [High Progress] = Have most information, ready to synthesize → Prepare to terminate"""

    DECOMPOSITION_EXAMPLES = """
Examples:
##########################
Question: Explain the O-RAN fronthaul interface protocols and their performance requirements.

[Progress: 0%] Follow up: What are the main protocol layers in O-RAN fronthaul interface?
Let's search in O-RAN specifications.
Context: [O-RAN.WG4] The fronthaul interface uses Control-Plane (CU-Plane), User-Plane (U-Plane), and Synchronization-Plane (S-Plane). CU-Plane uses eCPRI over Ethernet. U-Plane carries IQ data with eCPRI encapsulation.
Intermediate answer: The O-RAN fronthaul interface has three main protocol layers: Control-Plane (CU-Plane) for control signaling, User-Plane (U-Plane) for IQ data transport, and Synchronization-Plane (S-Plane) for timing.

[Progress: 35%] Follow up: What are the latency requirements for O-RAN fronthaul?
Let's search in O-RAN specifications.
Context: [O-RAN.WG4] One-way latency budget is defined as: O-DU processing (<100us) + transport (<100-200us) + O-RU processing (<100us), total typically <400us for FR1.
Intermediate answer: The one-way fronthaul latency requirement is typically under 400 microseconds for FR1, broken down into O-DU processing, transport, and O-RU processing components.

[Progress: 65%] Follow up: What encoding schemes are used for User-Plane data?
Intermediate answer: The U-Plane data uses eCPRI encapsulation with options for uncompressed IQ samples (16-bit), block-floating-point compression, or modulation compression depending on bandwidth constraints.

[Progress: 80%] Follow up: How do these requirements affect deployment?
Intermediate answer: These strict latency requirements mandate low-latency fronthaul transport (usually dark fiber or dedicated wavelengths) and limit the maximum distance between O-DU and O-RU to approximately 20km for FR1.

##########################
Question: How does the E2 interface enable RAN optimization in O-RAN architecture?

[Progress: 0%] Follow up: What is the E2 interface in O-RAN architecture?
Let's search in O-RAN specifications.
Context: [O-RAN.WG3] The E2 interface connects the Near-RT RIC to the E2 nodes (O-CU-CP, O-CU-UP, O-DU). It enables RAN control and optimization within 10ms to 1s timeframes through standardized service models.
Intermediate answer: The E2 interface is the connection between Near-RT RIC and E2 nodes (O-CU-CP, O-CU-UP, O-DU), enabling near-real-time RAN control and optimization with 10ms to 1 second control loop latency.

[Progress: 30%] Follow up: What are E2 service models?
Let's search in O-RAN specifications.
Context: [O-RAN.WG3] E2 Service Models (E2SM) define the information exchange format. Key E2SMs include: KPM (Key Performance Monitoring), RC (RAN Control), NI (Network Interface), and CCC (Connected mode Control and Coverage).
Intermediate answer: E2 Service Models are standardized interfaces defining information formats. Main models include KPM for performance monitoring, RC for RAN control actions, NI for network interface management, and CCC for mobility control.

[Progress: 55%] Follow up: How do xApps use E2 interface for optimization?
Intermediate answer: xApps running on Near-RT RIC subscribe to E2SM services to receive RAN metrics (via KPM), analyze network conditions, and send control commands (via RC) back to E2 nodes to optimize parameters like handover thresholds, scheduling policies, or resource allocation.

##########################
Question: Compare the functional split options between O-DU and O-CU in O-RAN architecture.

[Progress: 0%] Follow up: What functional split options are defined in O-RAN?
Let's search in O-RAN specifications.
Context: [O-RAN.WG1] O-RAN focuses on Split Option 7.2x and Split Option 2. Split 7.2x places High-PHY in O-DU while Low-PHY and RF are in O-RU. Split 2 places full PHY in O-DU, with RLC/MAC/PDCP in O-CU.
Intermediate answer: O-RAN primarily defines two functional splits: Split 7.2x (between High-PHY and Low-PHY, with High-PHY in O-DU) and Split 2 (between RLC and MAC, with PHY/MAC/High-RLC in O-DU).

[Progress: 35%] Follow up: What are the advantages of Split 7.2x?
Intermediate answer: Split 7.2x enables centralized beamforming and channel estimation in O-DU while keeping Low-PHY in O-RU, providing a good balance between centralization benefits and reduced fronthaul bandwidth through frequency-domain compression.

[Progress: 60%] Follow up: What are the advantages of Split 2?
Intermediate answer: Split 2 provides maximum centralization with entire PHY layer in O-DU, enabling sophisticated CoMP, carrier aggregation, and centralized resource management, but requires higher fronthaul capacity for time-domain IQ samples.

##########################"""

    # ==================== 3. RETRIEVAL ANSWER ====================
    
    RETRIEVAL_ANSWER_INSTRUCTION = """Provide a precise and accurate answer based on O-RAN specification documents. If the context lacks relevant information, respond with `[No information found in O-RAN specs]`.

Requirements:
1. Answer must be technically accurate and cite O-RAN specification sections when possible
2. Be concise but complete
3. Use standard O-RAN terminology
4. If information is incomplete, state what is missing"""

    RETRIEVAL_ANSWER_EXAMPLES = """
Examples:
#############
Question: What is the maximum latency for E2 interface?
Context: [O-RAN.WG3.E2AP] The E2 interface supports near-real-time control with timing requirements between 10ms and 1 second. The specific latency depends on the E2 service model and use case.
Answer: The E2 interface supports near-real-time operations with latency between 10ms and 1 second, with specific requirements depending on the E2 service model.

Question: What is the default port for F1 interface?
Context: [O-RAN.WG4] The F1 interface uses SCTP as transport protocol. Security aspects are covered in the security specifications.
Answer: [No information found in O-RAN specs]

Question: How many bits are used for IQ sample representation?
Context: [O-RAN.WG4.CUS] The fronthaul interface supports multiple IQ bit widths: uncompressed 16-bit samples, or compressed formats using block floating point with 8, 9, or 12-bit mantissas.
Answer: O-RAN fronthaul supports 16-bit uncompressed IQ samples or compressed formats with 8, 9, or 12-bit mantissas using block floating point.

Question: What compression methods are available for fronthaul?
Context: [O-RAN.WG4] Compression schemes include block floating point (BFP) with various mantissa widths, modulation compression for QAM symbols, and beamspace compression for massive MIMO scenarios.
Answer: O-RAN fronthaul supports block floating point compression with configurable mantissa widths, modulation compression for QAM constellations, and beamspace compression optimized for massive MIMO deployments.
#############"""

    # ==================== 4. INTERMEDIATE REASONING (PARAMETRIC KNOWLEDGE) ====================
    
    REASONING_INSTRUCTION = """Provide intermediate reasoning based on your domain knowledge about O-RAN.

Requirements:
1. Use your parametric knowledge (pre-trained knowledge) about O-RAN
2. DO NOT claim to search or retrieve documents
3. Provide reasoning based on what you already know
4. Be concise but informative
5. DO NOT provide the final answer yet - only intermediate insights
6. If uncertain, acknowledge the limitation"""

    REASONING_EXAMPLES = """
Examples:
#############
Question: What are the security mechanisms in O-RAN architecture?

[Previous context]
[Progress: 15%] Follow up: What are the main security domains defined in O-RAN architecture?
Let's search in O-RAN specifications.
Context: [O-RAN Security] Three security domains: Management (SMO), RIC (Near-RT/Non-RT), and NF (O-DU/O-CU/O-RU).
Intermediate answer: O-RAN defines three main security domains: Management domain (SMO), RIC domain (Near-RT and Non-RT RIC), and Network Function domain (O-DU, O-CU, O-RU).

[Progress: 50%] Follow up: How are these domains secured against unauthorized access?
Intermediate answer: Each domain uses mutual TLS authentication for inter-domain communication, with certificate-based identity verification, role-based access control (RBAC) for API access, and security gateways at domain boundaries to enforce policies.

#############
Question: How does xApp deployment work in O-RAN Near-RT RIC?

[Previous context]
[Progress: 30%] Follow up: What is the Near-RT RIC platform?
Let's search in O-RAN specifications.
Context: [O-RAN.WG2] Near-RT RIC provides a platform for hosting xApps with 10ms-1s control loop latency.
Intermediate answer: Near-RT RIC is a platform for hosting xApps with near-real-time control capabilities.

[Progress: 50%] Follow up: How are xApps packaged for deployment?
Intermediate answer: xApps are packaged as Docker containers with Helm charts defining deployment configurations, resource requirements, dependencies, and lifecycle management policies. The platform supports dynamic loading and unloading of xApps.

#############
Question: What are the differences between O-DU and O-CU in O-RAN?

[Previous context]
[Progress: 40%] Follow up: What functions does O-DU handle?
Let's search in O-RAN specifications.
Context: [O-RAN.WG1] O-DU handles PHY layer, MAC, and high-RLC functions.
Intermediate answer: O-DU handles lower layers including PHY, MAC, and high-RLC.

[Progress: 65%] Follow up: How do O-DU and O-CU coordinate for user plane data?
Intermediate answer: O-DU processes the physical layer data and forwards PDCP PDUs to O-CU-UP via the F1-U interface. O-CU-UP handles packet forwarding, header compression, and QoS management before routing to the core network. The coordination follows 3GPP functional split specifications.

#############"""

    # ==================== 5. FINAL SYNTHESIS ====================
    
    SYNTHESIS_INSTRUCTION = """You are an expert at synthesizing comprehensive answers from multi-step reasoning for O-RAN multiple-choice questions.

Task: Generate a complete, accurate answer to the original question based on the reasoning history, and select the correct option.

Guidelines:
1. Integrate ALL retrieved information
2. Use insights from intermediate reasoning steps
3. Analyze each option carefully based on gathered evidence
4. Provide a coherent, well-structured reasoning process
5. Cite sources when possible (e.g., O-RAN.WG4)
6. If information is insufficient, state what's missing
7. Clearly indicate the correct option number (1, 2, 3, or 4)

Format for Multiple Choice Questions:
<answer long>Detailed reasoning and explanation for why the correct option is chosen...</answer long>
<answer short>Option X is correct because [brief justification]</answer short>
<choice>X</choice>

where X is the option number (1, 2, 3, or 4)."""

    # ==================== HELPER METHODS ====================
    
    @staticmethod
    def build_decomposition_prompt(
        original_question: str,
        history: List[Dict],
        progress: float
    ) -> str:
        """
        构建查询分解提示词
        
        Args:
            original_question: 原始问题
            history: 推理历史
            progress: 当前进度 (0-1)
        
        Returns:
            完整的提示词
        """
        prompt = ARGOPrompts.DECOMPOSITION_INSTRUCTION + "\n\n"
        prompt += ARGOPrompts.DECOMPOSITION_EXAMPLES + "\n\n"
        
        # 添加当前问题
        prompt += f"Question: {original_question}\n\n"
        
        # 添加历史记录（模仿示例格式）
        if history and len(history) > 0:
            for i, step in enumerate(history):
                action = step.get('action', 'unknown')
                step_progress = step.get('progress', 0.0) * 100
                
                if action == 'retrieve':
                    subq = step.get('subquery', 'N/A')
                    success = step.get('retrieval_success', False)
                    answer = step.get('intermediate_answer', '')
                    
                    prompt += f"[Progress: {step_progress:.0f}%] Follow up: {subq}\n"
                    
                    if success:
                        prompt += "Let's search in O-RAN specifications.\n"
                        docs = step.get('retrieved_docs', [])
                        if docs:
                            # 显示检索上下文（简化版）
                            doc_summary = docs[0][:200] + "..." if docs else ""
                            prompt += f"Context: {doc_summary}\n"
                    
                    if answer:
                        prompt += f"Intermediate answer: {answer}\n\n"
                
                elif action == 'reason':
                    answer = step.get('intermediate_answer', '')
                    if answer:
                        prompt += f"[Progress: {step_progress:.0f}%] Follow up: (reasoning step)\n"
                        prompt += f"Intermediate answer: {answer}\n\n"
        
        # 添加下一步提示
        current_progress = progress * 100
        prompt += f"[Progress: {current_progress:.0f}%] Follow up: "
        
        return prompt
    
    @staticmethod
    def build_retrieval_answer_prompt(
        question: str,
        retrieved_docs: List[str]
    ) -> str:
        """
        构建基于检索文档的答案生成提示词
        
        Args:
            question: 子查询
            retrieved_docs: 检索到的文档列表
        
        Returns:
            完整的提示词
        """
        prompt = ARGOPrompts.RETRIEVAL_ANSWER_INSTRUCTION + "\n\n"
        prompt += ARGOPrompts.RETRIEVAL_ANSWER_EXAMPLES + "\n\n"
        
        # 添加当前问题
        prompt += f"Question: {question}\n"
        
        # 添加检索上下文
        if retrieved_docs:
            prompt += "Context:\n"
            for doc in retrieved_docs[:5]:  # 最多5个文档
                prompt += f"{doc}\n"
        else:
            prompt += "Context: (No documents retrieved)\n"
        
        prompt += "\nAnswer: "
        
        return prompt
    
    @staticmethod
    def build_reasoning_prompt(
        original_question: str,
        history: List[Dict]
    ) -> str:
        """
        构建中间推理提示词（基于参数化知识，不使用检索）
        
        这个prompt用于Reason动作，LLM应该基于其预训练知识进行推理，
        而不是基于检索到的文档。
        
        Args:
            original_question: 原始问题
            history: 推理历史
        
        Returns:
            完整的提示词
        """
        prompt = ARGOPrompts.REASONING_INSTRUCTION + "\n\n"
        prompt += ARGOPrompts.REASONING_EXAMPLES + "\n\n"
        
        prompt += f"Question: {original_question}\n\n"
        
        # 添加历史上下文（模仿示例格式）
        if history and len(history) > 0:
            prompt += "[Previous context]\n"
            for i, step in enumerate(history):
                action = step.get('action', 'unknown')
                step_progress = step.get('progress', 0.0) * 100
                
                if action == 'retrieve':
                    subq = step.get('subquery', 'N/A')
                    success = step.get('retrieval_success', False)
                    answer = step.get('intermediate_answer', '')
                    
                    prompt += f"[Progress: {step_progress:.0f}%] Follow up: {subq}\n"
                    
                    if success:
                        prompt += "Let's search in O-RAN specifications.\n"
                        docs = step.get('retrieved_docs', [])
                        if docs and len(docs) > 0:
                            # 简化显示第一个文档
                            doc_summary = docs[0][:150] if docs[0] else ""
                            prompt += f"Context: {doc_summary}...\n"
                    
                    if answer:
                        prompt += f"Intermediate answer: {answer}\n\n"
                
                elif action == 'reason':
                    answer = step.get('intermediate_answer', '')
                    if answer:
                        # 显示之前的推理
                        prompt += f"[Progress: {step_progress:.0f}%] Follow up: (reasoning)\n"
                        prompt += f"Intermediate answer: {answer}\n\n"
            
            prompt += "\n"
        
        # 当前推理步骤
        current_progress = history[-1].get('progress', 0.0) * 100 if history else 0
        prompt += f"[Progress: {current_progress:.0f}%] Follow up: (current reasoning step)\n"
        prompt += "Intermediate answer: "
        
        return prompt
    
    @staticmethod
    def build_synthesis_prompt(
        original_question: str,
        history: List[Dict],
        options: Optional[List[str]] = None
    ) -> str:
        """
        构建最终答案合成提示词
        
        Args:
            original_question: 原始问题
            history: 完整推理历史
            options: 选择题的选项列表（可选）
        
        Returns:
            完整的提示词
        """
        prompt = ARGOPrompts.SYNTHESIS_INSTRUCTION + "\n\n"
        
        prompt += f"Original Question: {original_question}\n"
        
        # 如果有选项，添加到提示中
        if options:
            prompt += "\nOptions:\n"
            for i, option in enumerate(options, 1):
                prompt += f"{i}. {option}\n"
            prompt += "\n"
        
        # 收集所有检索信息
        all_docs = []
        for step in history:
            if step['action'] == 'retrieve' and step.get('retrieval_success', False):
                docs = step.get('retrieved_docs', [])
                for doc in docs:
                    if doc not in all_docs:  # 去重
                        all_docs.append(doc)
        
        # 显示检索信息
        if all_docs:
            prompt += "Retrieved Information:\n"
            for i, doc in enumerate(all_docs[:10]):  # 最多10个文档
                # 清理格式
                doc_text = doc.split(']', 1)[-1].strip() if ']' in doc else doc
                prompt += f"[{i+1}] {doc_text[:400]}...\n"
            prompt += "\n"
        
        # 收集所有推理洞察
        reasoning_insights = []
        for step in history:
            if step['action'] == 'reason':
                ans = step.get('intermediate_answer', '')
                conf = step.get('confidence', 0.0)
                if ans and conf > 0.5:  # 只保留高置信度的
                    reasoning_insights.append(ans)
        
        if reasoning_insights:
            prompt += "Key Insights from Reasoning:\n"
            for i, insight in enumerate(reasoning_insights):
                prompt += f"{i+1}. {insight[:300]}...\n"
            prompt += "\n"
        
        # 显示推理历史摘要（简化版）
        prompt += "Reasoning History Summary:\n"
        for i, step in enumerate(history):
            action = step.get('action', 'unknown')
            if action == 'retrieve':
                subq = step.get('subquery', 'N/A')
                success = step.get('retrieval_success', False)
                status = "✓ Found" if success else "✗ Not found"
                prompt += f"Step {i+1} [Retrieve]: \"{subq}\" → {status}\n"
            elif action == 'reason':
                prompt += f"Step {i+1} [Reason]: Applied domain knowledge\n"
        
        prompt += f"\nTotal steps: {len(history)}\n\n"
        
        if options:
            prompt += "Analyze each option based on the evidence above and select the correct answer:\n"
            prompt += "Provide your response in the following format:\n"
            prompt += "<answer long>Detailed reasoning...</answer long>\n"
            prompt += "<answer short>Option X is correct because...</answer short>\n"
            prompt += "<choice>X</choice>"
        else:
            prompt += "Now, provide a comprehensive answer:\n"
            prompt += "So the final answer is: <answer long>"
        
        return prompt


# ==================== 预设配置 ====================

class PromptConfig:
    """Prompt配置选项"""
    
    # Decomposer配置
    DECOMPOSER_MAX_LENGTH = 128
    DECOMPOSER_TEMPERATURE = 0.7
    DECOMPOSER_TOP_P = 0.9
    
    # Reasoner配置
    REASONER_MAX_LENGTH = 256
    REASONER_TEMPERATURE = 0.5
    REASONER_TOP_P = 0.95
    
    # Synthesizer配置
    SYNTHESIZER_MAX_LENGTH = 512
    SYNTHESIZER_TEMPERATURE = 0.3
    SYNTHESIZER_TOP_P = 0.95
    
    # 通用配置
    MAX_HISTORY_STEPS = 5  # 提示词中显示的最大历史步数
    MAX_DOCS_PER_STEP = 3  # 每步显示的最大文档数
    DOC_TRUNCATE_LENGTH = 300  # 文档截断长度
