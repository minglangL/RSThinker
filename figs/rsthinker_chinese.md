# 📌 最新论文 | RSThinker：让遥感大模型“学会思考”的Geo-CoT推理框架，性能显著跃升！数据代码模型开源

---

# 📘 论文信息
- 题目：Towards Faithful Reasoning in Remote Sensing: A Perceptually-Grounded GeoSpatial Chain-of-Thought for Vision-Language Models
- 论文：https://arxiv.org/abs/2509.22221
- 代码：https://github.com/minglangL/RSThinker
- 模型：https://huggingface.co/minglanga/RSThinker
- 日期：2025.09
- 单位：吉林大学

 🖼️ 论文标题截图
<p align="center">
  <img width="100%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/title.png" />
</p>


---
# 摘要
现有的视觉语言模型在遥感领域的复杂分析任务中表现不佳，其原因在于端到端的训练范式跳过了关键的推理步骤，导致模型输出难以验证。针对这一问题，我们提出了**Geo-CoT**，基于感知锚定的地理空间思维链（Perceptually-Grounded Geospatial Chain-of-Thought）框架，将遥感分析过程建模为一个可验证的多阶段推理流程。在此基础上，我们构建了首个大规模的遥感推理思维链数据集**Geo-CoT380k**，并设计了一个两阶段的对齐策略：首先通过SFT建立模型的认知与推理框架；随后采用GRPO进一步引导模型的推理策略，提升其推理思维链的一致性与可验证性。基于Geo-CoT框架训练得到的模型**RSThinker**，不仅能够输出最终结论，还能同时生成其推理过程的可验证思维链。在多项遥感基础任务上，RSThinker均显著超越现有的主流模型，展现出卓越的可解释推理能力。我们开源了Geo-CoT380k数据集与RSThinker模型权重，为从“黑箱感知”迈向“结构化可验证推理”的地球观测智能提供了新的路径。


# 创新点
我们通过构建大规模可验证推理数据集Geo-CoT380k并采用两阶段对齐策略，有效地将“可验证的链式推理机制”注入遥感视觉语言模型。我们的主要贡献如下：
- 提出并形式化了Geo-CoT——这一遥感推理范式，要求在遥感问题分析中，每一个推理步骤都必须与其对应的视觉证据建立可验证的关联，从而实现可解释与可信的推理。
- 构建了首个遥感领域的大规模思维链监督微调数据集 Geo-CoT380k，该数据集显式构造了任务分解、证据逐步对齐及结果综合的认知结构，为模型推理能力的学习提供了坚实基础。
- 提出了基于两阶段训练的遥感视觉语言推理模型RSThinker。实验结果表明，以SFT作为GRPO强化学习的前置阶段是获得高质量推理能力的关键。RSThinker在一系列典型遥感任务上均取得领先性能。

<p align="center">
  <img width="90%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/overview.png" /><br>
  图1：Geo-CoT的整体框架
</p>

---

# 方法
Geo-CoT框架通过两阶段对齐策略实现：第一阶段利用SFT构建模型的认知结构；第二阶段采用GRPO进一步提升模型推理结果的事实一致性与可验证性。
- SFT训练。SFT阶段的关键在于依托一个体现Geo-CoT原则的大规模结构化推理思维链数据集。为此，我们设计了一条可扩展的思维链标注流水线，在多个现有的公开遥感数据集基础上，利用GPT-4V自动生成推理思维链。与传统的开放式推理生成不同，我们在标注过程中提供了**真实标注信息**与**思维链示例**，以约束 GPT-4V的生成任务，使其输出与真实证据一致、逻辑可验证的逐步分析推理思维链。这一策略确保了数据的真实性与逻辑一致性，从而构建了高保真的SFT-CoT数据集，Geo-CoT380k，共包含 384,591 条结构化推理样本，覆盖各项遥感基础任务。

<p align="center">
 表1：Geo-CoT380k覆盖的数据集<br>
 <img width="50%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/cot380k.png" /><br>
</p>

- GRPO训练。虽然SFT阶段成功地为模型注入了Geo-CoT的结构化模板，但其基于最大似然的逐词优化目标仍可能导致模型在局部层面生成“看似合理但事实不符”的推理链。为克服这一问题，我们在第二阶段引入GRPO强化学习，通过最终输出的结果质量来驱动策略更新。每个任务的奖励函数均由其标准评估指标直接定义，确保优化目标与实际任务性能严格一致。

<p align="center">
 表2：GRPO训练中不同任务的奖励设计<br>
 <img width="50%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/reward.png" /><br>
</p>

---

# 实验
我们在各项典型遥感任务上对所提出的模型RSThinker进行了系统实验评估。实验涵盖了从细粒度目标感知（目标计数、目标检测与视觉定位）到全局场景理解（场景分类与图像描述）以及复杂地理空间推理（视觉问答）的完整任务体系。结果表明，RSThinker在各项任务上均取得显著优于现有主流视觉语言模型的性能，充分验证了所提出框架在复杂遥感分析中的可靠性与泛化性。
在细粒度感知任务中，Geo-CoT框架要求模型显式地将推理过程与视觉证据对应，使得RSThinker在视觉定位、目标检测与计数任务上均取得显著优势。其逐步的“规划–对齐–综合”结构有效降低了重复计数与漏检问题，显著提升了模型的空间精确性与稳定性。
<p align="center">
 表3：RSThinker在视觉定位、目标检测与目标计数任务上的结果<br>
 <img width="90%"  alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/vg.png" />
 <img width="50%"  alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/oc.png" />
 <img width="42%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/det.png" /><br>
</p>

在场景分类与图像描述任务中，RSThinker能够通过思维链推理聚合细粒度证据，从而在分类与描述任务中生成更具事实依据的判断与文本。这种基于证据的综合能力避免了传统模型依赖全局统计特征所导致的偏差，展现出更强的鲁棒性与可解释性。
<p align="center">
  表4：RSThinker在图像描述任务上的结果<br>
 <img width="90%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/ic.png" /><br>
</p>
在复杂地理空间推理任务（如VQA）中，RSThinker 能够在多步逻辑问题（如存在判断与数量比较）中稳定输出符合事实的答案，体现了 Geo-CoT 在促进多层次推理一致性方面的优势。
<p align="center">
 表5：RSThinker在场景分类和视觉问答任务上的结果<br>
 <img width="90%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/sc_vqa.png" /><br>
</p>
此外，我们进行了系统的消融实验以剖析各组件的独立贡献。
<p align="center">
 表6：消融实验结果<br>
 <img width="90%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/ablation.png" /><br>
</p>
结果显示，仅在不包含推理思维链的遥感数据集上直接微调（SFT w/o CoT）虽可提升性能，但引入结构化推理思维链（SFT w/ CoT）后模型表现显著跃升。进一步地，在此基础上结合 GRPO 强化优化后，模型在复杂推理任务中取得最佳性能，而若缺少 CoT 支撑结构（SFT w/o CoT + GRPO），强化学习无法有效收敛。这表明：SFT 阶段的认知结构学习与 GRPO 阶段的事实对齐优化具有互补与协同作用。整体而言，实验结果充分证明了Geo-CoT框架能够显著提升遥感视觉语言模型的推理可信度与可验证性。


---

# 可视化展示

<p align="center">
 <img width="90%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/example_1.png" /><br>
 图2：RSThinker的测试样例（目标计数任务）
</p>

<p align="center">
 <img width="90%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/example_2.png" /><br>
 图3：RSThinker的测试样例（目标检测任务）
</p>

<p align="center">
 <img width="90%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/example_3.png" /><br>
 图4：RSThinker的测试样例（视觉定位任务）
</p>

<p align="center">
 <img width="90%" alt="image" src="https://github.com/minglangL/RSThinker/blob/main/figs/figs_chinese/example_4.png" /><br>
 图5：RSThinker的测试样例（视觉定位任务）
</p>
