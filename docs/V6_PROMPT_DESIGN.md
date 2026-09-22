# V6 Prompt 设计：Local Evidence + Global Evidence + Shared Mechanism Knowledge

## 1. 设计目标

V6-A 只修改 Prompt 的信息内容与组织，不改 TimeCMA-FD 的网络结构。以下部分全部保持 V5 不变：

- DE+FE 双传感器输入；
- 1024 点样本窗口；
- 256 点 patch、stride 128，共 7 个 patch token；
- Signal Encoder、RevIN、CLS、Prompt Encoder；
- GPT-2 模型与 last-valid-token pooling；
- Cross-Attention 与分类头；
- LOLO 划分、训练预算和评价指标。

因此 V5 与 V6-A 的差异可以解释为“Prompt 信息质量变化”，而不是模型结构变化。

## 2. 三层 Prompt

每个 patch 仍对应一个 Prompt token，但文本由三部分构成。

### 2.1 Local Evidence：每个 patch 不同

来源：当前 256 点同步 DE/FE patch。

每个传感器保留：

- mean
- std
- RMS
- peak
- skewness
- kurtosis
- crest factor
- raw-FFT dominant frequency

新增跨传感器局部证据：

- DE/FE RMS ratio
- DE/FE peak ratio
- DE-FE waveform correlation

这部分描述“当前局部时间片发生了什么”。

### 2.2 Global Evidence：同一 1024 点样本的 7 个 patch 共用

来源：完整 1024 点 DE/FE 窗口。

包含：

- DE/FE 全窗口 RMS、kurtosis、crest factor；
- raw-FFT dominant frequency；
- spectral centroid；
- 基于 Hilbert 解析信号得到的 envelope-spectrum dominant frequency；
- 全窗口 DE/FE RMS ratio、peak ratio、waveform correlation。

这部分描述“整个样本有哪些更稳定的全局振动与频谱证据”。频谱相关量使用 1024 点窗口而不是 256 点 patch，以获得更合理的频率分辨率。

## 3. 固定故障机理知识

下面的知识文本对所有样本、所有 patch 完全相同，同时包含四类状态，绝不根据真实标签选择某一段，因此不是标签泄漏。

> General bearing-fault mechanism knowledge shared by every sample: Healthy bearings usually do not exhibit persistent defect-related periodic impacts or stable fault-characteristic harmonic patterns. An outer-race defect is stationary relative to the housing; rolling elements repeatedly pass the defect and can generate periodic impacts associated with BPFO-related components and harmonics. An inner-race defect rotates with the shaft; repeated contacts can generate BPFI-related components and harmonics, and their amplitudes may be modulated by shaft rotation as the defect moves through the load zone. A rolling-element defect can generate BSF-related responses; because the damaged element contacts both races while rotating and orbiting, its impulsive and modulation patterns can be less stable and may contain cage- or shaft-related modulation. Fault characteristic frequencies depend on shaft speed and bearing geometry, so a single maximum peak in the raw FFT must not by itself be interpreted as BPFO, BPFI, or BSF evidence.

这部分的作用不是直接告诉模型类别，而是提供“观测证据应该如何与轴承故障概念建立关系”的共享语义先验。

## 4. Prompt 模板

每个 patch 最终近似组织为：

    Bearing vibration diagnostic evidence.
    Operating condition: load {load_hp} horsepower.

    Local evidence for temporal patch {i} of {P}:
    Drive-end sensor: ...
    Fan-end sensor: ...
    Local cross-sensor relation: ...

    Global evidence shared by all patches in this 1024-sample observation:
    DE global evidence: ...
    FE global evidence: ...
    Global cross-sensor relation: ...

    General bearing-fault mechanism knowledge shared by every sample:
    Healthy ...
    Outer-race ...
    Inner-race ...
    Rolling-element ...
    Characteristic-frequency caution ...

为了保证本轮只检验 Prompt 内容，V6-A 仍保持 V5 的尾部空格和 last-valid-token pooling。该 pooling 问题留到后续独立版本处理。

## 5. 本版本刻意没有加入的内容

V6-A 当前没有从 MAT 文件向 Prompt 传入可靠的 shaft RPM，也没有在代码中硬编码轴承几何参数。因此本版本不生成：

- order spectrum 数值；
- BPFO/BPFI/BSF 的数值预测频率；
- “检测到 inner/outer/ball 故障”之类样本级结论。

这样可以避免把 raw FFT 最大峰伪装成故障特征频率，也避免通过人工规则提前完成分类。

如果后续确认 RPM 与轴承几何参数可靠，再创建新的 Prompt template version，引入真正的 order-domain 与 characteristic-frequency evidence。

## 6. 缓存隔离

V6-A 的 cache spec 使用新的：

    prompt_template_version = v6_local_global_mechanism_v1

因此不能静默复用 V5 的旧 embedding 缓存。请为 V6 使用新的 embedding-root，重新生成全部 GPT-2 embedding。
