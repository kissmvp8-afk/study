# L24 seed232/233 D/S独立审阅

入口：[结论](review.txt)、[后续交接](next-plan.txt)、[参数](parameters.json)、[逐例表](cases.csv)、[配对图](DS-pairs.svg)。输入与版本见input-evidence.json；新增417文件实际下载核验，原231D及C1复用已核验版本。

`audit_ds.py`只读标准库脚本，默认工作区路径写在参数中，可用--work、--c1、--old-l24指定。需要已解压的三个新增归档、原231D和C1、L24-verification.json及model-inference-reference.json；L24验证文件的位置在源码中明确列出。复算54例31258步，逐项比较原protocol-result，再独立复算四窗口和通过门槛。产物包含完整逐例表、D/S与L24/L48配对、源码指纹和审计回执，不启动Isaac或加载优化器。原噪声历史未保存，所以不伪造共同输入。

P/W分别是两段停车窗口平均速度/平均绝对wz的较大值，Mavg是正反MAE均值，Mmax是验收用较大值；角速度rad/s、速度m/s、高度m。step从0起，dt=.02。瞬时采样偏差使用同一S访问状态的mu及已核验动作边界重建确定性动作，与实际D/S闭环差不同。

验证：实际运行脚本通过，54例与原评分一致、同seed初态/RNG/15源码指纹一致、D/S首个推理mu/std/速度预测相同、冻结前后不变；D动作重建误差<2e-5。本轮没有新训练、物理仿真、参数调整。234/235仍按用户报告记运行中；C2原始证据待独立核验。
