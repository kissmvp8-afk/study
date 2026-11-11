# SAC-R9L 本机执行规范

本目录补齐此前只作为聊天附件交付的本机质量优先任务书，基于固定R9/H1审阅提交 `19ae9a0a27ef321e638af58e298c9a8af235f927`。本次只发布规范和参数，没有访问用户电脑或启动训练。

本地Codex先读 [LOCAL_3060_NEXT.txt](LOCAL_3060_NEXT.txt)，然后读 [R9L_LOCAL_EXECUTION.txt](R9L_LOCAL_EXECUTION.txt)、[parameters.json](parameters.json) 和 [FILE_SHA256.json](FILE_SHA256.json)。用户已报告RTX 3060 Laptop单路128环境采集通过；完整CUDA学习、恢复与本机评价验收仍须本机实际完成。

保持现有SAC，R8 A1/1502-L48为研究暖启动，R6正式父模型只读。A为Z0/Z1各三条训练fork，B在A合格后做B0/B1各三条；本设备固定单进程调度六条轨迹，不要求四路并发或9小时截止。

[budget-and-scale-verification.json](budget-and-scale-verification.json) 仅验证128/64/32环境的预算等式，不是仿真或学习通过证据。完整指令明确固定版本、控制变量、计数、开发/留出门控、停止与交付要求；本目录不是已验证的一键训练程序。
