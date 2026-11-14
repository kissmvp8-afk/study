# R11 中文总结报告（Dot 独立归档）

## 结论
720 个评价案例执行完毕，性能门槛未通过。执行成功不等于性能验收通过；本报告不构成后续开训或释放旧服务器批准。

## 完成证据
评价为 6 分支 × L32/L64/L128 × development/heldout，共 36 组，每组 20 案例。
manifest 结束时间：2026-11-14T09:11:38.910711Z（北京时间 17:11:38）。日志 RUNNER_DONE total=36 failed=0。
09:13:09 UTC 核验时评价与协调器进程已退出，GPU 为 0%、2 MiB；资源仅为该时点快照。
gate-summary 于 09:15:11 UTC 用既有 r10_gate.py 生成，未重跑评价、未修改训练脚本。

## 门槛统计
36 组，qualified 0；survival 719/720；任务通过 319/720（44.3%）。这一合计覆盖全部检查点，不是最终模型通过率。
门槛要求：20 个有效案例、存活 20/20、任务通过 20/20，mean_P <= 0.04、mean_W <= 0.08、mean_Mavg <= 0.08。
P=max(stop1.speed,stop2.speed)；W=max(stop1.abs_wz,stop2.abs_wz)；Mavg=(forward.vx_mae+reverse.vx_mae)/2。
precision-3/L32/heldout 出现 missing_or_nonfinite_window，仅 19 个有效案例，严格均值为空；不能用 19 案例均值替代全样本门槛。

## L128 明细
| 分支/集 | 存活/20 | 通过/20 | mean_P | mean_W | mean_Mavg | qualified |
|---|---:|---:|---:|---:|---:|---|
| control-1/L128/development | 20 | 9 | 0.04811 | 0.08887 | 0.04556 | False |
| control-1/L128/heldout | 20 | 5 | 0.04601 | 0.09293 | 0.05120 | False |
| control-2/L128/development | 20 | 7 | 0.06062 | 0.07192 | 0.04965 | False |
| control-2/L128/heldout | 20 | 9 | 0.05616 | 0.07404 | 0.04803 | False |
| control-3/L128/development | 20 | 11 | 0.05074 | 0.05944 | 0.04955 | False |
| control-3/L128/heldout | 20 | 9 | 0.05581 | 0.05863 | 0.05633 | False |
| precision-1/L128/development | 20 | 13 | 0.04702 | 0.06124 | 0.04645 | False |
| precision-1/L128/heldout | 20 | 8 | 0.05264 | 0.07499 | 0.04860 | False |
| precision-2/L128/development | 20 | 10 | 0.04844 | 0.06805 | 0.04404 | False |
| precision-2/L128/heldout | 20 | 5 | 0.05464 | 0.08134 | 0.04813 | False |
| precision-3/L128/development | 20 | 5 | 0.06188 | 0.06546 | 0.06638 | False |
| precision-3/L128/heldout | 20 | 8 | 0.05475 | 0.08248 | 0.06661 | False |

所有 L128 组均未满足 P 和全案例任务通过要求，部分组也未满足 W；不据此宣称研究赢家。

## 资源边界
历史 4 路评价快照显示 GPU 利用率约 14%–94% 波动、显存约 5.7–8.9 GiB/24 GiB，不能宣称持续满载或显存达到 80%。未用空占显存、改并发或重启任务凑指标。
历史容器 CPU 约 5.1 核等效使用（16 核额度），内存使用包含大量缓存；均不是连续性能监测结论。

## 备份与迁移边界
2026-11-14 08:45:48 UTC 附近核验：R11 六个 full-L0128.pt 本地 SHA256 与 GitHub Release 服务端 digest 一致，每个 216117465 字节，总约 1.297 GB。标签 r11-precision-replication-20261114。
旧 upload-verify 文件存在 ERROR 和 downloaded_sha256=null，不能把 ALL DONE 当成功。后续服务端 digest 比对是新证据，但不是本次全部下载读回或新机恢复加载。
R10P first-upload.json 有四个检查点 VERIFIED 且本地和下载 SHA 一致，属于历史记录，本次未重新下载。
历史 R10P/R11 du 约 13G/18G，为当时占用，并非精确完整迁移清单。

### 新机进度（主协调员提供，本报告未独立复核）
报告编制时收到更新：新机已登录，GPU 标识 RTX 4080 SUPER，报告显存 32760 MiB；六个 R11 L128 检查点中 control-1 已下载且 SHA 一致，其余遭遇网络超时、正在断点续传。协调员未提供该快照的精确采集时间；最新全量状态尚未核实。
该信息不能证明六个文件全部迁移，更不能证明完整捕获数据、评价结果、代码配置环境已迁移或新机可恢复训练。不得将检查点备份当全部迁移完成，不得据此删除旧机原件。

## 证据与后续
证据：logs/eval-manifest-r11.json、logs/eval-runner.log、各组结果 JSON、r10_gate.py、eval/R11/gate-summary.json。
建议：先定位缺失/非有限窗口与停稳指标失败原因；完成完整迁移清单、哈希核对和新机恢复验证，保留回滚原件。任何重训或重评另行协调。
本报告仅新增独立 UTF-8 文本，不覆盖既有文件，不修改训练或原始数据，不含密码、令牌、密钥或原始捕获数据；尚未推送或合并。

生成时间 UTC：2026-11-14T09:21:15.306703+00:00
gate-summary SHA256：769ebbee6685b641b2524a1af9114508a32356160f7e8941c01b2c4986cb6caa
