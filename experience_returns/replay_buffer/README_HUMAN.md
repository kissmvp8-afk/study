# D02 · 经验池写入与抽样｜人类阅读版

[AI 实施规范](README.md) · [返回大模块人类版](../README_HUMAN.md) · [Codex](../../CODEX.md) · [Gemini](../../GEMINI.md)

| 字段 | 当前内容 |
|---|---|
| 文档状态 | 技术初稿完成，待 Gemini 润色 |
| 技术核对状态 | 已对照实际实现、9 项测试及 D01/D03 调用核对接口、形状、复制与覆盖规则；本次不扩展为全面代码审核 |
| 语言润色状态 | 待 Gemini 润色，尚未润色 |
| 已核对代码版本 | 审核基准 `7071e41f27f0b5e56b9bed3ef3e796741f96ee40`；实现与测试来自 `3838fe4a0d00bf2e8e102edda39a58d5c21e9f0c`，读取时无后续源码变化 |
| 待同步代码版本 | 无 |
| 实际阅读材料 | 根规范、上层及本目录 AI README；[buffer.py](buffer.py)、[test_buffer.py](test_buffer.py) 全文；D01 `schema.py`/测试、D03 `returns.py`/测试，D04/D05 衔接测试；固定 PPO `rollout_storage.py` 相关片段。全局材料见[大模块阅读记录](../README_HUMAN.md#阅读材料与证据) |
| 验证依据与限制 | 用户报告 D01～D05 共 55 项 CPU 检查已独立通过；本模块 9 项的既有结果见 [PR #5](https://github.com/kissmvp8-afk/study/pull/5) 和 [AI 交接表](README.md#实现证据与文档交接)。本次只核对文档，不重跑算法测试；人工数据，未接入仿真或训练 |

## 要解决的问题

SAC（Soft Actor-Critic，软演员—评论家算法）会反复使用过去的经历。经验池不能只把记录排进数组：容量用完后会覆盖旧数据，不同环境的写入也可能交错。如果把相邻槽位直接当成相邻时刻，就可能拼出从旧一轮跳到新一轮的假轨迹。

D02 保存 D01 的原始 `TransitionBatch`，按显式编号寻找连续片段，再把候选起点交给 D03。这里的“环形缓冲区”是槽位用到末尾后回到开头；“写入序号”则持续递增，用来辨认同一个槽位里是哪一代内容。它不计算回报、不归一化，也不缓存会随网络训练改变的历史编码。

## 从 D01 写入，到 D03 读取

设 E=`num_envs`、K=`capacity_per_env`、C=`q_dim`，M 是抽取的起点数，L 是每个起点读取的记录数。K 的单位是**每个环境的记录条数**，总容量为 E×K。

| 数据 | 输入/存储/输出形状 |
|---|---|
| D01 来源 | 历史 `[B,10,33]`、Q 观测 `[B,C]`、动作 `[B,8]` 等 13 个字段 |
| 池内每个字段 | `[E,K,*字段形状]`；例如历史 `[E,K,10,33]`，编号 `[E,K]` |
| 起点 | `env_ids[M]` 与 `slots[M]`；slot 是池内槽位，不是动作步号 |
| `SequenceBatch.fields` | 历史 `[M,L,10,33]`、奖励 `[M,L,1]`、编号 `[M,L]` 等 |
| `SequenceBatch.valid` | `[M,L] bool`，表示从起点到该位置是否连续有效 |
| `SequenceBatch.at(offsets)` | 每行选一条，回到 D01 的 `[M,...]` 布局 |

```text
D01 make_transition → ReplayBuffer.append
                        ↓
               valid_starts(n) → sample_starts(M,n)
                                      ↓
                         read_sequence(...,length=n)
                                      ↓
                         D03 build_n_step_batch
```

`append` 允许每批只包含部分环境，但同一批不能出现重复 `env_id`：批内没有定义同一环境两条记录的先后。连续多步应分次写入。记录必须与池同设备，环境号在 `[0,E)`，并通过 D01 校验。

写入通过索引赋值复制；读取字段通过高级索引复制。来源之后被改写、读出结果被改写，都不会影响池内字段。`SequenceBatch` 虽然是冻结数据类，其字典和张量仍可修改；不能因此把读出对象视为不可变。D01 的 `detach().clone()` 仍是采集数据入口的责任，D02 不取代它，也不能纠正错误的观测来源。

## `buffer.py`：逐函数职责

| 符号 | 输入、输出与副作用 |
|---|---|
| `ReplayBufferError` | 经验池参数、读写约定不满足时的异常类型，无自定义方法；D01 校验失败仍抛 `TransitionSchemaError` |
| `WRITE_SEQ_DTYPE` | `torch.int64`；槽位写入序号 -1 代表未写入 |
| `bytes_per_record(q_dim)` | 按 D01 布局与实际 dtype 字节数，返回单条记录加一个写入序号的字节数；不分配经验池 |
| `estimate_storage_bytes(num_envs, capacity_per_env, q_dim)` | 返回 E×K×单条字节数，供分配前估算，不含每环境写指针等少量簿记 |
| `ReplayBuffer.__init__(num_envs, capacity_per_env, q_dim, device='cpu')` | 要求 E、K 为正，一次分配全部字段、`_write_seq[E,K]` 与 `_writes[E]`；没有默认容量 |
| `ReplayBuffer.size` | 返回已写入且仍占用槽位的记录总数，不是有效 n 步起点数 |
| `ReplayBuffer.storage_bytes` | 统计实际字段存储、写入序号和写指针的张量字节数，不含 Python 对象及读取临时张量 |
| `ReplayBuffer.append(batch)` | 调用 D01 `validate_transition`，检查设备、环境号范围和批内唯一性；覆盖对应环境的下一槽位，返回 `[B]` 槽位；会修改池内状态 |
| `ReplayBuffer._continuity()` | 返回 `[E,K]` 的 `present`、`linked`、`boundary`：是否已写、到下一槽位的编号连接是否成立、是否为结束记录。供起点选择与读取复用 |
| `ReplayBuffer.valid_starts(n_steps)` | 要求 1≤n≤K，返回全部有效 `(env_ids, slots)`。有效表示已凑满 n 条，或在完整有效前缀内遇到结束；未结束且缺后继则暂不可用 |
| `ReplayBuffer.sample_starts(batch_size, n_steps, generator=None)` | 先调用 `valid_starts`，再有放回均匀抽取 M 个起点；M 必须为正，无候选时报错。传 CPU `torch.Generator` 可控制随机数；会推进所用随机数状态，不修改池内容 |
| `ReplayBuffer.read_sequence(env_ids, slots, length)` | 参数为 `[M]` 起点及 1≤L≤K，检查范围，沿槽位环读取 L 条并生成连续有效标记，返回 `SequenceBatch`；不会等待缺少的后继 |
| `SequenceBatch.length` | 读取 `valid.shape[1]`，返回 L |
| `SequenceBatch.at(offsets)` | 用 `[M]` 偏移逐行选记录；偏移会转到对应设备/int64，越界或选到无效位置报错。返回按字段高级索引得到的 `TransitionBatch`，不再调用 D01 校验 |

存储分配直接导入 D01 的内部 `_expected_layout(q_dim)`，确保形状与 dtype 只有一个来源。这是当前实际依赖，尚未改成公开接口；D01 改动该函数时必须同步本模块。各目录目前通过 `sys.path` 互相导入，不是已安装的 Python 包。

## 环形槽位为什么还需要写入序号

`append` 的关键代码是：

```python
seq = self._writes[env]
slot = seq % self.capacity_per_env
```

`seq` 是各环境自己的累计写入次数，取模仅决定落在哪个物理槽位。写入后，该槽位的 `_write_seq` 保存 seq，环境的 `_writes` 加一。因此槽位可重复，序号不能混淆。

现有测试给一个环境容量 4，依次写 step 0..9，最后布局为：

| 槽位 | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 当前 step / 写入序号 | 8 | 9 | 6 | 7 |

时间顺序是槽位 `2→3→0→1`，即 step `6→7→8→9`。从槽位 1 再绕到 2，会由序号 9 跳回 6，不能视为后继。`_continuity` 除了检查已写入，还要求：

```python
& (self._write_seq[:, nxt] == self._write_seq + 1)
& (episode[:, nxt] == episode)
& (step[:, nxt] == step + 1)
```

三行分别防止接到另一代覆盖内容、另一轮任务，以及重复/跳过的动作步。同一环境由存储行保证。`nxt` 也是取模后的下一槽位，故合法的跨数组末尾连接可以保留。

`read_sequence` 按这些连接递推 `valid`，断开后不会重新变真。它不靠结束标记单独截断连接；正常输入在结束后换 `episode_id`，自然断开，D03 还会显式在首个 `terminated | time_out` 处停止回报累计。上游必须正确维护编号和结束标记。

## 有效起点不是“剩下几条就用几条”

`valid_starts` 的循环维护 `ok` 和 `open_`：前者表示尚合格的起点，后者表示尚未遇到结束、仍需后继的起点。

```python
link_j = torch.roll(linked, shifts=-j, dims=1)
ok &= ~open_ | link_j
open_ &= link_j & ~torch.roll(boundary, shifts=-(j + 1), dims=1)
```

第一行把第 j 次连接与各起点对齐；第二行只要求仍开放的起点具有连接，已经在结束步闭合的起点无需继续延伸；第三行更新下一轮循环中还需后继的集合。结束步本身保留。

在上面的容量 4 例子中，没有结束标记：n=3 仅 step 6、7 有效，n=4 仅 step 6 有效；step 8、9 缺未来记录，不能把缺数据当成任务结束。另一项测试让某轮 step 2 真终止，则从 step 1 读三条可得到 `valid=[True,True,False]`，从该起点实际只有两步，但已经合法到达边界。

读取的无效位置可能是空槽、另一轮或较旧记录，不可直接拿 `fields` 的全部 L 条训练。应使用 D03 检查有效前缀，或经 `SequenceBatch.at` 选取有效记录。起点槽位也会随后续写入被覆盖；本接口没有承诺跨任意多次写入仍保持同一个起点内容。

抽样使用 `torch.randint`，因此是有放回抽样，同一条可能出现多次。测试在池内容相同、CPU Generator 都设种子 123 时，抽到的 16 个 n=2 起点一致；它不保证环境发生变化后结果仍相同。

## 容量：两段历史占了多少空间

单条含两段 `[10,33] float32` 历史，单这部分就是 `2×10×33×4=2640` 字节。加两段 Q、动作、奖励、两个 bool、五个 int64 字段和写入序号，总计：

```text
bytes_per_record(C) = 2640 + 8C + 32 + 4 + 2 + 40 + 8
                    = 2726 + 8C
C=54  → 3158 字节/条
C=241 → 4654 字节/条
```

4096 个环境、每环境 1221 条是 5,001,216 条，C=241 时字段与写入序号约占 23.3 GB（十进制），还未计分配器、Python 对象和抽样临时内存。`storage_bytes` 比估算额外包含 E 个 int64 写指针；测试 E=3、K=5 时两者相差 24 字节。这是存储计算，不是已测得的大容量运行性能，也不是推荐默认容量。

## `test_buffer.py`：怎样验证

辅助函数 `code_of(env,episode,step)` 用 `env×10000+episode×100+step` 编码；`make_records(entries, terminated, time_out, rewards, q_dim)` 经 D01 构造记录，默认奖励为 code，真实下一观测为 code+0.5。`write_episode` 逐步 append，`start_steps` 把有效槽位读回为编号列表。这些全是人工数据，Q_DIM=6 不是实际机器人 Q 维数。

| 测试类/方法 | 预期与覆盖范围 |
|---|---|
| `WriteReadTest.test_round_trip_all_fields` | 写入环境 2/0 两条记录，经读取及 `at` 后全部 D01 字段一致，size=2 |
| `WriteReadTest.test_copies_on_write_and_read` | 改来源奖励和历史、再改读出奖励，都不改变池中原值 0 和 0.5 |
| `OverwriteTest.test_wrapped_ring_keeps_only_valid_starts` | 容量 4 写 10 次，核对 step 6..9 及 n=3/4 的有效起点 |
| `OverwriteTest.test_sampled_sequences_are_consecutive_after_many_overwrites` | 两环境多次覆盖后抽 64 个起点，检查同环境同轮、步号连续，凑满 3 条或已结束 |
| `MultiEnvAndBoundaryTest.test_interleaved_envs_only_join_same_env` | 不同步写入仍只拼同环境片段 |
| `MultiEnvAndBoundaryTest.test_episode_boundaries` | 真终止/纯截断均可提前闭合；新一轮未凑够的起点不可用 |
| `MultiEnvAndBoundaryTest.test_reproducible_sampling` | 同种子、同池内容，抽样一致 |
| `CapacityAndErrorTest.test_bytes_follow_actual_fields` | C=54/241 的手算字节数及写指针额外开销 |
| `CapacityAndErrorTest.test_rejects_bad_writes_and_reads` | 拒绝重复/越界环境号、空候选、n 超容量、非正容量、`at` 指向无效位置 |

各测试由 `unittest.main()` 入口执行。已有 9 项 CPU 通过记录，不包含实际采集循环、GPU 或大容量性能。本次未重跑；需要复现时可在根目录执行原命令：

```bash
uv run --no-project --python 3.11 --with torch==2.15.0 python -B experience_returns/replay_buffer/test_buffer.py -v
```

排错时，先区分 `size` 与有效起点数量；片段错误查 `_write_seq`、环境/轮次/步号和 `valid`，内存问题查容量单位与 C。D02 无法证明观测是重置前捕获的，E05 仍负责来源。

## 下一步与润色交接

继续阅读 [D03](../n_step_returns/README_HUMAN.md) 看如何把连续片段变成 R、k 与尾部；[D04](../normalization/README_HUMAN.md)、[D05](../history_windows/README_HUMAN.md) 在读出原始历史上工作。完整字段含义见 [D01](../transition_schema/README_HUMAN.md)。

Gemini 只润色本文件的表达，保留接口名、形状、槽位示例、容量单位与未验证项；不把人工衔接检查描述为仿真或训练通过。
