# AI 编程助手（v1.1 · 仅免费模型）

基于 OpenRouter **免费模型**的命令行编程 Agent，约 750 行 Python。

## 快速开始
```bash
pip install openai
python agent.py --init-config      # 生成全局配置 ~/.ai-coder/config.toml，编辑填入 api_key
python agent.py --dir 你的项目目录   # 在项目目录里工作（不写 --dir 则用当前目录）
```
**API Key 的两种设置方式**（环境变量优先于 config.toml）：
- 写进 `config.toml` 的 `api_key = "..."`（Windows 在 `C:\Users\你\.ai-coder\config.toml`）
- 或环境变量：cmd `set OPENROUTER_API_KEY=你的key` · PowerShell `$env:OPENROUTER_API_KEY="你的key"` · Mac/Linux `export OPENROUTER_API_KEY=你的key`

> ⚠️ AI 只能读写**工作目录**。不要在 `ai-coder` 自己的目录里工作（它会改到自己的源码），请用 `--dir` 指向你的项目。
> `AGENTS.md`（项目规范）也是放在**工作目录**里才生效，把 `AGENTS.md.example` 复制过去改名即可。

## 运行测试
```bash
cd ai-coder
python -m unittest          # 63 个离线测试，不联网、不花钱
```

## 文件结构
| 文件 | 职责 | 为什么必须有 |
|---|---|---|
| `agent.py` | Agent 循环、命令、REPL | 把"想→做→看结果"串起来，没有它就只是聊天机器人 |
| `llm.py` | 调 OpenRouter、流式解析 | 屏蔽 API 细节；换服务商只改这里 |
| `textproto.py` | 文本工具协议（给不支持原生工具调用的模型） | 免费模型的 function calling 常常不稳定或不支持，文本协议是兜底 |
| `tools.py` | 工具说明书 + 实现 + 安全沙箱 | 模型的"手"；也是安全的核心控制点 |
| `memory.py` | history.jsonl 读写、修复、压缩 | 模型无状态，必须自己管记忆；长对话要压缩 |
| `git_util.py` | 自动提交 / 撤销 | AI 会改错，必须有后悔药 |
| `models.json` | 模型与参数配置 | 配置与代码分离，加模型不改代码 |
| `tests/test_all.py` | 离线测试（63 个用例） | 改代码不怕改坏：`python -m unittest discover -s tests` |

## 版本演进（每个版本都可独立运行）
| 版本 | 功能 | 对应位置 |
|---|---|---|
| v0.1 | 读/写文件 + Agent 循环 + 记忆 | `agent.py` 的 `run_turn` |
| v0.2 | `list_files`、`run_command`（需确认，可自动修 bug） | `tools.py` |
| v0.3 | 路径沙箱、命令黑名单 | `Tools.safe_path` / `DANGEROUS` |
| v0.4 | 流式输出、彩色终端 | `llm.complete` |
| v0.5 | `edit_file` 局部修改 | `Tools.t_edit_file` |
| v0.6 | `AGENTS.md` 项目规范 | `Agent.system_prompt` |
| v0.7 | 历史自动压缩、持久化 | `memory.compact` |
| v0.8 | `grep` 代码搜索 | `Tools.t_grep` |
| v1.0 | `/plan` 计划模式、Git 自动提交与 `/undo`、日志 | `Agent.plan` / `git_util` |

## 免费模型说明
- 代码层强制只允许 `:free` 模型和 `openrouter/free`（官方免费自动路由）；`models.json` 里写了付费模型会直接拒绝启动，`/model` 也不接受付费 ID，不会误扣费。
- 免费模型**会轮换、会限流**：程序遇到限流自动等待重试（指数退避），仍失败则按 `fallback_models` 依次换备用模型；模型不支持工具调用（404）则立即换。
- `/models` 联网列出当前免费且支持工具调用的模型，`/model 模型ID` 直接切换，不必改配置文件。
- 免费模型能力参差，复杂任务建议用 `/plan` 先看计划再执行。

## 两种工具调用方式
| | 原生 function calling（默认） | 文本协议 |
|---|---|---|
| 做法 | 工具通过 API 的 `tools` 参数传递，模型返回结构化调用 | 工具说明写进提示词，模型用 `<tool name="..."><path>..</path></tool>` 标签调用，程序解析 |
| 优点 | 标准、省 token | 任何模型都能用；参数是原文，不用 JSON 转义，写代码更稳 |
| 缺点 | 部分免费模型不支持或不稳定；参数是 JSON，输出被截断就会变成半截 JSON | 依赖模型遵守格式 |

- 模型报"不支持工具调用"时，程序会**自动**对该模型改用文本协议重试。
- 也可手动切换：`/toolmode text`，或在 models.json 里给模型加 `"tool_mode": "text"`。
- 历史记录始终用原生格式存储，发送时才转换，所以两种方式可随时切换。

## 常见问题
- **404 "This model is unavailable for free"**：该模型不再免费（OpenRouter 会提示付费版 slug）。本项目不会自动改用付费版，请用 `/models` 查当前免费模型，或用 `auto`。
- **400 "arguments must be a valid JSON"**：旧版本在模型输出被截断时把半截 JSON 存进了历史，之后每次请求都失败。新版已修复，并会在启动时自动修复旧的 history.jsonl。
- **输出被截断**：写长文件时模型可能超出 `max_tokens`。程序会提示模型"先写骨架，再用 edit_file 分段补充"。

## 命令
`/model [名]` `/toolmode [native|text]` `/models` `/plan 任务` `/compact` `/undo` `/clear` `/help` `exit`

## 配置
- **全局 `config.toml`**：`api_key`、`default` 默认模型、`[settings]` 覆盖参数、`[models.xxx]` 追加免费模型。对所有项目生效。
- **`models.json`**：内置的模型列表和默认参数。优先级：config.toml > models.json。

### models.json 说明
- `models`：加一行即新增模型（ID 以 openrouter.ai/models 为准，需支持 tools）
- `settings`：`context_limit_chars` 压缩阈值、`auto_commit` 自动提交开关、`command_timeout`、`max_steps`

## 安全说明
1. 文件操作限制在工作目录内（含符号链接逃逸、`.git` 保护）
2. 命令黑名单只是"尽力拦截"，**真正的防线是每条命令都需你确认**——请看清再按 y
3. 自动提交只在 git 仓库且你没有未提交改动时进行，不会混入你的修改

## 下一步可以扩展
- 写文件前显示 diff 并确认 · Token 精确计数 · 多文件重构的子任务 · 支持 MCP 工具 · Web 界面
