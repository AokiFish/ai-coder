"""llm.py —— 和 OpenRouter 通信（支持流式输出）。"""
import json
import os
import urllib.request
import uuid

DEFAULT_SETTINGS = {
    "context_limit_chars": 60000,  # 历史超过这个字符数就触发压缩
    "keep_recent_turns": 3,        # 压缩时保留最近几轮用户对话
    "auto_commit": True,           # 每轮改完文件自动 git 提交
    "command_timeout": 60,         # run_command 超时秒数
    "max_steps": 25,               # 单轮最多调用模型几次，防死循环
    "retries": 2,                  # 限流/临时故障时，同一模型重试次数
    "retry_wait": 3,               # 首次重试等待秒数（之后翻倍）
    "fallback_models": [],         # 当前模型不可用时，依次尝试的备用模型名
}

FREE_ROUTER = "openrouter/free"    # OpenRouter 官方"免费模型自动路由"
MODELS_URL = "https://openrouter.ai/api/v1/models"


def is_free(model_id):
    """本项目只允许免费模型：ID 以 :free 结尾，或是 openrouter/free 路由。"""
    return model_id == FREE_ROUTER or model_id.endswith(":free")


def config_path():
    """全局配置文件位置：~/.ai-coder/config.toml（Windows 即 C:\\Users\\你\\.ai-coder\\config.toml）。
    可用环境变量 AI_CODER_CONFIG 指定其他位置。"""
    return os.environ.get("AI_CODER_CONFIG") or os.path.join(
        os.path.expanduser("~"), ".ai-coder", "config.toml")


CONFIG_TEMPLATE = """# AI 编程助手 · 全局配置（对所有项目生效）
# 注意：这个文件含有你的 key，不要提交到 git，也不要分享给别人。

# OpenRouter 的 API Key（环境变量 OPENROUTER_API_KEY 优先于这里）
api_key = ""

# 默认模型（填 models.json 里的名字）
# default = "auto"

# 覆盖 models.json 里的 settings
[settings]
# auto_commit = false
# command_timeout = 120

# 额外的免费模型（只允许 :free 结尾）
# [models.mymodel]
# model = "某厂商/某模型:free"
# max_tokens = 4000
"""


def load_user_config(path=None):
    path = path or config_path()
    if not os.path.exists(path):
        return {}
    try:
        import tomllib                      # Python 3.11+ 自带
    except ModuleNotFoundError:
        try:
            import tomli as tomllib         # 低版本: pip install tomli
        except ModuleNotFoundError:
            raise SystemExit("读取 config.toml 需要 Python 3.11+，或先执行: pip install tomli")
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        raise SystemExit(f"config.toml 格式有误（{path}）: {e}")


def init_user_config(path=None):
    """生成配置模板，已存在则不覆盖。返回 (路径, 是否新建)。"""
    path = path or config_path()
    if os.path.exists(path):
        return path, False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(CONFIG_TEMPLATE)
    return path, True


def load_config(path, user_path=None):
    """读取 models.json，再用全局 config.toml 覆盖（若存在）。"""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    user = load_user_config(user_path)
    settings = dict(DEFAULT_SETTINGS)
    settings.update(cfg.get("settings", {}))
    settings.update(user.get("settings", {}))
    cfg["settings"] = settings
    cfg["models"].update(user.get("models", {}))
    if user.get("default"):
        cfg["default"] = user["default"]
    cfg["api_key"] = user.get("api_key", "")
    paid = [n for n, m in cfg["models"].items() if not is_free(m["model"])]
    if paid:
        raise SystemExit(f"这些模型不是免费模型（需以 :free 结尾）: {', '.join(paid)}")
    if cfg["default"] not in cfg["models"]:
        raise SystemExit(f"default 模型 {cfg['default']} 不在 models 中")
    return cfg


def resolve_api_key(cfg):
    """key 的查找顺序：环境变量 > config.toml。"""
    return os.environ.get(cfg["api_key_env"]) or (cfg.get("api_key") or "").strip()


def make_client(cfg):
    key = resolve_api_key(cfg)
    if not key:
        raise SystemExit(
            "没有找到 OpenRouter API Key。任选一种方式设置：\n"
            f"  1) 运行 python agent.py --init-config ，然后编辑 {config_path()} 填入 api_key\n"
            f"  2) Windows cmd:        set {cfg['api_key_env']}=你的key\n"
            f"     Windows PowerShell: $env:{cfg['api_key_env']}=\"你的key\"\n"
            f"     Mac/Linux:          export {cfg['api_key_env']}=你的key")
    from openai import OpenAI  # 延迟导入：测试时不需要安装
    return OpenAI(base_url=cfg["base_url"], api_key=key)


def fetch_free_models(opener=urllib.request.urlopen):
    """联网查询 OpenRouter 当前可用的免费且支持工具调用的模型（接口公开，无需 key）。"""
    req = urllib.request.Request(MODELS_URL, headers={"User-Agent": "ai-coder"})
    with opener(req, timeout=15) as r:
        data = json.load(r)["data"]
    out = []
    for m in data:
        price = m.get("pricing") or {}
        if (is_free(m["id"]) and str(price.get("prompt")) == "0"
                and str(price.get("completion")) == "0"
                and "tools" in (m.get("supported_parameters") or [])):
            out.append({"id": m["id"], "context": m.get("context_length") or 0})
    return sorted(out, key=lambda x: -x["context"])


def classify_error(e):
    """决定出错后怎么办: retry=等一下重试 / switch=换模型 / fatal=直接报错。"""
    code = getattr(e, "status_code", None)
    msg = str(e).lower()
    if code in (401, 402, 403):
        return "fatal"          # key 无效或无权限，换模型也没用
    if code in (408, 429, 500, 502, 503, 504) or "rate limit" in msg:
        return "retry"          # 免费模型最常见：限流
    if code in (400, 404) or "tool" in msg:
        return "switch"         # 模型下线/不支持工具调用
    if code is None and (isinstance(e, (OSError, TimeoutError)) or "connection" in msg or "timeout" in msg):
        return "retry"
    return "fatal"


def complete(client, mcfg, system, messages, tools=None, stream=True, on_text=None):
    """调用一次模型。返回 {"content": str, "tool_calls": [{"id","name","arguments"}]}"""
    kwargs = dict(
        model=mcfg["model"],
        max_tokens=mcfg.get("max_tokens", 4000),
        messages=[{"role": "system", "content": system}] + messages,
    )
    if tools:
        kwargs["tools"] = tools

    if not stream:
        m = client.chat.completions.create(**kwargs).choices[0].message
        calls = [{"id": t.id, "name": t.function.name,
                  "arguments": t.function.arguments or "{}"}
                 for t in (m.tool_calls or [])]
        return {"content": m.content or "", "tool_calls": calls}

    text, calls = [], {}
    for chunk in client.chat.completions.create(stream=True, **kwargs):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            text.append(delta.content)
            if on_text:
                on_text(delta.content)
        # 工具调用是"碎片"流式到达的，要按 index 拼起来
        for tc in getattr(delta, "tool_calls", None) or []:
            c = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                c["id"] = tc.id
            if tc.function:
                if tc.function.name and not c["name"]:
                    c["name"] = tc.function.name
                if tc.function.arguments:
                    c["arguments"] += tc.function.arguments
    ordered = [calls[i] for i in sorted(calls)]
    for i, c in enumerate(ordered):
        if not c["id"]:
            c["id"] = f"call_{i}_{uuid.uuid4().hex[:8]}"
        if not c["arguments"]:
            c["arguments"] = "{}"
    return {"content": "".join(text), "tool_calls": ordered}
