"""textproto.py —— 文本工具协议（给不支持原生 function calling 的模型用）。

原生模式：工具通过 API 的 tools 参数传递，模型返回结构化的 tool_calls。
文本模式：我们把工具说明写进提示词，模型用下面的标签格式"写出"调用，再由我们解析：

    <tool name="write_file">
    <path>hello.py</path>
    <content>
    print("hi")
    </content>
    </tool>

参数值直接写原文，不用 JSON 转义——写代码时比 JSON 稳得多，小模型也不容易出错。
历史记录始终以原生格式存储，发送给模型前才转换，所以两种模式可随时切换。
"""
import json
import re

TAG = "<tool name="
BLOCK = re.compile(r'<tool\s+name="(\w+)"\s*>(.*?)</tool>', re.S)
PARAM = re.compile(r"<(\w+)>(.*?)</\1>", re.S)


def render_call(name, args):
    lines = [f'<tool name="{name}">']
    for k, v in args.items():
        v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        lines.append(f"<{k}>\n{v}\n</{k}>" if "\n" in v else f"<{k}>{v}</{k}>")
    lines.append("</tool>")
    return "\n".join(lines)


def protocol_prompt(schemas):
    tools = []
    for s in schemas:
        f = s["function"]
        req = set(f["parameters"].get("required", []))
        params = ", ".join(p if p in req else p + "?" for p in f["parameters"]["properties"])
        tools.append(f"- {f['name']}({params}): {f['description']}")
    return (
        "【工具调用格式】你不能使用原生函数调用，请在回复中用下面的文本格式调用工具：\n"
        '<tool name="工具名">\n<参数名>参数值</参数名>\n</tool>\n'
        "规则：参数值直接写原文，无需转义；参数名后带 ? 表示可选；一次回复可包含多个工具调用；"
        "调用工具后立即停止输出，等待 <tool_result> 返回再继续。\n"
        "示例：\n"
        '<tool name="write_file">\n<path>hello.py</path>\n<content>\nprint("hi")\n</content>\n</tool>\n'
        "可用工具：\n" + "\n".join(tools))


def parse(text):
    """返回 (去掉工具块的正文, [(工具名, 参数dict)], 是否有未闭合的工具块)。"""
    calls = []
    for m in BLOCK.finditer(text):
        args = {}
        for pm in PARAM.finditer(m.group(2)):
            v = pm.group(2)
            if v.startswith("\r\n"):
                v = v[2:]
            elif v.startswith("\n"):
                v = v[1:]
            if v.endswith("\n"):
                v = v[:-1]
            args[pm.group(1)] = v
        calls.append((m.group(1), args))
    clean = BLOCK.sub("", text)
    cut = clean.find("<tool ")
    truncated = cut >= 0            # 开了标签却没写完 → 输出被截断
    if truncated:
        clean = clean[:cut]
    return clean.strip(), calls, truncated


def _safe_args(raw):
    try:
        d = json.loads(raw or "{}")
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def to_text_messages(messages):
    """把原生格式的历史转换成纯文本对话：tool_calls → 标签，tool 结果 → 用户消息。"""
    out, names = [], {}
    for m in messages:
        role = m["role"]
        if role == "assistant":
            parts = [m.get("content") or ""]
            for t in m.get("tool_calls") or []:
                names[t["id"]] = t["function"]["name"]
                parts.append(render_call(t["function"]["name"], _safe_args(t["function"]["arguments"])))
            out.append({"role": "assistant", "content": "\n".join(p for p in parts if p) or "（无内容）"})
        elif role == "tool":
            txt = f'<tool_result name="{names.get(m.get("tool_call_id"), "?")}">\n{m.get("content") or ""}\n</tool_result>'
            if out and out[-1].get("_tr"):
                out[-1]["content"] += "\n" + txt   # 连续的工具结果合并成一条
            else:
                out.append({"role": "user", "content": txt, "_tr": True})
        else:
            out.append({"role": "user", "content": m.get("content") or ""})
    for o in out:
        o.pop("_tr", None)
    return out


class TextFilter:
    """流式输出时，遇到 <tool name= 就不再显示（那是给程序看的，不是给人看的）。"""
    def __init__(self, on_text):
        self.on, self.buf, self.hidden = on_text, "", False

    def feed(self, t):
        if self.hidden or not self.on:
            return
        self.buf += t
        i = self.buf.find(TAG)
        if i >= 0:
            if i > 0:
                self.on(self.buf[:i])
            self.on("\n（正在生成工具调用…）")
            self.buf, self.hidden = "", True
            return
        keep = len(TAG) - 1                      # 末尾可能是半个标签，先扣住
        if len(self.buf) > keep:
            self.on(self.buf[:-keep])
            self.buf = self.buf[-keep:]

    def flush(self):
        if not self.hidden and self.buf and self.on:
            self.on(self.buf)
        self.buf = ""
