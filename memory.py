"""memory.py —— 对话记忆：history.jsonl 存取、修复、压缩。"""
import copy
import json
import os

import llm

SUMMARY_MARK = "[对话摘要——以下是此前对话的压缩记录]"


class History:
    def __init__(self, path):
        self.path = path

    def load(self):
        msgs = []
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            msgs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # 跳过损坏行
        fixed = sanitize(msgs)
        if fixed != msgs:
            self.rewrite(fixed)
        return fixed

    def append(self, msg):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def rewrite(self, msgs):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for m in msgs:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)  # 原子替换，不会写一半


def sanitize(msgs):
    """修复历史，避免 API 报错：
    1) 工具参数不是合法 JSON（比如输出被截断）→ 改成 {}，否则之后每次请求都会 400
    2) 末尾"调用了工具但结果不全"的记录 → 删掉"""
    msgs = [copy.deepcopy(m) for m in msgs if isinstance(m, dict) and "role" in m]
    for m in msgs:
        for t in m.get("tool_calls") or []:
            try:
                json.loads(t["function"]["arguments"] or "{}")
            except (json.JSONDecodeError, TypeError, KeyError):
                t["function"]["arguments"] = "{}"
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            need = {t["id"] for t in m["tool_calls"]}
            got = {x.get("tool_call_id") for x in msgs[i + 1:] if x["role"] == "tool"}
            if not need <= got:
                msgs = msgs[:i]
            break
    while msgs and msgs[0]["role"] != "user":  # 开头必须是用户消息
        msgs.pop(0)
    return msgs


def estimate(msgs):
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs)


def _transcript(msgs, limit=40000):
    lines = []
    for m in msgs:
        c = m.get("content") or ""
        if m["role"] == "tool":
            lines.append(f"[工具结果] {c[:300]}")
        elif m["role"] == "assistant":
            if c:
                lines.append(f"AI: {c[:1500]}")
            for t in m.get("tool_calls") or []:
                lines.append(f"[调用 {t['function']['name']}] {t['function']['arguments'][:200]}")
        else:
            lines.append(f"用户: {c[:1500]}")
    return "\n".join(lines)[-limit:]


def compact(client, mcfg, msgs, keep_recent):
    """把较早的对话压成一条摘要，保留最近 keep_recent 轮。无需压缩返回 None。"""
    keep_recent = max(1, keep_recent)
    users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    if len(users) <= keep_recent:
        return None
    cut = users[-keep_recent]  # 从"用户消息"处切，保证工具调用不被拆开
    old, recent = msgs[:cut], msgs[cut:]
    if not old:
        return None
    system = "你是对话压缩器。把下面的编程助手对话压缩成简洁摘要。"
    ask = ("请用中文总结，保留：用户的目标、已创建/修改的文件及要点、重要决定、"
           "尚未解决的问题。不要寒暄。\n\n" + _transcript(old))
    res = llm.complete(client, mcfg, system, [{"role": "user", "content": ask}], stream=False)
    summary = res["content"].strip()
    if not summary:
        return None
    return [{"role": "user", "content": SUMMARY_MARK + "\n" + summary}] + recent
