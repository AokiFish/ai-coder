#!/usr/bin/env python3
"""agent.py —— AI 编程助手主程序（v1.0）

用法：
    pip install openai
    export OPENROUTER_API_KEY=你的key
    cd 你的项目目录
    python /path/to/ai-coder/agent.py          # 或 --dir 项目目录 --model deepseek
"""
import argparse
import json
import logging
import os
import platform
import sys
import time

import git_util
import llm
import memory
from tools import Tools

HERE = os.path.dirname(os.path.abspath(__file__))

BASE_PROMPT = """你是一个严谨的编程助手，通过工具在用户的项目目录中工作。
工作原则：
1. 先了解再动手：用 list_files / grep / read_file 看清现状，不要凭空猜测文件内容。
2. 小改动用 edit_file，新建文件或大幅重写才用 write_file。
3. 写完代码后，用 run_command 验证；报错就读错误、自己修，直到通过。
   不要直接运行会弹窗口或无限循环的程序（游戏、GUI、服务器），它会卡住；
   这类程序用 python -m py_compile 检查语法，或写不依赖窗口的小测试来验证逻辑。
   缺少第三方库时，告诉用户需要 pip install 什么，不要自己反复尝试。
4. 回答简洁，说明你做了什么、结果如何。无法完成时如实说明。"""

TRUNCATED_HINT = ("你的上一条回复因长度限制在工具调用中途被截断，该调用没有执行。"
                  "请把内容拆小：先用 write_file 写出不超过 120 行的骨架，再用 edit_file 分段补充。")
TRUNCATED_ARGS = "错误: 你的输出因长度限制被截断，工具参数不完整，未执行。" + \
                 "请把内容拆小：先用 write_file 写不超过 120 行的骨架，再用 edit_file 分段补充。"
BAD_ARGS = "错误: 工具参数不是合法 JSON，未执行。请重新调用，并正确转义换行和引号。"

PLAN_SUFFIX = """【计划模式】你现在只能读取和搜索，不能修改任何东西。
请调研后输出一份清晰的分步计划（要改哪些文件、怎么改、如何验证），等待用户批准。"""

HELP = """命令:
  /model [名称]   查看/切换模型（可填 models.json 里的名字，或任意 :free 模型 ID）
  /toolmode [native|text]  查看/设置当前模型的工具调用方式（text=文本协议，适合不支持原生工具调用的模型）
  /models         联网列出当前可用的免费模型（支持工具调用）
  /plan 任务      先出计划，你批准后再执行
  /compact        立即压缩历史对话
  /undo           撤销 AI 最近一次自动提交
  /clear          清空对话历史
  /help           显示帮助
  exit            退出"""


def color(text, code):
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def preview(name, result, max_lines=10):
    """工具结果在终端里的预览。命令失败时显示输出末尾（报错信息通常在最后），
    否则只显示第一行，避免刷屏。"""
    lines = result.strip().split("\n")
    if name == "run_command" and not result.startswith("exit_code=0") and len(lines) > 1:
        body = lines[1:]
        shown = [lines[0]] + (["..."] if len(body) > max_lines else []) + body[-max_lines:]
        return [l[:200] for l in shown]
    return [lines[0][:150]]


def ask_yes_no(prompt):
    try:
        return input(color(prompt, "33") + " [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def make_logger(root):
    log = logging.getLogger("agent." + root)
    log.propagate = False
    if not log.handlers:
        h = logging.FileHandler(os.path.join(root, "agent.log"), encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


class Agent:
    def __init__(self, workdir, cfg, client, confirm=None, model=None):
        self.root = os.path.realpath(workdir)
        self.cfg, self.client = cfg, client
        self.settings = cfg["settings"]
        self.current = model or cfg["default"]
        self.mode = "normal"
        self.confirm = confirm or ask_yes_no
        self.log = make_logger(self.root)
        self.tools = Tools(self.root, self.confirm, self.settings, self.log)
        self.history = memory.History(os.path.join(self.root, "history.jsonl"))
        self.messages = self.history.load()
        self.agent_commits = []
        # 工具模式: native=原生 function calling / text=文本协议
        self.tool_modes = {n: m.get("tool_mode", "native") for n, m in cfg["models"].items()}
        self.sleep = time.sleep                  # 可替换，方便测试
        self.fetch_models = llm.fetch_free_models

    # ---------- 调用模型：限流重试 + 自动换备用模型 ----------
    def call_model(self, readonly, on_text):
        models = self.cfg["models"]
        order = [self.current] + [m for m in self.settings["fallback_models"]
                                  if m != self.current and m in models]
        last = None
        for name in order:
            for attempt in range(self.settings["retries"] + 1):
                try:
                    res = llm.complete(self.client, models[name], self.system_prompt(),
                                       self.messages, self.tools.schemas(readonly), on_text=on_text,
                                       tool_mode=self.tool_modes.get(name, "native"))
                    if name != self.current:
                        print(color(f"（本次由备用模型 {name} 完成）", "2"))
                    return res
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    last = e
                    kind = llm.classify_error(e)
                    self.log.warning("model %s failed (%s): %s", name, kind, e)
                    if kind == "fatal":
                        raise
                    # 模型不支持原生工具调用 → 同一个模型改用文本协议重试，而不是直接放弃
                    if llm.tools_unsupported(e) and self.tool_modes.get(name, "native") == "native":
                        self.tool_modes[name] = "text"
                        print(color(f"\n（{name} 不支持原生工具调用，自动改用文本协议重试）", "33"))
                        continue
                    if kind == "retry" and attempt < self.settings["retries"]:
                        wait = self.settings["retry_wait"] * 2 ** attempt
                        print(color(f"\n（{name} 繁忙或限流，{wait} 秒后重试…）", "33"))
                        self.sleep(wait)
                        continue
                    hint = "（该模型已不再免费）" if "unavailable for free" in str(e) else ""
                    print(color(f"\n（{name} 暂不可用{hint}: {str(e)[:200]}\n  → 尝试下一个模型）", "33"))
                    break
        raise last

    # ---------- 提示词（每轮重新读取 AGENTS.md，改了立即生效） ----------
    def system_prompt(self):
        parts = [BASE_PROMPT, f"工作目录: {self.root}\n操作系统: {platform.system()}"]
        path = os.path.join(self.root, "AGENTS.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                parts.append("【项目规范 AGENTS.md】\n" + f.read()[:8000])
        if self.mode == "plan":
            parts.append(PLAN_SUFFIX)
        return "\n\n".join(parts)

    # ---------- 记忆 ----------
    def add(self, msg):
        self.messages.append(msg)
        self.history.append(msg)

    def repair(self):
        fixed = memory.sanitize(self.messages)
        if fixed != self.messages:
            self.messages = fixed
            self.history.rewrite(fixed)

    def maybe_compact(self, force=False):
        if not force and memory.estimate(self.messages) <= self.settings["context_limit_chars"]:
            return False
        try:
            new = memory.compact(self.client, self.cfg["models"][self.current],
                                 self.messages, self.settings["keep_recent_turns"])
        except Exception as e:
            self.log.warning("compact failed: %s", e)
            print(color(f"（历史压缩失败，已跳过: {e}）", "2"))
            return False
        if new is None:
            return False
        self.messages = new
        self.history.rewrite(new)
        self.log.info("history compacted -> %d messages", len(new))
        print(color("（早期对话已压缩为摘要）", "2"))
        return True

    # ---------- 核心：Agent 循环 ----------
    def run_turn(self, user_text):
        use_git = (self.settings["auto_commit"] and self.mode == "normal"
                   and git_util.is_repo(self.root))
        was_dirty = use_git and git_util.dirty(self.root)
        self.add({"role": "user", "content": user_text})
        readonly = self.mode == "plan"

        for _ in range(self.settings["max_steps"]):
            self.maybe_compact()
            started = []

            def on_text(t):
                if not started:
                    print(color("AI> ", "1;36"), end="")
                    started.append(1)
                print(t, end="", flush=True)

            res = self.call_model(readonly, on_text)
            if started:
                print()
            calls = res["tool_calls"]
            cut_off = res.get("finish_reason") == "length"
            # 参数不是合法 JSON（常见原因：输出被 max_tokens 截断）。
            # 历史里必须存合法 JSON，否则之后每次请求都会被服务端拒绝（400）
            parsed = []
            for c in calls:
                try:
                    parsed.append(json.loads(c["arguments"] or "{}"))
                except json.JSONDecodeError:
                    parsed.append(None)
                    c["raw"], c["arguments"] = c["arguments"], "{}"
            record = {"role": "assistant",
                      "content": res["content"] or (None if calls else "")}
            if calls:
                record["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls]
            self.add(record)
            if res.get("truncated"):       # 文本协议：工具块写到一半被截断
                print(color("（输出被长度限制截断，已提示模型拆小后重试）", "33"))
                self.add({"role": "user", "content": TRUNCATED_HINT})
                continue
            if not calls:
                if cut_off:
                    print(color("（回答因长度限制被截断，可输入“继续”）", "33"))
                break
            for c, args in zip(calls, parsed):
                shown = (json.dumps(args, ensure_ascii=False) if args is not None
                         else c["raw"])[:120]
                print(color(f"  🔧 {c['name']} {shown}", "2"))
                if args is None:
                    result = TRUNCATED_ARGS if cut_off else BAD_ARGS
                else:
                    result = self.tools.run(c["name"], args, readonly)
                self.log.info("tool %s -> %s", c["name"], result[:200].replace("\n", " "))
                for line in preview(c["name"], result):
                    print(color(f"     ↳ {line}", "2"))
                self.add({"role": "tool", "tool_call_id": c["id"], "content": result})
        else:
            print(color(f"（已达到单轮最大步数 {self.settings['max_steps']}，已停止）", "33"))

        if use_git:
            self.git_after(user_text, was_dirty)

    def git_after(self, user_text, was_dirty):
        if was_dirty:
            print(color("（你有未提交的改动，本轮跳过自动提交，避免混入你的修改）", "2"))
            return
        if not git_util.dirty(self.root):
            return
        ok, info = git_util.commit_all(self.root, "agent: " + user_text.strip().split("\n")[0][:50])
        if ok:
            self.agent_commits.append(info)
            print(color(f"  ✅ 已自动提交 {info[:7]}（/undo 可撤销）", "2"))
        else:
            print(color(f"  （自动提交失败: {info}）", "33"))

    def safe_turn(self, text):
        try:
            self.run_turn(text)
        except KeyboardInterrupt:
            print(color("\n已中断本轮", "33"))
            self.repair()
        except Exception as e:
            self.log.exception("turn failed")
            print(color(f"请求失败: {e}", "31"))
            self.repair()

    # ---------- 计划模式 ----------
    def plan(self, task):
        self.mode = "plan"
        try:
            self.safe_turn(task)
        finally:
            self.mode = "normal"
        if self.messages and self.messages[-1]["role"] == "assistant" \
                and self.confirm("\n按这个计划执行吗?"):
            self.safe_turn("好的，请按上面的计划开始执行。")

    # ---------- 命令 ----------
    def command(self, line):
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd == "/help":
            print(HELP)
        elif cmd == "/model":
            if arg in self.cfg["models"]:
                self.current = arg
                print("已切换到", arg)
            elif arg and llm.is_free(arg):          # 直接填任意免费模型 ID
                self.cfg["models"][arg] = {"model": arg, "max_tokens": 8000}
                self.tool_modes.setdefault(arg, "native")
                self.current = arg
                print("已切换到", arg)
            elif arg:
                print("本项目只支持免费模型（ID 以 :free 结尾，或 openrouter/free）")
            else:
                print("当前:", self.current, "| 可选:", ", ".join(self.cfg["models"]))
        elif cmd == "/toolmode":
            if arg in ("native", "text"):
                self.tool_modes[self.current] = arg
                print(f"{self.current} 的工具调用方式 → {arg}")
            else:
                print(f"{self.current} 当前: {self.tool_modes.get(self.current, 'native')}（可选 native / text）")
        elif cmd == "/models":
            try:
                found = self.fetch_models()
            except Exception as e:
                print(f"查询失败: {e}")
                return
            for m in found[:20]:
                print(f"  {m['id']}  ({m['context'] // 1000}K 上下文)")
            print(f"共 {len(found)} 个免费且支持工具调用的模型，用 /model 模型ID 切换")
        elif cmd == "/clear":
            self.messages = []
            self.history.rewrite([])
            print("历史已清空")
        elif cmd == "/compact":
            if not self.maybe_compact(force=True):
                print("没有需要压缩的内容")
        elif cmd == "/plan":
            if arg:
                self.plan(arg)
            else:
                print("用法: /plan 你想做的事")
        elif cmd == "/undo":
            self.undo()
        else:
            print("未知命令，输入 /help 查看")

    def undo(self):
        if not self.agent_commits:
            print("本次运行中还没有 AI 的自动提交可撤销")
            return
        if git_util.head(self.root) != self.agent_commits[-1]:
            print("当前最新提交不是 AI 做的（你可能手动提交过），为安全起见不自动撤销")
            return
        if not self.confirm(f"将回退提交 {self.agent_commits[-1][:7]}，该次改动会丢失。继续?"):
            return
        ok, info = git_util.undo_last(self.root)
        if ok:
            self.agent_commits.pop()
            print("已撤销")
        else:
            print("撤销失败:", info)

    def repl(self):
        print(f"AI 编程助手 v1.1（仅免费模型）| 模型: {self.current} | 目录: {self.root}")
        print(f"已载入 {len(self.messages)} 条历史 | 输入 /help 查看命令")
        while True:
            try:
                line = input(color("\n你> ", "1;32")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("exit", "quit"):
                break
            if line.startswith("/"):
                self.command(line)
            else:
                self.safe_turn(line)


def main():
    ap = argparse.ArgumentParser(description="AI 编程助手")
    ap.add_argument("--dir", default=".", help="工作目录（默认当前目录）")
    ap.add_argument("--model", help="models.json 中的模型名")
    ap.add_argument("--init-config", action="store_true", help="生成全局配置模板 config.toml 后退出")
    args = ap.parse_args()
    if args.init_config:
        path, created = llm.init_user_config()
        print(("已生成: " if created else "已存在，未覆盖: ") + path)
        return
    if os.name == "nt":
        os.system("")  # 让 Windows 终端支持彩色
    root = os.path.realpath(args.dir)
    if not os.path.isdir(root):
        raise SystemExit(f"目录不存在: {root}")
    cfg = llm.load_config(os.path.join(HERE, "models.json"))
    if args.model and args.model not in cfg["models"]:
        raise SystemExit("可选模型: " + ", ".join(cfg["models"]))
    Agent(root, cfg, llm.make_client(cfg), model=args.model).repl()


if __name__ == "__main__":
    main()
