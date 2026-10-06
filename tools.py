"""tools.py —— 工具说明书 + 工具实现 + 安全沙箱。

安全分三层：
  1. 路径沙箱：所有文件操作限制在工作目录内（含符号链接逃逸）
  2. 命令黑名单：拦截明显危险的命令（尽力而为，不是万能的）
  3. 人工确认：run_command 每次都要你按 y（最后一道、也是最可靠的防线）
"""
import json
import os
import re
import subprocess

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".mypy_cache"}
MAX_OUT = 8000
READ_ONLY = {"read_file", "list_files", "grep"}

DANGEROUS = [
    (r"\brm\s+(-[a-zA-Z]+\s+)*(/\*?|~/?|\*|\.\.)(\s|$)", "删除根目录/家目录/上级目录"),
    (r"\bsudo\b", "提权命令"),
    (r"\bmkfs", "格式化磁盘"),
    (r"\bdd\s+if=", "dd 直接写磁盘"),
    (r":\(\)\s*\{", "fork 炸弹"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "关机/重启"),
    (r"(curl|wget)[^|;]*\|\s*(ba|z)?sh", "下载后直接执行脚本"),
    (r"\bgit\s+push\b", "推送代码（请手动操作）"),
    (r"\bformat\s+[a-zA-Z]:", "格式化磁盘"),
    (r"\b(rd|rmdir)\s+(/[sSqQ]\s+)*[a-zA-Z]:\\?(\s|$)", "递归删除整个磁盘"),
    (r">\s*/dev/sd", "直接写磁盘设备"),
]


def truncate(s, n=MAX_OUT):
    return s if len(s) <= n else s[:n] + f"\n...[已截断，共 {len(s)} 字符]"


def check_command(cmd):
    """返回危险原因；安全则返回 None。"""
    for pattern, reason in DANGEROUS:
        if re.search(pattern, cmd):
            return reason
    return None


def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


_S = {"type": "string"}
SCHEMAS = [
    _fn("read_file", "读取文件内容", {"path": _S}, ["path"]),
    _fn("write_file", "创建或整体覆盖文件。小改动请用 edit_file", {"path": _S, "content": _S}, ["path", "content"]),
    _fn("edit_file", "把文件中唯一出现的 old_str 替换为 new_str（局部修改）",
        {"path": _S, "old_str": _S, "new_str": _S}, ["path", "old_str", "new_str"]),
    _fn("list_files", "列出目录结构（最多3层），path 默认为当前目录", {"path": _S}, []),
    _fn("grep", "在文件中用正则搜索，返回 文件:行号: 内容", {"pattern": _S, "path": _S}, ["pattern"]),
    _fn("run_command", "在工作目录执行 shell 命令（需用户确认），用于运行代码、测试", {"command": _S}, ["command"]),
]


class Tools:
    def __init__(self, root, confirm, settings, log=None):
        self.root = os.path.realpath(root)
        self.confirm = confirm
        self.settings = settings
        self.log = log

    # ---- 沙箱 ----
    def safe_path(self, p):
        full = os.path.realpath(os.path.join(self.root, p or "."))
        try:
            inside = os.path.commonpath([full, self.root]) == self.root
        except ValueError:  # Windows 不同盘符
            inside = False
        if not inside:
            raise PermissionError(f"路径超出工作目录: {p}")
        return full

    def _check_writable(self, full):
        rel = os.path.relpath(full, self.root).split(os.sep)
        if ".git" in rel:
            raise PermissionError("不允许修改 .git 目录")

    # ---- 调度 ----
    def schemas(self, readonly=False):
        return [s for s in SCHEMAS if not readonly or s["function"]["name"] in READ_ONLY]

    def run(self, name, args, readonly=False):
        if not isinstance(args, dict):
            return "错误: 参数必须是 JSON 对象"
        if readonly and name not in READ_ONLY:
            return f"错误: 计划模式下不允许使用 {name}（只能读取和搜索）"
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return f"错误: 未知工具 {name}"
        try:
            return fn(**args)
        except TypeError as e:
            return f"错误: 参数不正确 ({e})"
        except Exception as e:
            if self.log:
                self.log.warning("tool %s failed: %s", name, e)
            return f"错误: {e}"

    # ---- 工具实现 ----
    def t_read_file(self, path):
        full = self.safe_path(path)
        with open(full, encoding="utf-8", errors="replace") as f:
            return truncate(f.read(), 100_000)

    def t_write_file(self, path, content):
        full = self.safe_path(path)
        self._check_writable(full)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {path}（{len(content)} 字符）"

    def t_edit_file(self, path, old_str, new_str):
        full = self.safe_path(path)
        self._check_writable(full)
        with open(full, encoding="utf-8") as f:
            text = f.read()
        n = text.count(old_str)
        if n == 0:
            return "错误: 没找到 old_str，请先 read_file 确认原文（注意空格和缩进）"
        if n > 1:
            return f"错误: old_str 出现了 {n} 次，请包含更多上下文使其唯一"
        with open(full, "w", encoding="utf-8") as f:
            f.write(text.replace(old_str, new_str, 1))
        return f"已修改 {path}"

    def t_list_files(self, path="."):
        base = self.safe_path(path)
        if not os.path.isdir(base):
            return "错误: 不是目录"
        out = []
        for dp, dns, fns in os.walk(base):
            dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
            rel = os.path.relpath(dp, base)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth >= 3:
                dns[:] = []
            prefix = "" if rel == "." else rel.replace(os.sep, "/") + "/"
            out += [prefix + d + "/" for d in dns]
            out += [prefix + f for f in sorted(fns)]
            if len(out) > 300:
                out.append("...（条目过多，已截断）")
                break
        return "\n".join(out) or "（空目录）"

    def t_grep(self, pattern, path="."):
        base = self.safe_path(path)
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"错误: 正则不合法 ({e})"
        files = [base] if os.path.isfile(base) else []
        if os.path.isdir(base):
            for dp, dns, fns in os.walk(base):
                dns[:] = [d for d in dns if d not in SKIP_DIRS]
                files += [os.path.join(dp, f) for f in sorted(fns)]
        hits = []
        for fp in files:
            try:
                if os.path.getsize(fp) > 1_000_000:
                    continue
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if "\0" in line:
                            break  # 二进制文件
                        if rx.search(line):
                            hits.append(f"{os.path.relpath(fp, self.root).replace(os.sep, '/')}:{i}: {line.strip()[:200]}")
                            if len(hits) >= 100:
                                return "\n".join(hits) + "\n...（结果过多，仅显示前100条）"
            except OSError:
                continue
        return "\n".join(hits) or "没有匹配"

    def t_run_command(self, command):
        reason = check_command(command)
        if reason:
            if self.log:
                self.log.warning("blocked command: %s (%s)", command, reason)
            return f"已拒绝: 命令被安全规则拦截（{reason}）"
        if not self.confirm(f"\n⚠️  AI 想执行命令:\n    {command}\n是否允许?"):
            if self.log:
                self.log.info("user denied command: %s", command)
            return "用户拒绝执行该命令"
        if self.log:
            self.log.info("run command: %s", command)
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True,
                               text=True, errors="replace", stdin=subprocess.DEVNULL,
                               timeout=self.settings["command_timeout"])
        except subprocess.TimeoutExpired:
            return f"错误: 命令超时（>{self.settings['command_timeout']}秒）已终止"
        return truncate(f"exit_code={r.returncode}\n{r.stdout}{r.stderr}")
