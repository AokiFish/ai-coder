"""git_util.py —— 每轮自动提交，改坏了可以 /undo 回退。"""
import subprocess

EXCLUDE = [":!history.jsonl", ":!agent.log", ":!history.jsonl.tmp"]


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def is_repo(root):
    try:
        r = _git(root, "rev-parse", "--is-inside-work-tree")
        return r.returncode == 0 and r.stdout.strip() == "true"
    except FileNotFoundError:
        return False


def dirty(root):
    return bool(_git(root, "status", "--porcelain", "--", ".", *EXCLUDE).stdout.strip())


def head(root):
    r = _git(root, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def commit_all(root, msg):
    """提交所有改动，返回 (是否成功, 提交哈希或错误信息)。"""
    _git(root, "add", "-A", "--", ".", *EXCLUDE)
    r = _git(root, "commit", "-m", msg)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip()
    return True, head(root)


def undo_last(root):
    r = _git(root, "reset", "--hard", "HEAD~1")
    return r.returncode == 0, (r.stderr or r.stdout).strip()
