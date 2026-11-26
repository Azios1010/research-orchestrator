"""DeepSeek diagnosis, redaction, GitHub and Telegram: every network call is injected."""

import json
import subprocess

import pytest

from labcore.diagnose import deepseek
from labcore.notify import telegram
from labcore.notify.github import GitHubNotifier, fenced
from labcore.redact import redact_for_model, redact_secrets

FAKE_KEY = {"DEEPSEEK_API_KEY": "placeholder-not-a-key"}


class Response:
    def __init__(self, content, error=None):
        self.content, self.error = content, error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class Post:
    def __init__(self, response):
        self.response, self.calls = response, []

    def __call__(self, url, **options):
        self.calls.append((url, options))
        return self.response


def never(*args, **kwargs):
    raise AssertionError("no request may be made")


def test_without_a_key_nothing_is_sent():
    reply = deepseek.diagnose("log", None, env={}, post=never)
    assert reply.startswith("diagnosis unavailable")


def test_request_and_reply():
    post = Post(Response("\n".join(f"line {i}" for i in range(20)) + "\n" + "x" * 1000))
    reply = deepseek.diagnose("Traceback ... KeyError", {"stage": "run"}, env=FAKE_KEY, post=post)
    assert reply.splitlines() == [f"line {i}" for i in range(8)]
    [(url, options)] = post.calls
    assert url == "https://api.deepseek.com/chat/completions"
    assert options["json"]["model"] == "deepseek-flash"
    assert options["json"]["thinking"] == {"type": "disabled"}
    assert options["timeout"] > 0


def test_model_and_base_url_can_be_overridden():
    post = Post(Response("ok"))
    env = {**FAKE_KEY, "DEEPSEEK_BASE_URL": "https://llm.example.invalid/v1/", "DEEPSEEK_MODEL": "other"}
    deepseek.diagnose("log", None, env=env, post=post)
    [(url, options)] = post.calls
    assert url == "https://llm.example.invalid/v1/chat/completions"
    assert options["json"]["model"] == "other" and "thinking" not in options["json"]


def test_only_the_log_tail_and_status_leave_the_machine():
    log = (
        "loading /kaggle/temp/cache/datasets/data/train-00001-of-00149.parquet\n"
        "case sub-stroke0042 failed reading D:\\data\\sub-stroke0042_ctp.nii.gz\n"
        'File "/tmp/lab_bundle_x/directions/isles24-ctp/src/data.py", line 10\n'
        "token hf_" + "Ab1" * 10 + "\n"
    )
    status = {
        "stage": "run", "ok": False, "error_type": "ValueError", "error_message": "bad sub-stroke0042",
        "traceback_tail": "x", "run_id": "r", "config": {"local_shard_dir": "/secret/data"},
    }
    post = Post(Response("ok"))
    deepseek.diagnose(log, status, env=FAKE_KEY, post=post)
    sent = json.dumps(post.calls[0][1]["json"]["messages"][1])
    for leaked in ("sub-stroke0042", ".parquet", ".nii.gz", "/kaggle/temp", "/secret/data", "hf_Ab1"):
        assert leaked not in sent, leaked
    assert "data.py" in sent and "ValueError" in sent


def test_the_prompt_forbids_moving_the_goalposts():
    system = deepseek.build_messages("log", None)[0]["content"].lower()
    for word in ("8 lines", "gate", "prereg", "evaluation"):
        assert word in system, word


@pytest.mark.parametrize(
    "post",
    [
        Post(Response("x", error=RuntimeError("401 with key placeholder-not-a-key"))),
        Post(Response("")),
        Post(Response(None)),
        never,
    ],
)
def test_any_failure_gives_the_fallback_without_details(post):
    reply = deepseek.diagnose("log", None, env=FAKE_KEY, post=post)
    assert reply.startswith("diagnosis unavailable") and "placeholder" not in reply


@pytest.mark.parametrize(
    "secret",
    [
        "hf_" + "a1B2" * 8,
        "sk-" + "a1b2" * 8,
        "ghp_" + "a1B2" * 9,
        "KGAT_" + "0a" * 12,
        "123456789:" + "AAbb11" * 6,
    ],
)
def test_secrets_are_redacted(secret):
    assert secret not in redact_secrets(f"value {secret} end")


def test_assignments_and_urls_are_redacted_but_ordinary_text_is_not():
    text = "KAGGLE_KEY=0123456789abcdef https://x.invalid/f?X-Amz-Signature=deadbeef&a=1"
    cleaned = redact_secrets(text)
    assert "0123456789abcdef" not in cleaned and "deadbeef" not in cleaned and "a=1" in cleaned
    ordinary = "token = get_token(user)\nsecret_santa = None\nmax_tokens: 600"
    assert redact_secrets(ordinary) == ordinary
    assert "main.py" in redact_for_model("File /tmp/x/src/main.py")


# ---------------------------------------------------------------- GitHub


class GH:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, command, **options):
        self.calls.append((command[1:], options))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        code, out = reply
        return subprocess.CompletedProcess(command, code, stdout=out, stderr="could not add label")


def notifier(*replies, env=None):
    gh = GH(*replies)
    return GitHubNotifier(runner=gh, env=env or {}), gh


def test_create_issue():
    github, gh = notifier((0, "https://github.com/o/r/issues/17\n"), env={"GITHUB_TOKEN": "placeholder"})
    assert github.create_issue("title", "body", ("run-failed",)) == 17
    args, options = gh.calls[0]
    assert args == ["issue", "create", "--title", "title", "--body-file", "-", "--label", "run-failed"]
    assert options["input"] == "body" and options["env"]["GH_TOKEN"] == "placeholder"


def test_a_missing_label_does_not_lose_the_issue():
    github, gh = notifier((1, ""), (0, "https://github.com/o/r/issues/3"))
    assert github.create_issue("t", "b", ("run-failed",)) == 3
    assert "--label" not in gh.calls[1][0]


def test_find_issue_matches_the_exact_title():
    found = json.dumps([{"number": 9, "title": "[tracking] a"}, {"number": 2, "title": "[tracking] ab"}])
    github, _ = notifier((0, found))
    assert github.find_issue("[tracking] a") == 9


@pytest.mark.parametrize("reply", [(1, ""), (0, "not json"), FileNotFoundError("gh")])
def test_gh_problems_are_swallowed(reply):
    github, _ = notifier(reply)
    assert github.find_issue("t") is None


def test_comment_and_long_bodies():
    github, gh = notifier((0, ""))
    assert github.comment(4, "z" * 100_000) is True
    assert len(gh.calls[0][1]["input"]) == 60000


def test_untrusted_text_cannot_leave_its_code_fence():
    hostile = "~~~\n@everyone\n~~~~~\nmore"
    wrapped = fenced(hostile)
    fence = wrapped.splitlines()[0].removesuffix("text")
    assert len(fence) > 5 and wrapped.endswith(fence) and wrapped.count(fence) == 2


# ---------------------------------------------------------------- Telegram

TELEGRAM = {"TELEGRAM_BOT_TOKEN": "placeholder-token", "TELEGRAM_CHAT_ID": "1"}


def test_telegram_is_silent_without_configuration(capsys):
    assert telegram.send("hi", env={}, post=never) is False
    assert telegram.send("hi", env={"TELEGRAM_BOT_TOKEN": "x"}, post=never) is False
    assert capsys.readouterr().out == ""


def test_telegram_sends():
    post = Post(Response("ok"))
    assert telegram.send("hi", env=TELEGRAM, post=post) is True
    assert post.calls[0][1]["json"] == {"chat_id": "1", "text": "hi"}


def test_telegram_failure_never_prints_the_token(capsys):
    post = Post(Response("x", error=RuntimeError("url had placeholder-token")))
    assert telegram.send("hi", env=TELEGRAM, post=post) is False
    assert "placeholder-token" not in capsys.readouterr().out
