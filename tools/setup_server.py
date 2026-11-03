#!/usr/bin/env python3
"""Install the accepted CUDA server profile; rollback failed cutovers. Standard library only."""
from __future__ import annotations
import argparse
import fcntl
import grp
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL_SHA = "d6aef57fa948e9bba3ca4959b3c237ed898c605471f48c73a32cedbd24aabe70"
SOURCES = {
    "src/kernels/cuda/qsa_prompt_attn_paired.cu": "2e5b8250f5d7953415fb5b8313c5d49242865ced4f42bd51c2d3b90fcb13fc96",
    "src/prefill/prefill_paired.cpp": "532b76ad7dbe30e224991b98b44a1557cf68aa6c902f3fe4a3288f3e743bcc12",
}
TUNE = {"--expert-cache": "auto", "--prefill": "auto", "--spec": "4",
        "--spec-min-p": "0.5", "--max-context": "262144", "--kv": "int8",
        "--vram-reserve-mib": "896"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path, data, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def normalized_config(cfg, root):
    """Build an exact argument list, never delete flags by guessing their arity."""
    args = cfg.get("args", [])
    def value(flag):
        matches = [i for i, x in enumerate(args) if x == flag]
        if len(matches) != 1 or matches[0] + 1 >= len(args) or args[matches[0] + 1].startswith("--"):
            raise ValueError(f"generated configuration needs exactly one {flag} value")
        return args[matches[0] + 1]
    native = Path(value("--native")).resolve()
    if not native.is_file() or sha(native) != MODEL_SHA:
        raise ValueError("canonical Ornith IQ4_XS model digest mismatch")
    result = []
    for flag in ("--pack", "--native", "--expert-profile", "--mtp-draft-vocab"):
        path = native if flag == "--native" else Path(value(flag)).resolve()
        if not path.exists():
            raise ValueError(f"missing generated artifact for {flag}: {path}")
        result.extend((flag, str(path)))
    result.extend(("--mtp", str(native)))
    for flag, value_ in TUNE.items():
        result.extend((flag, value_))
    tokenizer = cfg.get("tokenizer")
    if not Path(value("--pack")).is_dir() or not tokenizer or not Path(tokenizer).is_dir():
        raise ValueError("generated pack and tokenizer must be existing directories")
    cfg = dict(cfg, args=result, cwd=str(root))
    cfg["sampling"] = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    cfg["anthropic_thinking"] = "model"
    for key in ("reasoning_effort", "think", "enable_thinking", "thinking", "vision"):
        cfg.pop(key, None)
    return cfg


def owns_port(pid, port):
    """Require the service itself to own the listener; an unrelated /health is insufficient."""
    inodes = set()
    try:
        for table in ("tcp", "tcp6"):
            for line in Path("/proc/net/" + table).read_text().splitlines()[1:]:
                fields = line.split()
                if fields[3] == "0A" and int(fields[1].split(":")[-1], 16) == port:
                    inodes.add(fields[9])
        return any(os.readlink(fd) in {"socket:[" + inode + "]" for inode in inodes}
                   for fd in Path(f"/proc/{pid}/fd").iterdir())
    except (OSError, ValueError, IndexError):
        return False


def unit_quote(value):
    return json.dumps(str(value).replace("%", "%%"), ensure_ascii=False)


class Installer:
    def __init__(self, root, env=None):
        self.root = Path(root).resolve()
        if any(ord(c) < 32 for c in str(self.root)):
            raise ValueError("installation path must not contain control characters")
        self.env = dict(os.environ if env is None else env)
        self.name = self.env.get("SERVER_SERVICE_NAME", "momoe-ai-server")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,56}", self.name):
            raise ValueError("SERVER_SERVICE_NAME must contain 1–56 letters, digits, '-' or '_'")
        if self.env.get("SERVER_MODEL", "ornith") != "ornith" or self.env.get("MODEL_SIZE", "IQ4_XS") != "IQ4_XS":
            raise ValueError("the dedicated profile requires SERVER_MODEL=ornith and MODEL_SIZE=IQ4_XS")
        if self.env.get("STRATA_AIXS_PAIRED_INT8", "1") != "1":
            raise ValueError("the dedicated installer requires STRATA_AIXS_PAIRED_INT8=1")
        mode = self.env.get("SERVER_USER_SERVICE", "1")
        if mode not in ("0", "1"):
            raise ValueError("SERVER_USER_SERVICE must be 0 or 1")
        self.user_mode = mode == "1"
        self.host = self.env.get("SERVER_HOST", "127.0.0.1")
        if self.host != "localhost":
            ipaddress.ip_address(self.host)
        self.key = self.env.get("SERVER_API_KEY", "")
        if any(ord(c) < 32 or ord(c) == 127 for c in self.key):
            raise ValueError("SERVER_API_KEY must be a single line")
        loopback = self.host == "localhost" or ipaddress.ip_address(self.host).is_loopback
        if not loopback and not self.key:
            raise ValueError("a non-loopback SERVER_HOST requires SERVER_API_KEY")
        self.port = int(self.env.get("SERVER_PORT", "8080"))
        self.timeout = int(self.env.get("SERVER_HEALTH_TIMEOUT", "180"))
        jobs = int(self.env.get("CMAKE_BUILD_PARALLEL_LEVEL", "2"))
        if not 1 <= self.port <= 65535 or not 1 <= self.timeout <= 3600 or jobs < 1:
            raise ValueError("invalid port, health timeout or build parallelism")
        self.env.update(STRATA_AIXS_PAIRED_INT8="1", CMAKE_BUILD_PARALLEL_LEVEL=str(jobs))
        self.priv = [] if os.geteuid() == 0 else ["sudo"]
        self.ctl = ["systemctl", "--user"] if self.user_mode else self.priv + ["systemctl"]
        cfg_home = Path(self.env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config"))
        self.unit = ((cfg_home / "systemd/user") if self.user_mode else Path("/etc/systemd/system")) / (self.name + ".service")
        self.envfile = self.root / ".momoe" / ("server-" + self.name + ".env")
        self.cutover = False
        self.was_active = False
        self.was_enabled = False
        self.saved_unit = None
        self.saved_env = None

    def run(self, args, check=True, capture=False):
        return subprocess.run([str(x) for x in args], cwd=self.root, env=self.env, check=check,
                              stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.PIPE if capture else None, text=True)

    def control(self, *args, check=True):
        return self.run(self.ctl + list(args), check=check, capture=True)

    def install_unit(self, data):
        if self.user_mode:
            atomic_bytes(self.unit, data, 0o644)
        else:
            staged = self.tx / "new-unit"
            staged.write_bytes(data)
            self.run(self.priv + ["install", "-m", "0644", staged, self.unit])

    def render_unit(self):
        user = pwd.getpwuid(os.getuid()).pw_name
        group = grp.getgrgid(os.getgid()).gr_name
        identity = "" if self.user_mode else f"User={user}\nGroup={group}\n"
        return (f"[Unit]\nDescription=MoMoE accepted Ornith paired INT8 server\nAfter=network.target\n\n"
                f"[Service]\nType=simple\n{identity}WorkingDirectory={str(self.root).replace('%', '%%')}\n"
                f"EnvironmentFile={str(self.envfile).replace('%', '%%')}\n"
                "Environment=PYTHONUNBUFFERED=1\nEnvironment=CUDA_DEVICE_ORDER=PCI_BUS_ID\n"
                "Environment=STRATA_QSA_SPLIT_ATTN=1\nEnvironment=STRATA_ATTN_CHUNK=256\n"
                "Environment=STRATA_ATTN_MERGE_FAST=1\n"
                f"ExecStart={unit_quote(self.root / 'run-server.sh')}\n"
                "Restart=on-failure\nRestartSec=15\nTimeoutStopSec=90\nKillSignal=SIGTERM\n\n"
                "[Install]\nWantedBy=" + ("default.target" if self.user_mode else "multi-user.target") + "\n").encode()

    def prepare(self):
        for path, digest in SOURCES.items():
            if sha(self.root / path) != digest:
                raise ValueError(f"accepted profile source mismatch: {path}")
        args = [self.root / "setup.sh", "--family", "ornith", "--model", "IQ4_XS",
                "--context", "262144", "--kv", "int8", "--vram-reserve-mib", "896",
                "--vision", "none", "--no-start", "--no-browser", "--no-data-migration", "--yes", "--setup",
                "--config-name", "ornith.json", "--data-dir", self.root, "--port", str(self.port)]
        if self.env.get("SERVER_GGUF_DIR"):
            args.extend(("--gguf-dir", self.env["SERVER_GGUF_DIR"]))
        # Generic setup migrates data from registered installations. Keep this dedicated
        # transaction's registry private; --no-data-migration also disables nearby-checkout discovery.
        previous_xdg = self.env.get("XDG_CONFIG_HOME")
        self.env["XDG_CONFIG_HOME"] = str(self.root / ".momoe" / "setup-state")
        try:
            self.run(args)
        finally:
            if previous_xdg is None:
                self.env.pop("XDG_CONFIG_HOME", None)
            else:
                self.env["XDG_CONFIG_HOME"] = previous_xdg
        cfgpath = self.root / "config/ornith.json"
        cfg = normalized_config(json.loads(cfgpath.read_text()), self.root)
        cfg["port"] = self.port
        exe = Path(cfg["exe"]).resolve()
        if exe.parent not in (self.root / "engine", self.root / "engine-cuda12"):
            raise ValueError("generated engine is outside this installation")
        if not exe.is_file():
            raise ValueError("generated engine is missing")
        meta = json.loads((exe.parent / "BUILD.json").read_text())
        if meta.get("profile") is not True:
            raise ValueError("engine was not built with the accepted paired INT8 profile")
        atomic_bytes(cfgpath, (json.dumps(cfg, indent=2) + "\n").encode())
        self.exe = exe

    def wait_healthy(self):
        host = "127.0.0.1" if self.host in ("0.0.0.0", "localhost") else "::1" if self.host == "::" else self.host
        url = f"http://{'[' + host + ']' if ':' in host else host}:{self.port}/health"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        end = time.monotonic() + self.timeout
        while time.monotonic() < end:
            try:
                with opener.open(url, timeout=2) as response:
                    health = json.loads(response.read(65536))
                pid = int(self.control("show", self.name, "--property=MainPID", "--value").stdout.strip())
                if (health.get("status") == "ok" and health.get("loaded") is True and health.get("max_context") == 262144
                        and self.control("is-active", "--quiet", self.name, check=False).returncode == 0
                        and owns_port(pid, self.port)):
                    return health
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
            time.sleep(1)
        raise RuntimeError(f"service did not become loaded/healthy at native 262144 on {url}")

    def rollback(self):
        if self.cutover:
            self.control("stop", self.name, check=False)
        for name in ("config", "engine", "engine-cuda12"):
            dest = self.root / name
            saved = self.tx / name
            if dest.exists():
                os.replace(dest, self.tx / ("failed-" + name))
            if saved.exists():
                os.replace(saved, dest)
        if self.cutover:
            if self.saved_unit is None:
                if self.user_mode:
                    self.unit.unlink(missing_ok=True)
                else:
                    self.run(self.priv + ["rm", "-f", self.unit])
            else:
                self.install_unit(self.saved_unit)
            if self.saved_env is None:
                self.envfile.unlink(missing_ok=True)
            else:
                atomic_bytes(self.envfile, self.saved_env)
            self.control("daemon-reload")
            self.control("enable" if self.was_enabled else "disable", self.name, check=False)
            if self.was_active:
                self.control("start", self.name)

    def install(self):
        if sys.platform != "linux":
            raise ValueError("the dedicated systemd/CUDA profile requires Linux")
        self.run(["nvidia-smi"], capture=True)
        self.control("show-environment")
        state = self.root / ".momoe"
        state.mkdir(exist_ok=True)
        with (state / "server-install.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.tx = Path(tempfile.mkdtemp(prefix="install-backup-", dir=state))
            self.was_active = self.control("is-active", "--quiet", self.name, check=False).returncode == 0
            self.was_enabled = self.control("is-enabled", "--quiet", self.name, check=False).returncode == 0
            if self.unit.exists():
                self.saved_unit = (self.unit.read_bytes() if self.user_mode else
                                   self.run(self.priv + ["cat", self.unit], capture=True).stdout.encode())
            if self.envfile.exists():
                self.saved_env = self.envfile.read_bytes()
            for name in ("config", "engine", "engine-cuda12"):
                source = self.root / name
                if source.is_symlink():
                    raise ValueError(f"installer-owned {name} must not be a symlink")
                if source.exists():
                    shutil.copytree(source, self.tx / name, symlinks=True)
            try:
                self.prepare()
                if self.user_mode:
                    user = pwd.getpwuid(os.getuid()).pw_name
                    if self.run(["loginctl", "show-user", user, "--property=Linger", "--value"], capture=True).stdout.strip() != "yes":
                        self.run(["loginctl", "enable-linger", user])
                        if self.run(["loginctl", "show-user", user, "--property=Linger", "--value"], capture=True).stdout.strip() != "yes":
                            raise RuntimeError("user linger is not enabled; cannot promise startup at boot")
                self.cutover = True
                if self.was_active:
                    self.control("stop", self.name)
                env = {"SERVER_CONFIG": str(self.root / "config/ornith.json"), "SERVER_HOST": self.host,
                       "SERVER_PORT": str(self.port), "SERVER_API_KEY": self.key}
                content = "".join(k + "=" + json.dumps(v, ensure_ascii=False) + "\n" for k, v in env.items())
                atomic_bytes(self.envfile, content.encode())
                self.install_unit(self.render_unit())
                self.control("daemon-reload")
                self.control("enable", "--now", self.name)
                health = self.wait_healthy()
                receipt = {"profile": "aixs-paired-int8-v1", "engine_sha256": sha(self.exe),
                           "model_sha256": MODEL_SHA, "service": self.name, "health": health,
                           "checked_epoch": time.time(), "hardware_policy_changed": False}
                atomic_bytes(state / "accepted-server-install.json", (json.dumps(receipt, indent=2) + "\n").encode())
            except BaseException:
                try:
                    self.rollback()
                except BaseException as recovery:
                    print(f"Recovery needs attention; retained backup at {self.tx}: {recovery}", file=sys.stderr)
                    raise
                else:
                    shutil.rmtree(self.tx)
                raise
            shutil.rmtree(self.tx)
        print(f"Loaded accepted profile at native 262144; service {self.name}.service is enabled.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Install the verified Ornith IQ4_XS paired INT8 server. "
        "Configure with SERVER_HOST, SERVER_PORT, SERVER_API_KEY, SERVER_GGUF_DIR, "
        "SERVER_SERVICE_NAME, SERVER_USER_SERVICE and CMAKE_BUILD_PARALLEL_LEVEL.")
    parser.parse_args(argv)
    def interrupted(signum, frame):
        raise InterruptedError("installation interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        Installer(Path(__file__).resolve().parents[1]).install()
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"Server installation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
