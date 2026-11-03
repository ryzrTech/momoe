"""CPU-only tests for the dedicated-server installer (tools/setup_server.py), run-server.sh and setup.py's build cache.
No GPU, no downloads, no sudo, no real service: every side effect is mocked (real files only inside temp roots), so the
tests check the argument list setup is handed, the installer's rollback, an actual owns_port() over a real loopback
socket, run-server.sh as a black box and setup.py's build cache.

    python -m unittest tools.test_setup_server
"""
from __future__ import annotations

import ast
import contextlib
import unittest
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import setup  # noqa: E402
import setup_server  # noqa: E402

setup_path = ROOT / "setup.py"
setup_server_src = ROOT / "tools" / "setup_server.py"

TUNE = ["--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
        "--max-context", "262144", "--kv", "int8", "--vram-reserve-mib", "896"]

DEFAULT_ENV = {
    "SERVER_SERVICE_NAME": "momoe-ai-server",
    "SERVER_MODEL": "ornith",
    "MODEL_SIZE": "IQ4_XS",
    "STRATA_AIXS_PAIRED_INT8": "1",
    "SERVER_USER_SERVICE": "1",
    "SERVER_HOST": "127.0.0.1",
    "SERVER_API_KEY": "",
    "SERVER_PORT": "8080",
    "SERVER_HEALTH_TIMEOUT": "180",
    "CMAKE_BUILD_PARALLEL_LEVEL": "2",
    "XDG_CONFIG_HOME": "",
}


def build_fixture(tmp: Path, *, native_ok=True):
    """config/ornith.json plus the artifacts normalized_config needs (native model, pack, expert profile, MTP draft
    vocab, all existing).  The tiny native file passes the canonical digest check only because sha() is mocked."""
    native = tmp / "Ornith-1.5-35B-A3B-iq4_xs-00001-of-00002.gguf"
    pack = tmp / "experts.iq4_xs.pack"
    profile = tmp / "expert-profile-qwen36.bin"
    draft = tmp / "experts.bin"
    pack.mkdir()
    (pack / "tokenizer").mkdir()
    for p in (native, profile, draft):
        p.write_bytes(b"x")
    cfg = {
        "exe": str(tmp / "engine" / "momoe"),
        "args": ["--pack", str(pack), "--native", str(native),
                 "--expert-profile", str(profile), "--mtp-draft-vocab", str(draft)],
        "tokenizer": str(pack / "tokenizer"),
        "sampling": {"temperature": 0.7},
        "reasoning_effort": "high", "think": True, "enable_thinking": True,
        "thinking": "on", "vision": {"gpu": True, "max_tokens": 100},
    }
    (tmp / "config").mkdir(parents=True, exist_ok=True)
    (tmp / "config" / "ornith.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp / "engine").mkdir(parents=True, exist_ok=True)
    (tmp / "engine" / "momoe").write_bytes(b"engine-binary")
    (tmp / "engine" / "BUILD.json").write_text(json.dumps({"profile": True}), encoding="utf-8")
    return native, cfg


def sha_returns(model_sha_for):
    """Patch setup_server.sha so only the fixture model file passes the canonical digest check."""
    model_sha_for = str(model_sha_for)
    name = Path(model_sha_for).name

    def _sha(path):
        p = str(path)
        if p.endswith(name):
            return setup_server.MODEL_SHA if model_sha_for else "0" * 64
        return "0" * 64

    return _sha


# --------------------------------------------------------------------------------------------------- normalized_config
class TestNormalizedConfig(unittest.TestCase):
    """The exact argument list setup_server builds from a generated run config (#EXP-014)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ss-norm-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_arguments_are_exact(self):
        native, _ = build_fixture(self.tmp)
        cfg = json.loads((self.tmp / "config" / "ornith.json").read_text())
        with mock.patch.object(setup_server, "sha", sha_returns(native)):
            out = setup_server.normalized_config(cfg, self.tmp)
        args = out["args"]
        self.assertEqual(args[:4], ["--pack", str((self.tmp / "experts.iq4_xs.pack").resolve()),
                                    "--native", str(native.resolve())])
        self.assertEqual(args[4:6], ["--expert-profile", str((self.tmp / "expert-profile-qwen36.bin").resolve())])
        self.assertEqual(args[6:8], ["--mtp-draft-vocab", str((self.tmp / "experts.bin").resolve())])
        # the model runs both as --native and as the MTP draft (native == mtp)
        i = args.index("--mtp")
        self.assertEqual(args[i + 1], str(native.resolve()))
        # the tuned flags, exactly
        idx = args.index("--expert-cache")
        self.assertEqual(args[idx:idx + len(TUNE)], TUNE)
        # native 262144, int8 KV, the reserve
        self.assertEqual(args[args.index("--max-context") + 1], "262144")
        self.assertEqual(args[args.index("--kv") + 1], "int8")
        self.assertEqual(args[args.index("--vram-reserve-mib") + 1], "896")

    def test_boolean_resident_flags_removed_without_orphans(self):
        native, _ = build_fixture(self.tmp)
        cfg = json.loads((self.tmp / "config" / "ornith.json").read_text())
        cfg["args"] += ["--resident-experts", "--prefill", "slow", "--mmap-experts", "--memory-limit-mib", "100"]
        with mock.patch.object(setup_server, "sha", sha_returns(native)):
            out = setup_server.normalized_config(cfg, self.tmp)
        self.assertEqual(out["args"][-len(TUNE):], TUNE)
        self.assertNotIn("--resident-experts", out["args"])
        self.assertNotIn("slow", out["args"])
        self.assertNotIn("100", out["args"])
        for key in ("reasoning_effort", "think", "enable_thinking", "thinking", "vision"):
            self.assertNotIn(key, out, key)

    def test_thinking_and_sampling_enabled(self):
        native, _ = build_fixture(self.tmp)
        cfg = json.loads((self.tmp / "config" / "ornith.json").read_text())
        with mock.patch.object(setup_server, "sha", sha_returns(native)):
            out = setup_server.normalized_config(cfg, self.tmp)
        self.assertEqual(out["anthropic_thinking"], "model")          # thinking stays enabled
        self.assertEqual(out["sampling"], {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        self.assertEqual(out["cwd"], str(self.tmp))


class TestNormalizedConfigValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ss-val-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_wrong_canonical_hash_fails(self):
        native, _ = build_fixture(self.tmp)
        cfg = json.loads((self.tmp / "config" / "ornith.json").read_text())
        with mock.patch.object(setup_server, "sha", lambda p: "1" * 64):
            with self.assertRaises(ValueError):
                setup_server.normalized_config(cfg, self.tmp)

    def test_missing_artifact_fails(self):
        native, _ = build_fixture(self.tmp)
        (native).unlink()                    # the native model is required to exist
        cfg = json.loads((self.tmp / "config" / "ornith.json").read_text())
        with mock.patch.object(setup_server, "sha", sha_returns(native)):
            with self.assertRaises(ValueError):
                setup_server.normalized_config(cfg, self.tmp)

    def test_duplicate_flag_fails(self):
        native, cfg = build_fixture(self.tmp)
        cfg["args"] = ["--native", str(native), "--native", str(native)]   # two --native: arity unknown
        (self.tmp / "config" / "ornith.json").write_text(json.dumps(cfg))
        with mock.patch.object(setup_server, "sha", sha_returns(native)):
            with self.assertRaises(ValueError):
                setup_server.normalized_config(cfg, self.tmp)


    def test_missing_tokenizer_is_rejected(self):
        native, cfg = build_fixture(self.tmp)
        cfg.pop("tokenizer")
        with mock.patch.object(setup_server, "sha", sha_returns(native)):
            with self.assertRaises(ValueError):
                setup_server.normalized_config(cfg, self.tmp)


# ----------------------------------------------------------------------------------------------------- the installer
class TestInstallerConstructor(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ss-inst-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def env(self, **over):
        e = dict(DEFAULT_ENV)
        e.pop("XDG_CONFIG_HOME", None)
        e.update(over)
        return e

    def test_accepts_root_with_spaces(self):
        with tempfile.TemporaryDirectory() as d:
            name = "momoe install and 'quote'"
            spaced = Path(d) / name
            spaced.mkdir()
            inst = setup_server.Installer(spaced, env=self.env())
            self.assertEqual(inst.root, spaced.resolve())

    def test_accepts_key_with_quotes_and_percent(self):
        key = 'a"b%c\\d/e'                      # control-free, so accepted
        inst = setup_server.Installer(self.root, env=self.env(SERVER_API_KEY=key))
        self.assertEqual(inst.key, key)

    def _reject(self, **over):
        with self.assertRaises(ValueError):
            setup_server.Installer(self.root, env=self.env(**over))

    def test_rejects_wrong_model(self):
        self._reject(SERVER_MODEL="llama")

    def test_rejects_wrong_quant(self):
        self._reject(MODEL_SIZE="Q8_0")

    def test_rejects_wrong_profile(self):
        self._reject(STRATA_AIXS_PAIRED_INT8="0")

    def test_rejects_nonlocal_without_key(self):
        self._reject(SERVER_HOST="0.0.0.0")

    def test_rejects_bad_unit_name(self):
        self._reject(SERVER_SERVICE_NAME="bad name!")

    def test_rejects_bad_port(self):
        self._reject(SERVER_PORT="70000")

    def test_rejects_bad_timeout(self):
        self._reject(SERVER_HEALTH_TIMEOUT="0")

    def test_rejects_bad_parallelism(self):
        self._reject(CMAKE_BUILD_PARALLEL_LEVEL="0")


class TestRenderUnit(unittest.TestCase):
    def test_has_three_persistent_tune_envs_and_no_key(self):
        with tempfile.TemporaryDirectory() as d:
            inst = setup_server.Installer(Path(d), env=DEFAULT_ENV | {"SERVER_API_KEY": "S3cr3t-K%EY"})
            text = inst.render_unit().decode()
            self.assertIn("STRATA_QSA_SPLIT_ATTN=1", text)
            self.assertIn("STRATA_ATTN_CHUNK=256", text)
            self.assertIn("STRATA_ATTN_MERGE_FAST=1", text)
            self.assertNotIn("S3cr3t-K%EY", text)          # the unit must never carry the real key
            self.assertNotIn("SERVER_API_KEY=", text)       # nor reference the key at all


# ------------------------------------------------------------------------------------------------ rollback (install)
class _TxInstaller(setup_server.Installer):
    """Installer with run/control/prepare/wait_healthy replaced by recorders, so install() drives a real rollback with
    real files but no service, sudo or network."""

    def __init__(self, root, env, prepare_fn, healthy_fn):
        super().__init__(root, env=env)
        self._prepare, self._healthy = prepare_fn, healthy_fn
        self.runs, self.controls = [], []

    def run(self, args, check=True, capture=False):
        self.runs.append([str(a) for a in args])
        class R:
            returncode = 0
            stdout = "yes"
            stderr = ""
        return R()

    def control(self, *args, check=True):
        self.controls.append(list(args))
        class R:
            returncode = 0
            stdout = ""
        if args[:1] == ("show",):
            R.stdout = "12345\n"
        return R()

    def prepare(self):
        self._prepare(self)

    def wait_healthy(self):
        return self._healthy()


class TestInstallRollback(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ss-tx-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.env = DEFAULT_ENV | {"SERVER_API_KEY": "", "XDG_CONFIG_HOME": str(self.tmp / "cfg")}
        (self.tmp / "cfg" / "systemd" / "user").mkdir(parents=True, exist_ok=True)

    def _prior_active(self, installer, *, with_unit=True):
        """A pre-existing install: an old config, an engine, an enabled+active unit (optional) and an env file."""
        (self.tmp / "config").mkdir(parents=True, exist_ok=True)
        (self.tmp / "config" / "ornith.json").write_text('{"old": "config", "exe": "x"}', encoding="utf-8")
        eng = self.tmp / "engine"; eng.mkdir()
        (eng / "BUILD.json").write_text('{"profile": true}', encoding="utf-8")
        (eng / "momoe").write_bytes(b"old-engine")
        if with_unit:
            installer.unit.parent.mkdir(parents=True, exist_ok=True)
            installer.unit.write_bytes(b"OLD-UNIT")
        (installer.root / ".momoe").mkdir(parents=True, exist_ok=True)
        installer.envfile.write_bytes(b"OLD=env")
        return installer

    def test_prepare_failure_restores_everything_and_does_not_stop_the_old_service(self):
        def prepare_fail(inst):
            # setup had already written its own (partial) artifacts, then blew up before the cutover
            (inst.root / "config").mkdir(parents=True, exist_ok=True)
            (inst.root / "config" / "ornith.json").write_text('{"new": "config"}', encoding="utf-8")
            eng = inst.root / "engine"; eng.mkdir(exist_ok=True)
            (eng / "momoe").write_bytes(b"partial")
            raise ValueError("build failed")

        inst = _TxInstaller(self.tmp, self.env, prepare_fail, lambda: {"status": "ok"})
        self._prior_active(inst)
        with self.assertRaises(ValueError):
            inst.install()
        self.assertEqual((self.tmp / "config" / "ornith.json").read_text(), '{"old": "config", "exe": "x"}')
        self.assertEqual((self.tmp / "engine" / "momoe").read_bytes(), b"old-engine")
        self.assertIn(b"OLD-UNIT", inst.unit.read_bytes())                 # unit unchanged
        self.assertTrue(inst.envfile.read_bytes().startswith(b"OLD="))     # env unchanged
        # the previously running model was NOT stopped: prepare failed before the cutover
        self.assertFalse(any(c[0] == "stop" for c in inst.controls))
        self.assertFalse(any(c[0] == "daemon-reload" for c in inst.controls))

    def test_fail_after_unit_write_removes_fresh_unit_and_restores_prior_state(self):
        def succeed(inst):
            (inst.root / "config").mkdir(parents=True, exist_ok=True)
            (inst.root / "config" / "ornith.json").write_text('{"new": "config"}', encoding="utf-8")
            eng = inst.root / "engine"; eng.mkdir(exist_ok=True)
            (eng / "momoe").write_bytes(b"new-engine")
            inst.exe = eng / "momoe"

        def unhealthy():
            raise RuntimeError("never became healthy")

        inst = _TxInstaller(self.tmp, self.env, succeed, unhealthy)
        self._prior_active(inst, with_unit=False)           # was_active and was_enabled both True; no prior unit
        with self.assertRaises(RuntimeError):
            inst.install()
        self.assertFalse(inst.unit.exists())                # the fresh unit is removed (no prior unit to keep)
        self.assertEqual((self.tmp / "config" / "ornith.json").read_text(), '{"old": "config", "exe": "x"}')
        self.assertIn(["daemon-reload"], inst.controls)
        # the failed new service was stopped, the enabled flag is restored and the prior active one restarted
        self.assertIn(["stop", inst.name], inst.controls)
        self.assertTrue(any(c[0] == "enable" and c[-1] == inst.name for c in inst.controls))
        self.assertTrue(any(c == ["start", inst.name] for c in inst.controls))

    def test_success_writes_receipt_profile_and_env_is_private(self):
        def succeed(inst):
            (inst.root / "config").mkdir(parents=True, exist_ok=True)
            (inst.root / "config" / "ornith.json").write_text('{"exe": "x"}', encoding="utf-8")
            eng = inst.root / "engine"; eng.mkdir(exist_ok=True)
            (eng / "momoe").write_bytes(b"new-engine")
            (eng / "BUILD.json").write_text('{"profile": true}', encoding="utf-8")
            inst.exe = eng / "momoe"

        inst = _TxInstaller(self.tmp, self.env, succeed, lambda: {"status": "ok", "loaded": True, "max_context": 262144})
        inst.install()
        receipt = json.loads((self.tmp / ".momoe" / "accepted-server-install.json").read_text())
        self.assertEqual(receipt["profile"], "aixs-paired-int8-v1")
        self.assertEqual(receipt["model_sha256"], setup_server.MODEL_SHA)
        self.assertEqual(receipt["health"]["status"], "ok")
        self.assertEqual(receipt["service"], inst.name)
        mode = os.stat(inst.envfile).st_mode & 0o777          # the env file holding the API key is 0600
        self.assertEqual(mode, 0o600)


# ------------------------------------------------------------------------------------------------------ owns_port
class TestOwnsPort(unittest.TestCase):
    """A real loopback socket: the owning process owns_port(), an unrelated one does not.  No GPU involved."""

    def test_current_process_owns_its_port_wrong_pid_does_not(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        httpd = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        import threading
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            port = httpd.server_address[1]
            self.assertTrue(setup_server.owns_port(os.getpid(), port))
        finally:
            httpd.shutdown()
            httpd.server_close()

        # a process that is not listening on the port does not own it
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertFalse(setup_server.owns_port(child.pid, port))
        finally:
            child.terminate()
            child.wait(timeout=5)


# -------------------------------------------------------------------------------------------------- setup.py args
class TestSetupCliArgs(unittest.TestCase):
    """setup_server hands setup.sh (and so setup.py) a fixed flag list.  Those flag names must all be real setup.py
    parser options, and the tuned ones (--context/--vision/--no-browser/--no-start) must be present."""

    @staticmethod
    def setup_arg_names():
        tree = ast.parse(setup_path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("--"):
                    names.add(first.value.lstrip("-"))
        return names

    @staticmethod
    def setup_server_flags():
        """The flags passed to setup.sh in Installer.prepare(), extracted from the source (the list holding
        `self.root / "setup.sh"`, whose first element is a binop rather than a string literal)."""
        tree = ast.parse(setup_server_src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.List) and any(
                    isinstance(e, ast.BinOp) and isinstance(e.right, ast.Constant) and e.right.value == "setup.sh"
                    for e in node.elts):
                consts = [e.value for e in node.elts
                          if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                return [c.lstrip("-") for c in consts if c.startswith("--")]   # every flag (valueless ones included)
        return []

    def test_setup_server_only_uses_real_setup_flags(self):
        names = self.setup_arg_names()
        flags = self.setup_server_flags()
        self.assertTrue(flags, "could not find the setup.sh invocation")
        for flag in flags:
            self.assertIn(flag, names, f"{flag} is not a setup.py parser option")

    def test_expected_tuned_flags_are_present(self):
        flags = self.setup_server_flags()
        for flag in ("context", "vision", "no-browser", "no-start", "family", "model", "kv",
                     "vram-reserve-mib", "yes", "setup", "config-name", "data-dir"):
            self.assertIn(flag, flags)


# ---------------------------------------------------------------------------------------------------- run-server.sh
class TestRunServerSh(unittest.TestCase):
    """Black box: run-server.sh under a path with spaces/quotes, with a fake .venv/bin/python that records argv
    instead of starting the server."""

    RUNNER = ROOT / "run-server.sh"

    def _make(self, name):
        d = Path(tempfile.mkdtemp(prefix=f"ss-rs-{name}-"))
        shutil.copy2(self.RUNNER, d / "run-server.sh")
        (d / "run-server.sh").chmod(0o755)
        (d / "config").mkdir(parents=True, exist_ok=True)
        (d / "config" / "ornith.json").write_text(
            json.dumps({"model_name": "Ornith-1.5-35B-A3B", "exe": "momoe"}), encoding="utf-8")
        vbin = d / ".venv" / "bin"
        vbin.mkdir(parents=True)
        (vbin / "python").write_text(
            '#!/bin/bash\nmkdir -p "$PWD/.momoe"\nfor a in "$@"; do printf "%s\\n" "$a"; done > "$PWD/.momoe/argv.txt"\n',
            encoding="utf-8")
        (vbin / "python").chmod(0o755)
        return d

    def _run(self, d, **over):
        env = dict(os.environ)
        env.pop("SERVER_CONFIG", None)
        for k, v in over.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        return subprocess.run([str(d / "run-server.sh")], cwd=str(d), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)

    def test_argv_and_metadata_under_a_spaced_quoted_path(self):
        d = self._make("spaced")
        # move everything under a path with a space and a quote
        weird = Path(tempfile.mkdtemp(prefix='ss-weird "'))
        moved = weird / "momoe"
        shutil.move(str(d), str(moved))
        d = moved
        cfg = str(d / "config" / "ornith.json")
        res = self._run(d, SERVER_CONFIG=cfg, SERVER_HOST="127.0.0.1", SERVER_PORT="41234", SERVER_API_KEY="k ey")
        self.assertEqual(res.returncode, 0, res.stderr)
        argv = (d / ".momoe" / "argv.txt").read_text().splitlines()
        self.assertIn(str(d / "serve" / "server.py"), argv)     # it was handed server.py, but never ran it
        self.assertIn("--engine", argv)
        self.assertIn("--config", argv)
        self.assertIn("--host", argv)
        self.assertIn("--port", argv)
        meta = json.loads((d / ".momoe" / "server.json").read_text())
        self.assertEqual(meta["config"], cfg)                   # the spaced path is stored, not mangled
        self.assertEqual(meta["host"], "127.0.0.1")
        self.assertEqual(meta["port"], 41234)
        self.assertEqual(meta["model"], "Ornith-1.5-35B-A3B")

    def test_api_key_passes_as_one_argument(self):
        d = self._make("key")
        res = self._run(d, SERVER_CONFIG=str(d / "config" / "ornith.json"),
                        SERVER_HOST="127.0.0.1", SERVER_PORT="8080", SERVER_API_KEY="abc123")
        self.assertEqual(res.returncode, 0, res.stderr)
        argv = (d / ".momoe" / "argv.txt").read_text().splitlines()
        i = argv.index("--api-key")
        self.assertEqual(argv[i + 1], "abc123")                 # a single argument, never interpolated

    def test_nonlocal_without_key_is_refused(self):
        d = self._make("nokey")
        res = self._run(d, SERVER_CONFIG=str(d / "config" / "ornith.json"),
                        SERVER_HOST="0.0.0.0", SERVER_PORT="8080")
        self.assertEqual(res.returncode, 1)
        self.assertIn("refusing to bind", res.stderr)
        self.assertFalse((d / ".momoe" / "argv.txt").exists())   # never reached the exec


# ----------------------------------------------------------------------------------------------------- build cache
class TestBuildCache(unittest.TestCase):
    """setup.build_engine's cache (STRATA_AIXS_PAIRED_INT8 profile flag) with hardware/build/stamp mocked."""

    def _run_build(self, stamp_profile, want_env, arch=89):
        td = tempfile.mkdtemp(prefix="ss-build-")
        self.addCleanup(shutil.rmtree, td, ignore_errors=True)
        eng = Path(td) / "engine"; eng.mkdir()
        (eng / setup.EXE).write_bytes(b"engine")
        (eng / "BUILD.json").write_text(json.dumps(
            {"source": "local", "version": "0.1.32", "archs": [arch], "vision": None,
             "src": "src", "isa_floor": "", "profile": stamp_profile}), encoding="utf-8")
        cmake = mock.Mock(); atom = mock.Mock()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(setup, "engine_dir", lambda toolkit=13: eng))
        stack.enter_context(mock.patch.object(setup, "source_hash", lambda parts: "src"))
        stack.enter_context(mock.patch.object(setup, "cpu_info", lambda: ("CPU", True, False)))
        stack.enter_context(mock.patch.object(setup, "cpu_floor", lambda _avx2: ""))
        stack.enter_context(mock.patch.object(setup, "stale_binary", lambda e: False))
        stack.enter_context(mock.patch.object(setup, "source_version", lambda: "0.1.32"))
        stack.enter_context(mock.patch.object(setup, "install_build_tools", lambda gpu, yes: ("/usr/local/bin/nvcc", None)))
        stack.enter_context(mock.patch.object(setup, "engine_runs_more", lambda e: True))
        stack.enter_context(mock.patch.object(setup, "cmake_build", cmake))
        stack.enter_context(mock.patch.object(setup, "_atomic_replace", atom))
        if want_env is None:
            os.environ.pop("STRATA_AIXS_PAIRED_INT8", None)
        else:
            os.environ["STRATA_AIXS_PAIRED_INT8"] = want_env
        self.addCleanup(lambda: os.environ.pop("STRATA_AIXS_PAIRED_INT8", None))
        eng_out = setup.build_engine({"arch": arch, "archs": [arch]}, "none", True, Path("/llama"))
        return eng_out, cmake, atom

    def test_profile_off_is_a_cache_hit(self):
        eng, cmake, atom = self._run_build(stamp_profile=False, want_env="0")
        self.assertEqual(eng, eng)
        self.assertEqual(cmake.call_count, 0)
        self.assertEqual(atom.call_count, 0)

    def test_unset_preserves_installed_profile(self):
        for profile in (True, False):
            with self.subTest(profile=profile):
                _, cmake, atom = self._run_build(profile, None)
                cmake.assert_not_called()
                atom.assert_not_called()

    def test_switching_on_forces_a_build_with_explicit_on(self):
        eng, cmake, atom = self._run_build(stamp_profile=False, want_env="1")
        self.assertEqual(cmake.call_count, 1)
        defs = cmake.call_args.args[3]
        self.assertIn("-DSTRATA_AIXS_PAIRED_INT8=ON", defs)
        atom.assert_any_call(eng / setup.EXE, setup.ROOT / "build" / setup.EXE)

    def test_explicit_off_forces_a_build_and_cannot_stay_on(self):
        # the old stamp built with the profile ON; setup must pass OFF explicitly, so an old CMakeCache.txt can't keep ON
        eng, cmake, atom = self._run_build(stamp_profile=True, want_env="0")
        self.assertEqual(cmake.call_count, 1)
        defs = cmake.call_args.args[3]
        self.assertIn("-DSTRATA_AIXS_PAIRED_INT8=OFF", defs)


# ------------------------------------------------------------------------------------------------- atomic replace
class TestAtomicReplace(unittest.TestCase):
    def test_can_replace_a_running_binary_without_etxtbsy(self):
        if not Path("/bin/sleep").exists():
            self.skipTest("no /bin/sleep")
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            src = d / "new-sleep"
            dst = d / "old-sleep"
            shutil.copy2("/bin/sleep", src)
            shutil.copy2("/bin/sleep", dst)
            os.chmod(dst, 0o755)
            src.write_bytes(b"MARKER" + b"\x00" * 128)      # the replacement is identifiable
            proc = subprocess.Popen([str(dst), "30"])              # a running process holds the old inode
            try:
                self.assertIsNone(proc.poll())
                setup._atomic_replace(dst, src)              # must not fail with ETXTBSY
                self.assertEqual(dst.read_bytes(), src.read_bytes())
                self.assertIsNone(proc.poll())               # the old process keeps running on its freed inode
            finally:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=5)


class TestPreparationIsolation(unittest.TestCase):
    def test_setup_uses_private_registry_and_restores_environment_on_failure(self):
        class PreparedFailure(Exception):
            pass
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original_xdg = str(root / "caller-settings")
            inst = setup_server.Installer(root, env=DEFAULT_ENV | {"XDG_CONFIG_HOME": original_xdg})
            observed = {}
            def run(args, **kw):
                observed["xdg"] = inst.env["XDG_CONFIG_HOME"]
                observed["args"] = [str(a) for a in args]
                raise PreparedFailure()
            inst.run = run
            def source_sha(path):
                return setup_server.SOURCES[Path(path).relative_to(root).as_posix()]
            with mock.patch.object(setup_server, "sha", source_sha):
                with self.assertRaises(PreparedFailure):
                    inst.prepare()
            self.assertEqual(observed["xdg"], str(root / ".momoe/setup-state"))
            self.assertEqual(inst.env["XDG_CONFIG_HOME"], original_xdg)
            self.assertIn("--no-start", observed["args"])
            self.assertIn("--no-browser", observed["args"])
            self.assertEqual(observed["args"][observed["args"].index("--context") + 1], "262144")
            self.assertEqual(observed["args"][observed["args"].index("--vision") + 1], "none")

class TestEntryHelp(unittest.TestCase):
    def test_help_does_not_start_installation(self):
        result = subprocess.run([sys.executable, str(setup_server_src), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SERVER_GGUF_DIR", result.stdout)


class TestNoMigration(unittest.TestCase):
    def test_no_migration_never_discovers_or_moves_other_installations(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "new"
            with mock.patch.object(setup, "ROOT", dest), \
                 mock.patch.object(setup, "load_settings", return_value={"data_dir": str(Path(td)/"old")}), \
                 mock.patch.object(setup, "save_settings"), \
                 mock.patch.object(setup, "other_installs", side_effect=AssertionError("discovery forbidden")), \
                 mock.patch.object(setup, "move_into", side_effect=AssertionError("migration forbidden")):
                self.assertEqual(setup.data_folder(str(dest), migrate=False), (dest, []))

    def test_dedicated_installer_explicitly_disables_migration(self):
        self.assertIn('"--no-data-migration"', setup_server_src.read_text())


class TestSystemdPathDirectives(unittest.TestCase):
    def test_paths_are_not_exec_argument_quoted(self):
        with tempfile.TemporaryDirectory(prefix="unit space %") as td:
            inst = setup_server.Installer(Path(td), env=DEFAULT_ENV)
            unit = inst.render_unit().decode()
            self.assertIn("WorkingDirectory=" + td.replace("%", "%%") + "\n", unit)
            self.assertIn("EnvironmentFile=" + str(inst.envfile).replace("%", "%%") + "\n", unit)
            self.assertIn('ExecStart="', unit)
