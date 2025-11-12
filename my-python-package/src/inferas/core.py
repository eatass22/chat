"""
Core utilities for the `inferas` package.

Features:
- install_packages: installs packages using the current Python interpreter (`pip`).
- install_and_run: install packages (optional) then run a module (via `python -m <module>`).
- simple JSON metadata (version + build) kept next to this file.
- small CLI to drive install / run / metadata operations.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Tuple

METADATA_FILENAME = "build_meta.json"
METADATA_PATH = Path(__file__).parent / METADATA_FILENAME


def load_metadata(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or METADATA_PATH
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": "0.0.0", "build": 0}
    return {"version": "0.0.0", "build": 0}


def save_metadata(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    path = path or METADATA_PATH
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def increment_build(path: Optional[Path] = None) -> Dict[str, Any]:
    meta = load_metadata(path)
    meta["build"] = int(meta.get("build", 0)) + 1
    save_metadata(meta, path)
    return meta


def set_version(version: str, path: Optional[Path] = None) -> Dict[str, Any]:
    meta = load_metadata(path)
    meta["version"] = version
    save_metadata(meta, path)
    return meta


def _run_subprocess(cmd: List[str]) -> Tuple[int, str]:
    """
    Run subprocess command, return (returncode, combined_output).
    """
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = proc.stdout or ""
        return proc.returncode, output
    except FileNotFoundError as exc:
        return 127, str(exc)
    except Exception as exc:
        return 1, str(exc)


def install_packages(packages: Iterable[str], upgrade: bool = False) -> int:
    """
    Install packages using python -m pip install ...
    Packages example: ["requests==2.31.0", "numpy"]
    Returns subprocess returncode.
    """
    packages = list(packages)
    if not packages:
        return 0
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(packages)
    rc, out = _run_subprocess(cmd)
    print(out)
    return rc


def install_from_requirements(requirements_path: str, upgrade: bool = False) -> int:
    p = Path(requirements_path)
    if not p.exists():
        print(f"requirements file not found: {requirements_path}", file=sys.stderr)
        return 2
    return install_packages([f"-r{str(p)}"], upgrade=upgrade)


def run_module(module_name: str, args: Optional[List[str]] = None) -> int:
    """
    Runs a module with: python -m <module_name> [args...]
    Returns the subprocess return code.
    """
    args = args or []
    cmd = [sys.executable, "-m", module_name] + args
    proc = subprocess.run(cmd)
    return proc.returncode


def install_and_run(
    packages: Optional[Iterable[str]],
    module_name: str,
    run_args: Optional[List[str]] = None,
    upgrade: bool = False,
    inc_build_before_run: bool = False,
    meta_path: Optional[Path] = None,
) -> int:
    """
    Helper that installs packages (if provided), optionally increments build,
    then runs the requested module. Returns the final subprocess return code.
    """
    meta_path = meta_path or METADATA_PATH

    if packages:
        rc = install_packages(packages, upgrade=upgrade)
        if rc != 0:
            return rc

    if inc_build_before_run:
        increment_build(meta_path)

    return run_module(module_name, run_args)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="inferas", description="inferas runtime helper")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--install", nargs="+", help="Install packages (space separated list)")
    group.add_argument("--requirements", "-r", help="Install from a requirements.txt file")
    parser.add_argument("--upgrade", action="store_true", help="Pass --upgrade to pip install")
    parser.add_argument("--run", help="Run a module name with python -m <module>")
    parser.add_argument("--run-args", nargs=argparse.REMAINDER, help="Arguments passed to the run module")
    parser.add_argument("--show-meta", action="store_true", help="Print metadata (version & build)")
    parser.add_argument("--inc-build", action="store_true", help="Increment build number in metadata")
    parser.add_argument("--set-version", help="Set metadata version string (e.g. 1.2.3)")
    parser.add_argument("--meta-file", help="Path to metadata JSON file (defaults next to this module)")
    parser.add_argument("--inc-before-run", action="store_true", help="Increment build before running module")

    ns = parser.parse_args(argv)

    meta_path = Path(ns.meta_file) if ns.meta_file else METADATA_PATH

    if ns.set_version:
        meta = set_version(ns.set_version, meta_path)
        print("metadata updated:", json.dumps(meta, indent=2))
        return 0

    if ns.inc_build:
        meta = increment_build(meta_path)
        print("metadata updated:", json.dumps(meta, indent=2))
        return 0

    if ns.show_meta:
        meta = load_metadata(meta_path)
        print(json.dumps(meta, indent=2))
        return 0

    if ns.install:
        rc = install_packages(ns.install, upgrade=ns.upgrade)
        return rc

    if ns.requirements:
        rc = install_from_requirements(ns.requirements, upgrade=ns.upgrade)
        return rc

    if ns.run:
        run_args = ns.run_args or []
        # If install was also requested, install packages then run:
        if ns.install:
            rc = install_and_run(ns.install, ns.run, run_args, upgrade=ns.upgrade, inc_build_before_run=ns.inc_before_run, meta_path=meta_path)
            return rc
        rc = install_and_run(None, ns.run, run_args, upgrade=ns.upgrade, inc_build_before_run=ns.inc_before_run, meta_path=meta_path)
        return rc

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())