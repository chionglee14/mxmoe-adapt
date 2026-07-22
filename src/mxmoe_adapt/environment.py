"""Read-only MXMACA/C500 environment collector."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _command(
    command: list[str], timeout: int = 10, cwd: Path | None = None
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": type(error).__name__}
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("torch", "triton", "flag-gems", "vllm"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            if distribution == "flag-gems":
                try:
                    versions[distribution] = importlib.import_module("flag_gems").__version__
                except (ImportError, AttributeError):
                    versions[distribution] = None
            else:
                versions[distribution] = None
    return versions


def _mx_smi_identity(result: dict[str, Any]) -> dict[str, Any]:
    """Extract stable hardware/software fields from volatile mx-smi output."""
    identity: dict[str, Any] = {
        "available": result.get("available", False),
        "returncode": result.get("returncode"),
    }
    stdout = str(result.get("stdout", ""))
    patterns = {
        "mx_smi_version": r"mx-smi\s+version:\s*([^\s]+)",
        "driver_version": r"Kernel Mode Driver Version:\s*([^\s|]+)",
        "maca_version": r"MACA Version:\s*([^\s|]+)",
        "device_name": r"\|\s*\d+\s+(MetaX\s+[^|]+?)\s*\|\s*\d+",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            identity[name] = match.group(1).strip()
    return identity


def _source_revisions(project_root: Path) -> dict[str, Any]:
    project_revision = _command(["git", "rev-parse", "HEAD"], cwd=project_root)
    marker = project_root / ".source-revision"
    if project_revision.get("returncode") != 0 and marker.is_file():
        project_revision = {
            "available": True,
            "returncode": 0,
            "stdout": marker.read_text(encoding="utf-8").strip(),
            "stderr": "",
            "source": "deterministic source snapshot",
        }
    revisions: dict[str, Any] = {"mxmoe_adapt": project_revision}
    try:
        flag_gems = importlib.import_module("flag_gems")
        flaggems_root = Path(flag_gems.__file__).resolve().parents[2]
        revisions["flag_gems"] = _command(
            ["git", "rev-parse", "HEAD"], cwd=flaggems_root
        )
    except (ImportError, AttributeError, IndexError):
        revisions["flag_gems"] = {"available": False}
    return revisions


def _torch_device() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    result: dict[str, Any] = {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_api_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        result.update(
            {
                "device_count": torch.cuda.device_count(),
                "current_device": current,
                "device_name": torch.cuda.get_device_name(current),
            }
        )
    return result


def collect_environment() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    mx_smi = _command(["mx-smi"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "packages": _package_versions(),
        "torch_device": _torch_device(),
        "mx_smi": mx_smi,
        "source_revisions": _source_revisions(project_root),
    }
    identity_fields = {
        "platform": payload["platform"],
        "packages": payload["packages"],
        "torch_device": payload["torch_device"],
        "mx_smi": _mx_smi_identity(mx_smi),
        "source_revisions": payload["source_revisions"],
    }
    stable = json.dumps(identity_fields, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["environment_id"] = hashlib.sha256(stable).hexdigest()[:16]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(collect_environment(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
