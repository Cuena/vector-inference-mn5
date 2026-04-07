#!/usr/bin/env python3
"""Inspect local vec-inf tunnels and print validation commands."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib import request


SCRIPT_DIR = Path(__file__).resolve().parent
SESSION_STATE_DIR = SCRIPT_DIR / ".tunnel-sessions"
HTTP_TIMEOUT_SECONDS = 2.0
SS_LISTEN_RE = re.compile(
    r"LISTEN\s+\S+\s+\S+\s+\S+:(?P<port>\d+)\s+\S+\s+users:\(\(\"ssh\",pid=(?P<pid>\d+),"
)
LSOF_LISTEN_RE = re.compile(
    r"^ssh\s+(?P<pid>\d+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+TCP\s+.+:(?P<port>\d+)\s+\(LISTEN\)$"
)


@dataclass
class TunnelRecord:
    """Metadata about a locally accessible tunnel."""

    local_port: int
    base_url: str
    job_id: str | None = None
    requested_model_name: str | None = None
    served_model_names: list[str] = field(default_factory=list)
    health_ok: bool | None = None
    ssh_target: str | None = None
    remote_server_host: str | None = None
    remote_server_ip: str | None = None
    remote_server_port: int | None = None
    owner_pid: int | None = None
    state_path: str | None = None
    ssh_pid: int | None = None
    sources: list[str] = field(default_factory=list)
    probe_error: str | None = None

    def display_model_name(self) -> str:
        """Return the best available model label for humans."""
        if self.served_model_names:
            return ", ".join(self.served_model_names)
        if self.requested_model_name:
            return self.requested_model_name
        return "<unknown>"


@dataclass(frozen=True)
class ValidationCommand:
    """A named validation command for a tunnel."""

    key: str
    label: str
    command: str


def run_command(command: list[str]) -> str | None:
    """Return stdout for a subprocess command, or None on failure."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout


def parse_local_forward_spec(spec: str) -> tuple[int, str, int] | None:
    """Parse an SSH local forward specification."""
    normalized = spec[2:] if spec.startswith("-L") else spec
    parts = normalized.split(":")
    if len(parts) == 3:
        local_port_text, remote_host, remote_port_text = parts
    elif len(parts) == 4:
        _, local_port_text, remote_host, remote_port_text = parts
    else:
        return None

    try:
        local_port = int(local_port_text)
        remote_port = int(remote_port_text)
    except ValueError:
        return None

    return local_port, remote_host, remote_port


def extract_local_forwards_from_ssh_command(
    command: str,
) -> tuple[str | None, dict[int, tuple[str, int]]]:
    """Extract SSH target and local forwards from a process command line."""
    try:
        args = shlex.split(command)
    except ValueError:
        return None, {}

    target = next((token for token in reversed(args) if not token.startswith("-")), None)
    forwards: dict[int, tuple[str, int]] = {}

    index = 0
    while index < len(args):
        token = args[index]
        spec: str | None = None
        if token == "-L" and index + 1 < len(args):
            spec = args[index + 1]
            index += 2
        elif token.startswith("-L") and token != "-L":
            spec = token[2:]
            index += 1
        else:
            index += 1

        if spec is None:
            continue

        parsed = parse_local_forward_spec(spec)
        if parsed is None:
            continue
        local_port, remote_host, remote_port = parsed
        forwards[local_port] = (remote_host, remote_port)

    return target, forwards


def is_process_alive(pid: int | None) -> bool:
    """Check whether a process exists for the current user."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_registered_sessions(
    session_dir: Path = SESSION_STATE_DIR,
) -> dict[int, TunnelRecord]:
    """Load live tunnel state emitted by launch_and_tunnel.sh."""
    records: dict[int, TunnelRecord] = {}
    if not session_dir.exists():
        return records

    for path in sorted(session_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        owner_pid_raw = payload.get("owner_pid")
        owner_pid = owner_pid_raw if isinstance(owner_pid_raw, int) else None
        if owner_pid is not None and not is_process_alive(owner_pid):
            continue

        local_port = payload.get("local_port")
        if not isinstance(local_port, int):
            continue

        base_url = payload.get("base_url") or f"http://127.0.0.1:{local_port}/v1"
        record = TunnelRecord(
            local_port=local_port,
            base_url=base_url,
            job_id=str(payload["job_id"]) if payload.get("job_id") else None,
            requested_model_name=payload.get("requested_model_name"),
            ssh_target=payload.get("ssh_target"),
            remote_server_host=payload.get("remote_server_host"),
            remote_server_ip=payload.get("remote_server_ip"),
            remote_server_port=payload.get("remote_server_port"),
            owner_pid=owner_pid,
            state_path=str(path),
            sources=["state"],
        )
        records[local_port] = record

    return records


def discover_ssh_tunnels() -> dict[int, TunnelRecord]:
    """Discover active local SSH tunnels from system listeners."""
    records: dict[int, TunnelRecord] = {}

    if shutil.which("ss"):
        output = run_command(["ss", "-ltnp"]) or ""
        for line in output.splitlines():
            match = SS_LISTEN_RE.search(line)
            if match is None:
                continue
            local_port = int(match.group("port"))
            ssh_pid = int(match.group("pid"))
            command = run_command(["ps", "-o", "args=", "-p", str(ssh_pid)]) or ""
            ssh_target, forwards = extract_local_forwards_from_ssh_command(command)
            remote = forwards.get(local_port)
            if remote is None:
                continue
            remote_host, remote_port = remote
            records[local_port] = TunnelRecord(
                local_port=local_port,
                base_url=f"http://127.0.0.1:{local_port}/v1",
                ssh_target=ssh_target,
                remote_server_host=remote_host,
                remote_server_port=remote_port,
                ssh_pid=ssh_pid,
                sources=["listener"],
            )
        if records:
            return records

    if shutil.which("lsof"):
        output = run_command(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"]) or ""
        for line in output.splitlines():
            match = LSOF_LISTEN_RE.match(line.strip())
            if match is None:
                continue
            local_port = int(match.group("port"))
            ssh_pid = int(match.group("pid"))
            command = run_command(["ps", "-o", "args=", "-p", str(ssh_pid)]) or ""
            ssh_target, forwards = extract_local_forwards_from_ssh_command(command)
            remote = forwards.get(local_port)
            if remote is None:
                continue
            remote_host, remote_port = remote
            records[local_port] = TunnelRecord(
                local_port=local_port,
                base_url=f"http://127.0.0.1:{local_port}/v1",
                ssh_target=ssh_target,
                remote_server_host=remote_host,
                remote_server_port=remote_port,
                ssh_pid=ssh_pid,
                sources=["listener"],
            )

    return records


def merge_tunnel_records(
    preferred: dict[int, TunnelRecord],
    discovered: dict[int, TunnelRecord],
) -> dict[int, TunnelRecord]:
    """Merge state-backed records with listener discovery."""
    merged = {
        port: TunnelRecord(**asdict(record))
        for port, record in preferred.items()
    }
    for port, record in discovered.items():
        if port not in merged:
            merged[port] = TunnelRecord(**asdict(record))
            continue

        current = merged[port]
        for field_name in (
            "job_id",
            "requested_model_name",
            "ssh_target",
            "remote_server_host",
            "remote_server_ip",
            "remote_server_port",
            "owner_pid",
            "state_path",
            "ssh_pid",
        ):
            current_value = getattr(current, field_name)
            if current_value in (None, ""):
                setattr(current, field_name, getattr(record, field_name))
        for source in record.sources:
            if source not in current.sources:
                current.sources.append(source)
    return merged


def probe_health(base_url: str) -> bool | None:
    """Return True if the local health endpoint is reachable."""
    url = f"{base_url.removesuffix('/v1')}/health"
    try:
        with request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return response.status == 200
    except Exception:
        return None


def probe_model_names(base_url: str) -> tuple[list[str], str | None]:
    """Query /v1/models to get the exact model ids being served."""
    models_url = f"{base_url}/models"
    try:
        with request.urlopen(models_url, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except Exception as exc:
        return [], str(exc)

    data = payload.get("data", [])
    model_names = [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]
    return [str(name) for name in model_names], None


def enrich_records(records: dict[int, TunnelRecord]) -> dict[int, TunnelRecord]:
    """Probe each record locally for health and served model ids."""
    record_list = list(records.values())
    if not record_list:
        return records

    max_workers = min(8, len(record_list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_enrich_record, record_list))
    return records


def build_validation_environment(record: TunnelRecord) -> list[str]:
    """Return shared shell environment assignments for validation commands."""
    model_name = record.served_model_names[0] if record.served_model_names else (
        record.requested_model_name or "<MODEL_NAME>"
    )
    return [
        f"BASE_URL={shlex.quote(record.base_url)}",
        f"MODEL={shlex.quote(model_name)}",
    ]


def build_validation_commands(record: TunnelRecord) -> list[ValidationCommand]:
    """Return named validation commands for a given tunnel."""
    commands = [
        ValidationCommand("h", "Health", 'curl -fsS "${BASE_URL%/v1}/health"'),
        ValidationCommand("m", "Models", 'curl -fsS "$BASE_URL/models" | python3 -m json.tool'),
        ValidationCommand("t", "Metrics", 'curl -fsS "${BASE_URL%/v1}/metrics" | head'),
        ValidationCommand(
            "c",
            "Completion",
            (
                'curl -fsS "$BASE_URL/completions" '
                '-H "Content-Type: application/json" '
                '-d "{\"model\":\"$MODEL\",\"prompt\":\"Tell a short joke about GPUs in 2 or 3 sentences.\",\"max_tokens\":96,\"temperature\":0.7}" '
                "| python3 -m json.tool"
            ),
        ),
    ]
    if record.job_id:
        commands.append(
            ValidationCommand("s", "Vec-Inf Status", f"vec-inf status {shlex.quote(record.job_id)}")
        )
    return commands


def pick_records(
    records: dict[int, TunnelRecord],
    port: int | None = None,
    job_id: str | None = None,
) -> list[TunnelRecord]:
    """Filter records by port or job id."""
    selected = list(records.values())
    if port is not None:
        selected = [record for record in selected if record.local_port == port]
    if job_id is not None:
        selected = [record for record in selected if record.job_id == job_id]
    return sorted(selected, key=lambda record: record.local_port)


def choose_single_record(
    records: dict[int, TunnelRecord],
    port: int | None = None,
    job_id: str | None = None,
) -> TunnelRecord:
    """Resolve a single record for show/curls commands."""
    selected = pick_records(records, port=port, job_id=job_id)
    if not selected:
        raise SystemExit("No matching active tunnel found.")
    if len(selected) > 1:
        raise SystemExit("More than one tunnel matched. Pass --port or --job-id.")
    return selected[0]


def render_table(records: list[TunnelRecord]) -> str:
    """Render a compact plain-text status table."""
    headers = ["PORT", "JOB", "HEALTH", "MODEL", "BASE URL", "SOURCE"]
    rows = []
    for record in records:
        health = (
            "ok"
            if record.health_ok is True
            else "down"
            if record.health_ok is None and record.probe_error
            else "unknown"
        )
        rows.append(
            [
                str(record.local_port),
                record.job_id or "-",
                health,
                record.display_model_name(),
                record.base_url,
                "+".join(record.sources),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def build_curl_commands(record: TunnelRecord) -> list[str]:
    """Return canned curl commands for a given tunnel."""
    return build_validation_environment(record) + [
        command.command for command in build_validation_commands(record)
    ]


def _enrich_record(record: TunnelRecord) -> None:
    """Probe a single record in a worker thread."""
    try:
        record.health_ok = probe_health(record.base_url)
        model_names, probe_error = probe_model_names(record.base_url)
        record.served_model_names = model_names
        record.probe_error = probe_error
    except Exception as exc:
        record.health_ok = None
        record.served_model_names = []
        record.probe_error = str(exc)


def print_show(record: TunnelRecord) -> None:
    """Print a detailed single-tunnel summary."""
    print(f"Local port:      {record.local_port}")
    print(f"Base URL:        {record.base_url}")
    print(f"Served model(s): {record.display_model_name()}")
    print(f"Requested model: {record.requested_model_name or '-'}")
    print(f"Job ID:          {record.job_id or '-'}")
    print(
        "Health:          "
        + (
            "ok"
            if record.health_ok is True
            else "unknown"
            if record.health_ok is None
            else "down"
        )
    )
    print(f"SSH target:      {record.ssh_target or '-'}")
    if record.remote_server_host or record.remote_server_port:
        remote_host = record.remote_server_host or record.remote_server_ip or "-"
        remote_port = record.remote_server_port or "-"
        print(f"Remote server:   {remote_host}:{remote_port}")
    if record.probe_error:
        print(f"Probe error:     {record.probe_error}")
    print("")
    print("Validation commands:")
    for command in build_curl_commands(record):
        print(command)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect active local vec-inf tunnels, resolve exact served model ids, "
            "and print validation curl commands."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=("status", "show", "curls"),
        help="Operation to perform",
    )
    parser.add_argument("--port", type=int, help="Filter by local port")
    parser.add_argument("--job-id", type=str, help="Filter by Slurm job id")
    parser.add_argument("--json", action="store_true", help="Output JSON for status/show")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the tunnel helper CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    registered = load_registered_sessions()
    discovered = discover_ssh_tunnels()
    records = enrich_records(merge_tunnel_records(registered, discovered))

    if args.command == "status":
        selected = pick_records(records, port=args.port, job_id=args.job_id)
        if args.json:
            print(json.dumps([asdict(record) for record in selected], indent=2))
            return 0
        if not selected:
            print("No active local vec-inf tunnels found.", file=sys.stderr)
            return 1
        print(render_table(selected))
        return 0

    record = choose_single_record(records, port=args.port, job_id=args.job_id)
    if args.command == "curls":
        for command in build_curl_commands(record):
            print(command)
        return 0

    if args.json:
        print(json.dumps(asdict(record), indent=2))
    else:
        print_show(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
