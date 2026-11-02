"""What a module needs, and nothing else. Stdlib-only (plus contract.py and paths.py).

A module is a folder  modules/<name>/run.py  that does:

    from sourcediff.module_api import run_module
    def run(ctx):            # ctx = Ctx, see below
        ...                  # read ctx.input_path, other results, the library
        return ModuleResult(module=NAME, evidence=[...])
    if __name__ == "__main__":
        run_module(NAME, run)

run_module() handles the boring part: load ingest.json, time the call, stamp the input
hash, validate against the contract, and write results/<name>.json. If the module raises,
it writes status="error" instead of crashing the pipeline. The runner never imports a
module; it launches run.py as a subprocess, so a module can live in its own virtualenv
(registry.json -> "python").
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from sourcediff import paths
from sourcediff.contract import Evidence, Gate, ModuleResult


@dataclass
class Ctx:
    case_dir: Path
    ingest: dict

    @property
    def door(self) -> str:
        return self.ingest["door"]

    @property
    def flags(self) -> dict:
        return self.ingest.get("flags", {})

    @property
    def sha256(self) -> str:
        return self.ingest.get("sha256", "")

    @property
    def input_path(self) -> Path:
        return self.case_dir / "input" / self.ingest["filename"]

    def path(self, rel: str) -> Path:
        return self.case_dir / rel

    def read_json(self, rel: str) -> dict:
        return paths.load_json(self.case_dir / rel)

    def write_json(self, rel: str, obj) -> str:
        paths.write_json(self.case_dir / rel, obj)
        return rel


def load_ctx(case_dir) -> Ctx:
    case_dir = Path(case_dir)
    return Ctx(case_dir=case_dir, ingest=paths.load_json(case_dir / "ingest.json"))


def skipped(module: str, reason: str, stub: bool = False) -> ModuleResult:
    return ModuleResult(module=module, status="skipped", reason=reason, stub=stub)


def genuine_range(module: str, metric: str):
    """The range the verdict engine will use. Modules show it; they never decide with it."""
    spec = paths.config("genuine_ranges").get(f"{module}.{metric}")
    return list(spec["range"]) if spec else None


def registry_entry(module: str) -> dict:
    for m in paths.config("registry")["modules"]:
        if m["name"] == module:
            return m
    raise KeyError(module)


# ---- stubs: a module that has not been built yet --------------------------------------
def build_result(module: str, door: str, data: dict, stub: bool = True) -> ModuleResult:
    """Turn a hand-written scenario fragment into a validated ModuleResult.

    data = {"evidence": [{"metric": ..., "value": ..., "timestamps": [...]}], "gate": {...}}
    Family and channel come from registry.json, so a scenario cannot mislabel a module.
    """
    reg = registry_entry(module)
    rows = []
    for e in data.get("evidence", []):
        rows.append(Evidence(
            module=module, door=door, family=reg["family"], channel=reg["channel"],
            metric=e["metric"], value=e["value"], unit=e.get("unit", ""),
            genuine_range=genuine_range(module, e["metric"]),
            timestamps=e.get("timestamps", []),
            supports="neutral", confidence=e.get("confidence", 0.5),
            limits=e.get("limits", []), stub=stub,
        ).validate())
    gate = None
    if "gate" in data:
        g = dict(data["gate"])
        g.setdefault("library", "trusted")
        gate = Gate.from_dict(g)
    return ModuleResult(module=module, status="ok", stub=stub, version="scenario",
                        evidence=rows, gate=gate).validate()


def in_scenario(ctx: Ctx) -> bool:
    """True when the case folder carries hand-written evidence (tests, demos). Real modules
    step aside in that case, so a dummy input file never reaches a real model."""
    return (ctx.case_dir / "scenario").exists()


def stub_run(ctx: Ctx, module: str) -> ModuleResult:
    """Default body of every unbuilt module. With <case>/scenario/<module>.json present it
    replays that hand-written result (used by tests and demos); otherwise it reports that
    it did not run, which the verdict engine treats as 'not run', never as 'agrees'."""
    f = ctx.case_dir / "scenario" / f"{module}.json"
    if not f.exists():
        return skipped(module, "stub: not built yet", stub=True)
    return build_result(module, ctx.door, paths.load_json(f), stub=True)


def run_module(name: str, fn, argv=None) -> int:
    argv = argv or sys.argv
    if len(argv) < 2:
        print(f"usage: python run.py <case_dir>   (module {name})", file=sys.stderr)
        return 2
    case_dir = Path(argv[1])
    t0 = time.time()
    ctx = load_ctx(case_dir)
    try:
        res = fn(ctx)
        res.module = name
        res.input_sha256 = ctx.sha256
        res.runtime_s = round(time.time() - t0, 3)
        res.validate()
    except Exception as exc:  # a broken module must not take the pipeline down
        traceback.print_exc()
        res = ModuleResult(module=name, status="error", input_sha256=ctx.sha256,
                           reason=f"{type(exc).__name__}: {exc}",
                           runtime_s=round(time.time() - t0, 3))
    paths.write_json(case_dir / "results" / f"{name}.json", res.to_dict())
    print(f"[{name}] {res.status}" + (f" ({res.reason})" if res.reason else ""))
    return 0
