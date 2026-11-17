"""The interactive report: one ``index.html`` that opens offline, plus each run's asset tree.

``index.html`` inlines the styles, the scripts (Plotly included) and one ``JM.registerRun(...)``
per run. Field and metric files stay as separate ``<script>`` files under ``runs/<run>/`` and are
loaded on demand, so the page opens fast however many fields a run has.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from junctionmapper.errors import ConfigError

HERE = Path(__file__).parent
STATIC = HERE / "static"
SOURCE_RUN = "r1"   # the run id of the asset tree inside a run directory

# Plotly's "Produced with Plotly" logo is the only URL it would ever put on the page. The logo is
# switched off in app.js; the link is neutralised too so the page carries no external reference.
_PLOTLY_LOGO = ('t.href="https://plotly.com/"', 't.href="#"')


def _script(text: str) -> str:
    """JavaScript made safe to sit inside an inline ``<script>`` element."""
    return text.replace("</script", "<\\/script")


def _plotly() -> str:
    text = (STATIC / "plotly-cartesian.min.js").read_text(encoding="utf-8")
    if text.count(_PLOTLY_LOGO[0]) != 1:
        raise RuntimeError("vendored Plotly changed: logo link not found exactly once")
    return _script(text.replace(*_PLOTLY_LOGO))


def _json(obj) -> str:
    """JSON for an inline script: no ``<`` survives, so no ``</script>`` or ``<!--``."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def _check(runs: list[tuple[Path, str]], out: Path) -> None:
    seen: set[str] = set()
    for _, label in runs:
        if label in seen:
            raise ConfigError(f"duplicate run label: {label!r}")
        seen.add(label)
    for k, (run_dir, _) in enumerate(runs, start=1):
        if not (run_dir / "report" / "runs" / SOURCE_RUN).is_dir() or not (run_dir / "run.json").is_file():
            raise ConfigError(f"{run_dir}: no report assets (run.json, report/runs/{SOURCE_RUN}/); "
                              "re-run it with this version of junctionmapper")
        if k > 1 and _is_own_report(run_dir, out):
            raise ConfigError(f"{out} is the report folder of {run_dir}; list that run as the first run "
                              "or write the report elsewhere")


def _is_own_report(run_dir: Path, out: Path) -> bool:
    return (run_dir / "report").resolve() == out.resolve()


def build_dashboard(runs: list[tuple[Path, str]], out: Path) -> Path:
    """Write ``out/index.html`` and ``out/runs/r1..rN/`` for ``runs`` = [(run directory, label)].

    Run ids are ``r1``, ``r2``, … in argument order. A run's asset tree is copied unless ``out`` is
    that run's own ``report/`` folder and it is the first run. Returns the path of ``index.html``.
    """
    from junctionmapper.report.data import run_summary   # data imports build; keep this lazy

    runs = [(Path(d), str(label)) for d, label in runs]
    out = Path(out)
    _check(runs, out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for k, (run_dir, label) in enumerate(runs, start=1):
        rid = f"r{k}"
        if not (k == 1 and _is_own_report(run_dir, out)):
            shutil.copytree(run_dir / "report" / "runs" / SOURCE_RUN, out / "runs" / rid,
                            dirs_exist_ok=True)
        summaries.append(run_summary(run_dir, rid, label))

    env = Environment(loader=FileSystemLoader(HERE / "templates"), autoescape=False,
                      keep_trailing_newline=True)
    html = env.get_template("index.html").render(
        title=" vs ".join(s["label"] for s in summaries),
        css=(STATIC / "app.css").read_text(encoding="utf-8"),
        plotly=_plotly(),
        logic=_script((STATIC / "logic.js").read_text(encoding="utf-8")),
        app=_script((STATIC / "app.js").read_text(encoding="utf-8")),
        runs="".join(f"JM.registerRun({_json(s)});\n" for s in summaries),
    )
    path = out / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
