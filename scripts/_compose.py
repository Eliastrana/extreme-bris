"""Compose a training config the way anemoi-training would, outside its CLI.

WHY THIS IS NOT JUST hydra.compose. The configs here name group defaults like
`training: ensemble` and `model: graphtransformer_ens`, and those groups live
inside the installed anemoi-training package, not in this repo. The anemoi CLI
puts that package on Hydra's search path before composing. A bare
initialize_config_dir does not, so composing by hand fails with

    Could not find 'training/ensemble'

which looks like a broken config and is not one. The config is fine; the
search path was short.

The package directory is discovered from the imported module rather than
written down as pkg://something, because the layout has moved between anemoi
versions and a wrong literal fails the same way as a missing search path.
"""

from __future__ import annotations

from pathlib import Path


def anemoi_config_dirs() -> list[Path]:
    """Directories inside anemoi-training that hold Hydra config groups."""
    import anemoi.training

    base = Path(anemoi.training.__file__).resolve().parent
    found = [d for d in (base / "config", base / "configs") if d.is_dir()]
    if not found:
        listing = ", ".join(sorted(p.name for p in base.iterdir() if p.is_dir()))
        raise SystemExit(
            f"no config directory inside {base}.\n"
            f"Directories there: {listing}\n"
            "This anemoi-training lays its configs out differently than assumed; "
            "point _compose.anemoi_config_dirs at the right one."
        )
    return found


def compose(config_dir: Path | str, config_name: str, overrides: list[str] | None = None):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    search = ",".join(f"file://{d}" for d in anemoi_config_dirs())
    args = [f"hydra.searchpath=[{search}]", *(overrides or [])]

    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()),
                               version_base=None):
        return hydra_compose(config_name=config_name, overrides=args)
