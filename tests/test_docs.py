# ruff: noqa: D100, D103
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import risb

api_dir = Path(__file__).parent.parent / "docs" / "api"


def test_every_public_object_is_in_the_api_docs() -> None:
    documented_modules = {
        module
        for page in api_dir.glob("*.rst")
        for module in re.findall(
            r"^\.\. automodule:: (\S+)$", page.read_text(), re.MULTILINE
        )
    }
    exported: set[object] = set()
    for module_name in documented_modules:
        module = importlib.import_module(module_name)
        exported.update(
            getattr(module, name) for name in getattr(module, "__all__", [])
        )

    missing = []
    for info in pkgutil.walk_packages(risb.__path__, prefix="risb."):
        if info.name in documented_modules or any(
            part.startswith("_") for part in info.name.split(".")
        ):
            continue
        for name, obj in vars(importlib.import_module(info.name)).items():
            is_public = not name.startswith("_") and (
                inspect.isclass(obj) or inspect.isfunction(obj)
            )
            if is_public and obj.__module__ == info.name and obj not in exported:
                missing.append(f"{info.name}.{name}")

    assert not missing, (
        f"Not in the API docs: {missing}. Add each to the __all__ of a package "
        "documented in docs/api/, or give its module a page there."
    )
