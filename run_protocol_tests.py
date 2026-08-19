"""Run the 23 release tests without requiring pytest to be installed.

This direct PyCharm runner supports the small pytest.raises/tmp_path subset used
by the frozen tests. Installing pytest and running the normal test command is
still recommended for development.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import tempfile
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class Raises:
    def __init__(self, exception_type, match: str | None = None):
        self.exception_type = exception_type
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, _traceback):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exception_type.__name__} was not raised")
        if not issubclass(exc_type, self.exception_type):
            return False
        if self.match and not re.search(self.match, str(exc_value)):
            raise AssertionError(
                f"Exception message {exc_value!r} does not match {self.match!r}"
            )
        return True


def install_pytest_subset() -> None:
    module = types.ModuleType("pytest")
    module.raises = lambda exception_type, match=None: Raises(exception_type, match)
    sys.modules["pytest"] = module


def load_module(path: Path):
    name = "release_test_" + path.stem + "_" + str(abs(hash(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    install_pytest_subset()
    test_files = [
        ROOT / "tests" / "test_revision_pipeline.py",
        ROOT / "tests" / "test_frozen_feature_configuration.py",
        ROOT / "route_a" / "tests" / "test_route_a.py",
    ]
    passed = 0
    for path in test_files:
        module = load_module(path)
        for name in sorted(value for value in dir(module) if value.startswith("test_")):
            function = getattr(module, name)
            parameters = inspect.signature(function).parameters
            if not parameters:
                function()
            elif list(parameters) == ["tmp_path"]:
                with tempfile.TemporaryDirectory() as directory:
                    function(Path(directory))
            else:
                raise TypeError(f"Unsupported test fixture in {path.name}::{name}: {parameters}")
            passed += 1
            print(f"[pass] {path.name}::{name}")
    if passed != 23:
        raise AssertionError(f"Expected 23 tests, ran {passed}")
    print(f"All {passed} release tests passed")


if __name__ == "__main__":
    main()
