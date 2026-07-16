from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

import jinja2
from jinja2.sandbox import SandboxedEnvironment
from rattler_build.jinja_config import JinjaConfig
from rattler_build.render import render_context as _render_context
from rattler_build.tool_config import PlatformConfig
from ruamel.yaml.scalarstring import DoubleQuotedScalarString, SingleQuotedScalarString

from rattler_build_conda_compat.jinja.filters import _bool, _split, _version_to_build_string
from rattler_build_conda_compat.jinja.objects import (
    _stub_compatible_pin,
    _stub_match,
    _stub_subpackage_pin,
    _StubEnv,
)
from rattler_build_conda_compat.jinja.utils import _MissingUndefined
from rattler_build_conda_compat.loader import load_yaml

if TYPE_CHECKING:
    from collections.abc import Mapping


class RecipeWithContext(TypedDict, total=False):
    context: dict[str, str]


def jinja_env(variant_config: Mapping[str, str] | None = None) -> SandboxedEnvironment:
    """
    Create a `rattler-build` specific Jinja2 environment with modified syntax.
    Target platform, build platform, and mpi are set to linux-64 by default.
    """

    env = SandboxedEnvironment(
        variable_start_string="${{",
        variable_end_string="}}",
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=jinja2.select_autoescape(default_for_string=False),
        undefined=_MissingUndefined,
    )

    env_obj = _StubEnv()

    # inject rattler-build recipe functions in jinja environment
    if not variant_config:
        variant_config = {"target_platform": "linux-64", "build_platform": "linux-64", "mpi": "mpi"}

    extra_vars = {}
    target_platform = variant_config.get("target_platform", "linux-64")
    if target_platform != "noarch":
        # set `linux` / `win`
        platform, arch = target_platform.split("-")
        extra_vars[platform] = True
        if arch == "64":
            extra_vars["x86_64"] = True
        elif arch == "32":
            extra_vars["x86"] = True
        else:
            extra_vars[arch] = True

    if target_platform.startswith("win"):
        extra_vars["unix"] = False
    else:
        extra_vars["unix"] = True

    env.globals.update(
        {
            "compiler": lambda x: x + "_compiler_stub",
            "stdlib": lambda x: x + "_stdlib_stub",
            "pin_subpackage": _stub_subpackage_pin,
            "pin_compatible": _stub_compatible_pin,
            "cdt": lambda *args, **kwargs: "cdt_stub",  # noqa: ARG005
            "env": env_obj,
            "match": _stub_match,
            "is_unix": lambda x: not x.startswith("win"),
            "is_win": lambda x: x.startswith("win"),
            "is_linux": lambda x: x.startswith("linux"),
            **extra_vars,
            **variant_config,
        }
    )

    # inject rattler-build recipe filters in jinja environment
    env.filters.update(
        {
            "version_to_buildstring": _version_to_build_string,
            "split": _split,
            "bool": _bool,
        }
    )
    return env


def load_recipe_context(context: dict[str, str], jinja_env: jinja2.Environment) -> dict[str, str]:
    """
    Load all string values from the context dictionary as Jinja2 templates.
    Use linux-64 as default target_platform, build_platform, and mpi.
    """

    # Process each key-value pair in the dictionary
    for key, value in context.items():
        # If the value is a string, render it as a template
        if isinstance(value, str):
            template = jinja_env.from_string(value)
            rendered_value = template.render(context)
            # In this repo, rumael.yaml is configured to return strings as special subtypes
            # depending on how the user specified them in the yaml. Two of the subtypes,
            # SingleQuotedScalarString and DoubleQuotedScalarString, correspond to strings
            # that are explicitly quoted in the yaml and thus are always string values in Python.
            # We skip the yaml inference for those types since it does not need to be done and
            # they are already strings. To properly do the yaml inference, we'd have
            # to requote the strings before passing them in.
            if type(value) in (SingleQuotedScalarString, DoubleQuotedScalarString):
                context[key] = rendered_value
            else:
                # We have to escape sequences like \n, \t, etc. because they would be
                # escaped if we wrote the rendered text to a yaml object. We don't directly
                # dump via yaml since that will cause more type errors due to things still
                # being strings (e.g., for an int 8 we have yaml.dump({"value": "8"}, fp)
                # which yields "value: '8'\n" which would then be read as a string.).
                # The sequence of calls `.encode("unicode_escape").decode("utf-8")`
                # escapes the escape sequences and then converts back to a string
                # from bytes, so we get "\n" -> "\\n". We then reverse the operations
                # if the output type is a string.
                _value = load_yaml(
                    "value: " + rendered_value.encode("unicode_escape").decode("utf-8")
                )["value"]
                if isinstance(_value, str):
                    _value = _value.encode("utf-8").decode("unicode_escape")
                context[key] = _value

    return context


def render_recipe_with_context(
    recipe_content: RecipeWithContext, variant_config: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """
    Render the recipe using known values from context section.
    Unknown values are not evaluated and are kept as it is.
    Target platform, build platform, and mpi are set to linux-64 by default.

    Examples:
    ---
    ```python
    >>> from pathlib import Path
    >>> from rattler_build_conda_compat.loader import load_yaml
    >>> recipe_content = load_yaml((Path().resolve() / "tests" / "data" / "eval_recipe_using_context.yaml").read_text())
    >>> evaluated_context = render_recipe_with_context(recipe_content)
    >>> assert "my_value-${{ not_present_value }}" == evaluated_context["build"]["string"]
    >>>
    ```
    """
    rendered = _render_context(recipe_content, _context_jinja_config(variant_config))
    return _apply_lint_stubs(rendered)  # type: ignore[no-any-return]


def _context_jinja_config(variant_config: Mapping[str, str] | None) -> JinjaConfig:
    """Build a lenient `JinjaConfig` from a conda-smithy variant mapping."""
    variant = dict(variant_config) if variant_config else {}
    target_platform = str(variant.get("target_platform", "linux-64"))
    platform = PlatformConfig(target_platform=target_platform)
    return JinjaConfig(platform=platform, variant=variant, allow_undefined=True)


# rattler-build's engine leaves build-phase helper calls (`compiler`,
# `pin_subpackage`, ...) verbatim during context rendering. conda-smithy expects
# these in a stubbed form, so we map the surviving `${{ ... }}` calls to the
# same stubs the pure-python environment used to produce.
_STUB_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"""\$\{\{\s*compiler\(\s*['"]([^'"]+)['"][^}]*\)\s*\}\}"""), r"\1_compiler_stub"),
    (re.compile(r"""\$\{\{\s*stdlib\(\s*['"]([^'"]+)['"][^}]*\)\s*\}\}"""), r"\1_stdlib_stub"),
    (
        re.compile(r"""\$\{\{\s*pin_subpackage\(\s*['"]([^'"]+)['"][^}]*\)\s*\}\}"""),
        r"subpackage_pin \1",
    ),
    (
        re.compile(r"""\$\{\{\s*pin_compatible\(\s*['"]([^'"]+)['"][^}]*\)\s*\}\}"""),
        r"compatible_pin \1",
    ),
    (re.compile(r"""\$\{\{\s*cdt\([^}]*\)\s*\}\}"""), "cdt_stub"),
    (
        re.compile(r"""\$\{\{\s*env\.exists\(\s*['"]([^'"]+)['"][^}]*\)\s*\}\}"""),
        r'env_exists_"\1" ',
    ),
    (re.compile(r"""\$\{\{\s*env\.get\(\s*['"]([^'"]+)['"][^}]*\)\s*\}\}"""), r'env_"\1" '),
]


def _apply_lint_stubs(obj: Any) -> Any:  # noqa: ANN401
    """Recursively map verbatim recipe helper calls to conda-smithy stubs."""
    if isinstance(obj, str):
        for pattern, replacement in _STUB_PATTERNS:
            obj = pattern.sub(replacement, obj)
        return obj
    if isinstance(obj, dict):
        return {key: _apply_lint_stubs(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_apply_lint_stubs(value) for value in obj]
    return obj
