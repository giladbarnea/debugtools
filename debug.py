"""
exec(compile(open("/home/gilad/debug.py").read(), "/home/gilad/debug.py", 'exec'))
exec(compile((Path.home() / Path('debug.py')).open().read(), "/home/gilad/debug.py", 'exec'))
exec(compile(Path(debug := os.getenv("PYTHONDEBUGFILE", Path(os.getenv("HOME")) / 'debug.py')).open().read(), debug, 'exec'))
# %run '/home/gilad/debug.py'

# DEBUGFILE_FORCE_INSTALL, DEBUGFILE_RICH_TB, DEBUGFILE_PATCH_PRINT, DEBUGFILE_LOADED
"""

# python main.py 2>&1 | python -m pygments -f terminal16m -P style=monokai -l py3tb
# python main.py 2>&1 | pym39 rich.syntax -x py3tb -t dracula -b 'rgb(0,0,0)' --indent-guides --line-numbers --wrap -
# sizecustomize.py: https://rich.readthedocs.io/en/stable/traceback.html#automatic-traceback-handler
from __future__ import annotations

import types
from typing import Callable, ForwardRef

# Guaranteed to exist:
_print: Callable
pp: Callable
#

termwidth: int
rich: ForwardRef("rich")
con: ForwardRef("Console")
startipy: Callable
builtin: Callable[[..., str | None, bool | None], None]
builtin_variable: Callable[[Callable[..., ...], str | None], None]
getcaller: Callable[[int | None], ForwardRef("inspect.FrameInfo")]
cleanstack: Callable[[int | None, tuple[str] | None], list[ForwardRef("inspect.FrameInfo")]]
getmeths: Callable
getprops: Callable
inquire: Callable
varinfo: Callable

pformat: Callable
pretty: Callable
pprint: Callable
pretty_inst: Callable
pretty_sig: Callable
pretty_sig2: Callable
pretty_type: Callable
Reprer: ForwardRef("Reprer")
reprer: Callable
what: Callable
whatt: Callable
whattt: Callable
whatttt: Callable
who: Callable

import os
import sys


def envbool(key: str, sysarg: str = None) -> bool:
    """Precedence to cmdline args"""
    if sysarg:
        for arg in sys.argv[1:]:
            if arg == sysarg:
                return True
            if arg.startswith(sysarg) and "=" in arg:
                *_, val = arg.partition("=")
                return val.lower() in ("1", "true", "yes")

    return os.getenv(key, "").lower() in ("1", "true", "yes")


def init_debug_module(
    force_install: str = "",
    rich_tb: bool = False,
    patch_print: bool = False,
):
    if envbool("DEBUGFILE_LOADED"):
        _print("Not loading debug.py because DEBUGFILE_LOADED", file=__import__("sys").stderr)
        return
    print("Loading debug.py")
    import builtins
    from copy import deepcopy

    _original_builtins_names = set(dir(builtins))
    builtins._print = deepcopy(print)  # many things depend on the existence of this
    import ast
    import functools
    import inspect
    import io
    import os
    import re
    import sys
    from contextlib import suppress
    from pathlib import Path
    from textwrap import wrap
    from types import ModuleType
    from typing import Any, Iterable, Literal, NoReturn, TypeVar, overload

    builtins.inspect = inspect
    builtins.os = os
    builtins.Path = Path
    debug_module = sys.modules[__name__]

    OBJ_RE = re.compile(r"<(?:[\w\d<]+\.)*([\w\d>]+)[ object]* at (0x[\w\d]{9,12})>")
    TYPE_RE = re.compile(r"<class '(?:[\w\d<>]+\.)*([\w\d]+)'>")  # why not just just type.__name__?
    WHITESPACE_RE = re.compile(r"\s+")
    COLOR_RE = re.compile(r"(\x1b\[(?:\d;?)*m)")
    COMMA_SEP_VALUE_RE = re.compile(r"[^,]+")
    RUN_BY_HUMAN = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()  # colorlogs uses stdout

    Obj = TypeVar("Obj")
    Ret = TypeVar("Ret")

    def builtin(obj: Obj, name=None, verbose=False) -> Obj:
        """Set 'obj' as a builtin, with optional 'name' (defaults to obj.__name__)"""
        if not name:
            if not hasattr(obj, "__name__"):
                con.log(f"[warn]builtin({obj=!r}): obj has no __name__. Specify name=...; skipping")
                return obj
            name = obj.__name__
        obj.__name__ = name
        obj.__qualname__ = name
        # if verbose:
        #     getattr(builtins, "_print", builtins.print)(f"builtin({name})")
        setattr(builtins, name, obj)
        return obj

    builtin(builtin)
    builtin(debug_module, "debug")

    @builtin
    def builtin_variable(func: Callable[[], Ret], name: str = None, verbose=False) -> Ret:
        """Sets `builtins.name = func()` right away at definition. Removes '_' suffix from name."""
        # for better @wraps: https://github.com/micheles/decorator/blob/ad013a2c1ad7969963acf3dea948632be387f5a0/src/decorator.py#L277
        rv = func()
        if not name:
            if func.__name__.endswith("_"):
                name = func.__name__[:-1]
            else:
                name = func.__name__
        # assert not hasattr(builtins, name), name
        if verbose:
            _print(f"builtin_variable({name})")
        for attrname, attrval in {
            "__doc__": func.__doc__,
            "__qualname__": name,
            "__name__": name,
        }.items():
            with suppress(AttributeError):
                setattr(rv, attrname, attrval)
        setattr(builtins, name, rv)
        return rv

    def setup_rich(*, _rich_tb: bool):
        """DEBUGFILE_RICH_TB or --rich-tb"""
        if _rich_tb:
            print("Patching rich traceback")
            # noinspection PyUnresolvedReferences
            from rich.traceback import install

            # from rich.pretty import install
            # install(console=con,indent_guides=True,expand_all=True)
            install(extra_lines=5, show_locals=True)

    def setup_ipython(*, force_install_ipython: bool):
        """'ipython' in DEBUGFILE_FORCE_INSTALL"""
        try:
            # noinspection PyUnresolvedReferences
            from IPython import start_ipython
        except ModuleNotFoundError:
            if force_install_ipython:
                pip_install_command = f"{sys.executable!r} -m pip install IPython"
                print(pip_install_command)
                os.system(pip_install_command)
            else:
                pass

    setup_rich(_rich_tb=rich_tb)
    setup_ipython(force_install_ipython="ipython" in force_install)

    # *** Global (builtin) variables

    @builtin_variable
    def termwidth() -> int:
        width = None
        for fd in (
            sys.__stdin__.fileno(),
            sys.__stdout__.fileno(),
            sys.__stderr__.fileno(),
        ):
            try:
                width = os.get_terminal_size(fd)[0]
            except (AttributeError, ValueError, OSError):
                pass
            else:
                break

        if not width or width <= 80:
            columns = os.environ.get("COLUMNS")
            if columns and columns.isdigit() and int(columns) > 80:
                width = int(columns)

        if not width or width <= 80:
            width = 140
        return width

    @builtin_variable
    def rich() -> "rich":
        # TODO: should maybe finally set sys.modules['rich'] = rich?
        try:
            import rich
        except ModuleNotFoundError:
            print(f"{sys.executable!r} -m pip install rich")
            os.system(f"{sys.executable!r} -m pip install rich")
            import rich
        finally:
            assert sys.modules.get("rich")
            return rich

    @builtin_variable
    def con():
        from rich.console import Console
        from rich.theme import Theme

        PYCHARM_HOSTED = os.getenv("PYCHARM_HOSTED")

        theme = {
            "debug": "dim",
            "warn": "yellow",
            "warning": "yellow",
            "error": "red",
            "fatal": "bright_red",
            "success": "green",
            "good": "green",
            "prompt": "b bright_cyan",
            "title": "b bright_white",
        }
        # noinspection PyTypeChecker
        return Console(  # force_terminal=True,  # Run with Python Console
            log_time_format="[%d.%m.%Y][%T]",
            color_system="auto" if PYCHARM_HOSTED else "truecolor",
            width=termwidth,
            tab_size=2,
            height=max(os.getenv("LINES", 100), 100) if PYCHARM_HOSTED else None,
            file=sys.stdout if PYCHARM_HOSTED else sys.stderr,
            theme=Theme({**theme, **{k.upper(): v for k, v in theme.items()}}),
        )

    @builtin_variable
    def startipy():
        with suppress(ModuleNotFoundError):
            # noinspection PyUnresolvedReferences
            from IPython import start_ipython

            def startipy__(offset=1, **extra_globals):
                frame_info = getcaller(offset=offset)
                start_ipython(
                    argv=[],
                    user_ns={
                        **frame_info.frame.f_locals,
                        **globals(),
                        **extra_globals,
                    },
                )

            return startipy__

    # *** Inspection

    @builtin
    def cleanstack(context=3, *, exclude_filenames: tuple[str] = ()) -> list[inspect.FrameInfo]:
        """A wrapper for `inspect.stack(context)` that filters out irrelevant frames."""
        exclude_filenames += (
            "site-packages",
            "dist-packages",
            "python3",
            "jetbrains",
            "pycharm",
            ".vscode",
            "<input>",
            "<string>",
            "<ipython-input",
        )
        frame_infos: list[inspect.FrameInfo] = [
            frame_info
            for frame_info in inspect.stack(context)
            if __file__ not in (filename_lower := frame_info.filename.lower())
            and not any(name in filename_lower for name in exclude_filenames)
            and not ("debug" in filename_lower and frame_info.function != "cleanstack")
        ]
        return frame_infos

    @builtin
    def getcaller(offset=2) -> inspect.FrameInfo | None:
        # TODO: compare with inspect.stack(context=...), and traceback.extract_stack()
        currframe = inspect.currentframe()
        outer = inspect.getouterframes(currframe)
        try:
            frameinfo = outer[offset]
        except IndexError:
            con.log(f"[warn]IndexError: getcaller({offset}) | outer[{offset}] | {outer = }")
            return None
        return frameinfo

    class VarInfo:
        def __init__(
            self,
            arg_idx=0,
            value="__UNSET__",
            offset_or_frameinfo: int | inspect.FrameInfo = 2,
            *,
            include_file_name=True,
            include_function_name=True,
        ):
            # frameinfo
            if isinstance(offset_or_frameinfo, int):
                offset = offset_or_frameinfo
                frameinfo = getcaller(offset)
            else:
                frameinfo = offset_or_frameinfo

            # self.filename
            if include_file_name:
                filename = frameinfo.filename.split("/")[-1]
            else:
                filename = ""
            self.filename = filename

            # self.fnname
            if include_function_name:
                fnname = frameinfo.function + "(...)"
            else:
                fnname = ""
            self.fnname = fnname

            # self.argname
            self.argname = ""
            if frameinfo.code_context:
                ctx = frameinfo.code_context[0].strip()
                if more_contexts := frameinfo.code_context[1:]:
                    con.log(f"[debug]VarInfo() | {ctx = } | {more_contexts = }")

                argnames: list[str] = VarInfo._get_argnames(ctx)
                self.argname = argnames[arg_idx]

            elif value != "__UNSET__":
                f_locals = frameinfo.frame.f_locals
                for k, v in f_locals.items():
                    with suppress(TypeError):
                        if v is value:
                            self.argname = k

        @staticmethod
        def _unclosed_str(s: str) -> bool:
            stack = []
            for char in s:
                if char == "(" or char == "[" or char == "{":
                    stack.append(char)
                elif char == ")" or char == "]" or char == "}":
                    if len(stack) == 0:
                        return True
                    elif char == ")" and stack[-1] == "(":
                        stack.pop()
                    elif char == "]" and stack[-1] == "[":
                        stack.pop()
                    elif char == "}" and stack[-1] == "{":
                        stack.pop()
                    else:
                        return True
            if len(stack) > 0:
                return True
            else:
                return False

        @staticmethod
        def _get_argnames(context: str) -> list[str]:
            # noinspection PyUnresolvedReferences
            """
            >>> _get_argnames('(self, *args, **kwargs):')
            ['self', '*args', '**kwargs']
            """
            try:
                inside_parenthesis = context[context.index("(") + 1 : context.rindex(")")]
            except ValueError:
                con.log(f"[warn]VarInfo._get_argnames({context = !r}) | no (parenthesis), returning []")
                return []

            # 'foo, bar=(1)' -> 'foo, bar='
            inside_parenthesis = re.sub(r"\(.*\)", "", inside_parenthesis)
            matches = list(COMMA_SEP_VALUE_RE.finditer(inside_parenthesis))
            argnames: list[str] = []
            i = 0
            matches_len = len(matches)
            while i < matches_len:
                match = matches[i]
                group = match.group().strip()
                group = re.sub(r"\s*=.*", "", group)
                if VarInfo._unclosed_str(group):
                    # bug? always i+=2?
                    next_group = matches[i + 1].group()
                    if VarInfo._unclosed_str(next_group):
                        _merged = group + next_group
                        argnames.append(_merged)
                    else:
                        # bummer
                        argnames.append(group)
                        argnames.append(next_group)
                    i += 2
                    continue
                argnames.append(group)
                i += 1
            return argnames

        def __repr__(self):
            string = ""
            if self.filename:
                string += f"[{self.filename}]"
            if self.fnname:
                string += f"[{self.fnname}]"
            if self.argname:
                if string:
                    string = "\n" + string
                string += f"{self.argname}"
            return string

    @builtin
    def varinfo(
        arg_idx=0,
        value="__UNSET__",
        offset_or_frameinfo: int | inspect.FrameInfo = 2,
        *,
        include_file_name=True,
        include_function_name=True,
    ) -> str:
        """
        >>> def foo(bar):
        ...     print(varinfo())
        [rs_events_handler.py][__init__(...)] devices
        """

        # todo: inspect.getargvalues() and inspect.formatargvalues()
        def _unclosed(_s: str) -> bool:
            return _s.count("(") != _s.count(")") or _s.count("[") != _s.count("]") or _s.count("{") != _s.count("}")

        def _get_argnames(_ctx: str) -> list[str]:
            """
            >>> _get_argnames('(self, *args, **kwargs):')
            ['self', '*args', '**kwargs']
            """
            _inside_parenthesis = _ctx[_ctx.index("(") + 1 : _ctx.rindex(")")]

            # 'foo, bar=(1)' -> 'foo, bar='
            _inside_parenthesis = re.sub(r"\(.*\)", "", _inside_parenthesis)
            _matches = list(COMMA_SEP_VALUE_RE.finditer(_inside_parenthesis))
            _argnames = []
            i = 0
            _matches_len = len(_matches)
            while i < _matches_len:
                _match = _matches[i]
                _group = _match.group().strip()
                # 'bar =' -> 'bar'
                _group = re.sub(r"\s*=.*", "", _group)
                if _unclosed(_group):
                    # bug? always i+=2?
                    _next_group = _matches[i + 1].group()
                    if _unclosed(_next_group):
                        _merged = _group + _next_group
                        _argnames.append(_merged)
                    else:
                        # bummer
                        _argnames.append(_group)
                        _argnames.append(_next_group)
                    i += 2
                    continue
                _argnames.append(_group)
                i += 1
            return _argnames

        output = ""
        try:
            if isinstance(offset_or_frameinfo, int):
                offset = offset_or_frameinfo
                frameinfo = getcaller(offset)
            else:
                frameinfo = offset_or_frameinfo

            # con.log(f'[debug] varinfo() | {frameinfo = }')
            if include_file_name:
                filename = frameinfo.filename
                output += pretty(f'[dim][{filename.split("/")[-1]}][/dim]')

            if include_function_name:
                output += pretty(f"[dim][{frameinfo.function}(...)][/dim]")

            try_get_argname_from_locals = not bool(frameinfo.code_context)
            if frameinfo.code_context:
                ctx = frameinfo.code_context[0].strip()
                if more_contexts := frameinfo.code_context[1:]:
                    con.log(f"[debug] varinfo() | {ctx = } | {more_contexts = }")

                try:
                    argnames: list[str] = _get_argnames(ctx)
                except ValueError:
                    # con.log(f'[WARN] varinfo._get_argnames({ctx = !r}) | no (parenthesis), trying to get argnames from locals')
                    try_get_argname_from_locals = True
                else:
                    try:
                        output += argnames[arg_idx]
                    except IndexError:
                        con.log(f"[WARN] IndexError: varinfo() | argnames[{arg_idx}] | {argnames = }")
                        try_get_argname_from_locals = True

            if try_get_argname_from_locals and value != "__UNSET__":
                f_locals = frameinfo.frame.f_locals
                for k, v in f_locals.items():
                    with suppress(TypeError):
                        if v is value:
                            output += k
            return output
        except Exception:
            # from pdbpp import post_mortem; post_mortem()
            # con.log(f'[WARN] {e.__class__.__qualname__}: varinfo({arg_idx = !r}, {value = !r}, {frameinfo.code_context = !r}) | {e}')
            return output

    class Inspector:
        Constraint = Literal["regular", "private", "dunder"]
        skip_obj_props = (
            # '__module__',
            "__annotations__",
            "__builtins__",
            "__class__",
            "__dict__",
            "__doc__",
            "__globals__",
            "__weakref__",
            "_filesbymodname",
            "mod_dict",
            "modulesbyfile",
        )
        skip_obj_meths = (
            "__delattr__",
            "__dir__",
            "__enter__",
            "__eq__",
            "__exit__",
            "__format__",
            "__ge__",
            "__getattribute__",
            "__gt__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__le__",
            "__lt__",
            "__ne__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__repr__",
            "__setattr__",
            "__sizeof__",
            "__str__",
            "__subclasshook__",
        )

        @classmethod
        def _build_constraint(
            cls,
            _constraint: Constraint,
            _regular: bool,
            _private: bool,
            _dunder: bool,
        ) -> tuple[bool, bool, bool]:
            if _constraint == "regular":
                _regular = True
            elif _constraint == "private":
                _private = True
            elif _constraint == "dunder":
                _dunder = True
            else:
                con.log(
                    "[WARN] 'only' arg can be (an iterable composed of) 'regular', 'private' or"
                    " 'dunder'. returning as-is"
                )
            return _regular, _private, _dunder

        @classmethod
        def _build_constraints(cls, only: Constraint | tuple[Constraint]):
            """If `only` is None, this means no constraints: `regular = private = dunder = True` (show all).
            If it's a string (e.g. 'regular'), switch on `regular = True`, and switch off others (show only regular props).
            If it's a tuple (e.g. `('regular', 'private')`), switch on `regular = True, private = True` (don't show dunder).
            """

            if only is not None:
                regular = private = dunder = False
                if isinstance(only, str):
                    # noinspection PyTypeChecker
                    regular, private, dunder = cls._build_constraint(only, regular, private, dunder)
                else:  # iterable
                    for constraint in only:
                        regular, private, dunder = cls._build_constraint(constraint, regular, private, dunder)
            else:
                regular = private = dunder = True
            return regular, private, dunder

        @staticmethod
        def _is_method(obj) -> bool:
            # From pydevd_resolver.py:
            # if inspect.isroutine(attr) or isinstance(attr, MethodWrapperType):
            string = str(type(obj))
            return "method" in string or "wrapper" in string or "function" in string

        @classmethod
        def _cheapen(cls, obj, len_limit=10_000):
            if isinstance(obj, (str, bytes)):
                return shorten(obj, len_limit)

            if isinstance(obj, dict):
                cheap_dict = {}
                for k, v in obj.items():
                    if k in cls.skip_obj_props:
                        continue
                    cheap_dict[k] = cls._cheapen(v, len_limit)
                return cheap_dict

            if hasattr(obj, "__iter__") and type(obj) is not type:
                cheap_iterable = []
                for value in obj:
                    cheap_value = cls._cheapen(value, len_limit)
                    cheap_iterable.append(cheap_value)
                return type(obj)(cheap_iterable)

            if len(str(obj)) < len_limit:
                return obj

            try:
                newobj = type(obj)()
            except TypeError:
                newobj = object()

            for prop in dir(obj):
                cheap_prop = cls._cheapen(getattr(obj, prop), len_limit)
                setattr(newobj, prop, cheap_prop)

            return newobj

        @classmethod
        def _should_skip(
            cls,
            prop,
            regular: bool,
            private: bool,
            dunder: bool,
            *,
            include: Iterable,
            exclude: Iterable,
        ) -> bool:
            if exclude and prop in exclude:
                return True
            if include and prop not in include:
                return True
            if prop.startswith("__"):
                if not dunder:
                    return True
            elif prop.startswith("_"):
                if not private:
                    return True
            elif not regular:
                return True
            return False

    @builtin_variable
    def getprops() -> ForwardRef("GetProps"):
        """
        Args:
            values: Specify True to return a { prop: val } dict
            only: Either 'regular', 'private', 'dunder' or a tuple containing any of them
            exclude: A str or a tuple of strings specifying which props to ignore. `None` means not ignoring even object dunder props like __dict__, __globals__, etc.
            include: A str or a tuple of strings specifying which props to include. All others are ignored.
            cheapen_if_long: If True and len(str(obj)) > 10_000, recursively `skip_obj_props` in __dict__, `shorten` strings, etc.

        Returns:
            If `values` is True, a dict of { prop: val }, else a list of props.

        `exclude` and `include` are mutually exclusive.
        """

        class GetProps(Inspector):
            @overload
            def __call__(
                self,
                obj,
                values: Literal[False],
                *,
                only: Inspector.Constraint | tuple[Inspector.Constraint],
                exclude: None | str | tuple,
                include: None | str | tuple,
                cheapen_if_long: bool,
            ) -> list[str]: ...

            @overload
            def __call__(
                self,
                obj,
                values: Literal[True],
                *,
                only: Inspector.Constraint | tuple[Inspector.Constraint],
                exclude: None | str | tuple,
                include: None | str | tuple,
                cheapen_if_long: bool,
            ) -> dict: ...

            def __call__(
                self,
                obj,
                values: bool = False,
                *,
                only: Inspector.Constraint | tuple[Inspector.Constraint] = None,
                exclude: None | str | tuple = (),
                include: None | str | tuple = (),
                cheapen_if_long: bool = False,
            ) -> dict | list[str]:
                # TODO: compare rv with inspect.getmembers()
                if include and exclude:
                    con.log(f"[WARN] getprops({obj!r}) | Can't have both include and exclude")
                    return []

                # Normalize `include` and `exclude` to tuple (if not None)
                if isinstance(include, str):
                    include = (include,)
                if isinstance(exclude, str):
                    exclude = (exclude,)
                if exclude is not None:
                    exclude = (*exclude, *self.skip_obj_props)

                try:
                    proplist = obj.__dir__()
                except:
                    proplist = dir(obj)

                if values:
                    props = {}
                else:
                    props = []

                regular, private, dunder = self._build_constraints(only)

                for prop in sorted(proplist):
                    if self._should_skip(
                        prop,
                        regular,
                        private,
                        dunder,
                        include=include,
                        exclude=exclude,
                    ):
                        continue
                    try:
                        value = getattr(obj, prop)
                    except Exception as e:
                        value = e
                    if self._is_method(value):
                        continue

                    if values:
                        if cheapen_if_long and len(str(value)) > 10_000:
                            try:
                                con.log(f"[WARN] getprops({obj!r}) | obj.{prop} str length:" f" {len(str(value))}")
                                value = self._cheapen(value, len_limit=10_000)
                                con.log(f"[WARN]   shortened to: {len(str(value))}")
                            except Exception as e:
                                con.log(
                                    f"[WARN] getprops({obj!r}) | {prop} |"
                                    f" {e.__class__.__qualname__} when trying to cheapen value: {e}"
                                )
                                continue

                        props[prop] = value
                    else:
                        props.append(prop)

                return props

        return GetProps()

    @builtin_variable
    def getmeths() -> ForwardRef("GetMeths"):
        """
        Args:
            sigs: Specify True to check and return method signatures
            only: Either 'regular', 'private', 'dunder' or a tuple containing any of them
            exclude: A str or a tuple of strings specifying which props to ignore
            include: A str or a tuple of strings specifying which props to include. All others are ignored.
            cheapen_if_long: Currently only used in `getprops()`.

        Returns:
            A dict of method names and their signatures if `sigs` is True, otherwise a list of method names.

        `exclude` and `include` are mutually exclusive.
        """

        class GetMeths(Inspector):
            def __call__(
                self,
                obj,
                sigs: bool = False,
                *,
                only: Inspector.Constraint | tuple[Inspector.Constraint] = None,
                exclude: str | tuple = (),
                include: str | tuple = (),
                cheapen_if_long: bool = False,
            ) -> dict | list[str]:
                if include and exclude:
                    con.log(f"[WARN] getmeths({obj!r}) | can't have both include and exclude")
                    return []

                # Normalize include and exclude to tuple
                if isinstance(include, str):
                    include = (include,)
                if isinstance(exclude, str):
                    exclude = (exclude,)
                if exclude is not None:
                    exclude = (*exclude, *self.skip_obj_meths)

                try:
                    proplist = dir(obj)
                except:
                    proplist = obj.__dir__()

                regular, private, dunder = self._build_constraints(only)
                if sigs:
                    meths = {}
                else:
                    meths = []

                for prop in sorted(proplist):
                    if self._should_skip(
                        prop,
                        regular,
                        private,
                        dunder,
                        include=include,
                        exclude=exclude,
                    ):
                        continue
                    try:
                        meth = getattr(obj, prop)
                    except Exception as e:
                        meth = e
                    if not self._is_method(meth):
                        continue
                    if sigs:
                        try:
                            sig: inspect.Signature = inspect.signature(meth)
                            meths[prop] = str(prop) + str(sig)
                        except ValueError as e:
                            con.log(
                                f'[WARN] ValueError: {", ".join(map(repr, e.args))}. Appending meth' " without args"
                            )
                            meths[prop] = ""
                    else:
                        meths.append(prop)

                return meths  # return sorted(meths, key=sort)

        return GetMeths()

    @builtin
    def inquire(obj, quiet=False) -> dict[str, Any]:
        """Calls all relevant functions from `inspect` module on `obj` and prints the results."""
        inquiries = dict()
        exclude_inspect_functions = (
            "currentframe",
            "findsource",
            "getmembers",
            "findsource",
            "getsource",
            "getsourcelines",
            # 'findsource', 'getmembers',
            # '_check_class', '_check_instance',
            # '_has_code_flag', '_main', 'getargspec',
            # '_missing_arguments', '_shadowed_dict',
            # '_signature_bound_method', '_signature_from_builtin',
            # '_signature_from_callable', '_signature_from_function',
            # '_signature_fromstr', '_signature_get_bound_param',
            # '_signature_get_partial', '_signature_get_user_defined_method',
            # '_signature_strip_non_python_syntax', '_static_getmro',
            # '_too_many', 'classify_class_attrs',
            # 'cleandoc',
            # 'formatargspec', 'formatargvalues',
            # 'getargs', 'getargvalues',
            # 'getattr_static', 'getblock',
            # 'getclasstree', 'getcoroutinestate',
            # 'getframeinfo', 'getgeneratorlocals',
            # 'getgeneratorstate', 'getinnerframes',
            # 'getlineno', 'getmodulename',
            # 'getmro', 'getouterframes',
            # 'indentsize', 'namedtuple',
            # 'stack', 'walktree'
        )
        for inspect_function_name in filter(
            lambda function: str(function) not in exclude_inspect_functions,
            getmeths(inspect),
        ):
            pretty_fn = pretty_inst(obj)
            try:
                inspect_function = getattr(inspect, inspect_function_name)
                rv = inspect_function(obj)
                if not quiet:
                    # pretty_rv = pretty(rv)
                    pp(
                        rv,
                        include_file_name=False,
                        include_function_name=False,
                        include_type=False,
                        include_arg_name=False,
                        title=(  # crop=True
                            f"\n{inspect_function.__name__}({pretty_fn}) -> {pretty_type(rv)}:"
                        ),
                    )
                inquiries[inspect_function_name] = rv
            except Exception as e:
                if not quiet:
                    _print(f"\t⚠️ {pretty_fn} | {inspect_function_name} | {e.__class__.__qualname__}:" f" {e!r}")
        return inquiries

    @builtin
    def pretty_sig2(method: Callable, *args, **kwargs) -> str:
        # todo: sync with internal_log.py pretty_signature
        prettysig = ""
        if hasattr(method, "__name__"):
            method_name = method.__name__
        elif hasattr(method, "__qualname__"):
            method_name = method.__qualname__
        else:
            method_name = str(method)
        if args:
            first_arg, *rest = args
            # if some_method(self, *args, **kwargs), then `self` arg is an instance
            if hasattr(first_arg, method_name):
                obj = first_arg
                args = rest
                # if some_method(Class, *args, **kwargs), then `Class` arg is a class
                if type(obj) is type:
                    instance_name = pretty_inst(obj.__qualname__)
                else:
                    instance_name = pretty_inst(obj.__class__.__qualname__)
                prettysig += f"{instance_name}."
        if args or kwargs:
            pretty_args = ""
            indent = len(method_name) + 1  # +1 for the open parentheses
            bound_arguments = inspect.signature(method).bind(*args, **kwargs)
            for arg, value in bound_arguments.arguments.items():
                pretty_value = "\n".join(
                    wrap(
                        pretty_inst(repr(value)),  # wrap expects a string, so repr(value)
                        width=termwidth - indent,
                        subsequent_indent=" " * (indent + len(arg) + 1),
                    )
                )  # +1 for the equals sign
                pretty_args += f"{arg}={pretty_value},\n" + " " * indent
        else:
            signature = inspect.signature(method)
            # for pval in signature.parameters.values():
            #     pval._annotation = inspect._empty()
            # breakpoint()
            pretty_args = str(signature)  # should split by arg and wrap
        prettysig += f"{method_name}({pretty_args})"
        return prettysig

    @builtin
    def pretty_sig(method: Callable, *args, **kwargs) -> str:
        # func_node = None
        # tree=ast.parse(inspect.getsource(inspect.getmodule(func)))
        # classes=list(filter(lambda x: isinstance(x, ast.ClassDef), tree.body))
        # for cls in classes:
        #   for node in cls.body:
        #     if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
        #       func_node = node
        # or:
        # from rich._inspect import Inspect
        # pretty(Inspect(func)._get_signature("", func))
        # or:
        # str(inspect.signature(func))
        # cool:
        # inspect.signature(func).bind(*args, **kwargs)
        prettysig = "\033[97;48;2;30;30;30m" if RUN_BY_HUMAN else ""
        if hasattr(method, "__name__"):
            method_name = method.__name__
        elif hasattr(method, "__qualname__"):
            method_name = method.__qualname__
        else:
            method_name = str(method)

        if args:
            first_arg, *rest = args
            # if some_method(self, *args, **kwargs), then `self` arg is an instance
            if hasattr(first_arg, method_name):
                obj = first_arg
                args = rest
                # if some_method(Class, *args, **kwargs), then `Class` arg is a class
                if type(obj) is type:
                    instance_name = pretty_inst(obj.__qualname__)
                else:
                    instance_name = pretty_inst(obj.__class__.__qualname__)
                prettysig += f"{instance_name}."

        pretty_kwargs = []
        for k, v in kwargs.items():
            pretty_kwargs.append(f"{k}={pretty(v)}")

        # Relies on pretty_inst returning as-is if obj doesn't match "<foo> at 0x123".
        # TODO (unclear): what if both pretty_inst and pretty modify the string?
        #                 what if pretty(v) gets an instance? (pretty_kwargs)
        pretty_args_joined = ", ".join(map(pretty, map(pretty_inst, args)))
        pretty_kwargs_joined = ", ".join(pretty_kwargs)
        prettysig += (
            f"{method_name}\033[0m("
            + pretty_args_joined
            + (", " if args and kwargs else "")
            + pretty_kwargs_joined
            + ")"
        )
        return prettysig

    @builtin
    def what(obj, **kwargs):
        """rich.inspect(methods=True)"""
        return rich.inspect(
            obj,
            methods=True,
            title=kwargs.pop("title", varinfo()),
            console=con,
            **kwargs,
        )

    @builtin
    def whatt(obj, **kwargs):
        """rich.inspect(methods=True, help=True)"""
        return rich.inspect(
            obj,
            methods=True,
            help=True,
            title=kwargs.pop("title", varinfo()),
            console=con,
            **kwargs,
        )

    @builtin
    def whattt(obj, **kwargs):
        """rich.inspect(methods=True, help=True, private=True)"""
        return rich.inspect(
            obj,
            methods=True,
            help=True,
            private=True,
            title=kwargs.pop("title", varinfo()),
            console=con,
            **kwargs,
        )

    @builtin
    def whatttt(obj, **kwargs):
        """rich.inspect(all=True)"""
        return rich.inspect(
            obj,
            all=True,
            title=kwargs.pop("title", varinfo()),
            console=con,
            **kwargs,
        )

    @builtin
    def who(
        condition: Callable[[str], bool] | Callable[[Any], bool] | Callable[[str, Any], bool] = None,
        apply: Callable[[Any], Any] = lambda _: _,
    ) -> NoReturn:
        """
        >>> who()
        >>> who(lambda k,v: 'foo' in k and isinstance(v, str) [, apply = lambda v: v.upper()])
        >>> who(lambda k: 'foo' in k [, apply = lambda v: v.upper()])
        >>> who(lambda v: isinstance(v, str) [, apply = lambda v: v.upper()])
        """
        if condition:
            condition_sig = inspect.signature(condition)
            normalized_condition: Callable
            if len(condition_sig.parameters) == 2:
                normalized_condition = condition
            else:
                param = condition_sig.parameters[next(iter(condition_sig.parameters))]
                if param.name == "k":
                    normalized_condition = lambda _k, _v: condition(_k)
                elif param.name == "v":
                    normalized_condition = lambda _k, _v: condition(_v)
                else:
                    con.log(
                        f"[ERROR] who({condition = }) | If condition accepts 1 param, must be named"
                        f' "k" or "v". Got {param.name = }'
                    )
                    return

            for k, v in getcaller().frame.f_locals.items():
                if normalized_condition(k, v):
                    applied_v = apply(v)
                    _print(pretty(f"{k} [dim i]{pretty_type(applied_v)}[/]: {pformat(applied_v)}"))
        else:
            con.log("locals:", log_locals=True, _stack_offset=2)

    # *** Formatting

    def decolor(s: str) -> str:
        return COLOR_RE.sub("", s)

    def shorten(string: str, limit: int = termwidth) -> str:
        if not string:
            return string
        if not isinstance(string, str):
            string = str(string)
        length = len(string)
        if length <= limit:
            return string
        if limit < 3:
            raise ValueError(f"{limit = } but must be >= 3 because the shortest possible string is 'x..'")
        small_half_of_limit = limit // 2
        big_half_of_limit = limit - small_half_of_limit
        if "\033[" in string:
            # Don't remove color codes.
            no_color = decolor(string)
            real_length = len(no_color)
            if real_length <= limit:
                return string
            color_matches: list[re.Match] = list(COLOR_RE.finditer(string))
            if len(color_matches) == 2:
                color_a, color_b = color_matches
                if color_a.start() == 0 and color_b.end() == length:
                    # Colors surround string from both ends
                    return f"{color_a.group()}{shorten(no_color, limit)}{color_b.group()}"
            return shorten(no_color, limit)

        if 3 <= limit <= 4:
            placeholder = ".."
            placeholder_length = 2
        else:
            placeholder = "..."
            placeholder_length = 3
        small_half_of_placeholder_length = placeholder_length // 2
        big_half_of_placeholder_length = placeholder_length - small_half_of_placeholder_length

        # Odd limits make symmetric strings, because the separator is odd too. Big–big cancel out, as do small–small.
        if limit % 2 == 1:
            left_cutoff_index: int = big_half_of_limit - big_half_of_placeholder_length
            right_cutoff_index: int = length - (small_half_of_limit - small_half_of_placeholder_length)
        else:
            # Even limits make asymmetric strings, so we give the extra char to the left half.
            left_cutoff_index: int = big_half_of_limit - small_half_of_placeholder_length
            right_cutoff_index: int = length - (small_half_of_limit - big_half_of_placeholder_length)

        start = string[:left_cutoff_index]
        end = string[right_cutoff_index:]
        return f"{start}{placeholder}{end}".replace("\n", " ")

    @builtin
    def pretty_inst(obj: Obj) -> str | Obj:
        """<foo.bar.Person object at 0x7f2137d027c0> -> Person (0x7f2137d027c0)"""
        if isinstance(obj, str):
            # In case obj doesn't match "<foo> at 0x123",
            # just return as-is without additional str(...).
            if not OBJ_RE.search(obj):
                return obj
            string = obj
        else:
            # In case str(obj) doesn't match "<foo> at 0x123",
            # just return as-is without converting to string.
            match = OBJ_RE.search(str(obj))
            if not match:
                return obj
            string = str(obj)
        return OBJ_RE.sub(
            lambda _match: f"{(groups := _match.groups())[0]} ({groups[1]})",
            string,
        )

    @builtin
    def pretty_type(obj: Obj) -> str:
        """pretty_type({"hi" : "bye"}) -> dict (1)"""
        stringified_type: str
        if type(obj) is type:
            stringified_type = str(obj)
        else:
            stringified_type = str(type(obj))
        rv = TYPE_RE.sub(lambda match: match.groups()[0], stringified_type)
        with suppress(TypeError):
            rv += f" ({len(obj)})"
        return rv

    @builtin
    class Reprer:
        def __init__(self, printer: Callable = None):
            if not printer:
                printer = functools.partial(con.print, soft_wrap=True)
            self._print = printer
            self._print_funcs = {  # { condition(obj) : repr_fn(obj) }
                lambda obj: hasattr(obj, "url_root"): lambda obj, *args, **kwargs: printer(
                    obj, f" url_root = {obj.url_root}", *args, **kwargs
                ),
                lambda obj: isinstance(obj, BaseException): lambda exc, *args, **kwargs: printer(repr(exc)),
                lambda obj: isinstance(obj, (types.FunctionType, types.MethodType)): lambda kallable,
                *args,
                **kwargs: printer(kallable),
                lambda obj: "DataFrame" in str(type(obj)) and callable(getattr(obj, "show", None)): lambda df,
                *args,
                **kwargs: df.show(),
            }
            ast_dump_kwargs = dict(annotate_fields=True, include_attributes=False)
            if sys.version_info[1] >= 9:
                ast_dump_kwargs["indent"] = 4

            self._print_funcs.update(
                {
                    lambda obj: isinstance(obj, ast.AST): lambda node, *args, **kwargs: printer(
                        node,
                        f" {ast.dump(node, **ast_dump_kwargs)}".replace(r"\n", "\n... "),
                        *args,
                        **kwargs,
                    )
                }
            )

        def repr(self, obj, *args, **kwargs):
            """Falls back to `printer` given to constructor if no condition is met."""
            for condition, print_func in self._print_funcs.items():
                if condition(obj):
                    print_func(obj, *args, **kwargs)
                    return
            self._print(pretty_inst(obj), *args, **kwargs)

    reprer = Reprer()
    builtin(reprer, name="reprer")

    @builtin
    def pformat(
        obj: Obj,
        *,
        max_width=termwidth,
        indent_size=4,
        max_length=None,
        max_string=None,
        max_depth=None,
        expand_all=False,
    ) -> str:
        """Pretty-formats (no markup / syntax highlighting etc)."""
        from rich.pretty import pretty_repr

        pformatted = pretty_repr(
            obj,
            max_width=max_width,
            indent_size=indent_size,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            expand_all=expand_all,
        )
        return pformatted

    @builtin
    def pretty(
        obj,
        *,
        no_wrap: bool = False,
        markup: bool = None,
        width: int = termwidth,
        height: int = None,
        crop: bool = True,
        soft_wrap: bool = True,
        new_line_start: bool = False,
    ) -> str:
        # no need to use pformat
        # todo: move all this logic into Reprer, and maybe move all Reprer logic into con.print overload
        with con.capture() as captured:
            if isinstance(obj, (dict, list)):
                # noinspection PyUnresolvedReferences
                from rich.json import JSON

                try:
                    pretty_data = JSON.from_data(obj, indent=4, sort_keys=True)
                    con.print(
                        pretty_data,
                        no_wrap=no_wrap,
                        markup=markup,
                        width=width,
                        height=height,
                        crop=crop,
                        soft_wrap=soft_wrap,
                        new_line_start=new_line_start,
                    )
                except TypeError:
                    # con.print(obj)
                    reprer.repr(
                        obj,
                        no_wrap=no_wrap,
                        markup=markup,
                        width=width,
                        height=height,
                        crop=crop,
                        soft_wrap=soft_wrap,
                        new_line_start=new_line_start,
                    )
            elif isinstance(obj, str) and obj.startswith('{"'):
                from rich.json import JSON

                pretty_data = JSON(obj, indent=4, sort_keys=True)
                con.print(
                    pretty_data,
                    no_wrap=no_wrap,
                    markup=markup,
                    width=width,
                    height=height,
                    crop=crop,
                    soft_wrap=soft_wrap,
                    new_line_start=new_line_start,
                )
            else:
                # con.print(obj)
                reprer.repr(
                    obj,
                    no_wrap=no_wrap,
                    markup=markup,
                    width=width,
                    height=height,
                    crop=crop,
                    soft_wrap=soft_wrap,
                    new_line_start=new_line_start,
                )
        return captured.get().rstrip()

    @builtin
    def pprint(
        *objs: Obj,
        no_wrap: bool = None,
        markup: bool = None,
        width: int = termwidth,
        height: int = None,
        crop: bool = True,
        soft_wrap: bool = True,
        new_line_start: bool = False,
    ) -> None:
        for obj in objs:
            _print(
                pretty(
                    obj,
                    no_wrap=no_wrap,
                    markup=markup,
                    width=width,
                    height=height,
                    crop=crop,
                    soft_wrap=soft_wrap,
                    new_line_start=new_line_start,
                )
            )

    # *** API
    @functools.partial(builtin, verbose=True)
    def pp(*args, **kwargs):
        """
        Keyword Args:
            offset: int (default 2)
            include_file_name: bool (default True, unless in IPython / Jupyter)
            include_function_name: bool (default True, unless in IPython / Jupyter)
            include_arg_name: bool (default True)
            include_type: bool (default True)
            single_line: bool (default False)
            title: str (default None)

        Calls vanilla `print` if:
            - Keyword args include 'sep' or 'end', 'file' or 'flush';
            - called inside a list / dict / set comprehension;
            - called by pdb, ipython, pycharm, jupyter etc.

        Behavior can be controlled via 'debug.conf.print.*' settings.
        """
        # con.log(f'[debug] pp({args = }, {kwargs = }')
        if "sep" in kwargs or "end" in kwargs or "file" in kwargs or "flush" in kwargs:
            return _print(*args, **kwargs)
        formatted_args = []

        caller_frameinfo = getcaller(offset=kwargs.pop("offset", 2))
        caller_frame_name = str(caller_frameinfo.frame)
        caller_frame_function = caller_frameinfo.function
        # con.log(f'[debug]{caller_frameinfo = }\n{caller_frameinfo.filename = }\n{caller_frameinfo.frame = }\n{caller_frame_name = }\n{caller_frameinfo.frame.f_back = }')

        # These turn off include_file_name and include_function_name
        uninteresting_frame_names = (
            "ipython-input",
            "ipykernel",
            "pylab",
            "stdin",  # '.pdbrc.py',
            "code <module>",
        )
        in_uninteresting_frame = any(name in caller_frame_name for name in uninteresting_frame_names)

        # Pop kwargs that builtin print doesn't support early in the flow
        include_file_name = kwargs.pop(
            "include_file_name",
            not in_uninteresting_frame and debug_module.conf.print.include_file_name,
        )
        include_function_name = kwargs.pop(
            "include_function_name",
            not in_uninteresting_frame and debug_module.conf.print.include_function_name,
        )
        include_arg_name = kwargs.pop("include_arg_name", debug_module.conf.print.include_arg_name)
        include_type = kwargs.pop("include_type", debug_module.conf.print.include_type)
        single_line = kwargs.pop("single_line", debug_module.conf.print.single_line)
        # crop = kwargs.pop('crop', False)

        if any(comprehension in caller_frame_function for comprehension in ("listcomp", "setcomp", "dictcomp")):
            # or any(path in caller_frame_name for path in non_user_files):
            return _print(*args, **kwargs)

        # Fallback to vanilla print if caller is pdb, ipython, pycharm, jupyter etc.
        non_user_files = (  # '/pdbpp/',
            "pdb.py",
            "code_executor",
            "pydevd",
            "traceback.py",
            "interactiveshell.py",
            "displayhook.py",
            "IPython/core/page.py",
            "ipython_autoimport",
            "namespace.py",
            "<magic-timeit>",
            "core/magics/",
            *debug_module.conf.print.ignore_files,
        )
        for non_user_file in non_user_files:
            if non_user_file in caller_frame_name:
                return _print(*args, **kwargs)

            if non_user_file in caller_frameinfo.filename:
                con.log(f"[WARN] Missed ignoring {non_user_file = } because only in" f" {caller_frameinfo.filename = }")

        if title := kwargs.pop("title", ""):
            formatted_args.append("\n" + pretty(f"[title]{title}[/]") + "\n")

        if include_file_name:
            file_name = caller_frameinfo.filename.split("/")[-1]
            formatted_args.append(pretty(f"[dim][{file_name}][/dim]") + ("" if include_function_name else "\n"))
        if include_function_name:
            # formatted_args.append(pretty(f'[dim][{caller_frame_function}(...)][/dim]') + '\n')
            formatted_args.append(f"\x1b[2;38;2;0;175;215m[{caller_frame_function}(...)]\x1b[0m\n")
        for i, arg in enumerate(args):
            ## Variable name and type
            if include_arg_name:
                arg_name = varinfo(
                    i,
                    value=arg,
                    offset_or_frameinfo=caller_frameinfo,
                    # Already grabbed at the beginning, no need again
                    include_file_name=False,
                    include_function_name=False,
                )
            else:
                arg_name = None
            if arg_name:
                if single_line and len(args) > 1:
                    arg_name = f"  {arg_name}"

                if "result_repr" in arg_name:
                    formatted_args.append(arg)
                    continue

                if i != 0:
                    # Separate args with a newline
                    arg_name = "\n" + arg_name
                # arg_name = pretty(f'[blue]{arg_name}[/]')
                if include_type:
                    formatted_args.append(f"{arg_name} " + pretty(f"[dim i]{pretty_type(arg)}[/]"))
                else:
                    formatted_args.append(f"{arg_name}: ")

            elif include_type:
                formatted_type = pretty(f"[dim i]{pretty_type(arg)}[/]")
                if single_line and len(args) > 1:
                    formatted_type = f"  {formatted_type}"
                formatted_args.append(formatted_type)

            ## Variable value
            # Don't format value if already colored
            try:
                if "\x1b" in arg:
                    formatted_args.append(arg)
                    continue
            except (TypeError, io.UnsupportedOperation, ValueError):
                pass

            pretty_value = pretty(arg)
            # if crop:
            #     pretty_value = shorten(pretty_value)
            if any(
                include_kwarg
                for include_kwarg in (
                    include_file_name,
                    include_function_name,
                    include_arg_name,
                    include_type,
                )
            ):
                if "\n" in pretty_value:
                    formatted = "\n" + pretty_value
                elif single_line:
                    formatted = pretty("[dim i] = [/]") + pretty_value
                else:
                    formatted = pretty("\n[black]└─ [/black]") + pretty_value
            else:
                formatted = pretty_value
            formatted_args.append(formatted)

        return _print(*formatted_args, **kwargs, sep="")

    def displayhook(val):
        if val is None:
            return
        builtins._ = None
        pprint(val)
        builtins._ = val

    @builtin
    def install():
        if sys.displayhook is displayhook:
            pprint(f"[good]Already installed {sys.displayhook = }[/]")
            return
        setattr(sys, "__displayhook", sys.displayhook)
        sys.displayhook = displayhook
        pprint(f"✅ [good]Installed displayhook. {sys.displayhook = }[/]")

    @builtin
    def uninstall():
        if not hasattr(sys, "__displayhook"):
            pprint("[good]Builtin displayhook was already active[/]")
            return
        sys.displayhook = getattr(sys, "__displayhook")
        delattr(sys, "__displayhook")
        pprint(f"✅ [good]Uninstalled displayhook. {sys.displayhook = }[/]")

    if patch_print:
        builtin(pp, name="print", verbose=True)

    # install()

    # builtin(pp, name='_', verbose=True)
    lokals = locals()

    class DebugModule(ModuleType):
        class conf:
            class print:
                single_line = False
                include_file_name = True
                include_function_name = True
                include_arg_name = True
                include_type = True
                ignore_files = ()

        def __getitem__(self, name):
            return self.__dict__[name] if name in self.__dict__ else lokals[name]

        def __getattr__(self, item):
            return self.__getitem__(item)

        def __repr__(self):
            return f"DebugModule({super().__repr__()})"

        @property
        def help(self) -> str:
            builtins_help = "Builtins:\n"
            for added_builtin_name in _added_builtins_names:
                added_builtin = getattr(builtins, added_builtin_name)
                try:
                    sig = inspect.signature(added_builtin)
                    if 4 + len(str(sig)) + len(added_builtin_name) >= termwidth:
                        sig = (",\n    " + len(added_builtin_name) * " ").join(str(sig).split(","))
                    builtins_help += f"    {added_builtin_name}\x1b[2m{sig}\x1b[0m\n"
                except TypeError:
                    builtins_help += f"    {added_builtin_name} \x1b[2m{pretty_type(added_builtin)}\x1b[0m\n"
                except ValueError:
                    if added_builtin_name in ("print", "_print"):
                        builtins_help += (
                            f"    {added_builtin_name}\x1b[2m(value, ..., sep=' ', end='\\n',"
                            " file=sys.stdout, flush=False)\x1b[0m\n"
                        )

            pprint_help = "Patched print:" + pp.__doc__
            debugmodule_help = "Debug module:\n\t" + "\n\t".join(
                [
                    "help",
                    *[
                        f"conf.print.{key} = {value!r}"
                        for key, value in self.conf.print.__dict__.items()
                        if not key.startswith("_")
                    ],
                ]
            )
            env_vars_help = "Environment variables:\n\t" + "\n\t".join(
                [
                    "DEBUGFILE_FORCE_INSTALL",
                    "DEBUGFILE_RICH_TB",
                    "DEBUGFILE_PATCH_PRINT",
                ]
            )
            help_str = "\n\n".join([builtins_help, pprint_help, debugmodule_help, env_vars_help])

            return "\n" + "─" * termwidth + help_str + "\n" + "─" * termwidth

    # Allow 'from debug import pp' etc
    _added_builtins_names = sorted(set(dir(builtins)) - _original_builtins_names)
    assert "pp" in _added_builtins_names, "pp not found in new builtins"
    assert "_print" in _added_builtins_names, "_print not found in new builtins"
    debug_module.__class__ = DebugModule
    for added_builtin_name in _added_builtins_names:
        if added_builtin_name in (
            "Path",
            "_print",
            "builtins",
            "inspect",
            "os",
        ):
            continue
        new_builtin = getattr(builtins, added_builtin_name)
        setattr(
            DebugModule,
            added_builtin_name,
            staticmethod(new_builtin) if callable(new_builtin) else new_builtin,
        )
    setattr(DebugModule, "print", staticmethod(pp))  # debug.print is always patched
    setattr(DebugModule, "pp", staticmethod(pp))
    for print_alias in ("print", "pp"):
        assert print_alias in DebugModule.__dict__, f"{print_alias} not found in DebugModule.__dict__"
    os.environ["DEBUGFILE_LOADED"] = "1"
    return debug_module


if __name__ == "__main__":
    if envbool("DEBUGFILE_LOADED"):
        _print("Not loading debug.py because DEBUGFILE_LOADED", file=sys.stderr)
    else:
        init_debug_module(
            force_install=os.getenv("DEBUGFILE_FORCE_INSTALL", "").lower(),
            rich_tb=envbool("DEBUGFILE_RICH_TB", "--rich-tb"),
            patch_print=envbool("DEBUGFILE_PATCH_PRINT", "--patch-print"),
        )
