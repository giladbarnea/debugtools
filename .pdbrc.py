import pdb
import os
import sys
from pathlib import Path

# pdbpp is supposed to monkeypatch pdb, but sometimes it doesn't work (maybe when pip install -e pdbpp)
if not hasattr(pdb, "DefaultConfig"):
    import pdbpp as pdb

try:
    # todo: all of this is unused! should just import debug
    import builtins
    import types

    if hasattr(builtins, "pp"):
        print(".pdbrc.py | hasattr(builtins, 'pp')", file=sys.stderr)

        def pprint(obj, *args, **kwargs):
            builtins.pp(obj, include_arg_name=False)

    elif not isinstance(builtins.print, types.BuiltinFunctionType):
        print(".pdbrc.py | not isinstance(builtins.print, types.BuiltinFunctionType)", file=sys.stderr)

        def pprint(obj, *args, **kwargs):
            builtins.print(obj, include_arg_name=False)

    else:
        # No builtin print patch in 'pp', '__print__' or 'print', so define one
        from rich.console import Console
        from rich.theme import Theme
        import os
        import sys

        print(".pdbrc.py | no patched print in builtins; defining with rich", file=sys.stderr)

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
        con = Console(  # force_terminal=True,
            log_time_format="[%d.%m.%Y][%T]",
            # color_system='auto' if PYCHARM_HOSTED else 'truecolor',
            # width=TERMWIDTH,
            tab_size=2,
            # file=sys.stdout if PYCHARM_HOSTED else sys.stderr,
            theme=Theme({**theme, **{k.upper(): v for k, v in theme.items()}}),
        )

        def pprint(
            obj,
            *,
            no_wrap: bool = False,
            markup: bool = None,
            width: int = None,
            height: int = None,
            crop: bool = True,
            soft_wrap: bool = True,
            new_line_start: bool = False
        ) -> str:
            # no need to use pformat
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
                        con.print(
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
                    con.print(
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

except ImportError:
    print(".pdbrc.py | ImportError, defining with stdlib pprint", file=sys.stderr)
    from pprint import pprint as stdlib_pprint

    def pprint(*args, **kwargs):
        return stdlib_pprint(*args, **kwargs, compact=False, indent=4)


class Config(pdb.DefaultConfig):
    sticky_by_default = True
    pygments_formatter_class = "pygments.formatters.TerminalTrueColorFormatter"
    pygments_formatter_kwargs = {"style": "monokai"}
    use_terminal256formatter = False
    filename_color = pdb.Color.lightgray
    display_unchanged = True
    # def __init__(self):
    #     try:
    #         from pygments.formatters import terminal
    #     except ImportError:
    #         pass
    #     else:
    #         self.colorscheme = terminal.TERMINAL_COLORS.copy()
    #         self.colorscheme.update({
    #             terminal.Keyword:        ('darkred', 'red'),
    #             terminal.Number:         ('darkyellow', 'yellow'),
    #             terminal.String:         ('brown', 'green'),
    #             terminal.Name.Function:  ('darkgreen', 'blue'),
    #             terminal.Name.Namespace: ('teal', 'turquoise'),
    #             })

    # def setup(self, _pdb):
    #     def _log(message):
    #         print(f'.pdbrc.py | Config.setup() | {message}', file=sys.stderr)
    #
    #     _log('before super().setup(_pdb)')
    #     super().setup(_pdb)
    #     _log('after super().setup(_pdb)')
    #     # Pdb = _pdb.__class__
    #
    #     # def before_interaction_hook(self, pdb):
    #     home_dir = Path.home()
    #     try:
    #         for path in os.getenv('PYTHONPATH', '').split(':') + [str(home_dir)]:
    #             if (debug_file := Path(path) / 'debug.py').exists():
    #                 debug_file_parent = str(debug_file.parent)
    #                 if debug_file_parent not in sys.path:
    #                     sys.path.insert(0, str(debug_file.parent))
    #                 break
    #         else:
    #             _log(f'Did not find debug.py at home or in PYTHONPATH')
    #             return
    #
    #         try:
    #             # exec(compile(debug.open().read(), str(debug), 'exec'))
    #             # import debug
    #             from importlib import import_module
    #             debug = import_module(debug_file.stem)
    #         except Exception as e:
    #             if os.getenv('PDBPP_RAISE_CONFIG_ERRORS', 'false').lower() in ('1', 'true'):
    #                 raise
    #             _log(f'while "import debug" | {e.__class__} {e}')
    #         # else: # import successful
    #             # del Config.pprint
    #             # Config.pprint = staticmethod(debug.pprint)
    #             # del _pdb.config.__class__.pprint
    #             # _pdb.__class__.do_pp = staticmethod(debug.pprint)
    #             # del _pdb.do_pp
    #             # _pdb.config.__class__.pprint = staticmethod(debug.pprint)
    #             # _log(f'Patched pdbpp.config.pprint to {debug.pprint}')
    #     except Exception as e:
    #         _log(f'{e.__class__}: {e}')