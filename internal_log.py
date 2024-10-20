from __future__ import annotations

import builtins
from datetime import datetime
import functools
import inspect
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from contextlib import suppress, contextmanager, redirect_stdout
from functools import partialmethod
from io import StringIO
from logging.handlers import RotatingFileHandler
from pprint import pprint
from typing import TypeVar, Union

try:
    from loguru import logger as loguru_logger
except ModuleNotFoundError:
    pass
else:
    loguru_logger.__class__.title = partialmethod(loguru_logger.__class__.log,
                                                  loguru_logger.level("title",
                                                                      no=25,
                                                                      color='<bold><white>').name)

TERMWIDTH = None
for fd in (sys.__stdin__.fileno(), sys.__stdout__.fileno(), sys.__stderr__.fileno()):
    try:
        TERMWIDTH = os.get_terminal_size(fd)[0]
    except (AttributeError, ValueError, OSError):
        pass
    else:
        break

if not TERMWIDTH or TERMWIDTH <= 80:
    columns = os.environ.get("COLUMNS")
    if columns and columns.isdigit() and int(columns) > 80:
        TERMWIDTH = int(columns)

if not TERMWIDTH or TERMWIDTH <= 80:
    TERMWIDTH = 170

OBJ_RE = re.compile(r'<(?:[\w\d<]+\.)*([\w\d>]+)[ object]* at (0x[\w\d]{12})>')
TYPE_RE = re.compile(r"<class '(?:[\w\d<>]+\.)*([\w\d]+)'>")
WHITESPACE_RE = re.compile(r'\s+')
COLOR_RE = re.compile(r'(\x1b\[(?:\d;?)*m)')

INTERACTIVE = hasattr(sys.stdin, "fileno") and os.isatty(sys.stdin.fileno())
RUN_BY_HUMAN = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

try:
    import rich
except ModuleNotFoundError:
    print('\n\x1b[97;1m[internal_log.py] Running: pip install rich\x1b[0m\n', file=sys.stderr)
    failed = bool(os.system('pip install --retries=2 rich'))
    if failed:
        msg = f'\n\x1b[91;1m[internal_log.py] Failed: pip install rich\x1b[0m\n'
    else:
        msg = f'\n\x1b[92;1m[internal_log.py] Success: pip install rich\x1b[0m\n'
    print(msg, file=sys.stderr)
finally:
    try:
        import rich
    except ModuleNotFoundError:
        pass
    else:
        sys.modules['rich'] = rich

if 'rich' in sys.modules:
    import rich
    from rich.console import Console
    from rich.theme import Theme

    PYCHARM_HOSTED = os.getenv('PYCHARM_HOSTED')
    theme = {
        'debug':   'dim',
        'warn':    'yellow',
        'warning': 'yellow',
        'error':   'red',
        'fatal':   'bright_red',
        'success': 'green',
        'prompt':  'b bright_cyan',
        'title':   'b bright_white',
        }
    con = Console(force_terminal=False,
                  log_time_format='[%T %X]',
                  color_system='auto' if PYCHARM_HOSTED else 'truecolor',
                  width=TERMWIDTH,
                  tab_size=2,
                  file=sys.stdout if PYCHARM_HOSTED else sys.stderr,
                  theme=Theme({**theme, **{k.upper(): v for k, v in theme.items()}}))
else:
    # Failed importing rich, probably network issue
    class CaptureShim(StringIO):
        def get(self):
            return self.getvalue()


    class ConsoleShim:
        print = staticmethod(pprint)
        print_json = staticmethod(pprint)

        @contextmanager
        def capture(self):
            file = CaptureShim()
            with redirect_stdout(file):
                try:
                    yield file
                except Exception:
                    pass


    con = ConsoleShim()
builtins.con = con


def decolor(s):
    return COLOR_RE.sub('', s)


def shorten(s, limit=TERMWIDTH):
    if not s:
        return s
    if limit < 4:
        logging.warning(f"shorten({shorten(repr(s), limit=20)}) was called with limit = %d, can handle limit >= 4", limit)
        return s
    length = len(s)
    if length <= limit:
        return s
    half_the_limit = limit // 2
    if '\x1b[' in s:
        no_color = decolor(s)
        real_length = len(no_color)
        if real_length <= limit:
            return s
        color_matches: list[re.Match] = list(COLOR_RE.finditer(s))
        if len(color_matches) == 2:
            # Only 2 color tags (open and close)
            color_a, color_b = color_matches
            if color_a.start() == 0 and color_b.end() == length:
                # Colors surround string from both ends
                return f'{color_a.group()}{shorten(no_color, limit)}{color_b.group()}'
        return shorten(no_color, limit)
    # escape_seq_start_rindex = s.rindex('\x1b')
    # left_cutoff = max(escape_seq_start_index + 4, half_the_limit)
    # right_cutoff = min((real_length - escape_seq_start_rindex) + 4, half_the_limit)
    # print(f'{limit = } | {length = } | {real_length = } | {left_cutoff = } | {right_cutoff = } | {half_the_limit = } | {escape_seq_start_index = } | {escape_seq_start_rindex = }')
    left_cutoff = max(half_the_limit - 3, 1)
    right_cutoff = max(half_the_limit - 4, 1)
    # print(f'{limit = } | {length = } | {left_cutoff = } | {right_cutoff = } | {half_the_limit = }')
    free_chars = limit - left_cutoff - right_cutoff
    assert free_chars > 0, f'{free_chars = } not > 0'
    beginning = s[:left_cutoff]
    end = s[-right_cutoff:]
    if free_chars >= 7:
        separator = ' [...] '
    elif free_chars >= 5:
        separator = '[...]'
    elif free_chars >= 4:
        separator = ' .. '
    else:
        separator = '.' * free_chars
    assert len(separator) <= free_chars, f'{len(separator) = } ! <= {free_chars = }'
    return WHITESPACE_RE.sub(' ', f'{beginning}{separator}{end}')


class Reprer:
    def __init__(self, printer: Callable = None):
        if not printer:
            printer = functools.partial(con.print, soft_wrap=True)
        self._print = printer
        self._repr_fns = {
            lambda obj: hasattr(obj, 'url_root'): lambda obj: printer(obj, f' url_root = {obj.url_root}'),
            # lambda obj: hasattr(obj, 'payload'):  lambda obj: printer(obj, f' payload = {obj.payload}'),
            }

    def repr(self, obj):
        for condition, print_fn in self._repr_fns.items():
            if condition(obj):
                print_fn(obj)
                return
        self._print(pretty_inst(obj))


reprer = Reprer()


def markup_repr(obj) -> str:
    with con.capture() as captured:
        if isinstance(obj, (dict, list)):
            from rich.json import JSON
            try:
                pretty = JSON.from_data(obj, indent=4, sort_keys=True)
                con.print(pretty)
            except TypeError:
                # con.print(obj)  # use reprer?
                reprer.repr(obj)
        elif isinstance(obj, str) and obj.startswith("{\""):
            con.print_json(obj, indent=4, sort_keys=True)
        elif isinstance(obj, BaseException):
            con.print(f'{obj.__class__.__qualname__}({", ".join(map(repr, obj.args))})')  # use reprer?
        else:
            reprer.repr(obj)
    return captured.get().rstrip()


T = TypeVar('T')


def pretty_inst(obj: T) -> Union[str, T]:
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

    return OBJ_RE.sub(lambda _match: f'{(groups := _match.groups())[0]} ({groups[1]})', string)


def pretty_type(obj) -> str:  # unused?
    """pretty_type({"hi" : "bye"}) -> dict (1)"""
    stringified_type: str
    if isinstance(obj, str):
        return 'str'
    elif type(obj) is type:
        stringified_type = str(obj)
    else:
        stringified_type = str(type(obj))
    rv = TYPE_RE.sub(lambda match: match.groups()[0], stringified_type)
    with suppress(TypeError):
        rv += f' ({len(obj)})'
    return rv


def pretty_signature(method, *args, **kwargs) -> str:
    pretty_sig = "\x1b[97;48;2;30;30;30m" if RUN_BY_HUMAN else ""
    if hasattr(method, '__name__'): # because __qualname__ can be e.g 'AccountServiceCallback.__init__'
        method_name = method.__name__
    elif hasattr(method, '__qualname__'):
        method_name = method.__qualname__
    else:
        method_name = str(method)
    left_offset = len(method_name)
    instance_name = None
    if args:
        first_arg, *rest = args
        # if 'AccountServiceCallback' in first_arg.__class__.__name__ and '__init__' in method_name:
        #     print(f'{args[0] = }', f'{first_arg = }', f'{method_name = }', f'{hasattr(first_arg, method_name) = }', sep='\n')

        # If 'some_method(foo, *args, **kwargs)', or 'some_method(Foo, *args, **kwargs)',
        # Display it as 'Foo.some_method(*args, **kwargs)'.
        if hasattr(first_arg, method_name): # __qualname__ can include class, e.g 'AccountServiceCallback.__init__'
            obj = first_arg
            args = rest
            # if some_method(Class, *args, **kwargs), then `Class` arg is a class
            if type(obj) is type:
                instance_name = pretty_inst(obj.__qualname__)
            else:
                instance_name = pretty_inst(obj.__class__.__qualname__)
            pretty_sig += f'{instance_name}.'
            left_offset += len(instance_name) + 1

    if '.' not in method_name and not instance_name:
        # 'put' -> 'platforms_conf.put'
        pretty_sig += f'{method.__module__}.'
        left_offset += len(method.__module__) + 1
    # pretty_args = []
    # args_len = len(args)
    # for i, arg in enumerate(args):
    #     # Relies on pretty_inst returning as-is if obj doesn't match "<foo> at 0x123".
    #     pretty_arg = markup_repr(pretty_inst(repr(arg)))
    #     if i < args_len - 2:
    #         pretty_arg += '\x1b[33m,\x1b[0m'
    #         if len(pretty_arg) >= TERMWIDTH:
    #             pretty_arg += '\n' + ' ' * (len(method_name) + 4)
    #         else:
    #             pretty_arg += ' '
    #     pretty_args.append(pretty_arg)
    # pretty_args_joined = ''.join(pretty_args)
    left_offset = ' ' * (left_offset + 4)

    pretty_args = list(map(markup_repr, map(pretty_inst, map(repr, args))))
    if False and sum(map(len, pretty_args)) >= TERMWIDTH:
        pretty_args_joined = ('\x1b[1;33m,\x1b[0m\n' + left_offset).join(pretty_args)
    else:
        pretty_args_joined = '\x1b[1;33m,\x1b[0m '.join(pretty_args)
    # pretty_args_joined = ", ".join(map(markup_repr, args))

    pretty_kwargs = []
    for k, v in kwargs.items():
        # pretty_kwargs.append(f'{k}={markup_repr(v)}')
        pretty_kwargs.append(f'\x1b[33m{k}\x1b[0m={markup_repr(pretty_inst(repr(v)))}')
    pretty_kwargs_joined = f",\n{left_offset}".join(pretty_kwargs)
    pretty_sig += f'{method_name}\x1b[0m(' + pretty_args_joined + (', ' if args and kwargs else '') + pretty_kwargs_joined + ')'
    return pretty_sig

global last_log_ts
last_log_ts = 0

def log_method_calls(maybe_class_or_fn: Callable | type = None,
                     *,
                     only: tuple[str, ...] | Callable[[str], bool] = (),
                     exclude: tuple[str, ...] | Callable[[str], bool] = (),
                     short=False,
                     ) -> Callable:
    """
     A class or function decorator, logs when a method is called, and when it returns (with args and return values).

     Examples:

         # Ex. 1
         @log_method_calls
         class Calculator:
             def add(self, a, b): return a+b

         # Ex. 2
         @log_method_call(only=['add'])
         class ProCalculator:
             def add(self, a, b): return a + b
             def divide(self, a, b): return a / b

         # Ex. 3
         @log_method_calls
         def say_hello(name): print(f'Hello, {name}!')

     Args:
         only: Optionally specify `only=['some_method', 'other_method']` to only log specific methods,
               or `only=lambda x: x.startswith('_')` to only log private methods.
         exclude: Optionally specify `exclude=['dont_care']` to skip specific methods.
               or `exclude=lambda x: x.startswith('_')` to only log public methods.
     """

    def cyan(s):
        return f'\x1b[0m\x1b[2;3;36m{s}\x1b[0m'

    def decorator(cls_or_fn: Callable):
        def wrap(_method: Callable, _isstatic=False) -> Callable:
            @functools.wraps(_method)
            def inner(*args, **kwargs):
                global last_log_ts
                now = int(time.time())
                if now - last_log_ts > 60:
                    last_log_ts = now
                    print(f'\x1b[2m── {datetime.fromtimestamp(now).isoformat(" ")} ──\x1b[0m', file=sys.stderr)
                try:
                    pretty_sig = pretty_signature(_method, *args, **kwargs)
                    short_pretty_sig = shorten(pretty_sig, TERMWIDTH - 25)
                    fn_qualname, open_parenthesis, inside_parenthesis = short_pretty_sig.partition('(')
                    short_pretty_sig = f"\x1b[97;48;2;30;30;30m{fn_qualname}\x1b[0m({inside_parenthesis}"
                    if short:
                        calling = '\n' + cyan('Calling: ') + ('\n' + ' ' * 6).join(short_pretty_sig.splitlines()) + '\x1b[0m'
                    else:
                        calling = '\n' + cyan('Calling: ') + ('\n' + ' ' * 6).join(pretty_sig.splitlines()) + '\x1b[0m'

                    if calling.count('\n') > 15:
                        calling += f'\t\x1b[30m# {_method.__name__}(...)\x1b[0m'

                    calling += '\n'
                    print(calling, file=sys.stderr)
                    if _isstatic and hasattr(args[0], _method.__name__):
                        rv = _method(*args[1:], **kwargs)
                    else:
                        rv = _method(*args, **kwargs)

                    print(f"\n  {cyan('· Returning from:')} " + ('\n' + ' ' * 17).join(short_pretty_sig.splitlines()) + "\n  "
                          f"{markup_repr('[black]└─[/black]')}{cyan('-> ')}{markup_repr(rv)}\n",
                          file=sys.stderr)
                    return rv
                except Exception as e:
                    con.print_exception(max_frames=2,show_locals=True, extra_lines=3, width=TERMWIDTH)
                    print(markup_repr(e), file=sys.stderr)
                    return _method(*args, **kwargs)
                finally:
                    last_log_ts = int(time.time())

            return inner

        if inspect.isfunction(cls_or_fn):
            # todo: check if static method
            wrap = functools.wraps(cls_or_fn)(wrap)
            return wrap(cls_or_fn)

        if only:
            if inspect.isfunction(only):
                condition = only
            else:
                condition = lambda x: x in only
        elif exclude:
            if inspect.isfunction(exclude):
                condition = lambda x: not exclude(x)
            else:
                condition = lambda x: x not in exclude
        else:
            condition = lambda x: True

        # inspect.ismethod is good only if the object is an instance, not a class.
        # methods = {attrname: attr for attrname, attr in vars(cls_or_fn).items()
        #            if inspect.isfunction(attr) and condition(attrname)}
        methods = {attrname: attr for attrname, attr in inspect.getmembers(cls_or_fn)
                   # if ((inspect.isroutine(attr) or isinstance(attr, types.MethodWrapperType))
                   #     and '__getattribute__' not in attrname)
                   if inspect.isfunction(attr)  # isroutine includes all magics including __getattr__, __new__ etc
                   and condition(attrname)}
        for methodname, method in methods.items():
            isstatic = isinstance(inspect.getattr_static(cls_or_fn, methodname), staticmethod)
            wrapped = functools.wraps(method)(wrap)(method, isstatic)
            setattr(cls_or_fn, methodname, wrapped)
        return cls_or_fn

    if maybe_class_or_fn:
        return decorator(maybe_class_or_fn)
    return decorator


PY_SITE_PKGS_RE = re.compile(r'.*(python[\d.]*)/site-packages')
OLD_FACTORY = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
    record = OLD_FACTORY(*args, **kwargs)

    # Trim paths of python libs (non-asm)
    record_pathname = record.pathname
    if 'site-packages' in record_pathname:
        record.pathname = PY_SITE_PKGS_RE.sub(lambda m: f'{m.group(1)}/...', record_pathname)

    # Add colors (if colors are supported)
    if not RUN_BY_HUMAN:
        return record

    path_stem = record_pathname.rpartition('.py')[0]
    path, _, filename = path_stem.rpartition('/')

    if hasattr(record, 'asctime'):
        record.asctime = f'\x1b[38;2;180;180;180m{record.asctime}\x1b[0m'
    record.pathname = f'\x1b[38;2;180;180;180m{path}/{filename}.py\x1b[0m'
    record.funcName = f'\x1b[38;2;180;180;180m{record.funcName}\x1b[0m'

    levelname = record.levelname.lower()
    if levelname.startswith('warn'):
        record.levelname = f'\x1b[33m{record.levelname}\x1b[0m'
    elif levelname == 'error':
        record.levelname = f'\x1b[31m{record.levelname}\x1b[0m'
    elif levelname in ('critical', 'fatal'):
        record.levelname = f'\x1b[1;91m{record.levelname}\x1b[0m'
    elif levelname == 'debug':
        record.levelname = f'\x1b[35m{record.levelname}\x1b[0m'
    elif levelname == 'info':
        record.levelname = f'\x1b[36m{record.levelname}\x1b[0m'

    return record


RECORD_FILTER_SKIP_PATHS = (
    'jaeger_client',
    'connexion/operations/abstract.py',
    'connexion/apis/abstract.py',
    'connexion/decorators',
    'connexion/apis/flask_api.py',
    'connexion/operations/secure.py',
    'connexion/operations/openapi.py',
    'connexion/apps/abstract.py',
    'gunicorn/glogging.py',
    'chardet/charsetgroupprober.py',
    'chardet/mbcharsetprober.py',
    'chardet/eucjpprober.py',

    # IPython stuff
    '/parso/',
    'asyncio/selector_events.py',
    )


def record_filter(record: logging.LogRecord):
    pathname = record.pathname
    for skip_path in RECORD_FILTER_SKIP_PATHS:
        if skip_path in pathname:
            return False
    if 'openapi_spec_validator/validators.py' in pathname and 'dereference' in record.funcName:
        return False

    if 'chardet/charsetgroupprober.py' in pathname and 'get_confidence' in record.funcName:
        return False

    return True


def start_internal_log(microservice='-'):
    local_microservice = microservice
    level = os.getenv("SM_LOGLEVEL", logging.INFO)
    if not level:
        level = logging.INFO
    if isinstance(level, str) and level.isdigit():
        # e.g "10" → 10
        level = int(level)
    logger = logging.getLogger()
    logger.handlers = list()
    logger.setLevel(level)
    console = logging.StreamHandler()
    console.setLevel(level)
    sm_log_format_envvar = os.getenv('SM_LOG_FORMAT')
    if sm_log_format_envvar:
        # Example: '%(asctime)s [%(levelname)s][{microservice}][%(pathname)s:%(lineno)d][%(funcName)s()] %(message)s'
        if '{microservice}' in sm_log_format_envvar:
            log_format = sm_log_format_envvar.format(microservice=microservice)
        else:
            log_format = sm_log_format_envvar
    else:
        log_format = f'%(asctime)s [%(levelname)s][{microservice}][%(pathname)s:%(lineno)d][%(funcName)s()] %(message)s'
    console.setFormatter(logging.Formatter(log_format, datefmt='%H:%M:%S'))
    logger.addHandler(console)
    logging.captureWarnings(True)
    logging.setLogRecordFactory(record_factory)

    console.addFilter(record_filter)
    logging.info("LogLevel: %s", logging.getLevelName(logging.getLogger().level))