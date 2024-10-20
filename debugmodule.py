from __future__ import annotations
from types import ModuleType
import sys


class DebugModule(ModuleType):
    def __new__(cls, name: str, doc: str | None = None):
        if hasattr(cls, '__instance__'):
            print('returning existing instance')
            return getattr(cls, '__instance__')
        print('creating new instance')
        inst = super().__new__(cls)
        setattr(cls, '__instance__', inst)
        return inst

    def __repr__(self):
        return f'{self.__class__.__qualname__}({super().__repr__()})'

this_module = sys.modules[__name__]
this_module.__class__ = DebugModule
print(f'{__name__ = !r} | {this_module = !r}')
