#  Copyright 2025
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#  and associated documentation files (the “Software”), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
#  subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
#  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#  OTHER DEALINGS IN THE SOFTWARE.

from typing import Iterable, Union, Optional, Dict, Any
import numpy as np


class Cluster(tuple):
    """
    Combines namedtuple and list:
    - Has ordered items accessible by index
    - Allows naming of items
    - Access per name or index
    """

    def __new__(cls, values: Iterable, names: Union[Iterable[Optional[str]], Dict[int, str]] = None):
        """
        Initialize a new cluster

        :param values: values (Iterable)
        :param kwargs:  Names. Either a list of names with None for unnamed
                        or a dict with indices as keys and names as values
        """

        o = super().__new__(cls, values)

        o._names = dict()  # Benannte Werte

        if isinstance(names, dict):
            for i, name in names.items():
                if name:
                    o._names[name] = i

        elif isinstance(names, Iterable):
            for i, name in enumerate(names):
                if name:
                    o._names[name] = i

        return o

    @property
    def names(self):
        return self._names

    def __getitem__(self, key: Any):
        """
        Zugriff auf Werte:
        - Integer: Gibt das Element an der Position zurück.
        - String: Gibt das benannte Element zurück.
        """

        if isinstance(key, str):  # Zugriff per Name
            key = self._names[key]

        if isinstance(key, int):  # Zugriff per Index auf unbenannte Werte
            return super().__getitem__(key)
        else:
            raise TypeError("Index muss ein int (Index) oder str (benannter Key) sein.")

    def __getattr__(self, name):
        """Ermöglicht Zugriff per `obj.attr` für benannte Werte."""

        if name in self._names:
            return self._names[name]
        raise AttributeError(f"'{self.__class__.__name__}' hat kein Attribut '{name}'")

    def __dict__(self):
        """Ermöglicht die Umwandlung in ein Dictionary mit dict()."""
        return {name: self[index] for name, index in self._names.items()}

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._names.keys()
        else:
            super().__contains__(item)


class NamedArray(np.ndarray):
    """Array with metadata."""

    def __new__(cls, array, name: Optional[str] = "", **kwargs):
        obj = np.asarray(array, **kwargs).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', "")
