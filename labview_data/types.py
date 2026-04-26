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
import dataclasses
import typing
from datetime import datetime
from typing import Iterable, Union, Optional, Dict, Any
from enum import IntEnum
import numpy as np
from numpy.typing import NDArray

try:
    # numpy >= 2.0
    FLEX_STRING_DTYPE = np.dtypes.StringDType

except AttributeError:
    # numpy <2.0
    FLEX_STRING_DTYPE = object


class TypedItem:
    def __init__(self, item_type: Any = None):
        self.item_type: Any = item_type


class Cluster(tuple):
    """
    Represents a cluster, which is a specialized tuple allowing access to
    elements by index or named keys. This class is designed to provide
    flexible element retrieval and supports dictionary-like behavior
    for named keys.

    Detailed description:
    The `Cluster` class is a subtype of the `tuple` class which extends
    its functionality by allowing elements to be retrieved either by their
    integer indexes or by assigned names (if provided). `Cluster` also
    supports conversion to a dictionary and attribute-style access for
    named elements.

    :ivar names: A dictionary mapping names to their respective indices.
    :type names: Dict[str, int]
    """

    def __new__(cls, values: Iterable, names: Union[Iterable[Optional[str]], Dict[int, str]] = None):
        """
        Creates a new instance of the class, allowing for optional association
        of names with indices.

        This method initializes the iteration-based object with a given iterable
        of values. Optionally, it allows assignment of meaningful names to specific
        indices within the iterable, either through a dictionary mapping indices to
        names, or through an iterable providing names correlatively by order.
        The assigned names will allow for easier identification of elements
        within the object in subsequent operations.

        :param values: The iterable of elements to initialize the instance with.
        :type values: Iterable
        :param names: Optional mapping or iterable of names corresponding to specific
                      indices in the values iterable. If a dictionary is provided,
                      keys represent indices and values represent names. If an iterable
                      is provided, the names are assigned based on their position
                      in the iterable.
        :type names: Union[Iterable[Optional[str]], Dict[int, str]]
        :return: An instance of the class with named indices, if names are provided.
        :rtype: object
        """

        o = super().__new__(cls, values)

        o._names = dict()  # Benannte Werte
        o._names_list = [None for _ in range(len(values))]

        if isinstance(names, dict):
            for i, name in names.items():
                if name:
                    o._names[name] = i

                o._names_list[i] = name

        elif isinstance(names, Iterable):
            for i, name in enumerate(names):
                if name:
                    o._names[name] = i

                o._names_list[i] = name

        return o

    @property
    def names(self):
        """
        Provides access to the `names` attribute.

        The `names` property retrieves a value representing a collection of names encapsulated
        in the private attribute `_names`. This property is read-only and ensures that the data
        within `_names` can only be accessed but not directly modified.

        :return: The collection of names.
        :rtype: Any
        """
        return self._names

    @property
    def name_list(self):
        return self._names_list

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
    """
    A subclass of numpy.ndarray that associates a name with the array.

    The NamedArray class extends the functionality of numpy.ndarray by allowing
    an optional name to be associated with the array. The name is stored as an
    instance attribute and can be used to label the data contained within
    the array.

    :ivar name: The name associated with the array.
    :type name: Optional[str]
    """

    def __new__(cls, array, name: Optional[str] = "", **kwargs):
        """
        Creates and initializes a new instance of the class. This method converts the given
        array to a NumPy array-like object and assigns a name to the new instance.

        :param array: The input data that will be converted to a NumPy array-like object
                      and used to initialize the new class instance.
        :type array: any

        :param name: An optional string representing the name of the new instance. Defaults
                     to an empty string.
        :type name: Optional[str]

        :param kwargs: Additional optional arguments to pass to the NumPy `asarray` method.
        :type kwargs: dict

        :returns: A new instance of the class initialized with the converted NumPy array.
        :rtype: cls
        """
        obj = np.asarray(array, **kwargs).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', "")


class StringArray(NamedArray):
    """
    A specific NamedArray designed to hold variable-length LabVIEW strings.
    Automatically forces the correct flexible string dtype.
    """
    string_dtype = FLEX_STRING_DTYPE

    def __new__(cls, array, name: Optional[str] = "", **kwargs):
        kwargs['dtype'] = cls.string_dtype
        obj = super().__new__(cls, array, name=name, **kwargs)

        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)

    @classmethod
    def is_string_array(cls, array: np.ndarray):
        if isinstance(array, cls):
            return True

        if array.dtype == object or FLEX_STRING_DTYPE == object:
            return all([isinstance(o, str) for o in array.flat])

        if array.dtype == FLEX_STRING_DTYPE:
            return True

        return False

class TypedList(list, TypedItem):
    pass


@dataclasses.dataclass
class NamedItem:
    """
    Represents an item with an optional name.

    This class is used to associate a generic item with an optional string name.
    It is particularly useful when you need to provide contextual names for items
    while maintaining flexibility with the type of the item.

    :ivar item: The main item associated with the instance.
    :ivar name: An optional name or label for the item.
    """
    value: Any
    name: Optional[str] = None


class ExtendedIntEnum(IntEnum):
    """
    Provides an extended enumeration type inheriting from IntEnum.

    This class enables creating enumeration members dynamically for
    integer values not explicitly defined in the enumeration by
    overriding the `_missing_` method.

    It is particularly useful in scenarios where missing integer values
    need to be represented safely without raising errors, while still
    leveraging the benefits of enumerations.
    """
    @classmethod
    def _missing_(cls, value):
        """
        Handles cases where an expected enum value is missing. This is invoked automatically
        when accessing an enum member with a value that doesn't correspond to any predefined
        enum members.

        :param cls: The enum class being used.
        :param value: The value for which an enum member is not found.
        :return: Returns a dynamically created enum instance with the given value.
        """
        obj = int.__new__(cls, value)  # Erstellt eine Instanz mit dem int-Wert
        obj._value_ = value
        obj._name_ = str(value)  # Setzt den Namen auf die String-Repräsentation
        return obj


class Signal(np.ndarray):
    """
    Represents a signal as a subclass of a numpy ndarray, with additional attributes
    and functionality to handle temporal information and metadata.

    This class is designed to represent ordered data points (signal) with associated
    temporal information (start time, time step) and optional metadata. The class adds
    several properties and methods to manage and query this information efficiently.

    :ivar t0: The initial time of the signal.
    :type t0: datetime
    :ivar dt: The time difference between consecutive signal samples.
    :type dt: float
    :ivar attributes: Optional metadata or attributes associated with the signal.
    :type attributes: Any
    """
    def __new__(cls, y, t0: datetime, dt: float, attributes: Any = None):
        # Input array wird in ein ndarray konvertiert und zur Instanz dieser Klasse gemacht
        obj = np.asarray(y).view(cls)
        obj.t0 = t0
        obj.dt = dt
        obj.attributes = attributes
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.t0 = getattr(obj, 't0', None)
        self.dt = getattr(obj, 'dt', None)
        self.attributes = getattr(obj, 'attributes', None)

    def __repr__(self):
        base_repr = repr(self.y)
        return (f"AnalogSignal(t0={self.t0!r}, dt={self.dt!r}, attributes={self.attributes!r},\n"
                f"            y={base_repr})")

    @property
    def y(self):
        """
        Provides access to a view of the current object as a NumPy ndarray.

        This property allows the user to retrieve a NumPy ndarray view
        of the current object for further manipulation or usage.

        :return: A NumPy ndarray view of the instance.
        :rtype: np.ndarray
        """
        return self.view(np.ndarray)

    @property
    def delta(self) -> np.timedelta64:
        """
        Property that calculates and returns a time delta as a NumPy timedelta64 object.

        :return: The calculated time delta as a NumPy timedelta64 object in nanoseconds.
        :rtype: numpy.timedelta64
        """
        return np.timedelta64(int(self.dt * 1e9), "ns")

    @property
    def start(self) -> np.datetime64:
        """
        A property that provides the starting time as a numpy datetime64 object.

        This property calculates and returns the starting time (`t0`) converted to
        a numpy datetime64 object.

        :return: Starting time as a numpy datetime64 object.
        :rtype: numpy.datetime64
        """
        return np.datetime64(self.t0)

    @property
    def end(self) -> np.datetime64:
        """
        Gets the calculated end time.

        The property computes the end time by adding the product of `size` and
        `delta` to the `start` time.

        :return: The calculated end time as a numpy datetime64 object.
        :rtype: np.datetime64
        """
        return self.start + self.size * self.delta

    @property
    def times(self) -> NDArray:
        """
        The `times` property calculates and returns an array of evenly spaced values
        based on the initial `start` value, `delta` increment, and the total number
        of elements defined by `size`. It is a computed property that does not take
        any external arguments and dynamically generates an array when accessed.

        :return: Computed array containing evenly spaced time values.
        :rtype: NDArray
        """
        return self.start + self.delta * np.arange(self.size)

    def to_timeseries(self) -> typing.Tuple[NDArray, NDArray]:
        """
        Converts and returns stored data into a time-series format.

        This method transforms the stored attributes into a tuple containing
        time and corresponding data values, formatted as NumPy arrays.

        :return: A tuple containing the time values and corresponding data
            values.
        :rtype: typing.Tuple[NDArray, NDArray]
        """
        return self.times, self.y


@dataclasses.dataclass
class Variant(NamedItem):
    pass
