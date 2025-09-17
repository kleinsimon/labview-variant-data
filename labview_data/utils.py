# -----------------------------------------------------------------------------
# Copyright (c) 2025 Simon Josef Klein
#
# This work is licensed under the MIT License.
# See the LICENSE file in this repository or visit:
# https://opensource.org/licenses/MIT
#
# Author: Simon Klein
# Date: 2025
# Source: https://github.com/kleinsimon/labview-variant-data
# -----------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Iterable, List, Optional, Type, Any, Dict, Collection, Union, Set
from numbers import Number
from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from .types import Cluster, NamedItem


class LVDtypes:
    """
    Encapsulation of common data types and code page used in a specific context.

    This class serves as a collection of predefined data types compatible with
    NumPy, formatted with specific endianness or encoding. It also includes the
    default code page for handling character encodings. The class does not provide
    methods or behaviors but acts as a static container for related properties.

    :ivar u1: Unsigned 1-byte integer (NumPy data type).
    :type u1: numpy.dtype
    :ivar u2: Unsigned 2-byte big-endian integer (NumPy data type).
    :type u2: numpy.dtype
    :ivar u4: Unsigned 4-byte big-endian integer (NumPy data type).
    :type u4: numpy.dtype
    :ivar u8: Unsigned 8-byte big-endian integer (NumPy data type).
    :type u8: numpy.dtype
    :ivar i1: Signed 1-byte integer (NumPy data type).
    :type i1: numpy.dtype
    :ivar i2: Signed 2-byte big-endian integer (NumPy data type).
    :type i2: numpy.dtype
    :ivar i4: Signed 4-byte big-endian integer (NumPy data type).
    :type i4: numpy.dtype
    :ivar i8: Signed 8-byte big-endian integer (NumPy data type).
    :type i8: numpy.dtype
    :ivar f4: 4-byte big-endian floating-point number (NumPy data type).
    :type f4: numpy.dtype
    :ivar f8: 8-byte big-endian floating-point number (NumPy data type).
    :type f8: numpy.dtype
    :ivar codepage: Default code page used for encoding, set to "cp1252".
    :type codepage: str
    """
    u1 = np.dtype("u1")
    u2 = np.dtype(">u2")
    u4 = np.dtype(">u4")
    u8 = np.dtype(">u8")
    i1 = np.dtype("i1")
    i2 = np.dtype(">i2")
    i4 = np.dtype(">i4")
    i8 = np.dtype(">i8")
    f4 = np.dtype(">f4")
    f8 = np.dtype(">f8")
    codepage = "cp1252"


def num2bytes(number: Union[Number, Iterable], dtype=LVDtypes.u2) -> bytes:
    """
    Converts a number or an iterable of numbers into a bytes object using the specified data type.
    The function utilizes NumPy array conversion to achieve this.

    :param number: A single number or an iterable of numbers to be converted into a bytes object.
    :param dtype: The data type used for the conversion. Default is `LVDtypes.u2`.
    :return: A bytes object representing the converted numerical value(s).
    :rtype: bytes
    """
    return np.array(number, dtype=dtype).tobytes()


def bytes2num(buffer: bytes, offset=0, count: int=1, dtype=LVDtypes.u2, scalar=True) -> Tuple[int, int]:
    """
    Convert a buffer of bytes into a numerical value or an array of numerical values
    based on the specified data type and count.

    The function reads the data from the provided byte buffer starting at the given
    offset, interprets the data as the specified type, and returns the resulting
    value(s) along with the updated offset. If the `scalar` parameter is set to True
    and only one value is requested, the function will return the numerical value as
    a scalar; otherwise, it will return as a numpy array.

    :param buffer: The input byte buffer containing the data to be converted.
    :param offset: The starting byte offset in the buffer. Defaults to 0.
    :param count: The number of elements to read from the buffer. Defaults to 1.
    :param dtype: The data type of the elements to interpret from the buffer.
    :param scalar: A flag indicating whether to return a single scalar value (if
        count is 1) or an array. Defaults to True.
    :return: A tuple consisting of the interpreted value(s) and the updated offset.
    """

    val = np.frombuffer(buffer, offset=offset, count=count, dtype=dtype)
    offset = int(offset) + int(val.itemsize) * count

    if scalar and count == 1:
        return val[0], offset
    else:
        return val, offset


def str2bytes(value: str, s_dtype=LVDtypes.u4, fill=False) -> bytes:
    """
    Convert a string to a bytes object with optional length prefix and padding.

    This function encodes the provided string into bytes using the defined
    codepage, prefixes the length of the encoded bytes (in a specified data type),
    and optionally ensures the resulting byte sequence has an even length by
    appending a padding byte.

    :param value: The string to be encoded into bytes.
    :param s_dtype: The data type to use for encoding the length prefix.
    :param fill: A boolean flag that, if True, appends a null byte to ensure
                 an even-length output.
    :return: A bytes object containing the length prefix, the encoded string,
             and optional padding.
    """
    value = value.encode(LVDtypes.codepage)

    buffer = num2bytes(len(value), dtype=s_dtype) + value
    if fill and len(buffer) % 2 != 0:
        buffer += b"\00"

    return buffer


def bytes2str(buffer: bytes, offset, s_dtype=LVDtypes.u4, fill=False) -> Tuple[str, int]:
    """
    Converts a binary byte buffer into a string, interpreting the data at a
    specified offset and decoding it using a provided string size type.

    This function decodes a string from binary data. The string size is
    retrieved from the buffer using the specified data type for size decoding.
    Optionally, padding may be added to maintain alignment if required.

    :param buffer: Byte buffer containing the binary data.
    :type buffer: bytes
    :param offset: Offset in the buffer where the string data begins.
    :type offset: int
    :param s_dtype: Data type describing the size of the string,
        typically one of `LVDtypes` types.
    :type s_dtype: Any
    :param fill: Whether to add padding to maintain alignment for even
        sizes, if applicable.
    :type fill: bool
    :return: A tuple containing the decoded string and the updated offset
        in the buffer after decoding the string.
    :rtype: Tuple[str, int]
    """

    size, offset = bytes2num(buffer, offset, dtype=s_dtype, count=1)
    size = int(size)
    string = buffer[offset:offset + size].decode(LVDtypes.codepage)
    offset += size

    if fill and size + size.itemsize % 2 != 0:
        offset += 1

    return string, offset


epoch = datetime(year=1904, month=1, day=1, tzinfo=timezone.utc)
frac_to_microseconds = 1e6 / (2 ** 64)


def bytes2date(buffer: bytes, offset_d: int) -> Tuple[datetime, int]:
    """
    Converts a byte buffer containing encoded date and time information into a datetime
    object and an updated offset value.

    The function reads 16 bytes from the provided buffer at the specified offset
    and interprets them to determine the number of seconds and the fraction of a
    second (in microseconds) since the epoch. It then computes the corresponding
    datetime value, updates the offset accordingly, and returns the result.

    :param buffer: The byte buffer containing encoded datetime information.
    :type buffer: bytes
    :param offset_d: The starting position in the buffer to read the datetime data.
    :type offset_d: int
    :return: A tuple containing the decoded datetime and the updated offset.
    :rtype: Tuple[datetime, int]
    """
    s, radix = np.frombuffer(buffer, offset=offset_d, dtype=[("", ">i8"), ("", ">u8")], count=1)[0]
    dt = timedelta(seconds=int(s), microseconds=int(radix * frac_to_microseconds))
    date = epoch + dt
    offset_d += 16
    return date, offset_d


def date2bytes(value: datetime) -> bytes:
    """
    Converts a given `datetime` object to a byte representation. The method calculates
    the total seconds and fractional microseconds since a predefined epoch time and
    converts these values into a byte format. This is particularly useful for encoding
    datetime data for serialization or storage.

    :param value: The datetime object to be converted to bytes.
    :type value: datetime
    :return: A byte array representing the encoded datetime object.
    :rtype: bytes
    """
    dif = value - epoch

    secs = np.int64(dif.total_seconds())
    usecs = np.uint64(dif.microseconds / frac_to_microseconds)
    data = num2bytes(secs, ">i8") + num2bytes(usecs, ">u8")
    return data


def splitnumber(number, dtype=">u4"):
    """
    Splits a given number array into an array of smaller components based on the provided data type.

    The function takes a numeric array, converts it into bytes using its internal representation,
    and then interprets those bytes as a new array of smaller components (e.g., lower and upper
    halves of integers) specified by the given data type.

    :param number: A numeric input array to be split into smaller components.
    :param dtype: The data type of the resulting array components (default: ">u4").
    :return: A numpy array containing the smaller components derived from the input array.
    """
    number = np.asarray(number)
    buf = number.tobytes()
    return np.frombuffer(buf, count=number.size*2, dtype=dtype)


@dataclass
class StructType:
    """
    Represents a structure type with specific data type properties.

    This class is used to define and manage structured data types, typically in cases
    involving numpy data types or other type systems. It provides a way to pair a
    generalized type with a specific numpy dtype for specialized data handling.

    :ivar type: The general type or numpy dtype associated with this structure.
    :ivar s_dtype: The specific numpy dtype tied to this structure instance.
    """
    type: Union[Type, np.dtype] = None
    s_dtype: np.dtype = LVDtypes.u4


@dataclass
class StructElement(StructType):
    """
    Represents a structure element which can hold a value of any type or another StructElement.

    This class is used to represent an element that belongs to a specific structure type while allowing
    its value to be dynamically assigned. The value can either be of any type or another instance of
    StructElement, enabling the creation of nested structures. The `type` attribute dynamically assigns
    itself based on the value provided if not explicitly defined.

    :ivar value: The actual value of the structure element. It can either be of any type or another
        `StructElement` instance.
    :type value: Union[Any, StructElement]
    """
    value: Union[Any, "StructElement"] = None

    def __post_init__(self):
        if self.type is None:
            self.type = type(self.value)


StructValue = Tuple[Type, Optional[int]]


def lv_dump(element: Union[Iterable[StructElement], StructElement], dtype=None, s_dtype=LVDtypes.u4) -> bytes:
    """
    Constructs a binary representation of the given element(s) based on the specified types.

    This function takes an element or a collection of elements and encodes them into a binary
    representation using the specified `dtype` and `s_dtype`. Elements can be numerical, string,
    boolean, or datetime values, and their specific encoding technique is determined by their
    associated types.

    :param element: A single `StructElement` or an iterable of `StructElement` objects that
        need to be serialized.
    :param dtype: Optional. The type to be used for numeric or data serialization of the
        `StructElement`.
    :param s_dtype: Defines the string dtype to be used specifically for encoding purposes.
        Defaults to `LVDtypes.u4`, which defines the data structure of encoded strings.
    :return: A serialized binary `bytes` object representing the given element(s).
    :rtype: bytes
    :raises ValueError: If the encoding fails or an invalid element is passed.
    """
    buffer = None
    if not isinstance(element, StructElement):
        element = StructElement(value=element, type=dtype, s_dtype=s_dtype)

    if isinstance(element, Iterable):
        buffer = b""
        for e in element:
            buffer += lv_dump(e)

    elif isinstance(element, StructElement):
        value = element.value
        e_t = element.type

        if np.issubdtype(e_t, np.number):
            buffer = num2bytes(value, dtype=e_t)

        elif issubclass(e_t, str):
            buffer = str2bytes(value, element.s_dtype)

        elif issubclass(e_t, bool):
            buffer = num2bytes(value, LVDtypes.u1)

        elif issubclass(e_t, datetime):
            buffer = date2bytes(value)

    if not buffer:
        raise ValueError

    return buffer


def lv_parse(dtype, buffer: bytes, offset: int, count=1, scalar=True, s_dtype=LVDtypes.u4, e_dtype=LVDtypes.f8) -> Tuple[List, int]:
    """
    Parses a given buffer and returns a deserialized value along with the updated offset.
    The function can handle different data types, including numeric types, strings, booleans,
    dates, and arrays, and it supports scalar and vector formats.

    :param dtype: Data type to parse (may include types like numeric, string, datetime, etc.).
    :type dtype: type
    :param buffer: The binary buffer from which data is parsed.
    :type buffer: bytes
    :param offset: The starting position in the buffer for parsing.
    :type offset: int
    :param count: The number of elements to parse (default is 1).
    :type count: int
    :param scalar: Indicates whether to parse data as a single scalar value or a list.
    :type scalar: bool
    :param s_dtype: Starting data type for array-based parsing.
    :param e_dtype: Element data type for array values when parsing.
    :return: Returns a tuple containing the parsed value and the updated offset.
    :rtype: Tuple[List, int]
    """
    offset = int(offset)

    value = None

    if isinstance(dtype, tuple):
        value = []
        for i in range(count):
            v, offset = lv_parse(dtype[i], buffer, offset)
            value.append(v)
        value = tuple(value)

    elif np.issubdtype(dtype, np.number):
        value, offset = bytes2num(buffer, offset, count, scalar=scalar, dtype=dtype)

    elif issubclass(dtype, datetime):
        value = []
        for i in range(count):
            v, offset = bytes2date(buffer, offset)
            value.append(v)
        if scalar:
            value = value[0]

    elif issubclass(dtype, str):
        value = []
        for i in range(count):
            v, offset = bytes2str(buffer, offset, s_dtype)
            value.append(v)
        if scalar:
            value = value[0]

    elif issubclass(dtype, bool):
        value, offset = bytes2num(buffer, offset, count, scalar=scalar, dtype=LVDtypes.u1)
        value = value == 1

    elif issubclass(dtype, np.ndarray):
        n, offset = bytes2num(buffer, offset, 1, s_dtype, True)
        value, offset = bytes2num(buffer, offset, n, e_dtype, False)

    return value, offset


@dataclass
class DeserializationData:
    """
    Holds data required for deserialization of binary structures.

    This class provides essential attributes for handling deserialization,
    including buffer details, position offsets, and hierarchical depth.
    It includes methods to replace attributes, parse numeric data, fork new
    instances with modified properties, and parse headers. It is designed
    to support complex deserialization tasks with optional handling for
    nested data structures and shape information.

    :ivar header: Reference to the `HeaderInfo` instance.
    :type header: HeaderInfo
    :ivar buffer: The binary data being deserialized.
    :type buffer: bytes
    :ivar offset_d: Current deserialization offset within the buffer.
    :type offset_d: int
    :ivar depth: Depth of the current deserialization operation in recursive
        processes. Defaults to 0.
    :type depth: int
    :ivar count: Number of elements to process during the deserialization. Defaults to 1.
    :type count: int
    :ivar version: Version of the serialization format. Defaults to 0.
    :type version: int
    :ivar scalar: A flag indicating whether the deserialization expects scalar
        data. Defaults to True.
    :type scalar: bool
    :ivar shape: Collection describing the expected shape of the deserialized
        data, if applicable. Defaults to None.
    :type shape: Collection[int]
    :ivar header_lut: Optional lookup table for headers, which allows efficient
        deserialization. Defaults to None.
    :type header_lut: List[HeaderInfo]
    :ivar fill_header_words: A flag specifying whether to populate header words during
        header parsing. Defaults to True.
    :type fill_header_words: bool
    """

    header: "HeaderInfo"
    buffer: bytes
    offset_d: int
    depth: int = 0
    count: int = 1
    version: int = 0
    scalar: bool = True
    shape: Collection[int] = None
    header_lut: List["HeaderInfo"] = None
    fill_header_words: bool = True

    def replace(self, **kwargs) -> "DeserializationData":
        """
        Replaces the current instance with a new instance of the same class, updating any specified
        attributes with new values.

        :param kwargs: Arbitrary keyword arguments used to set or override attributes in the new
            instance. The keys correspond to attribute names, and values replace current values
            with new ones.
        :return: A new instance of the same class with updated attributes.
        :rtype: DeserializationData
        """
        return replace(self, **kwargs)

    def parse_num(self, offset, count=1, dtype=LVDtypes.u2) -> Tuple[int, int]:
        """
        Parses a numeric value from the buffer and returns it along with the next
        offset to be read. The method converts bytes from the buffer into a numeric
        representation based on the provided data type (`dtype`). The `count`
        determines the number of elements to interpret, and `offset` specifies the
        position in the buffer where reading should begin.

        :param offset: The starting position in the buffer for reading the numeric
            value.
        :type offset: int
        :param count: The number of numeric values to interpret. Defaults to 1.
        :type count: int, optional
        :param dtype: The data type used for converting bytes to a numeric
            representation. Defaults to LVDtypes.u2.
        :type dtype: type
        :return: A tuple where the first item is the interpreted numeric value(s)
            and the second is the next offset position in the buffer.
        :rtype: Tuple[int, int]
        """
        return bytes2num(self.buffer, count=count, dtype=dtype, offset=offset)

    def fork(self, **kwargs) -> "DeserializationData":
        """
        Creates a new instance of DeserializationData with updated parameters.

        This method generates a forked version of the current DeserializationData
        object, updating or overriding certain parameters provided in **kwargs. When
        parameters are not explicitly provided in **kwargs, they are reset to their
        default values from the class level.

        :param kwargs: Arbitrary keyword arguments that override or update the
                       default properties of the new DeserializationData instance.
        :return: A new instance of DeserializationData with updated properties.
        :rtype: DeserializationData
        """
        # reset specific params to default
        if "count" not in kwargs:
            kwargs["count"] = DeserializationData.count
        if "scalar" not in kwargs:
            kwargs["scalar"] = DeserializationData.scalar
        if "shape" not in kwargs:
            kwargs["shape"] = DeserializationData.shape
        return self.replace(depth=self.depth+1, **kwargs)

    def parse_header(self, offset_h):
        """
        Parses a header at the specified offset in the buffer and returns the
        corresponding header object along with the updated offset value.

        This method checks the existence of a header lookup table (header_lut).
        If it exists, it retrieves and processes the header index and offset
        from the buffer, utilizing the lookup table to fetch the corresponding
        header. Otherwise, the header is parsed directly using the `parse`
        method of the header object with optional padding based on the
        `fill_header_words` property.

        :param offset_h: The offset position in the buffer where the parsing
            starts.
        :type offset_h: int
        :return: A tuple containing the header object and the updated offset
            in the buffer.
        :rtype: tuple
        """
        if self.header_lut:
            h_idx, offset_h = bytes2num(self.buffer, offset=offset_h, count=1, dtype=LVDtypes.u2)
            return self.header_lut[h_idx], int(offset_h)
        else:
            h = self.header.parse(self.buffer, offset_h=offset_h, fill=self.fill_header_words)
            return h, int(h.start) + int(h.size)


@dataclass
class SerializationData:
    """
    Represents data used for serialization with attributes for version, name, depth,
    and dtype. Provides utility methods to create modified or derived instances.

    This class is intended to handle serialization-related data and provides
    functionalities to update or fork existing instances, enabling modifications
    or creation of subitems while preserving immutability.

    :ivar version: The version number relevant to the serialized data.
    :type version: int
    :ivar name: An optional name or identifier for the serialized data.
    :type name: str
    :ivar depth: Indicates the depth level of the item in a hierarchy. Defaults to 0.
    :type depth: int
    :ivar dtype: An optional data type descriptor for the serialized data.
    :type dtype: np.dtype
    """

    version: int
    name: str = None
    depth: int = 0
    dtype: np.dtype = None

    def replace(self, **kwargs) -> "SerializationData":
        """
        Replaces the attributes of the current instance with new values provided
        in the `kwargs` parameter. This method enables creating a new object with
        updated data while maintaining immutability of the original instance.

        :param kwargs: The attributes and their respective new values to replace
            in a copy of the current instance.
        :return: A new instance of the class with updated attributes of type
            ``SerializationData``.
        """
        return replace(self, **kwargs)

    def fork(self, **kwargs) -> "SerializationData":
        """
        Creates and returns a new instance with an incremented depth and updated attributes.

        This method is used to create a new instance of the object while modifying the
        `depth` attribute by incrementing it by 1 and optionally replacing other attributes
        by using keyword arguments.

        :param kwargs: Arbitrary attributes to be replaced or added to the new instance.
        :type kwargs: dict
        :return: A new instance with updated `depth` and other specified attributes.
        :rtype: SerializationData
        """

        return self.replace(depth=self.depth+1, **kwargs)


@dataclass
class SerializationResult:
    """
    Represents the result of a serialization process, encapsulating metadata
    and serialized components.

    This class is part of a serialization system that stores metadata, serialized
    data, and structural hierarchy of objects. It includes functionality to access,
    flatten, and manipulate serialization details, such as headers, buffers, sub-results,
    and their interrelations. Instances of this class can construct hierarchical
    serialization data and manage its representation.

    :ivar code: An integer representing the serialization code or identifier.
    :type code: int
    :ivar header: A bytes object containing the header information of the serialized object.
    :type header: bytes
    :ivar buffer: A bytes object containing the serialized data buffer.
    :type buffer: bytes
    :ivar depth: Indicates the depth level of the object in a hierarchy.
    :type depth: int
    :ivar sub_results: Iterable containing `SerializationResult` objects representing
        subordinate serialized results. Defaults to None if not applicable.
    :type sub_results: Iterable[SerializationResult]
    :ivar header_indices: Iterable of integers specifying indices to a subset of
        sub_results for header computation purposes. Defaults to None if not applicable.
    :type header_indices: Iterable[int]
    :ivar name: An optional string name associated with the serialized object. Defaults to None.
    :type name: str
    :ivar shape: A collection of integers representing the shape of the serialized data,
        if applicable. Defaults to None.
    :type shape: Collection[int]
    """

    code: int
    header: bytes
    buffer: bytes
    depth: int
    sub_results: Iterable["SerializationResult"] = None
    header_indices: Iterable[int] = None
    name: str = None
    shape: Collection[int] = None

    @property
    def header_q(self):
        """
        Constructs the `header_q` property by combining various components based on the
        object state and attributes. Prepends the resulting bytes sequence with an
        appropriate value depending on whether the `name` attribute is defined and then
        appends other calculated byte sequences.

        :return: A bytes sequence representing the constructed header.
        :rtype: bytes
        """
        return (b"\x00" if not self.name else b"\x40") + num2bytes(self.code, dtype=LVDtypes.u1) + self.header

    def all_sub_results(self, res=None) -> List["SerializationResult"]:
        """
        Generates a flat list of all the current object's results including its sub-results.

        This method recursively collects the current serialization result and its nested
        sub-results into a flat list. The main result is always included, followed by
        all sub-results, maintaining the hierarchical traversal order.

        :param res: Accumulator list to collect all results, starting with the current
            result itself. If not provided, an empty list is initialized.
        :return: A list of all `SerializationResult` objects, including the current
            result and all sub-results gathered recursively.
        :rtype: List[SerializationResult]
        """
        if res is None:
            res = []

        res.append(self)

        if self.sub_results:
            for sub in self.sub_results:
                sub.all_sub_results(res)
        return res

    def flat_header(self, include_sub_results=True, lut=None, force_empty_string=False) -> bytes:
        """
        Creates a flattened header in bytes format including optional sub-results and LUT transformation.

        The method combines the header of the current instance with headers from sub-results if required.
        It supports incorporation of a lookup table (LUT) mapping and allows forcing an empty string
        as part of the header when no name is available.

        :param include_sub_results: Whether to include headers from sub-results. Defaults to True.
        :type include_sub_results: bool
        :param lut: Optional lookup table (LUT) mapping for header transformation.
        :type lut: dict, optional
        :param force_empty_string: Whether to force including an empty string in the header. Defaults to False.
        :type force_empty_string: bool
        :return: A byte sequence of the flattened header with combined results.
        :rtype: bytes
        """
        h = self.header_q

        if include_sub_results and self.sub_results:
            results = self.sub_results
            if self.header_indices:
                results = [results[i] for i in self.header_indices]
            for res in results:
                subh = res.flat_header(include_sub_results, lut, force_empty_string)
                if lut:
                    h += lut[subh]
                else:
                    h += subh

        if self.name:
            h += str2bytes(self.name, s_dtype=LVDtypes.u1, fill=True)

        elif force_empty_string:
            h += b"\00"

        return num2bytes(len(h)+2) + h

    def flat_buffer(self) -> bytes:
        """
        Flattens the buffer and all sub-results into a single bytes object.

        This method concatenates the `buffer` with the flattened buffers of all
        sub-results, if any exist. It recursively collects the results from all
        nested sub-results and returns a single binary representation.

        :return: A single binary sequence combining the `buffer` and all nested
                 sub-results.
        :rtype: bytes
        """
        b = self.buffer
        if self.sub_results:
            b += b"".join([res.flat_buffer() for res in self.sub_results])
        return b

    def replace(self, **kwargs) -> "SerializationResult":
        """
        Replaces the properties of the current instance with new values, using the
        provided keyword arguments, and returns a new instance of the class with the
        updated properties.

        This method leverages the `replace` function to create a modified copy of the
        current instance while keeping other attributes unchanged.

        :param kwargs: Keyword arguments with names corresponding to the properties
            to replace and their respective new values.
        :return: A new instance of the class with updated values for the specified
            properties.
        :rtype: SerializationResult
        """
        return replace(self, **kwargs)


NameStruct = Union[str, Tuple[str, "NameStruct"]]


@dataclass
class DeserializationResult:
    """
    Represents the result of a deserialization process.

    DeserializationResult is used to encapsulate the outcome of deserializing
    data, including offsets, scalar values, associated metadata, and any nested
    deserialization results. This class provides utility methods and properties
    to access and manipulate the deserialization outcome.

    :ivar offset_d: The offset in deserialized data.
    :type offset_d: int
    :ivar offset_h: The header offset in deserialized data.
    :type offset_h: int
    :ivar scalar: The scalar value obtained from deserialization, if any.
    :type scalar: Any
    :ivar info: Metadata associated with the deserialization process.
    :type info: DeserializationData
    :ivar items: Nested deserialization results, if applicable.
    :type items: Optional[Collection[DeserializationResult]]
    """

    offset_d: int
    offset_h: int
    scalar: Any = None
    info: DeserializationData = None
    items: Optional[Collection["DeserializationResult"]] = None

    def replace(self, **kwargs) -> "DeserializationResult":
        """
        Replace the current instance with a new instance, optionally updated with
        provided keyword arguments.

        :param kwargs: Optional keyword arguments to update the instance.
        :type kwargs: dict
        :return: A new instance of `DeserializationResult` with updated values.
        :rtype: DeserializationResult
        """
        return replace(self, **kwargs)

    @property
    def depth(self):
        """
        Property that retrieves the depth information from the object's info attribute.

        :rtype: int
        :return: The depth value as an integer derived from the info attribute of the object.
        """
        return self.info.depth

    @property
    def has_name(self) -> bool:
        """
        Determines if a name is present or reachable based on the offset and
        header information.

        :return: A boolean indicating whether a name exists.
        :rtype: bool
        """
        return self.offset_h < self.info.header.end or self.info.header.name

    @property
    def name(self) -> Optional[str]:
        """
        Retrieve the name of the current object, if available.

        This property attempts to extract the name from the object's header
        or buffer based on its defined structure and offsets. It first checks
        if the `name` is directly available in the header, then tries to
        determine the name from its buffer if the `has_name` attribute is set.
        If no name is available, the property will return `None`.

        :rtype: Optional[str]
        :return: The name of the object if available, otherwise None.
        """
        if self.info.header.name:
            return self.info.header.name
        if self.has_name:
            name, offset = bytes2str(self.info.buffer, offset=self.offset_h, s_dtype=LVDtypes.u1)
            return name

    @property
    def value(self):
        """
        Gets the current scalar value.

        This property provides access to the scalar value stored in the object.
        It can be used to retrieve the current state of the scalar without
        directly modifying it.

        :return: The current scalar value.
        :rtype: Any
        """
        return self.scalar

    @property
    def named_item(self) -> NamedItem:
        """
        Provides access to a specific property representing a named item. The named item
        is a data structure that combines a value and its associated name.

        :rtype: NamedItem
        :return: A NamedItem instance containing the value and name.
        """
        return NamedItem(self.value, self.name)


class ArrayDeserializationResult(DeserializationResult):
    """
    Represents the result of deserialization of an array structure.

    The class provides access to the deserialized values and named
    items, with potential reshaping based on additional shape
    information. It extends the functionality of a generic
    `DeserializationResult` by handling multidimensional data.

    :ivar items: A collection of deserialized items. Each `item`
        is expected to have a `value` and `named_item` attribute.
    :type items: Iterable[Any]
    :ivar info: Contains metadata about the deserialized structure,
        such as shape information used for reshaping arrays.
    :type info: Any
    :ivar name: The name assigned to the deserialized data item.
    :type name: str
    """
    @property
    def value(self) -> Iterable[Any]:
        values = [item.value for item in self.items]
        ndim = len(self.info.shape) if self.info.shape else 0
        if ndim > 1:
            values = np.array(values, dtype=object)
            values = values.reshape(self.info.shape)
        return values

    @property
    def named_item(self) -> NamedItem:
        values = [item.named_item for item in self.items]
        ndim = len(self.info.shape) if self.info.shape else 0
        if ndim > 1:
            values = np.array(values, dtype=object)
            values = values.reshape(self.info.shape)
        return NamedItem(values, self.name)


class ClusterDeserializationResult(DeserializationResult):
    """
    Represents the result obtained from deserializing data into a cluster-like structure.

    This class is used to structure deserialization results into a `Cluster` instance
    and retrieve related cluster attributes such as named items. The class extends
    functionality by working with items to compose their values into a cluster format.

    :ivar items: A collection of deserialized items that constitute the content of
        the Cluster.
    :type items: List[DeserializationResult]
    :ivar name: Name of the cluster being deserialized.
    :type name: str
    """
    @property
    def value(self) -> Cluster:
        return Cluster([item.value for item in self.items], [item.name for item in self.items])

    @property
    def named_item(self) -> NamedItem:
        value = Cluster([item.named_item for item in self.items], [item.name for item in self.items])
        return NamedItem(value, self.name)


class MapDeserializationResult(DeserializationResult):
    """
    Represents the result of deserializing a map structure.

    This class is a specific implementation of the DeserializationResult, which is tailored
    to handle map-like data structures. It provides properties to extract key-value pairs
    from deserialized results and enables working with named items built from the deserialization
    process.

    :ivar name: The name associated with the deserialization result.
    :type name: str
    :ivar items: A collection of items representing the deserialized key-value pairs.
    :type items: List[Tuple[DeserializationResult, DeserializationResult]]
    """
    @property
    def value(self) -> Dict:
        return {k_item.value: v_item.value for k_item, v_item in self.items}

    @property
    def named_item(self) -> NamedItem:
        value = {k_item.named_item: v_item.named_item for k_item, v_item in self.items}
        return NamedItem(value, self.name)


class SetDeserializationResult(DeserializationResult):
    """
    Represents the result of deserialization for a set of items.

    This class is used to handle and encapsulate the results when a set of
    items is deserialized. It provides properties to access specific
    deserialized values or related information.
    """
    @property
    def value(self) -> Set:
        return {item.value for item in self.items}

    @property
    def named_item(self) -> NamedItem:
        value = {item.named_item for item in self.items}
        return NamedItem(value, self.name)


@dataclass
class HeaderInfo:
    """
    Represents the header information used within a specific context involving
    buffer parsing or processing.

    This class encapsulates details such as code, data segment offsets, size,
    start location, and other metadata for a header structure. It provides
    convenient access to computed properties like end location of the data
    segment and a converter associated with the header code. It also includes
    methods to parse a buffer into a header structure and replace attributes
    in an immutable fashion.

    :ivar code: The type identifier code of the header.
    :type code: int
    :ivar offset_h: The offset in the buffer where the data begins.
    :type offset_h: int
    :ivar size: The size of the data segment defined by this header.
    :type size: int
    :ivar start: The starting index of the header in the buffer.
    :type start: int
    :ivar name: The name or identifier for the header. Defaults to None.
    :type name: str or None
    :ivar has_str: Indicates if the header contains string data. Defaults to False.
    :type has_str: bool
    """
    code: int
    offset_h: int
    size: int
    start: int
    name: str = None
    has_str: bool = False

    @property
    def end(self) -> int:
        """
        Computes and returns the end point by adding the `start` and `size` attributes.

        :return: The computed end value as an integer.
        :rtype: int
        """
        return int(self.start) + int(self.size)

    @property
    def converter(self):
        """
        Provides access to the type converter associated with the current code.

        This property retrieves an instance of `LVTypeConverter` corresponding to
        the `code` attribute of the object. It serves as a utility to perform
        various data conversions based on the specific type code.

        :return: An instance of `LVTypeConverter` suitable for the object's
                 associated code.
        :rtype: LVTypeConverter
        """
        return LVTypeConverter.get_converter_for_code(self.code)

    @staticmethod
    def parse(buffer: bytes, offset_h: int, fill=True) -> "HeaderInfo":
        """
        Parses a buffer to extract header information starting from a specified offset.

        This method interprets the buffer content to determine properties such as
        size, type, and flags, and may also adjust the size for alignment considerations
        or handle corrections for certain anomalies. The method returns a `HeaderInfo`
        object containing the parsed information.

        :param buffer: The input byte buffer from which the header information is
            extracted.
        :type buffer: bytes
        :param offset_h: The initial position in the buffer where parsing begins.
        :type offset_h: int
        :param fill: Indicates whether size alignment corrections should be applied.
        :type fill: bool
        :return: A `HeaderInfo` object containing details about the parsed header.
        :rtype: HeaderInfo
        """
        start = offset_h
        size, offset_h = bytes2num(buffer, offset_h, count=1, dtype=LVDtypes.u2)
        flags = buffer[offset_h]
        p_type = buffer[offset_h + 1]
        offset_h += 2
        if fill and size % 2 != 0:
            size += 1

        if fill and size == 0:
            size = 8 #Fix for corrupt empty variants from labview with zero size

        return HeaderInfo(
            code=p_type,
            offset_h=offset_h,
            size=size,
            start=start,
            has_str=flags
        )

    def replace(self, **kwargs) -> "HeaderInfo":
        """
        Replaces the current instance with a new one, updating specified fields.

        This method allows you to generate a new instance of the class by copying
        the current instance and updating specified fields with new values. The
        method maintains the immutability of the original instance by returning a
        new modified instance without altering the original one.

        :param kwargs: Arbitrary keyword arguments for updating specific fields
            of the current instance.
        :return: A new instance of the class with updated fields, based on the
            provided keyword arguments.
        :rtype: HeaderInfo
        """
        return replace(self, **kwargs)


class LVTypeConverter(ABC):
    """
    Abstract Interface of all type-convertes
    """
    supported_codes: Iterable[int] = []
    supported_types: Iterable[Type] = []
    order = 0

    serializers: Dict[Type, Type["LVTypeConverter"]] = {}
    serializers_list: List[Tuple[Type, Type["LVTypeConverter"]]] = []
    deserializers: Dict[int, Type["LVTypeConverter"]] = {}

    default_array_converter: "LVTypeConverter" = None

    @classmethod
    @abstractmethod
    def _serialize(cls, value: Any, info: SerializationData) -> SerializationResult:
        """
        Serialize a value into a serialization result. This method is designed to be
        implemented by subclasses to handle specific serialization logic. The method
        converts the input value into a serialized representation including buffer data,
        header information, and other serialization metadata.

        :param value: The value to serialize.
        :type value: Any
        :param info: Additional serialization-related data, which may include depth or
            other information required for serialization.
        :type info: SerializationData
        :return: A `SerializationResult` object containing the serialized code, buffer,
            header, and depth metadata of the serialized content.
        :rtype: SerializationResult
        """
        return SerializationResult(
            code=0,
            buffer=b"",
            header=b"",
            depth=info.depth
        )

    @classmethod
    def serialize(cls, value: Union[Any, NamedItem], info: SerializationData) -> SerializationResult:
        """
        Serializes a given value into a `SerializationResult` using the provided
        serialization data, while handling cases where the value is an instance of
        `NamedItem`. The `name` attribute of the `NamedItem` is extracted and used
        during serialization.

        :param value: The input value or `NamedItem` to be serialized.
        :param info: A `SerializationData` instance containing auxiliary
            information for the serialization process.
        :return: A `SerializationResult` object containing the serialized output.
        """

        if isinstance(value, NamedItem):
            info.name = value.name
            value = value.item

        res = cls._serialize(value, info)

        res.name = info.name

        return res

    @classmethod
    @abstractmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        """
        Deserialize data from the provided deserialization information.

        This class method is abstract and must be implemented by subclasses
        to handle the deserialization of specific objects based on the given
        `DeserializationData`. The method returns the `DeserializationResult`,
        containing the deserialized object and updated offset information.

        :param info: Deserialization data containing the source information
                     and required context for deserialization.
        :type info: DeserializationData
        :return: Result of the deserialization process, including the object
                 deserialized and the offset information.
        :rtype: DeserializationResult
        """
        return DeserializationResult(None, info.offset_d)

    @classmethod
    def deserialize(cls, info: DeserializationData) -> DeserializationResult:
        """
        Deserializes the provided deserialization data and returns the result. This method
        utilizes the internal deserialization process and attaches the original
        info to the result for reference.

        :param info: The deserialization data to be processed
        :type info: DeserializationData
        :return: The result of the deserialization process with associated info
        :rtype: DeserializationResult
        """
        res = cls._deserialize(info)
        res.info = info
        return res

    @classmethod
    def serialize_array(cls, value, info: SerializationData, object_mode=False) -> SerializationResult:
        """
        Serializes an array using a default array converter.

        This method is a class method designed to handle the serialization of an
        array using the default array converter available in the class. It uses
        the provided serialization information and can operate in object mode if
        specified.

        :param value: The array or list of items to be serialized.
        :type value: Any
        :param info: Additional serialization data or metadata used during the
            serialization process.
        :type info: SerializationData
        :param object_mode: A flag indicating whether object mode serialization
            should be used. Defaults to False.
        :type object_mode: bool
        :return: An instance of `SerializationResult` containing the serialized
            data.
        :rtype: SerializationResult
        """
        return cls.default_array_converter.serialize_array(value, info=info, object_mode=object_mode)

    @classmethod
    def deserialize_array(cls, info: DeserializationData) -> DeserializationResult:
        """
        Deserializes the input deserialization data into a specific result using the default
        array converter. This method is typically used to process an array-like data structure
        during deserialization.

        :param info: DeserializationData
            The input data required for deserialization. Must be an object adhering to the
            DeserializationData structure.
        :return: DeserializationResult
            The deserialized output generated from the input, formatted according to the
            default array conversion logic.
        """
        return cls.default_array_converter.deserialize_array(info)

    def __init_subclass__(cls, **kwargs):
        for code in cls.supported_codes:
            LVTypeConverter.deserializers[code] = cls

        for t in cls.supported_types:
            LVTypeConverter.serializers[t] = cls
            LVTypeConverter.serializers_list.append((t, cls))

        LVTypeConverter.serializers_list.sort(key=lambda s: -s[1].order)

    @classmethod
    def get_converter_for_value(cls, value) -> Type["LVTypeConverter"]:
        """
        Retrieve the appropriate converter for a given value.

        This method determines the type of the provided value and attempts to find a
        matching converter in the registered serializers. If a direct match is not
        found, it iterates through the list of available serializers to locate the
        appropriate converter for the value.

        :param value: The input value for which a converter is to be found.
        :return: The corresponding type converter for the provided value.
        :rtype: Type[LVTypeConverter]
        :raises ValueError: If no suitable converter is found for the type of the value.
        """

        if isinstance(value, NamedItem):
            value = value.item

        try:
            return cls.serializers[type(value)]

        except KeyError:
            pass

        for t, ser in cls.serializers_list:
            if isinstance(value, t):
                return ser

        raise ValueError(f"no converter found for type {str(type(value))}")

    @classmethod
    def get_converter_for_type(cls, dtype) -> Type["LVTypeConverter"]:
        """
        Gets the appropriate converter for the provided data type.

        Searches through the registered serializers to find a converter that is
        compatible with the given data type. If no converter is found, raises
        a ValueError indicating that no converter exists for the specified type.

        :param dtype: The data type for which a converter is being requested.
        :type dtype: Type
        :return: A converter class that matches the given data type.
        :rtype: Type[LVTypeConverter]
        :raises ValueError: If no converter is found for the specified data type.
        """
        for t in cls.serializers.keys():
            if issubclass(dtype, t):
                return cls.serializers[t]

        raise ValueError(f"no converter found for type {str(dtype)}")

    @classmethod
    def get_converter_for_code(cls, code: int) -> Type["LVTypeConverter"]:
        """
        Retrieves the appropriate converter class for the given code.

        This method looks up the `deserializers` dictionary within the class to find
        the corresponding converter class for the provided integer code.

        :param code: The integer code representing the desired converter.
        :return: The converter class corresponding to the given code.
        :rtype: Type[LVTypeConverter]
        """

        return cls.deserializers[code]


