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
    err_type = (datetime, f8, u4)


def num2bytes(number: Union[Number, Iterable], dtype=LVDtypes.u2) -> bytes:
    """
    convert a number to bytes using s specific type
    :param number:
    :param dtype:
    :return:
    """
    return np.array(number, dtype=dtype).tobytes()


def bytes2num(buffer: bytes, offset=0, count: int=1, dtype=LVDtypes.u2, scalar=True) -> Tuple[int, int]:
    """
    parse a number from bytes
    :param buffer:
    :param offset: the offset of the start of the number or array
    :param count: number of scalars to retrieve
    :param dtype: type of all numbers
    :param scalar: if true, return a scalar, otherwise an array
    :return:
    """

    val = np.frombuffer(buffer, offset=offset, count=count, dtype=dtype)
    offset = int(offset) + int(val.itemsize) * count

    if scalar and count == 1:
        return val[0], offset
    else:
        return val, offset


def str2bytes(value: str, s_dtype=LVDtypes.u4, fill=False) -> bytes:
    """
    convert a string to a pascal string by prepending the size of the following string
    :param value:
    :param s_dtype: the type of the number, defaults to >u4
    :param fill: if true, fill to full words
    :return:
    """
    value = value.encode(LVDtypes.codepage)

    buffer = num2bytes(len(value), dtype=s_dtype) + value
    if fill and len(buffer) % 2 != 0:
        buffer += b"\00"

    return buffer


def bytes2str(buffer: bytes, offset, s_dtype=LVDtypes.u4, fill=False) -> Tuple[str, int]:
    """
    parse a pascal string from a buffer
    :param buffer:
    :param offset:
    :param s_dtype:
    :param fill: if true, skip uneven end byte
    :return:
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
    s, radix = np.frombuffer(buffer, offset=offset_d, dtype=[("", ">i8"), ("", ">u8")], count=1)[0]
    dt = timedelta(seconds=int(s), microseconds=int(radix * frac_to_microseconds))
    date = epoch + dt
    offset_d += 16
    return date, offset_d


def date2bytes(value: datetime) -> bytes:
    dif = value - epoch

    secs = np.int64(dif.total_seconds())
    usecs = np.uint64(dif.microseconds / frac_to_microseconds)
    data = num2bytes(secs, ">i8") + num2bytes(usecs, ">u8")
    return data


def splitnumber(number, dtype=">u4"):
    """
    split a number into values with smaller bytesize (i64 -> 2x i32)
    :param number:
    :param dtype:
    :return:
    """
    number = np.asarray(number)
    buf = number.tobytes()
    return np.frombuffer(buf, count=number.size*2, dtype=dtype)


@dataclass
class StructType:
    type: Union[Type, np.dtype] = None
    s_dtype: np.dtype = LVDtypes.u4


@dataclass
class StructElement(StructType):
    value: Union[Any, "StructElement"] = None

    def __post_init__(self):
        if self.type is None:
            self.type = type(self.value)


StructValue = Tuple[Type, Optional[int]]


def lv_dump(element: Union[Iterable[StructElement], StructElement], dtype=None, s_dtype=LVDtypes.u4) -> bytes:
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
    Holds information needed to deserialize a specific entry
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
        return replace(self, **kwargs)

    def parse_num(self, offset, count=1, dtype=LVDtypes.u2) -> Tuple[int, int]:
        return bytes2num(self.buffer, count=count, dtype=dtype, offset=offset)

    def fork(self, **kwargs) -> "DeserializationData":
        # reset specific params to default
        if "count" not in kwargs:
            kwargs["count"] = DeserializationData.count
        if "scalar" not in kwargs:
            kwargs["scalar"] = DeserializationData.scalar
        if "shape" not in kwargs:
            kwargs["shape"] = DeserializationData.shape
        return self.replace(depth=self.depth+1, **kwargs)

    def parse_header(self, offset_h):
        if self.header_lut:
            h_idx, offset_h = bytes2num(self.buffer, offset=offset_h, count=1, dtype=LVDtypes.u2)
            return self.header_lut[h_idx], int(offset_h)
        else:
            h = self.header.parse(self.buffer, offset_h=offset_h, fill=self.fill_header_words)
            return h, int(h.start + h.size)


@dataclass
class SerializationData:
    """
    Holds information of a specific value in a structure
    """

    version: int
    name: str = None
    depth: int = 0

    def replace(self, **kwargs) -> "SerializationData":
        """
        return a copy od this item with modified values
        :param kwargs:
        :return:
        """
        return replace(self, **kwargs)

    def fork(self, **kwargs) -> "SerializationData":
        """
        returns information for a subitem (depth+1) with optional changed params
        :param kwargs:
        :return:
        """

        return self.replace(depth=self.depth+1, **kwargs)


@dataclass
class SerializationResult:
    """
    Holds data of a serialized item in the structure
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
        return (b"\x00" if not self.name else b"\x40") + num2bytes(self.code, dtype=LVDtypes.u1) + self.header

    def all_sub_results(self, res=None) -> List["SerializationResult"]:
        if res is None:
            res = []

        res.append(self)

        if self.sub_results:
            for sub in self.sub_results:
                sub.all_sub_results(res)
        return res

    def flat_header(self, include_sub_results=True, lut=None, force_empty_string=False) -> bytes:
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
        b = self.buffer
        if self.sub_results:
            b += b"".join([res.flat_buffer() for res in self.sub_results])
        return b

    def replace(self, **kwargs) -> "SerializationResult":
        return replace(self, **kwargs)


NameStruct = Union[str, Tuple[str, "NameStruct"]]


@dataclass
class DeserializationResult:
    """
    Holds data of a deserialized item in the structure
    """

    offset_d: int
    offset_h: int
    scalar: Any = None
    info: DeserializationData = None
    items: Optional[Collection["DeserializationResult"]] = None

    def replace(self, **kwargs) -> "DeserializationResult":
        return replace(self, **kwargs)

    @property
    def depth(self):
        return self.info.depth

    @property
    def has_name(self) -> bool:
        return self.offset_h < self.info.header.end

    @property
    def name(self) -> Optional[str]:
        if self.has_name:
            name, offset = bytes2str(self.info.buffer, offset=self.offset_h, s_dtype=LVDtypes.u1)
            return name

    @property
    def value(self):
        return self.scalar

    @property
    def named_item(self) -> NamedItem:
        return NamedItem(self.value, self.name)


class ArrayDeserializationResult(DeserializationResult):
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
    @property
    def value(self) -> Cluster:
        return Cluster([item.value for item in self.items], [item.name for item in self.items])

    @property
    def named_item(self) -> NamedItem:
        value = Cluster([item.named_item for item in self.items], [item.name for item in self.items])
        return NamedItem(value, self.name)


class MapDeserializationResult(DeserializationResult):
    @property
    def value(self) -> Dict:
        return {k_item.value: v_item.value for k_item, v_item in self.items}

    @property
    def named_item(self) -> NamedItem:
        value = {k_item.named_item: v_item.named_item for k_item, v_item in self.items}
        return NamedItem(value, self.name)


class SetDeserializationResult(DeserializationResult):
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
    provides information of a type-header.

    code the lv-datacode
    offset_h offset AFTER thy typecode
    size of the complete header
    resolves to the name of the entry if given
    """
    code: int
    offset_h: int
    size: int
    start: int
    name: str = None

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def converter(self):
        return LVTypeConverter.get_converter_for_code(self.code)

    @staticmethod
    def parse(buffer: bytes, offset_h: int, fill=True) -> "HeaderInfo":
        start = offset_h
        size, offset_h = bytes2num(buffer, offset_h, count=1, dtype=LVDtypes.u2)

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
            start=start
        )

    def replace(self, **kwargs) -> "HeaderInfo":
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
        return SerializationResult(
            code=0,
            buffer=b"",
            header=b"",
            depth=info.depth
        )

    @classmethod
    def serialize(cls, value: Union[Any, NamedItem], info: SerializationData) -> SerializationResult:
        """
        serialize the given value with the information given in info.
        :param value:
        :param info:
        :return:
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
        return DeserializationResult(None, info.offset_d)

    @classmethod
    def deserialize(cls, info: DeserializationData) -> DeserializationResult:
        """
        deserialize the portion of the input buffer described in 'info'
        :param info:
        :return:
        """
        res = cls._deserialize(info)
        res.info = info
        return res

    @classmethod
    def serialize_array(cls, value, info: SerializationData, object_mode=False) -> SerializationResult:
        """
        specific method to serialize arrays.
        :param value:
        :param info:
        :param object_mode:
        :return:
        """
        return cls.default_array_converter.serialize_array(value, info=info, object_mode=object_mode)

    @classmethod
    def deserialize_array(cls, info: DeserializationData) -> DeserializationResult:
        """
        specific method to deserialize arrays.
        :param info:
        :return:
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
        get the type converter for a specific value
        :param value:
        :return:
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
        get the type converter for a specific value
        :param value:
        :return:
        """
        for t in cls.serializers.keys():
            if issubclass(dtype, t):
                return cls.serializers[t]

        raise ValueError(f"no converter found for type {str(dtype)}")

    @classmethod
    def get_converter_for_code(cls, code: int) -> Type["LVTypeConverter"]:
        """
        get the type converter for a lv type code
        :param value:
        :return:
        """

        return cls.deserializers[code]


