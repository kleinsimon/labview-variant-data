#  Copyright 2025 Simon Klein
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

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Iterable, List, Optional, Type, Any, Dict
from numbers import Number
from dataclasses import dataclass, replace


class LVDtypes:
    u1 = np.dtype("u1")
    u2 = np.dtype(">u2")
    u4 = np.dtype(">u4")
    u8 = np.dtype(">u8")
    i1 = np.dtype("i1")
    i2 = np.dtype(">i2")
    i4 = np.dtype(">i4")
    i8 = np.dtype(">i8")
    codepage = "cp1252"


def num2bytes(number: Number, dtype=LVDtypes.u2) -> bytes:
    return np.array(number, dtype=dtype).tobytes()


def bytes2num(buffer: bytes, offset=0, count: int=None, dtype=LVDtypes.u2, scalar=True) -> Tuple[int, int]:
    val = np.frombuffer(buffer, offset=offset, count=count, dtype=dtype)
    offset += val.itemsize

    if scalar and count == 1:
        return val[0], offset
    else:
        return val, offset


def str2bytes(value: str, s_dtype=LVDtypes.u4, fill=False) -> bytes:
    value = value.encode(LVDtypes.codepage)

    buffer = num2bytes(len(value), dtype=s_dtype) + value
    if fill and len(buffer) % 2 != 0:
        buffer += b"\00"

    return buffer


def bytes2str(buffer: bytes, offset, s_dtype=LVDtypes.u4, fill=False) -> Tuple[str, int]:
    size, offset = bytes2num(buffer, offset, dtype=s_dtype, count=1)
    string = buffer[offset:offset + size].decode(LVDtypes.codepage)
    offset += size

    if fill and size + size.itemsize % 2 != 0:
        offset += 1

    return string, offset


def splitnumber(number, dtype=">u4"):
    number = np.asarray(number)
    buf = number.tobytes()
    return np.frombuffer(buf, count=number.size*2, dtype=dtype)


@dataclass
class DeserializationData:
    header: "HeaderInfo"
    buffer: bytes
    offset_d: int
    depth: int = 0
    count: int = 1
    version: int = 0
    scalar: bool = True
    header_lut: List["HeaderInfo"] = None
    fill_header_words: bool = True

    def next(self, offset_d, count=1) -> "DeserializationData":
        h_start = self.header.offset_h + self.header.size
        if self.version == 0:
            h = HeaderInfo.parse(self.buffer, h_start)

        elif self.version == 0x18008000:
            idx = bytes2num(self.buffer, offset=h_start, count=1, dtype=LVDtypes.u2)
            h = self.header_lut[idx]

        return self.replace(header=h, offset_d=offset_d, count=count)

    def replace(self, **kwargs) -> "DeserializationData":
        return replace(self, **kwargs)

    def parse_num(self, offset, count=1, dtype=LVDtypes.u2) -> Tuple[int, int]:
        return bytes2num(self.buffer, count=count, dtype=dtype, offset=offset)

    def fork(self, **kwargs) -> "DeserializationData":
        return self.replace(depth=self.depth+1, **kwargs)

    def parse_header(self, offset_h):
        if self.header_lut:
            h_idx, offset_h = bytes2num(self.buffer, offset=offset_h, count=1, dtype=LVDtypes.u2)
            return self.header_lut[h_idx], offset_h
        else:
            h = self.header.parse(self.buffer, offset_h=offset_h, fill=self.fill_header_words)
            return h, h.start + h.size


@dataclass
class SerializationData:
    version: int
    name: str = None
    depth: int = 0

    def replace(self, **kwargs) -> "SerializationData":
        return replace(self, **kwargs)

    def fork(self, **kwargs) -> "SerializationData":
        return self.replace(depth = self.depth+1, **kwargs)


@dataclass
class SerializationResult:
    code: int
    header: bytes
    buffer: bytes
    depth: int
    sub_results: Iterable["SerializationResult"] = None
    header_indices: Iterable[int] = None
    name: str = None

    @property
    def header_q(self):
        return num2bytes(self.code, dtype=LVDtypes.u2) + self.header

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


@dataclass
class DeserializationResult:
    value: object
    offset_d: int
    offset_h: int
    depth: int = 0
    info: DeserializationData = None

    def replace(self, **kwargs) -> "DeserializationResult":
        return replace(self, **kwargs)

    @property
    def has_name(self) -> bool:
        return self.info and self.offset_h < self.info.header.end

    @property
    def name(self) -> Optional[str]:
        if self.has_name:
            name, offset = bytes2str(self.info.buffer[:self.info.header.end], offset=self.offset_h, s_dtype=LVDtypes.u1)
            return name


@dataclass
class HeaderInfo:
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
    supported_codes: Iterable[int] = []
    supported_types: Iterable[Type] = []

    serializers: Dict[Type, Type["LVTypeConverter"]] = {}
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
    def serialize(cls, value: Any, info: SerializationData) -> SerializationResult:
        res = cls._serialize(value, info.replace(name=None))

        if info.name is not None and info.name != "":
            res.replace(
                name=info.name
            )

        return res

    @classmethod
    @abstractmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        return DeserializationResult(None, info.offset_d)

    @classmethod
    def deserialize(cls, info: DeserializationData) -> DeserializationResult:
        res = cls._deserialize(info)
        res.info = info
        return res

    @classmethod
    def serialize_array(cls, value, info: SerializationData, object_mode=False)\
            -> Tuple[SerializationResult, Iterable[int]]:
        return cls.default_array_converter.serialize_array(value, info=info, object_mode=object_mode)

    @classmethod
    def deserialize_array(cls, info: DeserializationData, shape: Iterable[int] = ()) -> DeserializationResult:
        return cls.default_array_converter.deserialize_array(info, shape)

    def __init_subclass__(cls, **kwargs):
        for code in cls.supported_codes:
            LVTypeConverter.deserializers[code] = cls

        for t in cls.supported_types:
            LVTypeConverter.serializers[t] = cls

    @classmethod
    def get_converter_for_value(cls, value) -> Type["LVTypeConverter"]:
        for t in cls.serializers.keys():
            if isinstance(value, t):
                return cls.serializers[t]
        return cls

    @classmethod
    def get_converter_for_code(cls, code: int) -> Type["LVTypeConverter"]:
        return cls.deserializers[code]

