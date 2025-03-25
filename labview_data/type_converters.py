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

import abc
from typing import MutableSequence, Union, Dict, Iterable, Optional, Any, Tuple, Type, List
from numbers import Number
from pathlib import Path
from datetime import datetime, timezone, timedelta
from enum import IntEnum, auto

import numpy as np

from .utils import HeaderInfo, DeserializationData, SerializationData, DeserializationResult, SerializationResult
from .utils import MapDeserializationResult, ArrayDeserializationResult, ClusterDeserializationResult
from .utils import (bytes2num, bytes2str, num2bytes, LVDtypes, LVTypeConverter, SetDeserializationResult, str2bytes,
                    date2bytes, bytes2date, lv_parse, lv_dump, StructElement)
from .types import ExtendedIntEnum, Signal, Signal


class NumericConverter(LVTypeConverter):
    num_data = {        # lv_type --> dtype
        0x01: ">i1",    #i8
        0x02: ">i2",    #i16
        0x03: ">i4",    #i32
        0x04: ">i8",    #i64
        0x05: ">u1",    #u8
        0x06: ">u2",    #u16
        0x07: ">u4",    #u32
        0x08: ">u8",    #u64
        0x09: ">f4",    #f32
        0x0B: ">g",     #f128
        0x0A: ">f8",    #f64
        0x0C: ">F",     #c64
        0x0D: ">D",     #c128
        0x0E: ">G",     #c256
        0x21: ">?",     #bool
    }
    num_types = {code: np.dtype(short) for code, short in num_data.items()}
    num_data_rev = {dtype.name: (code, dtype) for code, dtype in num_types.items()}

    supported_codes = list(num_data.keys())
    supported_types = (Number, bool)

    @classmethod
    def _deserialize(cls, info: DeserializationData):
        dtype = np.dtype(cls.num_data[info.header.code])
        size = int(dtype.itemsize)
        offset_d = int(info.offset_d)
        value = np.frombuffer(info.buffer, offset=offset_d, dtype=dtype, count=info.count)
        offset_d += size * int(value.size)

        if info.scalar and value.size == 1:
            value = value[0]

        offset_h = int(info.header.offset_h)

        if value.dtype != np.bool_ and info.version >= 0x08508002:
            offset_h += 1

        return DeserializationResult(
            scalar=value,
            offset_d=offset_d,
            offset_h=offset_h,
            info=info
            )

    @classmethod
    def deserialize_array(cls, info: DeserializationData) -> DeserializationResult:
        result = cls.deserialize(info.replace(scalar=False))
        value = np.asarray(result.value)
        ndim = len(info.shape) if info.shape else 0

        if ndim == 0:
            value = value[0]

        else:
            value = value.reshape(info.shape)

        result.scalar = value
        return result

    @classmethod
    def _serialize(cls, value, info: SerializationData):
        value = np.asarray(value)
        dtype = value.dtype
        code, dtype = cls.num_data_rev[dtype.name]
        value = value.astype(dtype)

        buffer = value.tobytes()
        header = b""

        if value.dtype != np.bool_ and info.version >= 0x08508002:
            header += b"\00"

        return SerializationResult(
            code=code,
            header=header,
            buffer=buffer,
            depth=info.depth,
            shape=value.shape
        )

    @classmethod
    def serialize_array(cls, value, info: SerializationData, object_mode=False):
        if object_mode:
            raise ValueError("Numpy converter supports no variant object type")
        result = cls.serialize(value, info)
        return result


class StringConverter(LVTypeConverter):
    supported_codes = (0x30, )
    supported_types = (str, )

    @classmethod
    def _deserialize(cls, info: DeserializationData):
        value, offset_d = bytes2str(info.buffer, info.offset_d, s_dtype=LVDtypes.u4)

        return DeserializationResult(
            scalar=value,
            offset_d=offset_d,
            offset_h=info.header.offset_h + 4,
            info=info
        )

    @classmethod
    def _serialize(cls, value: str, info: SerializationData):
        buffer = value.encode(LVDtypes.codepage)

        return SerializationResult(
            code=0x30,
            header=b"\xff\xff\xff\xff",
            buffer=num2bytes(len(buffer), LVDtypes.u4) + buffer,
            depth=info.depth
        )


class PathConverter(LVTypeConverter):
    supported_codes = (0x32,)
    supported_types = (Path,)

    @classmethod
    def _serialize(cls, value: Path, info: SerializationData):
        size = len(value.parts)
        buffer = b""

        for part in value.parts:
            part = part.replace("\\", "")
            part = part.replace(":", "")
            part = part.replace("..", "")
            buffer += num2bytes(len(part), "u1") + str(part).encode("ansi")

        p_type = b"\00\00" if value.is_absolute() else b"\00\x01"

        return SerializationResult(
            code=0x32,
            depth=info.depth,
            header=b"\xff\xff\xff\xff",
            buffer=b"PTH0"
                   + num2bytes(len(buffer) + 4, LVDtypes.u4)
                   + p_type + num2bytes(size, LVDtypes.u2)
                   + buffer
        )

    @classmethod
    def _deserialize(cls, info: DeserializationData):
        offset_d = int(info.offset_d)
        if not info.buffer[offset_d:offset_d+4] == b"PTH0":
            raise ValueError
        offset_d += 4
        size, offset_d = bytes2num(info.buffer, offset=offset_d, count=1, dtype=LVDtypes.u4)
        end = int(offset_d + size)

        ptype, offset_d = bytes2num(info.buffer, offset=offset_d, count=1, dtype=LVDtypes.u2)
        count, offset_d = bytes2num(info.buffer, offset=offset_d, count=1, dtype=LVDtypes.u2)

        parts = []
        for i in range(count):
            s, offset_d = bytes2num(info.buffer, offset=offset_d, count=1, dtype=LVDtypes.u1)
            s = int(s)
            if s == 0:
                parts.append("..\\")
            else:
                parts.append(info.buffer[offset_d:offset_d+s].decode("ansi") + "\\")
            offset_d += int(s)

        if ptype == 0:
            #absolute path. This obviusly works only in windows
            parts[0] = parts[0][0] + ":\\"

        return DeserializationResult(
            scalar=Path(*parts),
            offset_d=end,
            offset_h=info.header.offset_h + 4,
            info=info
        )


class ArrayConverter(LVTypeConverter):
    supported_codes = (0x40,)
    supported_types = (np.ndarray, np.generic, MutableSequence)

    @classmethod
    def _serialize(cls, value: Any, info: SerializationData) -> SerializationResult:
        item_types = [type(i) for i in value]

        kwargs = {}
        if all([t == item_types[0] for t in item_types]):
            subt_converter = LVTypeConverter.get_converter_for_value(value[0])
            kwargs["object_mode"] = False
        else:
            subt_converter = cls

        sub_result = subt_converter.serialize_array(value, info.fork(), **kwargs)
        ndim = len(sub_result.shape)

        return SerializationResult(
            code=0x40,
            header=num2bytes(ndim, LVDtypes.u2) + b"\xff\xff\xff\xff",
            buffer=num2bytes(sub_result.shape, LVDtypes.u4),
            sub_results=(sub_result, ),
            header_indices=(0, ),
            depth=info.depth
        )

    @classmethod
    def _deserialize(cls, info: DeserializationData):
        offset_h = int(info.header.offset_h)
        ndim, offset_h = bytes2num(info.buffer, offset=offset_h, count=1, dtype=LVDtypes.u2)
        offset_h += 4

        subh, offset_h = info.parse_header(offset_h)

        shape, offset_d = bytes2num(info.buffer, offset=info.offset_d, count=ndim, dtype=LVDtypes.u4, scalar=False)

        subt_converter = LVTypeConverter.get_converter_for_code(subh.code)

        res = subt_converter.deserialize_array(info.fork(header=subh, offset_d=offset_d, count=np.prod(shape), shape=shape))
        res.offset_h = offset_h

        return res

    @classmethod
    def serialize_array(cls, value, info: SerializationData, object_mode=False) -> SerializationResult:
        o_array = np.array(value, dtype=object)
        shape = o_array.shape
        o_array = o_array.flatten()

        if not object_mode:
            converter = LVTypeConverter.get_converter_for_value(o_array[0])
        else:
            converter = VariantConverter

        results = [converter.serialize(o, info) for o in o_array]

        return results[0].replace(buffer=b"".join([res.flat_buffer() for res in results]), shape=shape)

    @classmethod
    def deserialize_array(cls, info: DeserializationData):
        items = []

        offset_h = info.header.offset_h
        offset_d = info.offset_d

        for i in range(info.count):
            item = info.header.converter.deserialize(info.replace(offset_d=offset_d))
            offset_d = item.offset_d
            offset_h = item.info.header.end
            items.append(item)

        return ArrayDeserializationResult(
            offset_d=offset_d,
            offset_h=offset_h,
            info=info,
            items=items
        )


LVTypeConverter.default_array_converter = ArrayConverter


class ClusterConverter(LVTypeConverter):
    supported_codes = (0x50, )
    supported_types = (Tuple, )

    @classmethod
    def _serialize(cls, value: Tuple, info: SerializationData) -> SerializationResult:
        n_items = len(value)

        items = []

        names: Iterable[str]

        for i, v in enumerate(value):
            item = LVTypeConverter.get_converter_for_value(v).serialize(v, info.fork())
            items.append(item)

        return SerializationResult(
            code=0x50,
            buffer=b"",
            header=num2bytes(n_items, LVDtypes.u2),
            sub_results=items,
            depth=info.depth
        )

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        offset_h = info.header.offset_h
        n_items, offset_h = bytes2num(info.buffer, offset=offset_h, dtype=LVDtypes.u2, count=1)

        items = []
        #names = []

        offset_d = info.offset_d

        for i in range(n_items):
            item_header, offset_h = info.parse_header(offset_h)
            item = item_header.converter.deserialize(info.fork(header=item_header, offset_d=offset_d))
            #names.append(item.name)
            items.append(item)
            offset_d = item.offset_d

        return ClusterDeserializationResult(
            offset_d=offset_d,
            offset_h=offset_h,
            info=info,
            items=items
        )


class SignalConverter(LVTypeConverter):
    supported_codes = (0x54, )
    supported_types = (Signal, datetime)

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        subtype, offset_h = bytes2num(info.buffer, offset=info.header.offset_h + 1, dtype=np.uint8, count=1)

        converter = None

        if subtype == 0x06:
            converter = TimeStampConverter
        elif subtype in AnalogSignalConverter.dtypes.keys():
            converter = AnalogSignalConverter

        if converter is None:
            raise ValueError("Signal subtype {%:02X} not supported", subtype)

        return converter.deserialize(info)

    @classmethod
    def _serialize(cls, value: Any, info: SerializationData) -> SerializationResult:
        converter = None

        if isinstance(value, Signal):
            converter = AnalogSignalConverter
        elif isinstance(value, datetime):
            converter = TimeStampConverter

        if converter is None:
            raise ValueError(f"Signal datatype {type(value)} not supported")

        return converter.serialize(value, info)


class AnalogSignalConverter(LVTypeConverter):
    dtypes = {
        0x14: LVDtypes.i1,
        0x02: LVDtypes.i2,
        0x15: LVDtypes.i4,
        0x19: LVDtypes.i8,
        0x11: LVDtypes.u1,
        0x12: LVDtypes.u2,
        0x13: LVDtypes.u4,
        0x20: LVDtypes.u8,
        0x05: LVDtypes.f4,
        0x03: LVDtypes.f8,
    }

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        offset_h = int(info.header.offset_h) + 1
        offset_d = int(info.offset_d)

        buffer = info.buffer

        s_type, offset_h = lv_parse(LVDtypes.u1,        buffer, offset=offset_h)  # get signal data type
        c1,     offset_h = lv_parse(LVDtypes.u2,        buffer, offset=offset_h)  # ??
        c2,     offset_h = lv_parse(LVDtypes.u2,        buffer, offset=offset_h)  # ??

        dtype = cls.dtypes[s_type]

        t0,     offset_d = lv_parse(datetime,           buffer, offset=offset_d)  # signal timestamp
        dt,     offset_d = lv_parse(LVDtypes.f8,        buffer, offset=offset_d)  # signal delta t
        values, offset_d = lv_parse(np.ndarray,         buffer, offset=offset_d, s_dtype=LVDtypes.u4, e_dtype=dtype)  # y
        err_f,  offset_d = lv_parse(bool,               buffer, offset=offset_d)  # signal error cluster bool
        err_n,  offset_d = lv_parse(LVDtypes.i4,        buffer, offset=offset_d)  # signal error cluster code
        err_s,  offset_d = lv_parse(str,                buffer, offset=offset_d)  # signal error cluster message

        attribs_r = VariantConverter.parse(info.buffer, offset_d)  # attributes variant

        attributes = {item.name: item.value for item in attribs_r.items} if attribs_r.items else None

        signal = Signal(t0=t0, dt=dt, attributes=attributes, y=values)

        return DeserializationResult(
            offset_d=attribs_r.offset_d,
            offset_h=offset_h,
            scalar=signal,
            info=info,
        )

    @classmethod
    def _serialize(cls, value: Signal, info: SerializationData) -> SerializationResult:
        header = b"\x00\x03"

        attribs_r = VariantConverter.serialize(value.attributes, info.fork())

        buffer = lv_dump(value.t0)                             # write t0 timestamp
        buffer += lv_dump(value.dt, dtype=LVDtypes.f8)          # write time diff
        buffer += lv_dump(value.size, dtype=LVDtypes.u4)        # write number of elements
        buffer += lv_dump(value.y, dtype=LVDtypes.f8)           # write signal data array
        buffer += b"\00\00\00\00\00\00\00\00\00"                # Fake empty error cluster
        buffer += attribs_r.buffer                              # include attributes, if any

        return SerializationResult(
            code=0x54,
            header=header,
            buffer=buffer,
            depth=info.depth
        )


class TimeStampConverter(LVTypeConverter):
    header_v0 = bytes.fromhex("0006 0016 0050 0004 0004 0003 0004 0003 0004 0003 0004 0003")

    @classmethod
    def _serialize(cls, value: datetime, info: SerializationData) -> SerializationResult:
        data = cls.serialize_date_raw(value)

        result = SerializationResult(
            code=0x54,
            header=cls.header_v0 if info.version == 0 else b"\x00\x06",
            buffer=data,
            depth=info.depth,
        )

        return result

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        offset_h = info.header.offset_h
        subtype, offset_h = bytes2num(info.buffer, offset=offset_h + 1, dtype=np.uint8, count=1)

        date, offset_d = cls.deserialize_date_raw(info, info.offset_d)

        return DeserializationResult(
            offset_d=offset_d,
            scalar=date,
            offset_h=offset_h,
            info=info
        )

    @classmethod
    def deserialize_date_raw(cls, info: DeserializationData, offset_d: int) -> Tuple[datetime, int]:
        return bytes2date(info.buffer, offset_d)

    @classmethod
    def serialize_date_raw(cls, value: datetime) -> bytes:
        return date2bytes(value)


class VoidConverter(LVTypeConverter):
    supported_types = (type(None), )
    supported_codes = (0, )

    @classmethod
    def _serialize(cls, value: object, info: SerializationData) -> SerializationResult:
        void = SerializationResult(
            code=0,
            buffer=b"",
            header=b"",
            depth=info.depth
        )
        res = VariantConverter.wrap_variant(void, info)
        return res

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        return DeserializationResult(
            offset_h=info.header.offset_h,
            offset_d=info.offset_d,
            info=info
        )

# region VariantVersion


class VariantVersionConverter:
    version = None
    converters: Dict[int, Type["VariantVersionConverter"]] = {}

    @classmethod
    def serialize(cls, value: Any, info: SerializationData) -> SerializationResult:
        if info.version is None:
            version = max(cls.converters.keys())
        else:
            version = info.version

        converter = cls.converters[version]
        return converter._serialize(value, info)

    @classmethod
    def _serialize(cls, value: Dict, info: SerializationData) -> SerializationResult:
        converter = LVTypeConverter.get_converter_for_value(value)
        res = converter.serialize(value, info)
        if value is None:
            return res
        return cls.wrap_variant(res, info)

    @classmethod
    def deserialize(cls, info: DeserializationData) -> DeserializationResult:
        try:
            v_converter = cls.converters[info.version]
        except KeyError:
            for v, converter in cls.converters.items():
                if v > info.version:
                    break
                else:
                    v_converter = converter
            else:
                v_converter = cls.converters[max(cls.converters.keys())]
        return v_converter._deserialize(info)

    @classmethod
    @abc.abstractmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        raise NotImplementedError

    @classmethod
    def wrap_variant(cls, data: SerializationResult, info: SerializationData) -> SerializationResult:
        return cls.converters[info.version]._wrap_variant(data, info)

    @classmethod
    @abc.abstractmethod
    def _wrap_variant(cls, data: SerializationResult, info: SerializationData) -> SerializationResult:
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        cls.converters[cls.version] = cls


class VariantVersionConverter0(VariantVersionConverter):
    version = 0x0

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        v_size, offset_d = bytes2num(info.buffer, offset=info.offset_d+2, count=1, dtype=LVDtypes.u2)

        sub_header = HeaderInfo.parse(info.buffer, offset_d)
        offset_d += int(sub_header.size)

        result = sub_header.converter.deserialize(info.replace(header=sub_header, offset_d=offset_d))
        result.offset_d = result.offset_d + 4
        result.offset_h = info.header.offset_h

        return result

    @classmethod
    def _serialize(cls, value: Dict, info: SerializationData) -> SerializationResult:
        converter = LVTypeConverter.get_converter_for_value(value)
        res = converter.serialize(value, info)

        return cls.wrap_variant(res, info)

    @classmethod
    def _wrap_variant(cls, res: SerializationResult, info: SerializationData) -> SerializationResult:
        #buffer = num2bytes(len(res.header) + 2, dtype=LVDtypes.u2) + res.header + res.buffer + b"\x00\x00\x00\x00"

        #buffer = num2bytes(len(buffer) + 4, LVDtypes.u4) + buffer

        buffer = res.flat_header() + res.flat_buffer() + b"\x00\x00\x00\x00"
        buffer = b"\00\00" + num2bytes(len(buffer) + 4, LVDtypes.u2) + buffer

        return SerializationResult(
            code=0x53,
            header=b"",
            buffer=buffer,
            depth=info.depth
        )


class VariantVersionConverter18008(VariantVersionConverter):
    version = 0x18008000

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        #1800 8000 [uint32: ntypes] [Nx Typeheader] [U2: Mx Datafields] [Mx Typeindex] [Data]

        offset_h = info.offset_d + 4
        n_headers, offset_h = bytes2num(info.buffer, offset=offset_h, count=1, dtype=LVDtypes.u4)

        headers = []

        info = info.replace(fill_header_words=False, header_lut=None, count=1)

        for i in range(n_headers):
            header, offset_h = info.parse_header(offset_h)
            headers.append(header)

        offset_d = offset_h
        info.header_lut = headers

        n_data, offset_d = bytes2num(info.buffer, offset=offset_d, count=1, dtype=LVDtypes.u2)

        assert n_data == 1

        sub_header, offset_d = info.parse_header(offset_d)

        result = sub_header.converter.deserialize(info.replace(header=sub_header, offset_d=offset_d))
        offset_d = result.offset_d

        # region Attributes

        n_attrs, offset_d = lv_parse(LVDtypes.u4, buffer=info.buffer, offset=offset_d, scalar=True)

        attrs = []
        for i_attr in range(n_attrs):
            attr_name, offset_d = lv_parse(str, buffer=info.buffer, s_dtype=LVDtypes.u4, offset=offset_d)
            attr = VariantConverter.parse(info.buffer, offset_d, name=attr_name)
            attrs.append(attr)
            offset_d = attr.offset_d

        if attrs:
            result.items = attrs

        # endregion Attributes

        result.offset_d = offset_d
        result.offset_h = info.header.start + info.header.size

        return result

    @classmethod
    def _wrap_variant(cls, res: SerializationResult, info: SerializationData) -> SerializationResult:
        sub_results = res.all_sub_results()
        sub_results.sort(key=lambda r: r.depth, reverse=True)
        depths: Dict[int, List[SerializationResult]] = {}
        for result in sub_results:
            if result.depth not in depths:
                depths[result.depth] = []
            depths[result.depth].append(result)

        headers_lut = {}
        headers = []

        for depth, results in depths.items():
            for result in results:
                subh = result.flat_header(lut=headers_lut, force_empty_string=False)
                if subh not in headers_lut:
                    headers.append(subh)
                    headers_lut[subh] = num2bytes(len(headers) - 1, dtype=LVDtypes.u2)

        tl_result = headers_lut[res.flat_header(lut=headers_lut, force_empty_string=False)]

        result = SerializationResult(
            code=0x53,
            header=b"",
            buffer=b"\x18\00\x80\00" + num2bytes(len(headers), LVDtypes.u4) + b"".join(headers)
                   + b"\x00\x01"
                   + tl_result
                   + res.flat_buffer()
                   + b"\00\00\00\00",
            depth=info.depth
        )

        return result


# endregion VariantVersion


class VariantConverter(LVTypeConverter):
    supported_codes = (0x53, )
    supported_types = ()

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        v, offset_d = info.parse_num(offset=info.offset_d)
        if v != 0:
            v, offset_d = info.parse_num(offset=info.offset_d, dtype=LVDtypes.u4)
        return VariantVersionConverter.deserialize(info.replace(version=v))

    @classmethod
    def _serialize(cls, value: Dict, info: SerializationData) -> SerializationResult:
        return VariantVersionConverter.serialize(value, info)

    @staticmethod
    def wrap_variant(res: SerializationResult, info: SerializationData) -> SerializationResult:
        return VariantVersionConverter.wrap_variant(res, info)

    @classmethod
    def parse(cls, buffer: bytes, offset: int, name=None) -> DeserializationResult:
        vheader = HeaderInfo(code=0x53, offset_h=offset, size=0, start=offset, name=name)
        return VariantConverter.deserialize(DeserializationData(header=vheader, buffer=buffer, offset_d=offset))


class MapConverter(LVTypeConverter):
    supported_codes = (0x74, )
    supported_types = (dict, )

    @classmethod
    def _serialize(cls, value: dict, info: SerializationData) -> SerializationResult:
        try:
            items = sorted([[k, v] for k, v in value.items()])
        except TypeError:
            items = value.items()

        keys = [item[0] for item in items]
        vals = [item[1] for item in items]

        k_types = [type(k) for k in keys]
        v_types = [type(v) for v in vals]

        if not all([kt == k_types[0] for kt in k_types]) or not all([vt == v_types[0] for vt in v_types]):
            k_conv = VariantConverter
            v_conv = VariantConverter
        else:
            k_conv = LVTypeConverter.get_converter_for_value(keys[0])
            v_conv = LVTypeConverter.get_converter_for_value(vals[0])

        sub_info = info.fork()

        k_res = [k_conv.serialize(k, sub_info) for k in keys]
        v_res = [v_conv.serialize(v, sub_info) for v in vals]

        return SerializationResult(
            code=0x74,
            header=b"\x00\x02",
            buffer=num2bytes(len(v_res), LVDtypes.u4),
            sub_results=[v for pair in zip(k_res, v_res) for v in pair],
            header_indices=(0, 1),
            depth=info.depth
        )

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        c_size, offset_h = bytes2num(info.buffer, offset=info.header.offset_h, count=1, dtype=LVDtypes.u2)

        if c_size != 2:
            raise ValueError("Map with more than 2 types encountered")

        k_header, offset_h = info.parse_header(offset_h)
        v_header, offset_h = info.parse_header(offset_h)

        k_conv = k_header.converter
        v_conv = v_header.converter

        n_items, offset_d = bytes2num(info.buffer, offset=info.offset_d, count=1, dtype=LVDtypes.u4)

        items = []

        for i in range(n_items):
            k_res = k_conv.deserialize(info.fork(header=k_header, offset_d=offset_d, count=1))
            offset_d = k_res.offset_d

            v_res = v_conv.deserialize(info.fork(header=v_header, offset_d=offset_d, count=1))
            offset_d = v_res.offset_d

            items.append((k_res, v_res))

        return MapDeserializationResult(
            offset_d=offset_d,
            offset_h=offset_h,
            info=info,
            items=items
        )


class SetConverter(LVTypeConverter):
    supported_codes = (0x73, )
    supported_types = (set, )

    @classmethod
    def _serialize(cls, value: set, info: SerializationData) -> SerializationResult:
        value = list(value)
        item_types = {type(i) for i in value}

        if len(item_types) == 1:
            subt_converter = LVTypeConverter.get_converter_for_value(value[0])
        else:
            subt_converter = cls

        sub_result = subt_converter.serialize(value, info.fork())

        if info.version == 0:
            header = b"\00\01"
        else:
            header = b""

        return SerializationResult(
            code=0x73,
            header=header,
            buffer=num2bytes(len(value), LVDtypes.u4),
            sub_results=(sub_result, ),
            depth=info.depth
        )

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        offset_h = int(info.header.offset_h)
        offset_d = int(info.offset_d)

        if info.version == 0:
            sub_size, offset_h = bytes2num(info.buffer, offset=offset_h, count=1, dtype=LVDtypes.u2)
            if sub_size != 1:
                raise ValueError

        subt, offset_h = info.parse_header(offset_h)
        converter = subt.converter

        s_size, offset_d = bytes2num(info.buffer, offset=offset_d, count=1, dtype=LVDtypes.u4)

        items = []

        for i in range(s_size):
            res = converter.deserialize(info.fork(offset_d=offset_d, header=subt))
            offset_d = res.offset_d
            items.append(res)

        return SetDeserializationResult(
            offset_d=offset_d,
            offset_h=offset_h,
            items=items,
            info=info
        )


class EnumConverter(LVTypeConverter):
    supported_codes = (0x15, 0x16, 0x17)
    supported_types = (IntEnum, )
    order = 1

    dtype_limits = [(dtype, np.iinfo(dtype).max) for dtype in (LVDtypes.u1, LVDtypes.u2, LVDtypes.u4)]

    @classmethod
    def _serialize(cls, value: IntEnum, info: SerializationData) -> SerializationResult:
        items = {item.value: item.name for item in type(value)}

        set_values = set(items.keys())
        maxvalue = max(set_values)

        for (dtype, dtmax), code in zip(cls.dtype_limits, cls.supported_codes):
            if dtmax >= maxvalue:
                break
        else:
            raise ValueError

        header = num2bytes(maxvalue+1, dtype=LVDtypes.u2)

        for i in range(maxvalue + 1):
            if i in set_values:
                header += str2bytes(items[i], s_dtype=LVDtypes.u1, fill=False)
            else:
                header += b"\00"

        buffer = num2bytes(value.value, dtype=dtype)

        if info.version == 0:
            header += b""
        else:
            header += b"\00"

        return SerializationResult(
            code=code,
            header=header,
            buffer=buffer,
            depth=info.depth
        )

    @classmethod
    def _deserialize(cls, info: DeserializationData) -> DeserializationResult:
        offset_h = int(info.header.offset_h)
        offset_d = int(info.offset_d)
        nitems, offset_h = bytes2num(info.buffer, offset=offset_h, dtype=LVDtypes.u2)

        items = {}

        for i in range(nitems):
            item_name, offset_h = bytes2str(info.buffer, offset_h, s_dtype=LVDtypes.u1, fill=False)
            if not item_name:
                item_name = str(i)
            items[item_name] = i

        if info.version >= 0x8508002:
            offset_h += 2
        else:
            offset_h += 1

        code_idx = cls.supported_codes.index(info.header.code)
        dtype = cls.dtype_limits[code_idx][0]

        enum_v, offset_d = bytes2num(info.buffer, offset_d, dtype=dtype)

        #python intenum supports no unnamed values, other than labview... and labview must start at 0 other than python
        enum_t = IntEnum("", items)
        value = enum_t(enum_v)

        #value = enum_v

        return DeserializationResult(
            offset_d=offset_d,
            offset_h=offset_h,
            scalar=value,
            info=info
        )


