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

import numpy as np

from .utils import HeaderInfo, DeserializationData, SerializationData, DeserializationResult, SerializationResult
from .utils import bytes2num, bytes2str, num2bytes, LVDtypes, LVTypeConverter
from .types import Cluster


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
        size = dtype.itemsize
        offset_d = info.offset_d
        value = np.frombuffer(info.buffer, offset=offset_d, dtype=dtype, count=info.count)
        offset_d += size * value.size

        if info.scalar and value.size == 1:
            value = value[0]

        offset_h = info.header.offset_h

        if value.dtype != np.bool_ and info.version >= 0x08508002:
            offset_h += 1

        return DeserializationResult(
            value=value,
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

        result.value = value
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
            value=value,
            offset_d=offset_d,
            offset_h=info.header.start + info.header.size,
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
        offset_d = info.offset_d
        if not info.buffer[offset_d:offset_d+4] == b"PTH0":
            raise ValueError
        offset_d += 4
        size, offset_d = bytes2num(info.buffer, offset=offset_d, count=1, dtype=LVDtypes.u4)
        end = int(offset_d + size)

        #offset_d += 2
        ptype, offset_d = bytes2num(info.buffer, offset=offset_d, count=1, dtype=LVDtypes.u2)
        count, offset_d = bytes2num(info.buffer, offset=offset_d, count=1, dtype=LVDtypes.u2)

        parts = []
        for i in range(count):
            s, offset_d = bytes2num(info.buffer, offset=offset_d, count=1, dtype=LVDtypes.u1)
            if s == 0:
                parts.append("..\\")
            else:
                parts.append(info.buffer[offset_d:offset_d+s].decode("ansi") + "\\")
            offset_d += s

        if ptype == 0:
            #absolute path. This obviusly works only in windows
            parts[0] = parts[0][0] + ":\\"

        return DeserializationResult(
            value=Path(*parts),
            offset_d=end,
            offset_h=info.header.offset_h + 4,
            depth=info.depth,
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
        offset_h = info.header.offset_h
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
        values = []

        offset_h = info.header.offset_h
        offset_d = info.offset_d

        for i in range(info.count):
            item = info.header.converter.deserialize(info.replace(offset_d=offset_d))
            offset_d = item.offset_d
            offset_h = item.info.header.end
            values.append(item.value)

        ndim = len(info.shape)
        if ndim > 1:
            values = np.array(values, dtype=object)
            values = values.reshape(info.shape)

        return DeserializationResult(
            value=values,
            offset_d=offset_d,
            offset_h=offset_h,
            depth=info.depth,
            info=info
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
        names = []

        offset_d = info.offset_d

        for i in range(n_items):
            item_header, offset_h = info.parse_header(offset_h)
            item = item_header.converter.deserialize(info.fork(header=item_header, offset_d=offset_d))
            names.append(item.name)
            items.append(item.value)
            offset_d = item.offset_d

        return DeserializationResult(
            value=Cluster(items, names),
            offset_d=offset_d,
            offset_h=offset_h,
            depth=info.depth,
            info=info
        )


class TimeStampConverter(LVTypeConverter):
    supported_types = (datetime, )
    supported_codes = (0x54, ) # This is the code for Signal
    epoch = datetime(year=1904, month=1, day=1, tzinfo=timezone.utc)
    header_v0 = bytes.fromhex("0006 0016 0050 0004 0004 0003 0004 0003 0004 0003 0004 0003")

    @classmethod
    def _serialize(cls, value: datetime, info: SerializationData) -> SerializationResult:
        dif = value - cls.epoch

        secs = np.int64(dif.total_seconds())
        usecs = np.uint64(dif.microseconds)
        data = num2bytes(secs, ">i8") + num2bytes(usecs, ">u8")

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

        if subtype != 0x06:
            raise ValueError("Only Timestamp supported as Signal (0x54)")

        s, ms = np.frombuffer(info.buffer, offset=info.offset_d, dtype=[("", ">i8"), ("", ">u8")], count=1)[0]
        dt = timedelta(seconds=int(s), microseconds=int(ms))

        return DeserializationResult(
            offset_d=info.offset_d + 16,
            value=cls.epoch + dt,
            offset_h=offset_h,
            depth=info.depth,
            info=info
        )


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
            value=None,
            offset_h=info.header.offset_h,
            offset_d=info.offset_d,
            depth=info.depth,
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

        return cls.wrap_variant(res, info)

    @classmethod
    def deserialize(cls, info: DeserializationData) -> DeserializationResult:
        return cls.converters[info.version]._deserialize(info)

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
        offset_d += sub_header.size

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

        for i in range(n_data): #there could be attributes
            sub_header, offset_d = info.parse_header(offset_d)

            result = sub_header.converter.deserialize(info.replace(header=sub_header, offset_d=offset_d))

            result.offset_d = result.offset_d + 4
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

        results = {}

        for i in range(n_items):
            k_res = k_conv.deserialize(info.fork(header=k_header, offset_d=offset_d, count=1))
            offset_d = k_res.offset_d

            v_res = v_conv.deserialize(info.fork(header=v_header, offset_d=offset_d, count=1))
            offset_d = v_res.offset_d

            results[k_res.value] = v_res.value

        return DeserializationResult(
            value=results,
            offset_d=offset_d,
            offset_h=offset_h,
            depth=info.depth,
            info=info
        )

