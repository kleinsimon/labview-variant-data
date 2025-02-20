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

from .type_converters import VariantConverter
from .utils import HeaderInfo, DeserializationData, LVDtypes, SerializationData, LVTypeConverter


def deserialize_variant(buffer: bytes):
    if len(buffer) == 0:
        return

    vheader = HeaderInfo(code=0x53, offset_h=0, size=0, start=0)
    res = VariantConverter.deserialize(DeserializationData(header=vheader, buffer=buffer, offset_d=0))
    return res.value


def serialize_variant(value, name=None, version=0x18008000) -> bytes:
    result = VariantConverter.serialize(value, SerializationData(name=name, version=version))
    return result.buffer

