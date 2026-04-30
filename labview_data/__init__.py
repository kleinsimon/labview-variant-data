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
from .utils import HeaderInfo, DeserializationData, LVDtypes, SerializationData, LVTypeConverter, NamedItem


def deserialize_variant(buffer: bytes, return_struct=False):
    if len(buffer) == 0:
        return

    res = VariantConverter.parse(buffer, 0)
    if return_struct:
        return res
    else:
        return res.value


def serialize_variant(value, version=0x18008000) -> bytes:
    result = VariantConverter.serialize(value, SerializationData(version=version))
    return result.flat_buffer()

