# (De-) Serialization for LabVIEW variant data
This package allows the serialization (parsing) and deserialization (creating)
of binary data compatible with LabVIEW created and evaluated with "to string" 
and "from string" functions. 

# Features
* Create and parse arbitrary binary data representing structured data
* Supports all basic data types and structures, even nested structures and variants
* Ready for different Versions, tested with 0, 18, 23. 
* Extendable for custom types
* Efficient handling of N-dim arrays in both directions

# Supported Types

| LabVIEW Type  | Python Type               |
| ------------- | ------------------------- |
| Integer       | Numpy scalars (Type safe) |
| String        | Python str                |
| Path          | pathlib.Path              |
| Timestamp     | datetime.datetime         |
| Void          | None                      |
| Variant       | Nested structure          |
| Array         | np.ndarray or list        |
| Cluster       | Tuple                     |
| Map           | Dict                      |

# Installation
Copy the package into your python project or clone it as a git submodule

# Usage

## Python
```
from labview import serialize_variant, deserialize_variant

# here version could be set explicitly, defaults to 0x18008000

buffer = serialize_variant(("Hello", "World"))
print(buffer.hex())

# 180080000000000200080030ffffffff000a0050000200000000000100010000000548656c6c6f00000005576f726c6400000000

result = deserialize_variant(buffer)
print(result)

# ('Hello', 'World')

```

## LabVIEW
Deserialze with "From String" and Variant as type

![image](https://github.com/user-attachments/assets/617208de-f434-4c5e-85c6-b51bde92a538)

Serialize with "To String"

![image](https://github.com/user-attachments/assets/3de37467-2593-4538-bc09-dbf456801c9d)

Care has to be taken regarding the version of the serialization algorithm in labview. Although tests seem to indicate,
that all versions up to now are supported for the listed types, edge cases could occur.
The deserialization algorithm on both sides detects the version of the data.

See https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z0000015CmaSAE&l=en-US for more information

# License
Released under MIT license. See License.txt for details.
