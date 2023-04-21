## I/O tools

To add your own I/O functions:
1. add a parser in `parsers.py`,
2. if necessary add an `if` statement or write your own collate function in
`collates.py`.


You can write your own sampling function in `samplers.py`.

### 1. Writing and Reading HDF5 Files

```yaml
iotool:
  writer:
    name: HDF5Writer
    file_name: output.h5
    input_keys: None
    skip_input_keys: []
    result_keys: None
    skip_result_keys: []
```
