## Output formatters

If you want to output your network results to a nice format for later analysis,
you are at the right place. Currently available formatters create a CSV file
with 5 columns: `x`, `y`, `z`, `type` and `value`.

To add your own formatter/postprocessing, create a file `my_formatter.py` and
add inside:
```python
def my_formatter(csv_logger, data_blob, res):
    # Do your formatting and output to some nice format.
```
These functions take as input `csv_logger` which is an instance of `CSVData`
and allows you to write stuff to a CSV file.
`data_blob` is the current event data. `res` is the output of the network.
