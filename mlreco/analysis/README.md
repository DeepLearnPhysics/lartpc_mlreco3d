## Analysis scripts

Put here your scripts to analyze the output of the networks.
Create a file `my_analysis.py` with:

```python
def my_analysis(data_blob, res, cfg, idx):
    # Do stuff
```

`data_blob` contains the data. `res` is the network outputs.
`cfg` comes from the YAML configuration file. `idx` is the iteration.


Don't forget to edit `__init__.py` and to add your script to the list there.
