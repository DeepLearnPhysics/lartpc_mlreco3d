## Tests

### Basics
Run all tests with `pytest`.
Run specific tests with `pytest test/test_parser.py` for example.

### Image size
To test against several image sizes you can use the `--N` command line option:
```
$ pytest --N 192 256
```
By default it will run with N = 192px.

### Test data file
There are 3 data files available to run the tests:
* [192px](http://stanford.edu/~ldomine/small_192px.root)
* [512px](http://stanford.edu/~ldomine/small_512px.root)
* [768px](http://stanford.edu/~ldomine/small_768px.root)
You can specify which one(s) to use in the `pytest.ini` file.

*Note: Each of them contain 5 events. Set the batch size accordingly (<5).*

### Exclude slow tests
You can mark tests that will take some time to complete with the decorator
`@pytest.mark.slow`.
If you are in a hurry, use `-m "no slow"` to run the tests excluding slow tests.

Currently only the full tests of the models are marked as slow. They involve
running all models on a small data sample in LArCV format, including parsers.
