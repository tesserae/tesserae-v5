# Tesserae Unit Tests

Tesserae v5 uses the `pytest` framework to perform unit testing. The tests are
arranged to have one-to-one correspondence with the `tesserae` package, meaning
one test module exists for each module in the Tesserae v5 source code. Package-
and module-level utility operations (e.g., setup, teardown, etc.) can be found
in `conftest.py` in each directory.

For information about `pytest`, see the [documentation](http://pytest.org).
