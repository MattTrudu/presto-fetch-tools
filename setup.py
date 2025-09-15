from setuptools import setup

setup(
    name="presto-fetch-tools",
    version="0.1.1",
    description="Convert PRESTO .singlepulse outputs into FETCH-compatible CSV files",
    scripts=["make_presto_csv.py"],   # installs a runnable 'make_presto_csv.py' into <env>/bin
    py_modules=["make_presto_csv"],   # also installs the module so you can do: python -m make_presto_csv
    python_requires=">=3.7",
)
