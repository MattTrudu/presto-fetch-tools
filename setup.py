from setuptools import setup

setup(
    name="presto-fetch-tools",
    version="0.1.0",
    description="Convert PRESTO .singlepulse outputs into FETCH-compatible CSV files",
    scripts=["bin/make_presto_csv.py"],  # <-- path to your script in the repo
    python_requires=">=3.7",
)

