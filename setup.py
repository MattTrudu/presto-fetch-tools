from setuptools import setup

setup(
    name="presto-fetch-tools",
    version="0.1.0",
    description="Convert PRESTO .singlepulse outputs into FETCH-compatible CSV files",
    author="Matteo Trudu",
    author_email="matteo@trudu.inaf",
    url="https://github.com/MattTrudu/presto-fetch-tools",
    py_modules=["make_presto_csv"],  # file must be make_presto_csv.py
    scripts=["make_presto_csv.py"], 
    python_requires=">=3.7",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)

