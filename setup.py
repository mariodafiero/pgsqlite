# coding: utf-8

"""
    pgsqlite

    pgsqlite  # noqa: E501
"""


from setuptools import setup, find_packages  # noqa: H301

NAME = "pgsqlite"
VERSION = "1.0b7"
# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

REQUIRES = ["sqlite-utils >= 3.28", "psycopg >= 3.1", "psycopg-binary >= 3.1", "structlog >= 22.1.0"]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=VERSION,
    description="Loader to import sqlite3 databases into Postgres",
    author="bit.io",
    author_email="python@bit.io",
    url="https://github.com/bitdotioinc/pgsqlite",
    keywords=["bit.io", "Database", "postgres", "postgresql", "sqlite", "sqlite3"],
    install_requires=REQUIRES,
    packages=find_packages(exclude=["test", "tests"]),
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/bitdotioinc/pgsqlite/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
