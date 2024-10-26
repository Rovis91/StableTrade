from setuptools import setup, find_packages

setup(
    name="stabletrade",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'celery',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
    ]
)