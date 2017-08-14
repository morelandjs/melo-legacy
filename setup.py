from setuptools import setup, find_packages

setup(
    name='melo',
    version='0.1',
    description='NFL ELO rankings.',
    url='https://github.com/nfl-elo.git',
    author='J. Scott Moreland',
    author_email='morelandjs@gmail.com',
    packages=find_packages(exclude=('tests', 'docs')),
    )
