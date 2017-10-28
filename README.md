# MELO Rating System
Module which implements the margin-dependent Elo rating system (MELO) for
predicting NFL spread and over/under point distributions.

## What's Here?
This git repository contains a Python module to calculate margin-dependent Elo
ratings for NFL games.

## Dependencies
 - Python 2.7 
 - [numpy](www.numpy.org)
 - [scipy](www.scipy.org)
 - [nfldb](https://github.com/BurntSushi/nfldb)
 - [skopt](https://scikit-optimize.github.io/)

## Setup
First, create a virtual enviroment (recommended),
```
virtualenv --python=2.7 .env
```
Activate the virtual environment and cd into the desired parent directory. Then clone this repository,
```
git clone git@github.com:morelandjs/melo.git
```
Once you've cloned the repository, cd into the repo directory and install using pip,
```
pip install -r requirements
```
Finally, you'll need to follow the steps to install [nfldb](https://github.com/BurntSushi/nfldb), a relational database bundled with a Python module to quickly and conveniently query and update active NFL game data, written by Andrew Gallant.
Installation instruction can be found on his project's [github page](https://github.com/BurntSushi/nfldb/wiki/Installation).

## Tests
Unit tests are included via [pytest](https://pypi.python.org/pypi/pytest/).
From inside the repository directory run,
```
python -m pytest
```

## Usage
