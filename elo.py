#!/usr/bin/env python2

import copy
import sqlite3
from collections import defaultdict
from functools import partial
from itertools import chain
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter1d

import matplotlib.pyplot as plt
import numpy as np

import nfldb


nweeks = 17
nested_dict = lambda: defaultdict(nested_dict)

class Rating:
    def __init__(self, kfactor=50, decay=0.7, database='elo.db'):
        self.kfactor = kfactor
        self.decay = decay

        self.nfldb = nfldb.connect()
        self.spread_prob = self.spread_probability()
        self.elodb = nested_dict()
        self.calc_elo()

    def hfa(self, handicap, year, week):
        """
        ELO bonus for home field advantage

        """
        hfa = []
        for yr, wk in self.rewind(year, week, n=17):
            home_rtg = self.query_elo('home', handicap, yr, wk)
            away_rtg = self.query_elo('away', handicap, yr, wk)

            hcap_diff = home_rtg['hcap'] - away_rtg['hcap']
            fair_diff = home_rtg['fair'] - away_rtg['fair']

            hfa.append(0.5*(hcap_diff + fair_diff))

        return np.mean(hfa)

    def starting_elo(self, margin):
        """
        Initialize the starting elo for each team. For the present
        database, team histories start in 2009. One exception is the LA
        rams.

        """
        elo_init = 1500.
        prob = max(0.5*self.spread_prob[margin], 1e-6)
        arg = max(1/prob - 1, 1e-6)
        elo_diff = 200*np.log10(arg)

        return {'fair': elo_init + elo_diff, 'hcap': elo_init - elo_diff}

    def spreads(self):
        """
        All point spreads (points winner - points loser) since 2009.

        """
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')
        spreads = [abs(g.home_score - g.away_score) for g in q.as_games()]

        return spreads

    def spread_probability(self):
        """
        Probabilities of observing each point spread.

        """
        bins = np.arange(-0.5, 41.5)
        hist, edges = np.histogram(self.spreads(), bins=bins, normed=True)
        spread = 0.5*(edges[:-1] + edges[1:])
        prob = np.cumsum(hist[::-1], dtype=float)[::-1]

        return dict(zip(spread, prob))

    def ngames(self, year, week):
        """
        Number of games scheduled in a given (year, week)

        """
        # nfldb database
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular', season_year=year, week=week)

        # integer number of games
        N = len(q.as_games())

        return float(N)

    def rewind(self, year, week, n=2):
        """
        Simple function to go back "one game in time".
        For example rewind(2016, 1) = (2015, 17).

        """
        for _ in range(n):
            if week > 1:
                week -= 1
            else:
                year -= 1
                week = nweeks
            yield year, week

    def regress(self, rtg, week):
        if week == 1:
            rtg['fair'] = 1500. + self.decay * (rtg['fair'] - 1500.)
            rtg['hcap'] = 1500. + self.decay * (rtg['hcap'] - 1500.)
        return rtg

    def query_elo(self, team, margin, year, week):
        """
        Queries the most recent ELO rating for a team, i.e.
        elo(year, week) for (year, week) < (query year, query week)

        If the team name is one of ("home", "away"), then return the
        rating from the current year, week if it exists.

        """
        def previous(year, week):
            if team in ('home', 'away'):
                yield year, week
            for yr, wk in self.rewind(year, week):
                yield yr, wk

        for yr, wk in previous(year, week):
            elo = self.elodb[team][margin][yr][wk]
            if elo:
                return self.regress(elo.copy(), wk)

        elo = self.starting_elo(margin)
        self.elodb[team][margin][year-1][nweeks] = elo

        return self.regress(elo, week)


    def elo_change(self, rating_diff, points, handicap):
        """
        Change in home team ELO rating after a single game

        """
        prob = self.win_prob(rating_diff)

        if points - handicap > 0:
            return self.kfactor * (1. - prob)
        else:
            return -self.kfactor * prob

    def calc_elo(self):
        """
        This function calculates ELO ratings for every team
        for every value of the spread.

        """
        # nfldb database
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')

        # small sorting function
        def time(game):
            return game.season_year, game.week

        # loop over historical games in chronological order
        for game in sorted(q.as_games(), key=lambda g: time(g)):

            # game time
            year = game.season_year
            week = game.week

            # point differential
            points = game.home_score - game.away_score

            # game team names
            team_names = [
                    ('home', 'away', self.ngames(year, week)),
                    (game.home_team, game.away_team, 1.),
                    ]
            
            # first update home/away ratings, then team ratings
            for home, away, scale in team_names:

                # loop over all possible spread margins
                for handicap in range(41):

                    # query current elo ratings from most recent game
                    home_rtg = self.query_elo(home, handicap, year, week)
                    away_rtg = self.query_elo(away, handicap, year, week)

                    # disable hfa for home vs away elo calc
                    hfa = (self.hfa(handicap, year, week)
                            if scale == 1 else 0.)

                    # elo change when home(away) team is handicapped
                    bounty_home_hcap, bounty_away_hcap = [
                            self.elo_change(
                                home_rtg[a] - away_rtg[b] + hfa,
                                points,
                                hcap
                                ) / scale
                            for (a, b, hcap) in [
                                ('hcap', 'fair', handicap),
                                ('fair', 'hcap', -handicap),
                                ]
                            ]

                    # scale update by ngames if necessary
                    home_rtg['hcap'] += bounty_home_hcap
                    away_rtg['fair'] -= bounty_home_hcap
                    home_rtg['fair'] += bounty_away_hcap
                    away_rtg['hcap'] -= bounty_away_hcap

                    # home and away team elo data
                    team_rtgs = [(home, home_rtg), (away, away_rtg)]

                    # update elo ratings
                    for team, team_rtg in team_rtgs:
                        self.elodb[team][handicap][year][week] = team_rtg

    def cdf(self, home, away, year, week):
        """
        Cumulative (integrated) probability that,
        score home - score away > x.

        """
        spreads = np.arange(-40, 41)
        cumulative_win_prob = []

        for handicap in spreads:
            hcap = abs(handicap)
            home_rtg = self.query_elo(home, hcap, year, week)
            away_rtg = self.query_elo(away, hcap, year, week)
            hfa = self.hfa(hcap, year, week)

            if handicap < 0:
                rtg_diff = home_rtg['fair'] - away_rtg['hcap'] + hfa
            else:
                rtg_diff = home_rtg['hcap'] - away_rtg['fair'] + hfa

            cumulative_win_prob.append(self.win_prob(rtg_diff))

        return spreads, gaussian_filter1d(
                cumulative_win_prob, 2, mode='constant'
                )


    def predict_spread(self, home, away, year, week):
        """
        Predict the spread for a matchup, given current knowledge of each
        team's ELO ratings.

        """
        # cumulative spread distribution
        cdf = self.cdf(home, away, year, week)

        # plot median prediction (compare to vegas spread)
        median, prob = min(zip(*cdf), key=lambda k: abs(k[1] - 0.5))

        return median, prob

        
    def predict_score(self, home, away, year, week):
        # cumulative spread distribution
        spreads, cprob = self.cdf(home, away, year, week)

        return sum(cprob) - 40.

    def win_prob(self, rtg_diff):
        """
        Probability that a team will win as a function of ELO difference

        """
        return 1./(10**(-rtg_diff/400.) + 1.)

    def model_accuracy(self):
        # nfldb database
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')

        residuals = []

        # loop over all historical games
        for n, game in enumerate(q.as_games()):
            year = game.season_year
            week = game.week
            home = game.home_team
            away = game.away_team

            # allow for one season burn-in
            if year > 2012 and week > 8:
                predicted = self.predict_score(home, away, year, week)
                observed = game.home_score - game.away_score
                residuals.append(observed - predicted)

        return np.std(residuals)

def main():
    rating = Rating(kfactor=60, decay=0.6)
    rms_error = rating.model_accuracy()
    print(rms_error)

if __name__ == "__main__":
    main()
