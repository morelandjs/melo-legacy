#!/usr/bin/env python2

import bisect
import sqlite3
from collections import defaultdict
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter1d

import matplotlib.pyplot as plt
import numpy as np

import nfldb


nweeks = 17
nested_dict = lambda: defaultdict(nested_dict)

class Rating:
    def __init__(self, kfactor=40, decay=0.7, hfa=60.0, database='elo.db'):
        self.kfactor = kfactor
        self.decay = decay
        self.hfa = hfa

        self.nfldb = nfldb.connect()
        self.spread_prob = self.spread_probability()
        self.elodb = nested_dict()
        self.calc_elo()

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

    def query_elo(self, team, margin, year, week):
        """
        Queries the most recent ELO rating for a team, i.e.
        elo(year, week) for (year, week) < (query year, query week)

        If the team name is one of ("home", "away"), then return the
        rating from the current year, week if it exists.

        """
        for yr, wk in self.rewind(year, week):
            elo = self.elodb[team][margin][yr][wk]
            if elo: return elo.copy()

        elo = self.starting_elo(margin)
        self.elodb[team][margin][year-1][nweeks] = elo

        return elo

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

            # team names
            home = game.home_team
            away = game.away_team

            # point differential
            points = game.home_score - game.away_score

            # loop over all possible spread margins
            for handicap in range(41):

                # query current elo ratings from most recent game
                home_rtg = self.query_elo(home, handicap, year, week)
                away_rtg = self.query_elo(away, handicap, year, week)

                # elo change when home(away) team is handicapped
                bounty_home_hcap, bounty_away_hcap = [
                        self.elo_change(
                            home_rtg[a] - away_rtg[b] + self.hfa,
                            points,
                            hcap
                            )
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
        cprob = []

        for handicap in spreads:
            hcap = abs(handicap)
            home_rtg = self.query_elo(home, hcap, year, week)
            away_rtg = self.query_elo(away, hcap, year, week)

            if handicap < 0:
                rtg_diff = home_rtg['fair'] - away_rtg['hcap'] + self.hfa
            else:
                rtg_diff = home_rtg['hcap'] - away_rtg['fair'] + self.hfa

            cprob.append(self.win_prob(rtg_diff))

        return spreads, gaussian_filter1d(cprob, 0, mode='constant')

    def predict_spread(self, home, away, year, week):
        """
        Predict the spread for a matchup, given current knowledge of each
        team's ELO ratings.

        """
        # cumulative spread distribution
        spreads, cprob = self.cdf(home, away, year, week)

        # plot median prediction (compare to vegas spread)
        index = np.square(cprob - 0.5).argmin()
        x0, y0 = (spreads[index - 1], cprob[index - 1])
        x1, y1 = (spreads[index], cprob[index])
        x2, y2 = (spreads[index + 1], cprob[index + 1])

        # fit a quadratic polynomial
        coeff = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)

        print(np.polyval(coeff, x1))
        quit()

        return median, prob
        
    def predict_score(self, home, away, year, week):
        """
        The model predicts the CDF of win margins, i.e. P(spread > x).
        One can use integration by parts to calculate the expected score
        from the CDF,

        E(x) = \int x P(x)

        """
        # cumulative spread distribution
        spreads, cprob = self.cdf(home, away, year, week)

        # Calc via integration byE(x) = \int x P(x)
        return sum(cprob) - 40.5

    def win_prob(self, rtg_diff):
        """
        Probability that a team will win as a function of ELO difference

        """
        return 1./(10**(-rtg_diff/400.) + 1.)

    def model_accuracy(self):
        """
        Calculate the mean and standard deviation of the model's
        residuals = observed - predicted.

        """
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')

        residuals = []

        # loop over all historical games
        for n, game in enumerate(q.as_games()):
            year = game.season_year
            week = game.week
            home = game.home_team
            away = game.away_team

            # allow for two season burn-in
            if year > 2011:
                predicted = self.predict_score(home, away, year, week)
                observed = game.home_score - game.away_score
                residuals.append(observed - predicted)

        return np.mean(residuals), np.std(residuals)

def main():
    """
    Main function prints the model accuracy parameters and exits

    """
    rating = Rating(kfactor=60, decay=0.6)
    rating.predict_spread('CLE', 'NE', 2016, 12)
    mean_error, rms_error = rating.model_accuracy()
    print(mean_error, rms_error)

if __name__ == "__main__":
    main()
