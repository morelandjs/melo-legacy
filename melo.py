#!/usr/bin/env python2

import bisect
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import minimize

import nfldb


nweeks = 17
nested_dict = lambda: defaultdict(nested_dict)

class Rating:
    def __init__(self, obs='score', kfactor=60, hfa=60, database='elo.db'):

        # point-spread interval attributes
        self.bins= self.range(obs)
        self.ubins = self.bins[-int(1+.5*len(self.bins)):]
        self.range = 0.5*(self.bins[:-1] + self.bins[1:])

        # model hyper-parameters
        self.obs = obs
        self.kfactor = kfactor
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
        prob = max(0.5*self.spread_prob[abs(margin)], 1e-6)
        arg = max(1/prob - 1, 1e-6)
        elo_diff = 200*np.log10(arg)

        # separate positive and negative ratings
        # i.e. home team wins by 40 =  team[40] << team[-40]
        return elo_init - elo_diff if margin > 0 else elo_init + elo_diff

    def spreads(self):
        """
        All point spreads (points winner - points loser) since 2009.

        """
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular', finished=True)
        spreads = [self.point_diff(g) for g in q.as_games()]

        return spreads

    def spread_probability(self):
        """
        Probabilities of observing each point spread.

        """
        spreads = np.abs(self.spreads())
        hist, edges = np.histogram(
                spreads, bins=self.ubins, normed=True
                )
        spread = 0.5*(edges[:-1] + edges[1:])
        prob = np.cumsum(hist[::-1], dtype=float)[::-1]

        return dict(zip(spread, prob))

    def cycle(self, year, week, n=2):
        """
        Simple function to go forward "one game in time".
        For example rewind(2015, 17) = (2016, 1).
        """
        for _ in range(n):
            if week < nweeks:
                week += 1
            else:
                year += 1
                week = 1
            yield year, week

    def query_elo(self, team, margin, year, week):
        """
        Queries the most recent ELO rating for a team, i.e.
        elo(year, week) for (year, week) < (query year, query week)
        If the team name is one of ("home", "away"), then return the
        rating from the current year, week if it exists.
        """
        elo = self.elodb[team][margin][year][week]
        if elo: return elo.copy()

        return self.starting_elo(margin)

    def yds(self, game, team):
        """
        Calculates the yards traversed by a team over the course of a
        game.

        """
        yards = sum(
                drive.yards_gained for drive in game.drives
                if team == nfldb.standard_team(drive.pos_team)
                )

        return yards

    def point_diff(self, game):
        """
        Function which returns the point difference.
        The point type is defined by the observable argument.

        """
        home, away = (game.home_team, game.away_team)
        point_dict = {
                "score": game.home_score - game.away_score,
                "yards": self.yds(game, home) - self.yds(game, away)
                }

        return point_dict[self.obs]

    def range(self, obs):
        """
        Returns an iterator over the range of reasonable point values.

        """
        edges_dict = {
                "score": np.arange(-40.5, 41.5, 1),
                "yards": np.arange(-355, 365, 10)
                }

        return edges_dict[obs]

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
        The rating reflects the posterior knowledge after the 
        given week.

        """
        # nfldb database
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular', finished=True)

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
            points = self.point_diff(game)

            # loop over all possible spread margins
            for handicap in self.range:

                # query current elo ratings from most recent game
                home_rtg = self.query_elo(home, handicap, year, week)
                away_rtg = self.query_elo(away, -handicap, year, week)

                # elo change when home(away) team is handicapped
                rtg_diff = home_rtg - away_rtg + self.hfa
                bounty = self.elo_change(rtg_diff, points, handicap)

                # scale update by ngames if necessary
                home_rtg += bounty
                away_rtg -= bounty

                # update elo ratings
                for yr, wk in self.cycle(year, week):
                    self.elodb[home][handicap][yr][wk] = home_rtg
                    self.elodb[away][-handicap][yr][wk] = away_rtg

    def cdf(self, home, away, year, week):
        """
        Cumulative (integrated) probability that,
        score home - score away > x.

        """
        cprob = []

        for handicap in self.range:
            home_rtg = self.query_elo(home, handicap, year, week)
            away_rtg = self.query_elo(away, -handicap, year, week)

            rtg_diff = home_rtg - away_rtg + self.hfa

            cprob.append(self.win_prob(rtg_diff))

        return self.range, cprob

    def predict_spread(self, home, away, year, week):
        """
        Predict the spread for a matchup, given current knowledge of each
        team's ELO ratings.

        """
        # cumulative spread distribution
        spreads, cprob = self.cdf(home, away, year, week)

        # plot median prediction (compare to vegas spread)
        index = np.square([p - 0.5 for p in cprob]).argmin()
        x0, y0 = (spreads[index - 1], cprob[index - 1])
        x1, y1 = (spreads[index], cprob[index])
        x2, y2 = (spreads[index + 1], cprob[index + 1])

        # fit a quadratic polynomial
        coeff = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
        res = minimize(lambda x: np.square(np.polyval(coeff, x) - 0.5), x1)
        median = 0.5 * round(res.x * 2)

        return median
        
    def predict_score(self, home, away, year, week):
        """
        The model predicts the CDF of win margins, i.e. P(spread > x).
        One can use integration by parts to calculate the expected score
        from the CDF,

        E(x) = \int x P(x)

        """
        # cumulative spread distribution
        spreads, cprob = self.cdf(home, away, year, week)
        spread_max = max(self.range)
        spread_step = spreads[1] - spreads[0]

        # Calc via integration by parts of E(x) = \int x P(x)
        return sum(cprob)*spread_step - spread_max - 0.5*spread_step

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
        q.game(season_type='Regular', finished=True)

        residuals = []

        # loop over all historical games
        for game in q.as_games():
            year = game.season_year
            week = game.week
            home = game.home_team
            away = game.away_team

            # allow for two season burn-in
            if year > 2010:
                predicted = self.predict_score(home, away, year, week)
                observed = self.point_diff(game)
                residuals.append(observed - predicted)

        return residuals

def main():
    """
    Main function prints the model accuracy parameters and exits

    """
    rating = Rating()
    residuals = rating.model_accuracy()
    mean_error = np.mean(residuals)
    rms_error = np.std(residuals)
    print(mean_error, rms_error)

if __name__ == "__main__":
    main()
