#!/usr/bin/env python2

from collections import defaultdict
from functools import total_ordering

import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize

import nfldb

import matplotlib.pyplot as plt

@total_ordering
class Date:
    """
    Creates a cyclic date object with year and week
    attributes.

    date.next and date.prev function calls increment
    and decrement the date object.

    """
    def __init__(self, year, week):
        self.nweeks = 17
        self.year = year
        self.week = week

    def __eq__(self, other):
        return (self.year, self.week) == (other.year, other.week)

    def __lt__(self, other):
        return (self.year, self.week) < (other.year, other.week)

    def __sub__(self, other):
        dy = (self.year - other.year)
        dw = (self.week - other.week)
        return dy*self.nweeks + dw

    @property
    def next(self):
        if self.week < self.nweeks:
            return Date(self.year, self.week + 1)
        else:
            return Date(self.year + 1, 1)

    @property
    def prev(self):
        if self.week > 1:
            return Date(self.year, self.week - 1)
        else:
            return Date(self.year - 1, self.nweeks)


class Rating:
    """
    Rating class calculates margin-dependent Elo ratings.

    """
    def __init__(self, obs='points', mode='spread', kfactor=59,
                 decay=50, regress=0.64, database='elo.db'):

        # function to initialize a nested dictionary
        def nested_dict():
            return defaultdict(nested_dict)

        # model hyper-parameters
        self.obs = obs
        self.mode = mode
        self.kfactor = kfactor
        self.decay = decay
        self.regress = regress

        # home field advantage
        self.hfa = {'spread': 56, 'total': 0}[mode]

        # point-spread interval attributes
        self.bins = self.bin_edges(obs)
        self.range = 0.5*(self.bins[:-1] + self.bins[1:])

        # list of team names
        self.teams = set()

        # calculate Elo ratings
        self.nfldb = nfldb.connect()
        self.last_game = self.last_played()
        self.spread_prob = self.spread_probability()
        self.elodb = nested_dict()
        self.calc_elo()

    def last_played(self):
        """
        Date(year, week) of most recent completed game.

        """
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular', finished=True)

        last_game = {}

        for game in sorted(
                q.as_games(),
                key=lambda g: Date(g.season_year, g.week)
                ):

            date = Date(game.season_year, game.week)
            last_game.update({game.home_team: date, game.away_team: date})
            self.teams.update([game.home_team, game.away_team])

        return last_game

    def starting_elo(self, margin):
        """
        Initialize the starting elo for each team. For the present
        database, team histories start in 2009. One exception is the LA
        rams.

        """
        TINY = 1e-3
        elo_init = 1500.

        margin = {'spread': margin, 'total': abs(margin)}[self.mode]
        spread_prob = self.spread_prob[margin]
        P = np.clip(spread_prob, TINY, 1 - TINY)

        elo_diff = (
                400*np.log10(1/P - 1)
                if margin < 0 and self.mode == 'total' else
                -400*np.log10(1/P - 1)
                )

        return elo_init + .5*elo_diff

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
        ub = np.append(self.bins[1:], 1)
        bins = {'spread': self.bins, 'total': self.bins[ub > 0]}[self.mode]

        hist, edges = np.histogram(self.spreads(), bins=bins, normed=True)
        spread = 0.5*(edges[:-1] + edges[1:])
        bin_width = edges[1] - edges[0]
        prob = bin_width * np.cumsum(hist[::-1], dtype=float)[::-1]

        return dict(zip(spread, prob))

    def regress_to_mean(self, rtg, margin, factor):
        """
        Regress Elo rating to the mean. Used to project future
        games and update ratings after the offseason.

        """
        default_rtg = self.starting_elo(margin)

        return default_rtg + factor * (rtg - default_rtg)

    def adjust_injuries(self, team, year, week):
        """
        Add or subtract an injury bonus/penalty to the Elo rating

        """
        andrew_luck = (team == 'IND' and year == 2017 and week > 12)
        carson_palmer = (team == 'ARI' and year == 2017 and 7 < week < 15)
        aaron_rodgers = (team == 'GB' and year == 2017 and week > 6)

        if andrew_luck:
            return 100
        elif aaron_rodgers:
            return -100
        elif carson_palmer:
            return -75
        else:
            return 0

    def elo(self, team, margin, year, week):
        """
        Queries the most recent ELO rating for a team, i.e.
        elo(year, week) for (year, week) < (query year, query week)

        """
        date = Date(year, week)
        date_last = self.last_game[team].next

        # extrapolate Elo with information decay for future dates
        if date > date_last:
            last_year, last_week = (date_last.year, date_last.week)
            elo = self.elodb[team][margin][last_year][last_week]

            # number of weeks since last game (don't count bye weeks)
            elapsed = date - date_last - 1
            factor = np.exp(-elapsed/self.decay)

            # adjust for injuries
            elo += self.adjust_injuries(team, year, week)

            # regress Elo to the mean
            return self.regress_to_mean(elo, margin, factor)

        # return the most recent Elo rating, account for bye weeks
        for d in date, date.prev:
            elo = self.elodb[team][margin][d.year][d.week]
            if elo:
                return elo.copy()

        return self.starting_elo(margin)

    def yards(self, game, team):
        """
        Calculates the yards traversed by a team over the course of a
        game.

        """
        def possession(drive):
            return team == nfldb.standard_team(drive.pos_team)

        def progress(drive):
            try:
                side, ydline = str(drive.end_field).split()
                return {'OPP': 100 - int(ydline), 'OWN': int(ydline)}[side]
            except ValueError:
                return 50

        home, away = (game.home_team, game.away_team)
        total = {home: game.home_score, away: game.away_score}[team]

        for drive in filter(lambda d: possession(d), game.drives):
            if drive.result not in ('Field Goal', 'Touchdown'):
                total += 7*progress(drive)/100.

        return total

    def point_diff(self, game):
        """
        Function which returns the point difference.
        The point type is defined by the observable argument.

        """
        home, away = (game.home_team, game.away_team)

        spread = {
                "points": game.home_score - game.away_score,
                "yards": self.yards(game, home) - self.yards(game, away),
                }[self.obs]

        total = {
                "points": game.home_score + game.away_score,
                "yards": self.yards(game, home) + self.yards(game, away),
                }[self.obs]

        return {"spread": spread, "total": total}[self.mode]

    def bin_edges(self, obs):
        """
        Returns an iterator over the range of reasonable point values.

        """
        spread = {
            "points": np.arange(-40.5, 41.5, 1),
            "yards": np.arange(-50.5, 51.5, 1),
            }[self.obs]

        total = {
            "points": np.arange(-100.5, 101.5, 1),
            "yards": np.arange(-100.5, 101.5, 1),
            }[self.obs]

        return {"spread": spread, "total": total}[self.mode]

    def elo_change(self, rating_diff, points, handicap):
        """
        Change in home team ELO rating after a single game

        """
        prob = self.win_prob(rating_diff)
        win = self.kfactor * (1. - prob)
        lose = -self.kfactor * prob

        # TODO fix this for point totals
        return win if points > handicap else lose

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

        # loop over historical games in chronological order
        for game in sorted(
                q.as_games(), key=lambda g: Date(g.season_year, g.week)
                ):

            # game date
            date = Date(game.season_year, game.week)
            year, week = (date.year, date.week)

            # team names
            home = game.home_team
            away = game.away_team

            # point differential
            points = self.point_diff(game)

            # hcap range changes based on mode
            hcap_range = {
                    'spread': self.range,
                    'total': self.range[self.range > 0],
                    }[self.mode]

            # loop over all possible spread margins
            for hcap in hcap_range:

                # query current elo ratings from most recent game
                home_rtg = self.elo(home, hcap, year, week)
                away_rtg = self.elo(away, -hcap, year, week)

                # elo change when home(away) team is handicapped
                rtg_diff = home_rtg - away_rtg + self.hfa
                bounty = self.elo_change(rtg_diff, points, hcap)

                # scale update by ngames if necessary
                home_rtg += bounty
                away_rtg -= bounty

                # update elo ratings
                next_year, next_week = (date.next.year, date.next.week)
                updates = [(home, hcap, home_rtg), (away, -hcap, away_rtg)]

                for (team, hcap, rtg) in updates:
                    if next_year > year:
                        rtg = self.regress_to_mean(rtg, hcap, self.regress)

                    self.elodb[team][hcap][next_year][next_week] = rtg

    def cdf(self, home, away, year, week):
        """
        Cumulative (integrated) probability that,
        score home - score away > x.

        """
        cprob = []

        hcap_range = {
                'spread': self.range,
                'total': self.range[self.range > 0],
                }[self.mode]

        for hcap in hcap_range:
            home_rtg = self.elo(home, hcap, year, week)
            away_rtg = self.elo(away, -hcap, year, week)
            elo_diff = home_rtg - away_rtg + self.hfa

            win_prob = self.win_prob(elo_diff)
            cprob.append(win_prob)

        return hcap_range, cprob

    def predict_spread(self, home, away, year, week, perc=0.5):
        """
        Predict the spread for a matchup, given current knowledge of each
        team's ELO ratings.
        """
        # cumulative spread distribution
        spreads, cprob = self.cdf(home, away, year, week)

        # plot median prediction (compare to vegas spread)
        index = np.square([p - perc for p in cprob]).argmin()
        if index in range(1, len(cprob) - 2):
            x0, y0 = (spreads[index - 1], cprob[index - 1])
            x1, y1 = (spreads[index], cprob[index])
            x2, y2 = (spreads[index + 1], cprob[index + 1])

            # fit a quadratic polynomial
            coeff = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
            res = minimize(
                    lambda x: np.square(np.polyval(coeff, x) - perc), x1
                  )

            return 0.5 * round(res.x * 2)

        return spreads[index]

    def predict_score(self, home, away, year, week):
        """
        The model predicts the CDF of win margins, i.e. F = P(spread > x).
        One can use integration by parts to calculate the expected score
        from the CDF,

        E(x) = \int x P(x)
             = x*F(x)| - \int F(x) dx

        """
        # cumulative spread distribution
        x, F = self.cdf(home, away, year, week)
        dx = (x[1] - x[0])
        int_term = sum(F)*dx
        x0, x1 = (min(x), max(x))
        bdry_term = x1*F[-1] - x0*F[0]

        return int_term - bdry_term - .5*dx

    def win_prob(self, rtg_diff):
        """
        Win probability as function of ELO difference

        """
        return 1./(10**(-rtg_diff/400.) + 1.)

    def model_accuracy(self, year=None):
        """
        Calculate the mean and standard deviation of the model's
        residuals = observed - predicted.

        """
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular', finished='True')
        if year is not None:
            q.game(season_year=year)

        residuals = []

        for game in sorted(
                q.as_games(),
                key=lambda g: Date(g.season_year, g.week)
                ):

            year = game.season_year
            week = game.week
            home = game.home_team
            away = game.away_team

            # allow for one season burn-in
            if year > 2009:
                predicted = self.predict_score(home, away, year, week)
                observed = game.home_score - game.away_score
                residuals.append(observed - predicted)

        return residuals

    def optimize(self):
        """
        Function to optimize model hyper-parameters

        """
        def objective(parameters):
            """
            Evaluates the mean absolute error for a set of input
            parameters: kfactor, decay, regress.

            """
            kfactor, decay, regress = parameters
            rating = Rating(kfactor=kfactor, decay=decay, regress=regress)
            residuals = rating.model_accuracy()
            mean_abs_error = np.abs(residuals).mean()
            return mean_abs_error

        bounds = [(10, 100), (10, 10**3, "log-uniform"), (0.1, 1)]
        res_gp = gp_minimize(objective, bounds, n_calls=10, random_state=0)

        print("Best score: {:.4f}".format(res_gp.fun))
        print("Best parameters: {}".format(res_gp.x))


def main():
    """
    Main function prints the model accuracy parameters and exits

    """
    rating = Rating(database='elo.db')
    residuals = rating.model_accuracy()
    mean_error = np.mean(residuals)
    rms_error = np.std(residuals)
    mean_abs_error = np.abs(residuals).mean()

    print("residual mean:", mean_error)
    print("residual std:", rms_error)
    print("residual abs error:", mean_abs_error)


if __name__ == "__main__":
    main()
