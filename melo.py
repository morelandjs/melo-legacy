#!/usr/bin/env python2

from collections import defaultdict
from functools import total_ordering
from math import sqrt

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf, erfinv
from skopt import gp_minimize

import nfldb

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
    def __init__(self, mode='spread', database='elo.db',
            kfactor=None, hfa=None, regress=None, decay=None, smooth=None):

        # function to initialize a nested dictionary
        def nested_dict():
            return defaultdict(nested_dict)

        # short function to toggle defaults
        def opt(defaults, manual):
            return defaults if manual is None else manual

        # model hyper-parameters
        self.mode = mode
        self.decay = decay
        self.kfactor = kfactor
        self.regress = regress
        self.smooth = smooth

        # Elo constants
        self.elo_init = 1500
        self.sigma = 300

        # default hyper-parameter settings
        self.kfactor = opt({'spread': 75., 'total': 38}[mode], kfactor)
        self.hfa = opt({'spread': 50., 'total': 0}[mode], hfa)
        self.regress = opt({'spread': .58, 'total': .7}[mode], regress)
        self.decay = opt({'spread': 51, 'total': 51}[mode], regress)
        self.smooth= opt(8.5, smooth)

        # handicap values
        self.hcaps = self.handicaps(mode)

        # list of team names
        self.teams = set()

        # calculate Elo ratings
        self.nfldb = nfldb.connect()
        self.last_game = self.last_played
        self.point_prob = self.point_probability
        self.elodb = nested_dict()
        self.calc_elo()

    @property
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
        database, team histories start in 2009. One exception is the
        LA Rams.

        """
        TINY = 1e-3

        phys_margin = {'spread': margin, 'total': abs(margin)}[self.mode]
        point_prob = self.point_prob[phys_margin]
        P = np.clip(point_prob, TINY, 1 - TINY)
        Delta_R = sqrt(2)*self.sigma*erfinv(2*P - 1)

        positive_rtg = (margin < 0 and self.mode == 'total')
        return self.elo_init + .5*(-Delta_R if positive_rtg else Delta_R)

    @property
    def game_points(self):
        """
        All game points (spreads or totals) since 2009.

        """
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular', finished=True)

        return [self.points(g) for g in q.as_games()]

    @property
    def point_probability(self):
        """
        Probabilities of observing each point spread.

        """
        points = np.array(self.game_points)
        prob = [(points > hcap).sum(dtype=float)/points.size
                for hcap in self.hcaps]

        return dict(zip(self.hcaps, prob))

    def elo(self, team, margin, year, week):
        """
        Queries the most recent ELO rating for a team, i.e.
        elo(year, week) for (year, week) < (query year, query week)

        """
        # used to predict performance against an average team
        if team == 'AVG':
            return self.starting_elo(margin)

        date = Date(year, week)
        date_last = self.last_game[team].next

        # extrapolate Elo with information decay for future dates
        if date > date_last:
            last_year, last_week = (date_last.year, date_last.week)
            elo = self.elodb[team][margin][last_year][last_week]

            # number of weeks since last game (don't count bye weeks)
            elapsed = date - date_last - 1
            week_decay = np.exp(-elapsed/self.decay)

            # regress Elo to the mean
            return self.regress_to_mean(elo, week_decay, margin)

        # return the most recent Elo rating, account for bye weeks
        for d in date, date.prev:
            elo = self.elodb[team][margin][d.year][d.week]
            if elo:
                return elo.copy()

        return self.starting_elo(margin)

    def points(self, game):
        """
        Function which returns either a point difference
        or a point total depending on the mode argument.

        """
        point_diff = game.home_score - game.away_score
        point_total = game.home_score + game.away_score

        return {"spread": point_diff, "total": point_total}[self.mode]

    def handicaps(self, mode):
        """
        Returns an iterator over the range of reasonable point values.

        """
        spread_range = np.arange(-50.5, 51.5, 1)
        total_range = np.arange(-100.5, 101.5, 1)

        return {"spread": spread_range, "total": total_range}[mode]

    def norm_cdf(self, x, loc=0, scale=1):
        """
        Normal cumulative probability distribution

        """
        return 0.5*(1 + erf((x - loc)/(sqrt(2)*scale)))

    def elo_change(self, rating_diff, points, hcap):
        """
        Change in home team ELO rating after a single game

        """
        sign = np.sign(hcap)
        pts = {'spread': points, 'total': sign*points}[self.mode]

        TINY = 1e-3

        prior = self.win_prob(rating_diff)
        exp_arg = (pts - hcap)/max(self.smooth, TINY)

        if self.smooth:
            post = self.norm_cdf(pts, loc=hcap, scale=self.smooth)
        else:
            post = (1 if pts > hcap else 0)

        return self.kfactor * (post - prior)

    def regress_to_mean(self, rtg, factor, hcap):
        """
        Regress an Elo rating to it's default (resting) value

        """
        default_rtg = self.starting_elo(hcap)

        return default_rtg + factor * (rtg - default_rtg)

    def calc_elo(self):
        """
        This function calculates ELO ratings for every team
        for every value of the point spread/total.
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
            yr, wk = (date.year, date.week)
            nxt_yr, nxt_wk = (date.next.year, date.next.week)

            # team names
            home = game.home_team
            away = game.away_team

            # point differential
            points = self.points(game)

            # loop over all possible spread/total margins
            for hcap in self.hcaps:

                # query current elo ratings from most recent game
                home_rtg = self.elo(home, hcap, yr, wk)
                away_rtg = self.elo(away, -hcap, yr, wk)

                # elo change when home(away) team is handicapped
                rtg_diff = home_rtg - away_rtg + self.hfa
                bounty = self.elo_change(rtg_diff, points, hcap)

                # scale update by ngames if necessary
                home_rtg += bounty
                away_rtg -= bounty

                # update elo ratings
                for (team, hcap, rtg) in [
                        (home, hcap, home_rtg),
                        (away, -hcap, away_rtg),
                        ]:

                    # regress Elo to the mean
                    if nxt_yr > yr:
                        rtg = self.regress_to_mean(rtg, self.regress, hcap)

                    self.elodb[team][hcap][nxt_yr][nxt_wk] = rtg

    def cdf(self, home, away, year, week):
        """
        Cumulative (integrated) probability that,
        score home - score away > x.

        """
        cprob = []

        hcap_range = {
                'spread': self.hcaps,
                'total': self.hcaps[self.hcaps > 0],
                }[self.mode]

        for hcap in hcap_range:
            home_rtg = self.elo(home, hcap, year, week)
            away_rtg = self.elo(away, -hcap, year, week)

            hfa = 0 if 'AVG' in (home, away) else self.hfa
            elo_diff = home_rtg - away_rtg + hfa

            win_prob = self.win_prob(elo_diff)
            cprob.append(win_prob)

        return hcap_range, sorted(cprob, reverse=True)

    def predict_spread(self, home, away, year, week, perc=0.5):
        """
        Predict the spread/total for a matchup, given current
        knowledge of each team's Elo ratings.
        """
        # cumulative point distribution
        points, cprob = self.cdf(home, away, year, week)

        # plot median prediction (compare to vegas spread/total)
        index = np.square([p - perc for p in cprob]).argmin()
        if index in range(1, len(cprob) - 2):
            x0, y0 = (points[index - 1], cprob[index - 1])
            x1, y1 = (points[index], cprob[index])
            x2, y2 = (points[index + 1], cprob[index + 1])

            # fit a quadratic polynomial
            coeff = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
            res = minimize(
                    lambda x: np.square(np.polyval(coeff, x) - perc), x1
                  )

            return 0.5 * round(res.x * 2)

        return points[index]

    def predict_score(self, home, away, year, week):
        """
        The model predicts the CDF of win margins, i.e. F = P(points > x).
        One can use integration by parts to calculate the expected score
        from the CDF,

        E(x) = \int x P(x)
             = x*F(x)| - \int F(x) dx

        """
        # cumulative point distribution
        x, F = self.cdf(home, away, year, week)
        x0, x1 = (min(x), max(x))
        dx = (x[1] - x[0])

        # integral and boundary terms
        int_term = sum(F)*dx
        bdry_term = x1*F[-1] - x0*F[0]

        return int_term - bdry_term - .5*dx

    def win_prob(self, rtg_diff):
        """
        Win probability as function of ELO difference

        """
        return self.norm_cdf(rtg_diff, scale=self.sigma)

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
                observed = self.points(game)
                residuals.append(observed - predicted)

        return residuals

    def optimize(self, mode):
        """
        Function to optimize model hyper-parameters

        """
        def obj(parameters):
            """
            Evaluates the mean absolute error for a set of input
            parameters: kfactor, decay, regress.

            """
            kfactor, regress, smooth = parameters
            rating = Rating(mode=mode, kfactor=kfactor,
                    regress=regress, smooth=smooth)
            residuals = rating.model_accuracy()
            mean_abs_error = np.abs(residuals).mean()
            return mean_abs_error

        bounds = [(65., 85.), (0.48, 0.68), (4.0, 10.0)]
        res_gp = gp_minimize(obj, bounds, n_calls=100, verbose=True)

        print("Best score: {:.4f}".format(res_gp.fun))
        print("Best parameters: {}".format(res_gp.x))


def main():
    """
    Main function prints the model accuracy diagnostics and exits

    """
    # Rating().optimize('spread')
    rating = Rating(mode='spread')
    residuals = rating.model_accuracy()
    mean_error = np.mean(residuals)
    rms_error = np.std(residuals)

    residuals = rating.model_accuracy()
    mean_abs_error = np.abs(residuals).mean()

    print("mean:", mean_error)
    print("std dev:", rms_error)
    print("mean abs error:", mean_abs_error)


if __name__ == "__main__":
    main()
