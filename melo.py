#!/usr/bin/env python2

from collections import defaultdict
from functools import total_ordering

import numpy as np
from scipy.optimize import minimize

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
    def __init__(self, obs='points', mode='spread', kfactor=60, hfa=58,
            decay=50, regress=0.7, database='elo.db'):

        # function to initialize a nested dictionary
        nested_dict = lambda: defaultdict(nested_dict)

        # model hyper-parameters
        self.obs = obs
        self.mode = mode
        self.kfactor = kfactor
        self.hfa = hfa
        self.decay = decay
        self.regress = regress

        # point-spread interval attributes
        self.bins= self.range(obs)
        self.range = 0.5*(self.bins[:-1] + self.bins[1:])

        # list of team names
        self.teams = set()

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

        # look up most recent game
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
        elo_init = 1500.
        prob = max(self.spread_prob[abs(margin)], 1e-6)
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
        spreads = self.spreads()
        reversed_spreads = [-s for s in spreads]
        hist, edges = np.histogram(
                spreads + reversed_spreads, bins=self.bins, normed=True
                )
        spread = 0.5*(edges[:-1] + edges[1:])
        prob = np.cumsum(hist[::-1], dtype=float)[::-1]

        return dict(zip(spread, prob))

    def regress_to_mean(self, rtg, margin, factor):
        """
        Regress Elo rating to the mean. Used to project future
        games and update ratings after the offseason.

        """
        default_rtg = self.starting_elo(margin)

        return default_rtg + factor * (rtg - default_rtg)

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

            elapsed = float(date - date_last)
            factor = np.exp(-elapsed/self.decay)

            return self.regress_to_mean(elo, margin, factor)

        # return the most recent Elo rating, account for bye weeks
        for d in date, date.prev:
            elo = self.elodb[team][margin][d.year][d.week]
            if elo: return elo.copy()

        return self.starting_elo(margin)

    def yds(self, game, team):
        """
        Calculates the yards traversed by a team over the course of a
        game.

        """
        yards = 0

        for drive in filter(
                lambda d: nfldb.standard_team(d.pos_team)==team,
                game.drives
                ):
    
            try:
                side, yardline = str(drive.end_field).split()
                field = {'OPP': 100 - int(yardline), 'OWN': int(yardline)}
                progress = field[side]
            except ValueError:
                progress = 50

            if drive.result == 'Touchdown':
                yards += 100
            elif drive.result == 'Field Goal':
                yards += progress + (3/7)*(100 - progress)
            else:
                yards += progress

        return yards

    def point_diff(self, game):
        """
        Function which returns the point difference.
        The point type is defined by the observable argument.

        """
        home, away = (game.home_team, game.away_team)

        spread_dict = {
                "points": game.home_score - game.away_score,
                "yards": self.yds(game, home) - self.yds(game, away)
                }

        total_dict = {
                "points": game.home_score + game.away_score,
                "yards": self.yds(game, home) + self.yds(game, away)
                }

        mode_dict = {
            "spread": spread_dict[self.obs],
            "total": total_dict[self.obs]
            }

        return mode_dict[self.mode]

    def range(self, obs):
        """
        Returns an iterator over the range of reasonable point values.

        """
        spread_dict = {
            "points": np.arange(-40.5, 41.5, 1),
            "yards": np.arange(-505, 515, 10)
            }

        total_dict = {
            "points": np.arange(-0.5, 90.5, 1),
            "yards": np.arange(-5, 1005, 10)
            }

        mode_dict = {
            "spread": spread_dict[obs],
            "total": total_dict[obs]
            }

        return mode_dict[self.mode]

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

        # loop over historical games in chronological order
        for game in sorted(q.as_games(),
                key=lambda g: Date(g.season_year, g.week)):

            # game date
            date = Date(game.season_year, game.week)
            year, week = (date.year, date.week)

            # team names
            home = game.home_team
            away = game.away_team

            # point differential
            points = self.point_diff(game)

            # loop over all possible spread margins
            for hcap in self.range:

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

        for handicap in self.range:
            home_rtg = self.elo(home, handicap, year, week)
            away_rtg = self.elo(away, -handicap, year, week)

            rtg_diff = home_rtg - away_rtg + self.hfa

            cprob.append(self.win_prob(rtg_diff))

        return self.range, cprob

    def predict_spread(self, home, away, year, week, perc=0.5):
        """
        Predict the spread for a matchup, given current knowledge of each
        team's ELO ratings.
        """
        # cumulative spread distribution
        spreads, cprob = self.cdf(home, away, year, week)

        # plot median prediction (compare to vegas spread)
        index = np.square([p - perc for p in cprob]).argmin()
        if index in (1, len(cprob) - 2): 
            x0, y0 = (spreads[index - 1], cprob[index - 1])
            x1, y1 = (spreads[index], cprob[index])
            x2, y2 = (spreads[index + 1], cprob[index + 1])

            # fit a quadratic polynomial
            coeff = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
            res = minimize(
                    lambda x: np.square(np.polyval(coeff, x) - perc), x1
                  )

            return res.x

        return spreads[index]

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
        for game in sorted(q.as_games(),
                key=lambda g: Date(g.season_year, g.week)):
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

    rank = {}

    for team in rating.teams:
        date = rating.last_game[team].next
        elo = rating.elo(team, 0, date.year, date.week)
        rank.update({team: elo})

    for team in sorted(rank, key=rank.get, reverse=True):
        print(team, rank[team])

if __name__ == "__main__":
    main()
