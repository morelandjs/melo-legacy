#!/usr/bin/env python2

import copy
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

import nfldb


nweeks = 17
AVG_ELO = 1500.

class Rating:
    def __init__(self, database='elo.db'):
        nested_dict = lambda: defaultdict(nested_dict)
        self.nfldb = nfldb.connect()
        self.hfa = self.home_field_advantage()
        self.elodb = nested_dict()

    def home_field_advantage(self):
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')

        spreads = [g.home_score - g.away_score for g in q.as_games()]
        hfa = sum(spreads)/float(len(spreads))

        return hfa

    def starting_elo(self, margin):
        """
        Initialize the starting elo for each team. For the present
        database, team histories start in 2009. One exception is the LA
        rams.

        """
        return {'fair': 1500, 'hcap': 1500}

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

        """
        for yr, wk in self.rewind(year, week):
            elo = self.elodb[team][margin][yr][wk]
            if elo:
                return elo.copy()

        elo = self.starting_elo(margin)
        self.elodb[team][margin][year-1][nweeks] = elo
        return elo

    def calc_elo(self, k_factor=40.):
        """
        This function calculates ELO ratings for every team
        for every value of the spread. The ratings are stored
        in an sqlite database for subsequent reference.

        """
        # nfldb database
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')

        # small sorting function
        def time(game):
            return game.season_year, game.week

        # elo point change
        def elo_change(rating1, rating2, home_wins=True):
            prob = self.win_prob(rating1, rating2)
            if home_wins:
                return k_factor * (1. - prob)
            return - k_factor * prob


        # loop over historical games in chronological order
        for game in sorted(q.as_games(), key=lambda g: time(g)):

            # game attributes
            year = game.season_year
            week = game.week
            home = game.home_team
            away = game.away_team

            # point differential
            points = game.home_score - game.away_score

            # loop over all possible spread margins
            for handicap in range(0, 40):

                # query current elo ratings from most recent game
                home_rtg = self.query_elo(home, handicap, year, week)
                away_rtg = self.query_elo(away, handicap, year, week)

                # handicap the home team
                if points - handicap >= 0:
                    # home team wins
                    bounty = elo_change(
                            home_rtg['hcap'],
                            away_rtg['fair'],
                            home_wins=True
                            )
                    home_rtg['hcap'] += bounty
                    away_rtg['fair'] -= bounty
                else:
                    # away team wins
                    bounty = elo_change(
                            home_rtg['hcap'],
                            away_rtg['fair'],
                            home_wins=False
                            )
                    home_rtg['hcap'] += bounty
                    away_rtg['fair'] -= bounty

                # handicap the away team
                if points + handicap >= 0:
                    # home team wins
                    bounty = elo_change(
                            home_rtg['fair'],
                            away_rtg['hcap'],
                            home_wins=True
                            )
                    home_rtg['fair'] += bounty
                    away_rtg['hcap'] -= bounty
                else:
                    # away team wins
                    bounty = elo_change(
                            home_rtg['fair'],
                            away_rtg['hcap'],
                            home_wins=False
                            )
                    home_rtg['fair'] += bounty
                    away_rtg['hcap'] -= bounty

                # update elo ratings
                for team, team_rtg in [(home, home_rtg), (away, away_rtg)]:
                    self.elodb[team][handicap][year][week] = team_rtg

    def elo_history(self, team, margin):
        """
        Plot the ELO rating history for a given team with a 
        specified margin of victory handicap.

        """
        def time(game):
            return game.season_year + game.week/float(nweeks)

        def elo(game):
            rtg = self.query_elo(team, margin, game.season_year, game.week)
            return rtg['hcap']

        q = nfldb.Query(self.nfldb)
        q.game(team=team, season_type='Regular')
        q.as_games()

        history = [(time(game), elo(game))
                for game in sorted(q.as_games(), key=lambda g: time(g))]

        plt.plot(*zip(*history), label=team)
        plt.xlabel('Time (year, week)')
        plt.ylabel('ELO rating'.format(margin))
        plt.title('Handicap={} pts'.format(margin))
        plt.legend()

    def predict_spread(self, team, opp, year, week):
        """
        Predict the spread for a matchup, given current knowledge of each
        team's ELO ratings.

        """
        def winprob(margin):
            team_rtg = self.query_elo(team, margin, year, week)
            opp_rtg = self.query_elo(opp, margin, year, week)
            return self.win_prob(team_rtg['hcap'], opp_rtg['fair'])

        prob = [(margin, winprob(margin)) for margin in range(40)]
        label = ' '.join([team, '@'+opp])
        plt.plot(*zip(*prob), label=label)

    def win_prob(self, team_rating, opp_rating):
        elo_diff = team_rating - opp_rating
        logistic_prob = 1/(10**(-elo_diff/400) + 1)

        return logistic_prob


rating = Rating(database='elo.db')
rating.calc_elo()
rating.elo_history('CLE', 0)
#rating.predict_spread('NE', 'CLE', 2016, 17)
plt.show()
