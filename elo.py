#!/usr/bin/env python2

import sqlite3
from collections import defaultdict, namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import nfldb
import numpy as np


WEEKS = 17.
AVG_ELO = 1500.

Rtg = namedtuple('Rtg', 'elo, elo_')

class Rating:
    def __init__(self, database='elo.db'):

        self.nfldb = nfldb.connect()
        self.elodb = self.build(database)
        #self.hfa = self.home_field_advantage()

    def build(self, name):
        dirname = Path('sqlite')
        if not dirname.exists():
            dirname.mkdir()
        dest = dirname / name

        return sqlite3.connect(str(dest))


    def starting_elo(self, margin):
        return Rtg(elo=1500, elo_=1500)

    def query_elo(self, team, year, week, margin):
        """
        Retrieve elo ratings for a given team
        for (year, week) with given margin

        """

        c = self.elodb.cursor()

        query = c.execute(
        '''SELECT elo, elo_ FROM rating
        WHERE team = ? AND year < ? AND week < ? AND margin = ?
        ORDER BY year, week DESC''',
        (team, year, week, margin)
        ).fetchone()

        if query:
            return Rtg(*query)
        else:
            rtg = self.starting_elo(margin)
            c.execute(
                    '''INSERT INTO rating
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (team, year, week, margin, rtg.elo, rtg.elo_)
                    )
            return rtg

    def set_elo(self, team, year, week, margin, elo, elo_):
        """
        Set elo ratings for a given team
        for (year, week) with given margin

        """
        c = self.elodb.cursor()

        c.execute(
                '''INSERT INTO rating
                VALUES (?, ?, ?, ?, ?, ?)''',
                (team, year, week, margin, elo, elo_)
                )

    def calc_elo(self, k_factor=40.):
        """
        This function calculates ELO ratings for every team
        for every value of the spread. The ratings are stored
        in an sqlite database for subsequent reference.

        """
        # nfldb database
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')

        # elo ratings database
        c = self.elodb.cursor()
        c.execute(
        '''CREATE TABLE IF NOT EXISTS rating
        (team text, year int, week int,
        margin int, elo float, elo_ float)'''
        )

        def time(game):
            return game.season_year, game.week

        # loop over historical games in chronological order
        for game in sorted(q.as_games(), key=lambda g: time(g)):

            # game attributes
            year = game.season_year
            week = game.week
            home = game.home_team
            away = game.away_team

            # loop over all possible spread margins
            for spread in range(0, 40):

                # retrieve elo ratings
                home_rtg = self.query_elo(home, year, week, spread)
                away_rtg = self.query_elo(away, year, week, spread)
                print(year, week, spread, home_rtg)

                # first handicap the home team
                score = game.home_score - game.away_score - spread

                def elo_change(rtg1, rtg2, win):
                    prob = self.win_prob(rtg1, rtg2)
                    if win:
                        return k_factor * (1. - prob)
                    return - k_factor * prob


                # handicapped home team covers spread
                if score > 0:
                    print('score > 0')
                    pts1 = elo_change(home_rtg.elo, away_rtg.elo_, True) 
                    pts2 = elo_change(home_rtg.elo_, away_rtg.elo, True) 

                    self.set_elo(
                            home, year, week, spread,
                            home_rtg.elo + pts1, home_rtg.elo_ + pts2
                            )

                    self.set_elo(
                            away, year, week, spread,
                            away_rtg.elo - pts2, away_rtg.elo_ - pts1
                            )
                else:
                    print('score < 0')
                    pts1 = elo_change(home_rtg.elo, away_rtg.elo_, False) 
                    pts2 = elo_change(home_rtg.elo_, away_rtg.elo, False) 

                    self.set_elo(
                            home, year, week, spread,
                            home_rtg.elo + pts1, home_rtg.elo_ + pts2
                            )

                    self.set_elo(
                            away, year, week, spread,
                            away_rtg.elo - pts2, away_rtg.elo_ - pts1
                            )

#    def current_elo(self, team, game):
#        q = nfldb.Query(self.nfldb)
#        q.game(team=team, season_type='Regular')
#
#        all_games = sorted(q.as_games(), key=lambda g: self.time(g))
#        new_team = all(self.time(game) <= self.time(g) for g in all_games)
#
#        if new_team:
#            self.elo[team][game.season_year][game.week] = AVG_ELO
#
#        return self.elo[team][game.season_year][game.week]
#
#    def update_elo(self, home, away, win_margin, handicap=0):
#        team_win_prob = self.win_prob(home, away)
#        opp_win_prob = 1 - team_win_prob
#
#        # handicapped home team
#        if win_margin > win_line:
#            self.elo[home.team][year][week]
#                    
#                    = home_elo + k_factor * opp_win_prob 
#            away_elo
#
#        win_margin - win_line
#
#        return home_elo, away_elo
#
#    def home_field_advantage(self):
#        q = nfldb.Query(self.nfldb)
#        q.game(season_type='Regular')
#
#        hfa = np.mean([g.home_score - g.away_score for g in q.as_games()])
#
#        return hfa
#
#    def increment(self, team, last_game):
#        q = nfldb.Query(self.nfldb)
#        q.game(team=team, season_type='Regular')
#
#        for game in sorted(q.as_games(), key=lambda g: self.time(g)):
#            if self.time(game) > self.time(last_game):
#                return game
#
#        return None
#
#    def plot(self, team):
#        q = nfldb.Query(self.nfldb)
#        q.game(team=team, season_type='Regular')
#        games = sorted(q.as_games(), key=lambda g: self.time(g))
#
#        team_elo = [
#                (g.season_year + g.week/WEEKS,
#                 self.elo[team][g.season_year][g.week]) for g in games
#        ]
#
#        plt.plot(*zip(*team_elo))
#
#    def predict_spread(self, home_team, away_team, year, week):
#        q = nfldb.Query(self.nfldb)
#        q.game(home_team=home_team, away_team=away_team, season_year=year,
#                week=week, season_type='Regular')
#
#        def most_recent_elo(team):
#            q = nfldb.Query(self.nfldb)
#            q.game(team=team, season_type='Regular')
#            team_games = sorted(q.as_games(), key=lambda g: self.time(g))
#            game_year, game_week = [
#                    (g.season_year, g.week) for g in team_games
#                    if self.time(g) <= (year, week)
#                    ].pop()
#
#            if game_week == WEEKS:
#                last_elo = self.elo[team][game_year + 1][1]
#            else:
#                last_elo = self.elo[team][game_year][game_week]
#
#            if year == game_year:
#                elapsed_weeks = week - game_week
#                decay = np.exp(-elapsed_weeks/self.elo_decay)
#            else:
#                weeks_left = (WEEKS - game_week)
#                elapsed_weeks = weeks_left + week
#                elapsed_years = year - game_year
#                year_penalty = (1 - self.regress_to_mean)**elapsed_years
#                decay = year_penalty*np.exp(-elapsed_weeks/self.elo_decay)
#
#            return AVG_ELO + decay * (last_elo - AVG_ELO)
#
#        home_elo = most_recent_elo(home_team)
#        away_elo = most_recent_elo(away_team)
#        elo_diff = home_elo - away_elo
#
#        return elo_diff/self.elo_point_conv + self.hfa
#
#
#    def validate(self):
#        red_elo = 1500.
#        blue_elo = 1500.
#
#        red_win_rate = np.random.rand()
#        blue_win_rate = 1 - red_win_rate
#        samples = 10**3
#        binom = np.random.binomial(n=1, p=red_win_rate, size=samples)
#        spreads = 2*binom - 1
#
#        red = plt.cm.Reds(.6)
#        blue = plt.cm.Blues(.6)
#
#        for game, point_margin in enumerate(spreads):
#            points = self.elo_change(red_elo, blue_elo, point_margin)
#
#            red_elo += points
#            blue_elo -= points
#
#            red_win_prob = self.win_prob(red_elo, blue_elo)
#            blue_win_prob = self.win_prob(blue_elo, red_elo)
#
#            plt.scatter(game, red_win_prob, color=red)
#            plt.scatter(game, blue_win_prob, color=blue)
#
#        plt.axhline(y=red_win_rate, color=red)
#        plt.axhline(y=blue_win_rate, color=blue)
#
#        plt.xlabel('Game')
#        plt.ylabel('Win probability')
#        plt.ylim(0, 1)
#        plt.show()
#            
#
#
    def win_prob(self, team_rating, opp_rating):
        elo_diff = team_rating - opp_rating
        logistic_prob = 1/(10**(-elo_diff/400) + 1)

        return logistic_prob


rating = Rating(database='elo.db')
rating.calc_elo()
