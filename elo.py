#!/usr/bin/env python2

import sqlite3
from collections import defaultdict
from pathlib import Path

import nfldb


nweeks = 17.
AVG_ELO = 1500.



class Rating:
    def __init__(self, database='elo.db'):

        self.nfldb = nfldb.connect()
        self.hfa = self.home_field_advantage()

    def home_field_advantage(self):
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')

        spreads = [g.home_score - g.away_score for g in q.as_games()]
        hfa = sum(spreads)/float(len(spreads))

        return hfa

    def starting_elo(self, margin):
        # (fair elo, handicapped elo)
        return {'fair': 1500, 'hcap': 1500}

    def calc_elo(self, k_factor=40.):
        """
        This function calculates ELO ratings for every team
        for every value of the spread. The ratings are stored
        in an sqlite database for subsequent reference.

        """
        # nfldb database
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')

        # create the elo record dictionary
        nested_dict = lambda: defaultdict(nested_dict)
        elodb = nested_dict()

        def time(game):
            return game.season_year, game.week

        def elo_change(rtg1, rtg2, home_wins=True):
            prob = self.win_prob(rtg1, rtg2)
            if home_wins:
                return k_factor * (1. - prob)
            return - k_factor * prob

        def query_elo(team, year, week, margin):
            """
            Retrieve elo ratings for a given team
            for (year, week) with given margin

            """
            while True:
                if week > 1:
                    week -= 1
                else:
                    year -= 1
                    week = nweeks

                if not elodb[team][year]:
                    starting_elo = self.starting_elo(margin)
                    print(starting_elo)
                    elodb[team][year][week][margin] = starting_elo
                    print(elodb[team][year][week][margin])
                    return starting_elo
                elif elodb[team][year][week][margin]:
                    print(elodb[team][year][week][margin])
                    return elodb[team][year][week][margin]

        # loop over historical games in chronological order
        for game in sorted(q.as_games(), key=lambda g: time(g)):

            # game attributes
            year = game.season_year
            week = game.week
            home = game.home_team
            away = game.away_team
            #print(year, week, home, '@' + away)

            # point differential
            points = game.home_score - game.away_score - self.hfa

            # loop over all possible spread margins
            for handicap in range(0, 1):

                # query current elo ratings from most recent game
                home_rtg = query_elo(home, year, week, handicap)
                away_rtg = query_elo(away, year, week, handicap)

                #print(home_rtg)
                #print(away_rtg)
                #print

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
                    elodb[team][year][week][handicap] = rating

    def predict_spread(self, team, opp, year, week):

        for margin in range(0, 40):
            team_rtg = query_elo(team, year, week, margin)
            opp_rtg = query_elo(opp, year, week, margin)
            prob = self.win_prob(team_rtg['hcap'], opp_rtg['fair'])
            print(margin, prob)



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

rating.predict_spread('NE', 'CLE', 2016, 17)
