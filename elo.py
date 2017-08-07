#!/usr/bin/env python2

import copy
import sqlite3
from collections import defaultdict
from functools import partial
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nfldb


nweeks = 17
nested_dict = lambda: defaultdict(nested_dict)

class Rating:
    def __init__(self, kfactor=40, kdecay=60, database='elo.db'):
        # k factor parameters
        self.kfactor = kfactor
        self.kdecay = kdecay

        # score and elo containers
        self.nfldb = nfldb.connect()
        self.spread_prob = self.spread_probability()
        self.elodb = self.calc_elo()

    def hfa(self, elodb, handicap, year, week):
        """
        ELO bonus for home field advantage

        """
        home_rtg = self.query_elo(elodb, 'home', handicap, year, week)
        away_rtg = self.query_elo(elodb, 'away', handicap, year, week)

        hcap_diff = home_rtg['hcap'] - away_rtg['hcap']
        fair_diff = home_rtg['fair'] - away_rtg['fair']

        return 0.5*(hcap_diff + fair_diff)

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

    def query_elo(self, elodb, team, margin, year, week):
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
            elo = elodb[team][margin][yr][wk]
            if elo:
                return elo.copy()

        elo = self.starting_elo(margin)
        elodb[team][margin][year-1][nweeks] = elo

        return elo


    def elo_change(self, rating_diff, points, handicap):
        """
        Change in home team ELO rating after a single game

        """
        prob = self.win_prob(rating_diff)
        K = self.kfactor * np.exp(-handicap/self.kdecay)

        # home team wins
        if points - handicap > 0:
            return K * (1. - prob)
        # home and away team tie
        elif points - handicap == 0:
            return K * (0.5 - prob)
        # away team wins
        elif points - handicap < 0:
            return -K * prob

    def calc_elo(self):
        """
        This function calculates ELO ratings for every team
        for every value of the spread.

        """
        # nfldb database
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')

        # store elo ratings
        elodb = nested_dict()
        elo = self.query_elo

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
                    home_rtg = elo(elodb, home, handicap, year, week)
                    away_rtg = elo(elodb, away, handicap, year, week)

                    # disable hfa for home vs away elo calc
                    hfa = (self.hfa(elodb, handicap, year, week)
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
                        elodb[team][handicap][year][week] = team_rtg

        # return final elo ratings
        return elodb

    def elo_history(self, team, margin):
        """
        Plot the ELO rating history for a given team with a 
        specified margin of victory handicap.

        """
        # convert (year, week) into a single date number
        def date(game):
            return game.season_year + game.week/float(nweeks)

        # query database for game dates
        q = nfldb.Query(self.nfldb)
        q.game(season_type='Regular')
        if team not in ('home', 'away'):
            q.game(team=team)

        # unpack historical elo ratings
        rtg_history = [
                (date(game),
                self.query_elo(
                    self.elodb,
                    team,
                    margin,
                    game.season_year,
                    game.week
                    ))
                for game in sorted(q.as_games(), key=lambda g: date(g))
                ]

        # plot "fair" elo rating history
        rtg_fair = [(date, rtg['fair']) for date, rtg in rtg_history]
        #plt.step(*zip(*rtg_fair), label=team)

        # plot "handicapped" elo rating history
        rtg_hcap = [(date, rtg['hcap']) for date, rtg in rtg_history]
        plt.step(*zip(*rtg_hcap), label=team)

        # figure properties
        plt.xlabel('Time (year, week)')
        plt.ylabel('ELO rating'.format(margin))
        plt.title('Handicap={} pts'.format(margin))
        #plt.legend()

    def cdf(self, home, away, year, week):
        """
        Cumulative (integrated) probability that,
        score home - score away > x.

        """
        for handicap in range(-40, 41):
            hcap = abs(handicap)
            home_rtg = self.query_elo(self.elodb, home, hcap, year, week)
            away_rtg = self.query_elo(self.elodb, away, hcap, year, week)
            hfa = self.hfa(self.elodb, hcap, year, week)

            if handicap < 0:
                rtg_diff = home_rtg['fair'] - away_rtg['hcap'] + hfa
            else:
                rtg_diff = home_rtg['hcap'] - away_rtg['fair'] + hfa

            yield handicap, self.win_prob(rtg_diff)

    def predict_spread(self, home, away, year, week):
        """
        Predict the spread for a matchup, given current knowledge of each
        team's ELO ratings.

        """
        # cumulative spread distribution
        cdf = list(self.cdf(home, away, year, week))

        # plot median prediction (compare to vegas spread)
        median, _ = min(cdf, key=lambda k: abs(k[1] - 0.5))

        return median

    def plot_spread(self, home, away, year, week, vegas=None):
        # median model prediction
        median = self.predict_spread(home, away, year, week)

        # determine winner
        if median > 0:
            winner = home
        elif median == 0:
            winner = 'TIE'
        else:
            winner = away

        # plot cumulative spread distribution
        label = ''.join([away, '@'+home])
        plt.step(*zip(*cdf), where='post')

        plt.axvline(
                median, linewidth=.5, color=offblack,
                label='model: {} ({})'.format(-abs(median), winner)
                )

        # plot Vegas spread if provided
        vegas_winner = home if vegas < 0 else away
        if vegas: plt.axvline(
                -vegas, linewidth=.5, color=plt.cm.Reds(.6),
                label='vegas: {} ({})'.format(-abs(vegas), vegas_winner)
                )

        # axes properties
        plt.xlim(-20, 20)
        plt.ylim(0, 1)
        plt.xlabel('{} - {} [points]'.format(home, away))
        plt.ylabel('Prob({} - {} > x)'.format(home, away))
        plt.title(label)
        plt.legend(handlelength=1)

        plt.savefig('{}.png'.format(label), dpi=200)

    def win_prob(self, rtg_diff):
        return 1./(10**(-rtg_diff/400.) + 1.)
