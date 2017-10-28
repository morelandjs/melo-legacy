#!/usr/bin/env python2

import numpy as np

from melo import Date, Rating

class TestElo(object):

    def test_elo_conserved(self):
        """
        Ensure that the total Elo rating in the system is conserved.
        Tolerance must be better than 1% across seasons.

        """
        date = Date(2009, 1)
        rtg = Rating(database='elo.db') 
        hcap = np.random.choice(rtg.range)
        tolerance = 1e-2

        while date < Date(2016, 17):
            elo = sum([
                rtg.elo(team, hcap, date.year, date.week)
                for team in rtg.teams
                ])

            try:
                assert abs((elo - elo_last)/elo_last) < tolerance
            except NameError:
                pass

            elo_last = elo
            date = date.next
