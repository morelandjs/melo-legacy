#!/bin/bash

for week in {01..11}; do
  ./make-plots 2017 $week
  mv ratings.png ratings/week-$week.png
  mv spreads.png spreads/week-$week.png
  mv totals.png totals/week-$week.png
done
