#!/usr/bin/env python
# coding: utf-8

# # DS 2500 HW 0
# 
# Due: Fri Feb 3 @ 11:59PM
# 
# ### Submission Instructions
# Please submit both of the following to the corresponding [gradescope](https://www.gradescope.com/courses/478298) assignment:
# - this `.ipynb` file 
#     - give a fresh `Kernel > Restart & Run All` just before uploading
# - a `.py` file consistent with your `.ipynb`
#     - `File > Download as ...`
#     
# Gradescope may run your `.py` submission to determine part of your score for this assignment.  See the [autograder instructions](https://github.com/matthigger/gradescope_auto_py/blob/main/stud_instruct.md) for details.
# 
# 
# ### Tips for success
# - Start early
# - Make use of [Piazza](https://course.ccs.neu.edu/ds2500/admin_piazza.html)
# - Make use of [Office Hours](https://course.ccs.neu.edu/ds2500/office_hours.html)
# - Remember that [Documentation / style counts for credit](https://course.ccs.neu.edu/ds2500/python_style.html)
# - [No student may view or share their ungraded homework with another](https://course.ccs.neu.edu/ds2500/syllabus.html#academic-integrity-and-conduct)
# 
# | question   |   points(+ex credit) |
# |:-----------|---------:|
# | part 1.1   |       17 |
# | part 1.2.1 |       10 |
# | part 1.2.2 |       13 |
# | part 1.2.3 |       15 |
# | part 1.2.4 |       10 |
# | part 1.3   |       35+2 |
# |extra credit| 0+4 |
# | total      |      100 |
# 

# # Markdown
# 
# ## Part 1.1 (17 points): 
# Use [markdown](https://www.markdownguide.org/cheat-sheet/) to create your own brief wikipedia-esque description of any subject of interest. 
# 
# Your mini-wiki page must include:
# - three headers: a title, subtitle and subsubtitle (the #, ##, ### syntax)
# - an embedded image from a web address (use an [image hosting site](https://makeawebsitehub.com/free-photo-hosting/) if you'd like to upload your own)
#     - note gradescope may not render the image, if it shows on jupyter locally we'll award credit
# - a table of size at least 3 rows x 3 columns (needn't be correct, but must make sense)
# - a list
# - a link to another website
# 
# Please be brief in your text.  Aim for roughly 3 sentences total of text.  We won't grade based on content, but keep it appropriate for class.  If you make the grader smile, no extra credit will be awarded beyond the satisfaction of having made somebody's day better :)
# 

# # KOREA #
# Seoul is the capital of South Korea. It is where the pop culture, modern skyscrapers, and high tech transportations meet the historical palaces and street markets. 
# 
# ## Places to visit #
# 1. Hongdae
#     - busking: dances & othe performances
#     - food:
#         - street food: fried chicken, tanghulu, takoyaki, skewers, etc.
#     - diy shops:
#         - make your own jewlery
#         - make your own phone case
#     - animal cafe:
#         - lots of dog, cats, and racoons cafes, where visitors can enjoy their food while interacting with cute animals
#     - shops:
#         - lots of shops selling cheap and good quality clothes and accessories
# 2. N Seoul Tower
#     - get to see the entire view of South Korea
#     - recommend to take the cable car up for full experience
# 3. Palaces
#     - Gyeongbokgung palace
#     - Cheonggyecheon palace
#     - rent a hanbok and visit these palaces & stroll around bukchon hanok village
# 4. myeongdong
#     - lots of street food & shops
#     - good skin care stores
# 5. lotte world
#     - similar to disneyland
#     - lots of rides indoor & outdoor
#     - recommend to go during warmer temperature and buy fast lane pass
#     
# Here is the link for some things do to while in seoul: [click here for recommendations](https://www.planetware.com/south-korea/top-rated-tourist-attractions-in-seoul-kor-1-3.htm)
# 
# ### Ratings & Comments #
# 
# | Places:       | rating:  | comments:                                                                                                                         |
# |---------------|----------|-----------------------------------------------------------------------------------------------------------------------------------|
# | Hongdae       | 10/10    | - so many places to go; one day is definitely not enough - the performances are so entertaining & fun to watch                    |
# | N Seoul Tower | 8/10     | - the view was amazing at night - however, I wasn't able to go in the tower since it was closed                                   |
# | Palaces       | 7/10     | - fun experience wearing hanbok in the palace - feel like I am back in the old period - very cold since I was there during winter |
# | Myeongdong    | 8/10     | - enjoy shopping a lot - wish I bought more things (clothes & skincare) - some skincare salespeople are very persistent           |
# | Lotte World   | 9/10     | - had so much fun - lines are so long and it was very cold (winter) - did not have enough time to go on every ride                |
# 
# 
# #### Here is a picture of lotte world: #
# 
# <img src= "https://i.ibb.co/SrWQ30v/lotte.jpg" width = 500>
# 
# 

# # Part 1.2
# 
# In [blackjack](https://en.wikipedia.org/wiki/Blackjack), players are given two cards from a [standard deck](https://en.wikipedia.org/wiki/Standard_52-card_deck#Composition) of cards and add their values together.  Values are assigned to cards as:
# - number cards use their own value
# - Jack, Queen, King cards all have a value of 10
# - **aces, in this problem will always have a value of 1**
# 
# Players may then choose to take as many cards as they'd like, one at a time, to increase their hand's sum.  The goal is to produce a value as high as possible without exceeding 21.  When a player's hand sum increases beyond 21 they are said to 'bust' and they lose the game.
# 
# The key to playing well is knowing when to stop taking cards.  We'll explore this issue by writing a program to have the computer simulate many hands of blackjack and count how often they bust.
# 

# ## Part 1.2.1 (10 points)
# Build a `draw_value()` function which accepts no inputs and returns the value (an integer from 1 to 10) of some card drawn from a standard deck of cards.  `random.choices()`, and its `weights` parameter may be helpful here.  Assume that you draw cards with replacement, so that its possible to draw the same card many times.  
# 
# (In practice, this isn't far from the truth as real casinos often draw from a pile of cards containing multiple, shuffled decks to mitigate the effectiveness of [card counting](https://en.wikipedia.org/wiki/Card_counting)).
# 

# We won't typically publish the rubrics to students.  However, we make an exception here so you can see how important [documentation, variable naming and style](https://course.ccs.neu.edu/ds2500/python_style.html) is in your grade.
# 
# rubric:
# - 2 pt any function created
# - 2 pt function created which returns a card value
# - 2 pt function created which returns a card value with proper `weights` param
#     - scaling `weights` by any reasonable constant acceptable for full credit
# - 2 pt docstring given in proper format
# - 2 pt variable names are informative and in proper style
# 
# TAs are encouraged to give other minor point penalties for other style issues which crop up to nudge students in the proper direction (otherwise students may repeat these bad habits!)
# 

# In[1]:


def draw_value():
    """ show the values of card drawn
    
    Returns:
        value (int) of card drawn
        
    """ 
    # import random library
    import random
    
    # initialize values for A, J, Q, K
    J = 10
    Q = 10
    K = 10
    A = 1
    
    # make a list of the card deck values 
    # no need for 52 cards because of card replacements
    cards = [A, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K]
    
    # select a random card
    card_value = random.choices(cards, weights = None, k=1)
    
    # convert item from list to int
    for value in card_value:
        # return the card value in integer
        return int(value)


# In[2]:


# test case
draw_value()


# ## Part 1.2.2 (13 points)
# Its challenging to test a function which returns a random value, it gives different values by design!  However, the more samples we get from `draw_value()` the closer the fraction of values equal to 10 should be to our expected number of 10s output: 4/13 of the total cards (see also "Law of Large Numbers").
# 
# - Sample 13000 values from `draw_value()`
# - record how many times you observe each card in a `defaultdict` named `card_count`
# - print `card_count` and write one sentence in a markdown cell which explains why your results validate that `draw_value()` works
# 

# In[3]:


# import defaultdict
from collections import defaultdict

# specify the type of dictionary and set to a variable
card_count = defaultdict(int)

# draw a random card for 13000 times
for i in range(13000):
    value = draw_value()
    
    # counting each type of card value in total 
    # add to dictionary by its respective key
    card_count[value] = card_count[value] + 1

# sort dictionaries by key
sort_card_count = dict(sorted(card_count.items()))

# print the dictionary
print(sort_card_count)


# ### Response: #
# Draw_value() worked because "10" has 4 times more counts than other numbers (from 2 to 9), and the counts for "1" has double the number of counts than other numbers (from 2 to 9).

# ## Part 1.2.3 (15 points)
# 
# One of the big questions facing blackjack players is, "Given that my hand already has a value of X, what's the probability that the next card will bust my hand?".  For every starting hand sum from 10 to 21, we'll estimate this probability by simulating how often a hand goes bust among many randomly drawn hands.
# 
# - initialize `num_bust_per_start` as a defaultdict
#     - keys are the starting hands from 10 to 21 (including both 10 and 21)
#     - values (initially) count how many hands went bust (defaults to 0)
#     - e.g. `num_bust_per_start = {19: 100, 15: 42}` indicates that, among all the simulations:
#         - 100 hands starting at 19 went bust
#         - 42 hands starting at 15 went bust
# - for every `start_hand` from 10 to 21:
#     - draw a single card via `draw_value()`
#     - record the hand as a bust if the `start_hand` + `card_val` exceeds 21
#     - repeat the above two steps so that you simulate 10,000 draws
#         - 10,000 draws starting from 10
#         - 10,000 draws starting from 11 
#         - ... 
# 

# In[4]:


def probability_bust():
    """ show the probabilty of bust for each type of starting value

    Returns:
        num_bust_per_start (dictionary):
            probabilities to their respective keys
            
    """
    # import defaultdict
    from collections import defaultdict
    
    # initialize num_bust_per_start as defaultdict
    num_bust_per_start = defaultdict(int)
    
    # for loop
    for start_hand in range(10,22):
        
        # initialize bust count as 0
        bust_count = 0
        
        # draw cards for 10000 times
        for turn in range(10000):
            card_val = draw_value()
            
            # if value exceeds 21, add times it bust to previous amount
            # add it to the dictionary with their respective keys
            if start_hand + card_val > 21:
                bust_count = bust_count + 1
                num_bust_per_start[start_hand] = bust_count
            
            # if value is less than or equal to 21 
            # bust_count will remain as original (0) and add to dictionary
            elif start_hand + card_val <= 21:
                bust_count = bust_count
                num_bust_per_start[start_hand] = bust_count
    
    # return the dictionary
    return num_bust_per_start            


# In[5]:


# test case
probability_bust()


# ## Part 1.2.4 (10 points)
# 
# Let's synthesize the results of part 1.2.3 immediately above by estimating the probability of going bust choosing to take 1 more card from every starting hand from 10 to 21
# 
# - make a new dictionary `prob_bust_given_start` whose keys are values 10 to 21 and values are probability estimates of going bust which correspond to `num_bust_per_start`
#     - for example, if 7000 of the 10,000 draws starting from 15 went bust, the prob estimate of busting from 15 is .7
#     
# - print `prob_bust_given_start` so it may be observed
#     - (optional: try using ["pretty print"](https://docs.python.org/3/library/pprint.html) to sort the output dictionary.  `from pprint import pprint` and then `pprint(prob_bust_given_tart)` on another line)
# 
# - A player wants to continue to add cards to their hand as long as the chance of going bust is no more than 50%.  What would you analysis suggest they do?  Write a one or two sentence response which is easily understood by everyone.
# 

# In[6]:


# import pprint
from pprint import pprint

# create a new dictionary & initalize the num of draws
prob_bust_given_start = {}
draws = 10000

# set proobability_bust() to a variable
bust_deck = probability_bust()

# find the probability (round to 2) of bust for each item in dictionary
for key, val in bust_deck.items():
    probability = val/draws
    prob_bust_given_start[key] = (round(probability, 2))

# print the dictionary
pprint(prob_bust_given_start)


# ### Response: #
# Although the probability of busting is less than 50%, there is a 50% (at most) chance that the next chosen card will bust. Every time a player draw an additional card, there is an increase of 7%-8% probability that the deck would bust. 

# # Triple-or-nothing
# Triple-or-nothing is a game where a `jar` is initially set up with 1 coin and placed between two opposing players.  In each round:
# - both players bet some fraction, `frac`
#     - neither player knows the other's `frac` before choosing their own
#     - fractions must be between 0 and 1 (including 0 and 1)
# - coins are distributed to the players as:
#     1. the player with the smaller $^1$ `frac` takes that fraction$^2$ of coins from the jar and places them in their `purse`
#     1. the player with the larger$^1$ `frac` takes that fraction$^2$ of coins from the **remaining** jar and places them in their `purse`
#     1. the "bank" (an infinite supply of coins) triples the value of the jar (fractional coins allowed)
#     1. If the jar has no coins in it or the players have already played 10 rounds, the game ends
#     
# Each player seeks to earn as many coins as possible.
# 
# Footnote 1: You may assume the players do not write the same value down.  (see also: extra credit below)
# 
# Footnote 2: In the event the jar is emptied, the game ends
# 

# # Part 1.3 (16+2 pts autograder, 19 pts)
# 
# Complete the `update_round()` function below, which updates a round of triple-or-nothing as described above.  Your function should pass all the given assert statements below.
# 
# For (+2 pts) extra credit, revise your `update_round()` to support the case when players give the same `frac`.  In this event, the total number of coins taken out of the jar if the players went (in any order) is: 
# 
#     frac * jar + frac * (1 - frac) * jar = (2 - frac) * frac * jar
# 
# and it is split evenly between the players.
# 
# #### Hints:
# - Study the docstring and testcases of `update_round()` before building it, make sure you understand why the test cases use the values they do
# - Notice that `update_round()` does not know anything baout round indices, game stopping conditions etc
# 

# In[7]:


def update_round(jar, frac0, frac1):
    """ runs a round of Triple-or-nothing
    
    Args:
        jar (float): number of coins in the jar at round start
        frac0 (float): player0's fraction bet
        frac1 (float): player1's fraction bet
            
    Returns:
        jar (float): number of coins in the jar at end of round
        new_coin0 (float): number of coins player0 has earned
            in this round
        new_coin1 (float): number of coins player1 has earned
            in this round
    """
    # player w/ the smallest frac his fraction of the coin from 
    # the jar first and other player will take his fraction of 
    # the remaining coin
    if frac0 > frac1:
        new_coin1 = (frac1 * jar)
        
        # calculate remaining coin
        jar = jar - new_coin1
        new_coin0 = (frac0 * jar)
        
        # calculate remaining coin
        jar = jar - new_coin0
    
    elif frac0 < frac1:
        new_coin0 = (frac0 * jar)
        
        # calculate remaining coin
        jar = jar - new_coin0
        new_coin1 = (frac1 * jar)
        
        # calculate remaining coin
        jar = jar - new_coin1
    
    # after each turn, jar triples
    jar = jar*3
     
    
    # check to ensure frac0 and frac1 are distinct
    # (discard this line if you begin work on extra credit)
    assert frac0 != frac1, 'invalid input, frac0 == frac1'
    
    # check that both fraction inputs are valid
    assert 0 <= frac0 <= 1, 'invalid input: frac0'
    assert 0 <= frac1 <= 1, 'invalid input: frac1'
    
    # return the result
    return jar, new_coin0, new_coin1
        
    
   


# In[8]:


assert update_round(jar=10, frac0=.1, frac1=0) == (27, 1, 0), '(2 pts)'
assert update_round(jar=10, frac0=.5, frac1=.2) == (12, 4, 2), '(2 pts)'
assert update_round(jar=10, frac0=1, frac1=.5) == (0, 5, 5), '(2 pts)'
assert update_round(jar=10, frac1=.9, frac0=1) == (0, 1, 9), '(2 pts)'

# swap frac0 / frac1 and new_coin0 / new_coin1 from previous 4 test cases
assert update_round(jar=10, frac1=.1, frac0=0) == (27, 0, 1), '(2 pts)'
assert update_round(jar=10, frac1=.5, frac0=.2) == (12, 2, 4), '(2 pts)'
assert update_round(jar=10, frac1=1, frac0=.5) == (0, 5, 5), '(2 pts)'
assert update_round(jar=10, frac1=1, frac0=.9) == (0, 9, 1), '(2 pts)'

# # extra credit test cases
# assert update_round(jar=10, frac0=1, frac1=1) == (0, 5, 5), '(1 pt)'
# assert update_round(jar=20, frac0=.5, frac1=.5) == (15, 7.5, 7.5), , '(1 pt)'


# ## Extra Credit (up to +4 points)
# 
# Program your own triple-or-nothing player.  Your new function must use the same inputs, outputs and name (`triple_nothing_player()`) as the function below for credit to be awarded.
# 
# - You needn't use all inputs provided, they're provided to help you build a player which is "smarter" or more fun
# - You may, if you choose, call a working version of `update_round()` as defined above
# - Do not `import` any other packages in building your `triple_nothing_player()` 
# 
# All student submissions will play one game against every other student submission.  The winner is the function with the largest coin total at the end of all games.
# 
# While technically, this is a competition, you're welcome to submit strategies which behave in funny ways without the intention of winning.  I'm hopeful that the data of all these games will be interesting to look at, if so we'll be seeing it again in DS2500.  To help identify which strategy is yours in the dataset, please output a unique (within the class) `name` to your strategy.  
# 
# All Extra Credit will be awarded based on the creativity, implementation and documentation of the submission. 
# 

# In[9]:


# set below to True if you wish to submit your extra credit
extra_credit_attempted = False


# In[10]:


def triple_nothing_player(jar, round_idx, frac_hist_opp, frac_hist_self):
    """ produces a fraction in a game of triple or nothing
    
    frac_hist objects are lists of previous fraction history in
    the game.  For example, frac_hist = [0, .01, .2] indicates
    a player had fraction 0 in the first round, .01 in the second
    and .2 in the third
    
    Args:
        jar (float): value of the jar in current round
        round_idx (int): current round index (0 for
            first round, 9 for final round)
        frac_hist_opp (list): a history of all fractions input
            by opposing player
        frac_hist_self (list): a history of all fractions input 
            by self
            
    Returns:
        frac (float): fraction for current round
        name (str): a unique name given to your strategy
    """

