#!/usr/bin/env python
# coding: utf-8

# # DS 2500 HW 2
# 
# Due: Fri Feb 17 @ 11:59PM
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
# | question                        |   points |
# |:--------------------------------|---------:|
# | part 1.1: circle implementation |       20 |
# | part 1.2: circle testing        |       15 |
# | part 2: monopolypropertyhand    |       31 |
# | total                           |       66 |
# 

# ### This hw has 66 (not 100) points
# Its about 2/3 the length of a typical HW and will be weighted as 66/100 as much as other HWs.  
# 
# (I hope this evens out the workload through the semester a bit, I shortened slightly by focusing on essential skills)
# 

# # Part 1: Circle
# 
# ### Part 1.1: Circle Implementation (20 points)
# Build a class which describes the radius and position of a circle
# - Attributes:
#     - radius (float): radius of the circle
#     - pos_x (float): position of center of circle (horizontal)
#     - pos_y (float): position of center of circle (vertical)
# - Methods:
#     - `Circle.__init__()`
#         - accepts & stores all attributes
#     - `Circle.__repr__()`
#         - example output: `Circle(radius=1, pos_x=1, pos_y=2)`
#     - `Circle.scale_radius()`
#         -  multiplies the radius of a circe by input `scale`
#             - for example: if `circ0.radius==1` and we call `circ0.scale_radius(10)` then `circ0.radius == 10`
#         - doesn't return anything
#     - `Circle.move()` 
#         - changes position (pos_x, pos_y) of circles 
#         - has inputs `offset_x` and `offset_y`
#             - default values are 0 for each
#         - adds offsets to corresponding position
#         - doesn't return anything
#         
# #### note:
# - [style](https://course.ccs.neu.edu/ds2500/python_style.html#class-docstrings) counts!
# - part 1.2 will ask you test this code, might be worth completing this first
# - of the 20 points in this problem, 11 are awarded by a "hidden" autograder whose results will be hidden from students until after the submission deadlines.
#     - this is to prevent us from "giving away" part 1.2's solution
# 

# In[1]:


class Circle:
    """ a plot of a circle (shape) whose radius and position can be adjusted
    
    Attributes:
        radius (float): radius of the circle
        pos_x (float): position of the center of circle (horizontal)
        pos_y (float): position of the center of circle (vertical)
    """
    def __init__(self, radius, pos_x, pos_y):
        self.radius = radius
        self.pos_x = pos_x
        self.pos_y = pos_y
    
    def __repr__(self):
        return f'Circle(radius={self.radius}, pos_x={self.pos_x}, pos_y={self.pos_y})'
    
    def scale_radius(self, scale):
        """ scale the radius by a given constant
        
        Args:
            scale (float): the constand
        """
        
        self.radius = self.radius * scale
    
    def move(self, offset_x = 0, offset_y = 0):
        """ changes the position of x and y of the circle
        
        Args:
            offset_x (float) = 0: default
            offset_y (float) = 0: default
        """
        
        self.pos_x = self.pos_x + offset_x
        self.pos_y = self.pos_y + offset_y
        


# ## Part 1.2: Circle Testing (15 points)
# 
# Write a set of test cases which validate that your code above works as intended:
# 1. build one (or more, if you'd like) circle objects
# 1. call the methods above
# 1. validate that the attributes of the circle objects are as expected between method calls
# 
# #### Note:
# - see our test cases in the monpoly problem below for an example of object testing.
# - "How many test cases do I need to build?"
#     - Build as few as possible such that the tests ensure the class works as expected.  (I ended up with 6 or 7 in my solution).  Ensure that you've run all the lines of code above and tested all inputs / defaults
# - "How do I check that all attributes of an object (its state) match some expected values?"
#     - Every object has an attribute `__dict__` which is a dictionary containing all object attributes (keys are strings, the name of the attribute, while values are the values of each).  I'd suggest comparing this dictionary to some expected state of the object:
# 
# ```python
# assert circ0.__dict__ == {'radius': 1, 'pos_x': 1, 'pos_y': 2}
# ```
# 

# In[2]:


# test all of the methods
# tests: __init__() and __repr__()
c = Circle(radius = 2, pos_x = 3, pos_y = 3)
assert str(c) == 'Circle(radius=2, pos_x=3, pos_y=3)'
assert c.__dict__ == {'radius': 2, 'pos_x': 3, 'pos_y': 3}

# test: scale_radius()
c.scale_radius(5)
assert str(c) == 'Circle(radius=10, pos_x=3, pos_y=3)'
assert c.__dict__ == {'radius': 10, 'pos_x': 3, 'pos_y': 3}

# test: move()
c.move(offset_x=2, offset_y=-1)
assert str(c) == 'Circle(radius=10, pos_x=5, pos_y=2)'
assert c.__dict__ == {'radius': 10, 'pos_x': 5, 'pos_y': 2}


# In[3]:


# test only input & move with positive int
# tests: __init__() and __repr__()
ci = Circle(radius = 4, pos_x = 6, pos_y = 7)
assert str(ci) == 'Circle(radius=4, pos_x=6, pos_y=7)'
assert ci.__dict__ == {'radius': 4, 'pos_x': 6, 'pos_y': 7}

# test: move()
ci.move(offset_x=3, offset_y=2)
assert str(ci) == 'Circle(radius=4, pos_x=9, pos_y=9)'
assert ci.__dict__ == {'radius': 4, 'pos_x': 9, 'pos_y': 9}


# In[4]:


# test only input & move with negative int
# tests: __init__() and __repr__()
cir = Circle(radius = 3, pos_x = 4, pos_y = -2)
assert str(cir) == 'Circle(radius=3, pos_x=4, pos_y=-2)'
assert cir.__dict__ == {'radius': 3, 'pos_x': 4, 'pos_y': -2}

# test: move()
cir.move(offset_x=-5, offset_y=2)
assert str(cir) == 'Circle(radius=3, pos_x=-1, pos_y=0)'
assert cir.__dict__ == {'radius': 3, 'pos_x': -1, 'pos_y': 0}


# In[5]:


# test only input & scale
# tests: __init__() and __repr__()
circ = Circle(radius = 3, pos_x = 4, pos_y = 3)
assert str(circ) == 'Circle(radius=3, pos_x=4, pos_y=3)'
assert circ.__dict__ == {'radius': 3, 'pos_x': 4, 'pos_y': 3}

# test: scale_radius()
circ.scale_radius(scale = 2)
assert str(circ) == 'Circle(radius=6, pos_x=4, pos_y=3)'
assert circ.__dict__ == {'radius': 6, 'pos_x':4, 'pos_y': 3}


# In[6]:


# test move with default offsets
# tests: __init__() and __repr__()
circl = Circle(radius = 2, pos_x = 0, pos_y = 1)
assert str(circl) == 'Circle(radius=2, pos_x=0, pos_y=1)'
assert circl.__dict__ == {'radius': 2, 'pos_x': 0, 'pos_y': 1}

# test: move()
circl.move()
assert str(circl) == 'Circle(radius=2, pos_x=0, pos_y=1)'
assert circl.__dict__ == {'radius': 2, 'pos_x': 0, 'pos_y': 1}

# test: move()
circl.move(offset_x = 1, offset_y = 3)
assert str(circl) == 'Circle(radius=2, pos_x=1, pos_y=4)'
assert circl.__dict__ == {'radius': 2, 'pos_x': 1, 'pos_y': 4}


# # Part 2: MonopolyPropertyHand (18 auto + 13 points)
# 
# <img src="https://m.media-amazon.com/images/I/81oC5pYhh2L.jpg" width=200>
# 
# Complete the `MonopolyPropertyHand` class below.  Using this class, one can: 
# - add and remove properties from a players hand (`add_prop()`, `rm_prop()`) 
#     - please do not modify these methods
# - `trade()` properties with another player
# - keep track of who has a "monopoly" as well as how many properties per group a player has via `update_group_mono()`
#     - a "monopoly" is the event a player has all properties in a group
#         - unlike real monopoly, one can have a monopoly on `'Stations'` or `'Utilities'` here
# 
# #### Note:
# - Read through the test cases below the code first to study the expected behavior of `MonopolyPropertyHand`
#     - notice: `MonopolyPropertyHand.prop_set` is a set, using a list can cause trouble
# - Hint: Calling some methods from within others may prevent you from duplicating code.  When one calls `add_prop()`, we'll need to update `group_count` and `mono_set`, right?  ... if only we had a method we could call to get that done for us ...
# - Hint: it may be easier to re-build `group_count` and `mono_set` with every call to `update_group_mono()` rather than update the existing values.
# 
# [Monopoloy Property Source](https://monopoly.fandom.com/wiki/List_of_Monopoly_Properties)
# 

# In[7]:


from monopoly_data import group_prop_dict

# keys are property groups, values are a tuple of strings containing all properties
group_prop_dict['Orange']


# In[8]:


class MonopolyPropertyHand:
    """ a collection of all properties in one player's hand

    Attributes:
        prop_set (set): a set of properties owned by player (each property is
            a string)
        group_count (dict): keys are property groups (e.g. 'Orange').  values
            are a count of how many properties this player owns in that group
            (e.g. group_count['Orange'] = 2 implies player has 2 orange props)
        mono_set (set): a set of all property groups where player owns all
            properties.  For example if mono_set includes 'Dark Purple' then
            prop_set includes both 'Mediterranean Avenue' and 'Baltic Avenue'.
    """

    def __init__(self, prop_set=None):     
        
        # initialize empty set for prop_set
        if prop_set == None:
            self.prop_set = set()
        else:
            self.prop_set = prop_set
        
        # initialize dictionary and empty set
        self.group_count = {'Dark Purple': 0, 'Light Blue': 0,'Pink': 0,
                            'Orange': 0, 'Red': 0,'Yellow': 0,'Green': 0,
                            'Dark Blue': 0,'Stations': 0, 'Utilities': 0}
        
        self.mono_set = set()
        
        
    def add_prop(self, prop):
        """ adds a property to players hand

        Args:
            prop (str): a monopoly property
        """
        self.prop_set.add(prop)
        self.update_group_mono()

    def rm_prop(self, prop):
        """ removes a property from players hand

        Args:
            prop (str): a monopoly property
        """
        self.prop_set.remove(prop)
        self.update_group_mono()

    def update_group_mono(self):       
        """ update the group_count, update the mono_set
        
        """ 
        # set group_count and mono_set back to default (0, empty set) 
        # to prevent any duplications from add_prop or rm_prop
        self.group_count = {'Dark Purple': 0, 'Light Blue': 0,'Pink': 0,
                            'Orange': 0, 'Red': 0,'Yellow': 0,'Green': 0,
                            'Dark Blue': 0,'Stations': 0, 'Utilities': 0}
        self.mono_set = set()
        
        # count the number of properties for each color in the prop_set 
        # and add to group count dictionary
        for key, value in group_prop_dict.items():
            for place in self.prop_set:          
                if place in value:
                    self.group_count[key] += 1
                
                # add the color set to mono_set if player has 
                # all prop of that color prop
                if self.group_count[key] == len(value):
                    self.mono_set.add(key)
                    

        
    def trade(self, other_player, give_prop_set, take_prop_set):
        """ trade a prop with a player
        Args:
            give_prop_set (set): set of properties that are given away
            take_prop_set (set): set of properties that are taking in
        
        """
        # remove the trading prop from one's set and add it to another's set 
        for give_prop in give_prop_set:
            self.rm_prop(give_prop)
            other_player.add_prop(give_prop)  
        
        for take_prop in take_prop_set:
            self.add_prop(take_prop)
            other_player.rm_prop(take_prop)
        
        # update the group count & mono_set 
        self.update_group_mono()
        other_player.update_group_mono()
  


# ### Test Cases
# 
# Notice: `MonopolyPropertyHand.__dict__` gives a dictionary of all attributes of an object.  The keys are the attribute names and values are the attribute values.
# 

# In[9]:


# test0: empty hand of properties
race_car = MonopolyPropertyHand()
assert race_car.__dict__ == {'prop_set': set(),
                             'group_count': {'Dark Purple': 0,
                                             'Light Blue': 0,
                                             'Pink': 0,
                                             'Orange': 0,
                                             'Red': 0,
                                             'Yellow': 0,
                                             'Green': 0,
                                             'Dark Blue': 0,
                                             'Stations': 0,
                                             'Utilities': 0},
                             'mono_set': set()}, '(3 pts)'

# test1: add properties (but no monopolies)
race_car.add_prop('Electric Company')
race_car.add_prop('Mediterranean Avenue')
assert race_car.__dict__ == {'prop_set': {'Electric Company',
                                          'Mediterranean Avenue'},
                             'group_count': {'Dark Purple': 1,
                                             'Light Blue': 0,
                                             'Pink': 0,
                                             'Orange': 0,
                                             'Red': 0,
                                             'Yellow': 0,
                                             'Green': 0,
                                             'Dark Blue': 0,
                                             'Stations': 0,
                                             'Utilities': 1},
                             'mono_set': set()}, '(4 pts)'

# test2: add a few properties, including 2 monopolies (Dark Purple &
# `Utilities`)
race_car.add_prop('Baltic Avenue')
race_car.add_prop('Water Works')

assert race_car.__dict__ == {'prop_set': {'Baltic Avenue',
                                          'Electric Company',
                                          'Mediterranean Avenue',
                                          'Water Works'},
                             'group_count': {'Dark Purple': 2,
                                             'Light Blue': 0,
                                             'Pink': 0,
                                             'Orange': 0,
                                             'Red': 0,
                                             'Yellow': 0,
                                             'Green': 0,
                                             'Dark Blue': 0,
                                             'Stations': 0,
                                             'Utilities': 2},
                             'mono_set': {'Dark Purple',
                                          'Utilities'}}, '(5 pts)'

# test3: build and trade with another player
battleship = MonopolyPropertyHand()
battleship.add_prop('Park Place')
race_car.add_prop('Boardwalk')
race_car.trade(battleship,
               give_prop_set={'Baltic Avenue'},
               take_prop_set={'Park Place'})

assert race_car.__dict__ == {'prop_set': {'Boardwalk',
                                          'Electric Company',
                                          'Mediterranean Avenue',
                                          'Park Place',
                                          'Water Works'},
                             'group_count': {'Dark Purple': 1,
                                             'Light Blue': 0,
                                             'Pink': 0,
                                             'Orange': 0,
                                             'Red': 0,
                                             'Yellow': 0,
                                             'Green': 0,
                                             'Dark Blue': 2,
                                             'Stations': 0,
                                             'Utilities': 2},
                             'mono_set': {'Dark Blue',
                                          'Utilities'}}, '(3 pts)'

assert battleship.__dict__ == {'prop_set': {'Baltic Avenue'},
                               'group_count': {'Dark Purple': 1,
                                               'Light Blue': 0,
                                               'Pink': 0,
                                               'Orange': 0,
                                               'Red': 0,
                                               'Yellow': 0,
                                               'Green': 0,
                                               'Dark Blue': 0,
                                               'Stations': 0,
                                               'Utilities': 0},
                               'mono_set': set()}, '(3 pts)'


# In[10]:


# test
race_car.__dict__


# In[11]:


# test
battleship.__dict__

