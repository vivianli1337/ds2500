#!/usr/bin/env python
# coding: utf-8

# # DS 2500 HW 7
# 
# Due: Mon Apr 03 @ 11:59PM
# 
# ### Submission Instructions
# Please submit both of the following to the corresponding [gradescope](https://www.gradescope.com/courses/478298) assignment:
# - this `.ipynb` file
#      -  <span style="color:red">give a fresh Kernel > Restart & Run All just before uploading</span>
#          - there is no autograder for hw7, so this step is extra important here!
# 
# - a `.py` file consistent with your `.ipynb`
#     - `File > Download as ...`
# 
# ### Tips for success
# - Start early
# - Make use of [Piazza](https://course.ccs.neu.edu/ds2500/admin_piazza.html)
# - Make use of [Office Hours](https://course.ccs.neu.edu/ds2500/office_hours.html)
# - Remember that [Documentation / style counts for credit](https://course.ccs.neu.edu/ds2500/python_style.html)
# - [No student may view or share their ungraded homework with another](https://course.ccs.neu.edu/ds2500/syllabus.html#academic-integrity-and-conduct)
# 
# | part                                        |       | ex cred   |   part total |
# |:--------------------------------------------|:------|:----------|-------------:|
# | Part 1: `BayesNetwork.add_prior_node`       | 20.0  |           |           20 |
# | Part 2: `BayesNetwork.add_conditional_node` | 25.0  |           |           25 |
# | Part 3: `BayesNet.get_prob`                 | 20.0  |           |           20 |
# | Part 4: `BayesNet.get_conditional_prob`     | 15.0  |           |           15 |
# | Part 5: Gardening                           | 15.0  |           |           15 |
# | Part 6: Memory Analysis                     | 5.0   |           |            5 |
# | Part 7: Build-your-own                      |       | 4.0       |            4 |
# | total                                       | 100.0 | 4.0       |          104 |

# # Suggestions:
# 
# - only modify the code in the cell immediately below
#     - modifying the tests can alter the intended behavior of the test
# - test your code by giving a fresh restart & run for each run
#     - the tests are built to be run in the given sequence, running a code cell twice 
#     
# # Hints:
# 
# - this `BayesNetwork` class operates just like the "manual spreadsheet" computation shown in class.  Before diving into the syntax and programming challenge of building it be sure you're comfortable with the mathematics and "manual" computation method shown in class first.
# - [hw7_hint](hw7_hint.ipynb) has a few constructions which could be useful   
# 

# In[1]:


from copy import copy

import pandas as pd


class BayesNetwork:
    """ Bayes Net, computes full joint table

    Attributes:
        df_joint (pd.DataFrame): a column per random variable plus another col
            for probability.  each row contains the outcomes of the
            corresponding random variable or the joint prob of entire row
    """

    def __init__(self):
        # note: we specify type of prob as float with 1.0 below
        self.df_joint = pd.DataFrame({'prob': [1.0]})

    def add_prior_node(self, rv_name, prob_dist):
        """ adds a nodes to joint distribution table

        Args:
            rv_name (str): name of random variable (must be unique in df_joint)
            prob_dist (dict): keys are outcomes of random variable, values are
                probability of each
        """
        assert rv_name not in self.df_joint.columns,             f'non-unique node: {rv_name}'
        
        
        # make a list of dictionaries (each will be a new col)
        col_list = []
        
        # iterate over each row
        for idx, row in self.df_joint.iterrows():
            
            # copy the original row
            copy_row = copy(row)
            
            # find the specific row
            old_prob = copy_row['prob']
            
            # for each variable, update the prob
            for key, val in prob_dist.items():
                new_col = {'prob': old_prob * val,
                        rv_name: key}
                # add it to the list
                col_list.append(new_col)
        
        # create the table with the additional new col
        self.df_joint = pd.DataFrame(col_list)
        
    def add_conditional_node(self, cond_dist):
        """ adds a nodes to joint distribution table

        Args:
            cond_dist (ConditionalProb): a conditional probability of some new
                random variable.  (conditioned on random variables already in
                df_joint)
        """
        # check that all conditioned variables are in joint already
        assert set(cond_dist.condition_list).issubset(self.df_joint.columns),             f'condition rvs not in joint table: {cond_dist.condition_list}'
        
        # check that target variable is not in joint already
        assert cond_dist.target not in self.df_joint.columns,             f'random variable already in network: {cond_dist.target}'
    
        # create a new list 
        row_list = []
        
        # iterate through each row
        for _, row in self.df_joint.iterrows():
            
            # find the variables through a for-loop
            for col_name in cond_dist.condition_list:
                # find the key (variables) in con_prob_dict using cond_dist.condition_list
                # matching the element in condition_list and find the associated value & store it in a tuple  
                c_key = [row[col_name] for col_name in cond_dist.condition_list]
                c_key = tuple(c_key)
                
                # find the other variables aka values of the c_key in con_prob_dict
                val_dict = cond_dist.cond_prob_dict[c_key]
            
            # for each variable, update the prob 
            for key, val in val_dict.items():
                # copy and modify the new copied row
                new_row = copy(row)
                # calculate prob
                new_row['prob'] = new_row['prob'] * val
                new_row[cond_dist.target] = key
                # add it to the list
                row_list.append(new_row)
                
        # create the table with the additional new col
        self.df_joint = pd.DataFrame(row_list)
        
        

    def get_prob(self, state):
        """ sums all rows which satisfy state (marginalization)

        Args:
            state (dict): keys are random variable, values are corresponding
                outcomes
                
        Returns:
            prob (float): probability of the given state
        """
        
        # initialize prob
        s_prob = 0
        
        # iterate through each row
        for _, row in self.df_joint.iterrows():
            # check to see if state is in each row
            if set(state.items()).issubset(row.items()):
                    # add prob if state is in each row
                    s_prob += row['prob']

        return s_prob   
        
    def get_conditional_prob(self, state, condition):
        """ computes conditional probability of state given condition:

        P(ABC|XYZ) = P(ABCXYZ) / P(XYZ)

        above ABC are state variables while XYZ are conditional variables

        Args:
            state (dict): keys are random variable, values are corresponding
                outcomes
            condition (dict): keys are random variable, values are
                corresponding outcomes
                
        Returns:
            prob (float): probability of the given state given condition
        """
        # check that no variable is in state & conditional
        rv_double = set(state.keys()).intersection(condition.keys())
        assert not rv_double,             f'same random variable before & after conditional: {rv_double}'
        
        # create a new dict of both state and condition & update
        state_condition = dict(state)
        state_condition.update(condition)
        
        # get numerator
        num = self.get_prob(state_condition)
        # get denominator
        den = self.get_prob(condition)
        
        # calculate the conditional prob
        prob = num/den

        return prob   


# # Part 1: `BayesNetwork.add_prior_node` (20 points)
# 
# We validate whether the nodes have been added properly by constructing a known example: 
# 
# <img src="https://miro.medium.com/max/640/1*9OsQV0PqM2juaOtGqoRISw.jpeg" width=500>
# 
# and comparing output `bayes_net.df_joint` to expected dataframes, which are stored in the [expected_csv](expected_csv) folder.

# In[2]:


# write the prob into the clouds & compute them

# for example, after adding the cloudy node to the network, bayes_net.df_joint should look as below:
df_expected = pd.read_csv('expected_csv/prob_cloudy.csv', index_col=False)
df_expected


# In[3]:


# build bayes net with cloudy node
bayes_net = BayesNetwork()
bayes_net.add_prior_node('Cloudy', prob_dist={'c0': .5, 'c1': .5})

# manually check output dataframe (just this first time, to see how to debug below)
bayes_net.df_joint


# In[4]:


from df_compare import assert_df_equal_no_idx

# automatically compare expected to actual dataframe
# (it ends up being somewhat challenging to do given that we 
# can shuffle order of cols or rows while the two are still
# equivilent, for our purposes ... see df_compare.py for details,
# but it isn't necessary to complete the assignment)
assert_df_equal_no_idx(bayes_net.df_joint, df_expected)


# # Part 2: `BayesNetwork.add_conditional_node` (25 points)
# 
# Hint:
# - Inspect and study the given output DataFrames via their [expected_csv](expected_csv) before implementing!

# In[5]:


from conditional import ConditionalProb

# add rain conditional prob
cond_prob_rain =     ConditionalProb(target='Rain',
                    condition_list=['Cloudy'],
                    cond_prob_dict={('c1',): {'r1': .8, 'r0': .2},
                                    ('c0',): {'r1': .2, 'r0': .8}})
bayes_net.add_conditional_node(cond_prob_rain)

# check that rain conditional prob was added properly
df_joint_expected = pd.read_csv('expected_csv/prob_cloudy_rain.csv', index_col=False)
assert_df_equal_no_idx(df_joint_expected, bayes_net.df_joint)


# In[6]:


df_joint_expected


# In[7]:


# add sprinkler conditional prob
cond_prob_sprinkler =     ConditionalProb(target='Sprinkler',
                    condition_list=['Cloudy'],
           
                    cond_prob_dict={('c1',): {'s1': .1, 's0': .9},
                                    ('c0',): {'s1': .5, 's0': .5}})
bayes_net.add_conditional_node(cond_prob_sprinkler)

# check that sprinkler conditional prob was added properly
df_joint_expected = pd.read_csv('expected_csv/prob_cloudy_rain_sprinkler.csv', index_col=False)
assert_df_equal_no_idx(df_joint_expected, bayes_net.df_joint)


# In[8]:


df_joint_expected


# In[9]:


# add wet grass conditional prob
cond_prob_grass_wet =     ConditionalProb(target='WetGrass',
                    condition_list=['Rain', 'Sprinkler'],
                    cond_prob_dict={('r1', 's1'): {'w1': .99, 'w0': .01},
                                    ('r0', 's1'): {'w1': 0.9, 'w0': .1},
                                    ('r1', 's0'): {'w1': 0.9, 'w0': .1},
                                    ('r0', 's0'): {'w1': 0.0, 'w0': 1}})
bayes_net.add_conditional_node(cond_prob_grass_wet)

# check that wet grass conditional prob was added properly
df_joint_expected = pd.read_csv('expected_csv/prob_cloudy_rain_sprinkler_grass.csv', index_col=False)
assert_df_equal_no_idx(df_joint_expected, bayes_net.df_joint)


# In[10]:


df_joint_expected


# # Part 3: `BayesNet.get_prob` (20 points)

# In[11]:


from math import isclose

assert isclose(bayes_net.get_prob({'Cloudy': 'c1'}), .5)

assert isclose(bayes_net.get_prob({'Sprinkler': 's1', 'Cloudy': 'c1'}), .05)
assert isclose(bayes_net.get_prob({'Sprinkler': 's1', 'Cloudy': 'c0'}), .25)
assert isclose(bayes_net.get_prob({'Sprinkler': 's1'}), .3)

assert isclose(bayes_net.get_prob({'Rain': 'r1', 'Cloudy': 'c1'}), .4)
assert isclose(bayes_net.get_prob({'Rain': 'r1', 'Cloudy': 'c0'}), .1)
assert isclose(bayes_net.get_prob({'Rain': 'r1'}), .5)


# #### extra math note (not needed for HW completion, helpful for probability fluency though)
# 
# The chunks of three assert statements immediately above demonstrate marginalization: 
# 
# - there's only two ways sprinkler is on: 
#     - when its cloudy or clear outside (.3 = .05 + .25)
# - there's only two ways its raining:     
#     - when its cloudy or clear outside (.5 = .1 + .4)

# # Part 4: `BayesNet.get_conditional_prob` (15 points)
# 
# To validate `.get_conditional_prob()` we reproduce known conditional probs from the bayes net definition:

# In[12]:


# whats the prob the sprinkler is on given its cloudy?
assert isclose(bayes_net.get_conditional_prob(state={'Sprinkler': 's1'}, condition={'Cloudy': 'c1'}), .1)


# In[13]:


# whats the prob its not raining given its not cloudy?
assert isclose(bayes_net.get_conditional_prob(state={'Rain': 'r0'}, condition={'Cloudy': 'c0'}), .8)


# In[14]:


# whats the prob lawn is wet given sprinkler is on and its raining?
assert isclose(bayes_net.get_conditional_prob(state={'WetGrass': 'w1'}, condition={'Sprinkler': 's1',
                                                                                    'Rain': 'r1'}), .99)


# # Part 5: Gardening (15 points)
# 
# A gardener wants their newly planted lawn to have (at least) a 70% chance of being wet while using their sprinkler as little as possible, to conserve water.  Each morning they step outside their house and observe only if it is cloudy or not.  With only this evidence, they want to know whether they must turn their sprinkler on.
# 
# - on clear days, should the gardener turn on their sprinkler?
# - on cloudy days, should the gardener turn on their sprinkler?
# - is it possible for the gardener to always ensure at least 70% chance of having a wet lawn?
# 
# Call a few methods of the bayes net above to investigate the questions immediately above.  Write a summary of results in 2-3 sentences which is easily understood by a garener who knows little of probability or Bayes Nets.

# In[15]:


# chances of wetgrass when sky is clear & sprinkler is on
bayes_net.get_conditional_prob(state={'WetGrass': 'w1'}, condition={'Cloudy': 'c0', 'Sprinkler': 's1'})


# In[16]:


# chances of wetgrass when sky is clear & sprinkler is off
bayes_net.get_conditional_prob(state={'WetGrass': 'w1'}, condition={'Cloudy': 'c0', 'Sprinkler': 's0'})


# In[17]:


# chances of wetgrass when sky is cloudy & sprinkler is on
bayes_net.get_conditional_prob(state={'WetGrass': 'w1'}, condition={'Cloudy': 'c1', 'Sprinkler': 's1'})


# In[18]:


# chances of wetgrass when sky is cloudy & sprinkler is off
bayes_net.get_conditional_prob(state={'WetGrass': 'w1'}, condition={'Cloudy': 'c1', 'Sprinkler': 's0'})


# In[19]:


# chances of rain when sky is cloudy 
bayes_net.get_conditional_prob(state={'Rain': 'r1'}, condition={'Cloudy': 'c1'})


# In[20]:


# chances of rain when sky is clear
bayes_net.get_conditional_prob(state={'Rain': 'r1'}, condition={'Cloudy': 'c0'})


# ### <font color=royalblue> Response: </font> 
# It is possible for the gardener to always ensure at least 70% chance of having a wet lawn by not turning on the sprinklers on cloudy days. During cloudy days, there is a 80% that there will be rain. Thus, by not turning on the sprinklers on cloudy days, there is a 72% that the grass will be wet. However, there is only a 20% chance of rain on clear days. Thus, there will be a 92% that the grass will be wet by keeping the sprinklers on during clear days.

# # Part 6: Memory Analysis (5 points)
# 
# Let's consider the liver disease bayes net example shown in class.  Assuming it has 40 total nodes, and each is binary, how much memory would it cost to store the probability column of `df_joint` as shown above?  Assume that every combination of variables must be stored as a float which uses `np.ones(1).nbytes / 1e6` megabytes of space.
# 
# Summarize your computation in 2 sentences so a non-technical reader can understand the drawback.  (Note: this memory problem lies with our implementation, there are methods to avoid it)
# 
# Hint:
# - its a big number, don't try this line of code as you'll run out of memory before you get an answer:
#     - `np.ones(2 ** 40).nbytes / 1e6` megabytes of space

# In[21]:


import numpy as np
(2 ** 40 * np.ones(1).nbytes) / 1e6


# ### <font color=black> Background: </font> 
# For every new nodes (aka the columns) and each is binary, the number of row would double from the previous number. For example, there is 2 nodes (columns) with 4 rows. If 1 node is added, there will be a total of 8 rows.
# 
# ### <font color=royalblue> Response to part 6: </font>
# With 40 nodes, there will be 2^40 (1,099,511,627,776) rows with 40 columns; the table would require to have 8,796,093.02 megabytes of space. Because it is a huge table, it can be very disorganized and difficult to locate certain conditions.

# # Part 7: Build-your-own (4 ex cred pts)
# 
# Build your own Bayes Net problem!
# 
# 1. Provide a graphical representation which contains a graph and all necessary distributions
#     - see the thief, alarm, dog, doorbell, earthquake example in class
#     - include it as an embedded image directly below
# 1. Implement it as a `BayesNet`
# 1. Write a few questions which tell a "data story".  Answer them by querying your network and interpretting results.
#     - again, see the thief example in class for a "data story"
#     
# Grab your project team's data if you'd like :)
# 
# I'd love a few more beautiful examples for use in future coursework.  Make a super-clean figure and a compelling datastory to earn the full four points of credit.  
# 
# If you're willing to share this in future coursework (for any course or instructor) please shoot me a copy via email saying "You, or other instructors, are welcome to use this in any future course".  Also, let us know if you'd like us to cite you or whether you'd like us to give credit to an anonymous DS2500 student.  Your consent to use / share won't impact whether you score extra credit points.
