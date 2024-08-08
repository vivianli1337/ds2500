import numpy as np


class ConditionalProb:
    """ a conditional probability over discrete prob distributions

    supports a single target variable A and as many conditionals as needed

    Attributes:
        target (str): name of target variable
        condition_list (list): a list of conditional random variables
        cond_prob_dict (dict): keys are tuples of condition outcomes.  values
            are dictionaries of target probability distributions (keys are
            outcomes of target, values are corresponding probs)

    Consider a ConditionalProb object representing P(LawnWet|Rain, Sprinkler)
    (taken from https://miro.medium.com/max/640/1*9OsQV0PqM2juaOtGqoRISw.jpeg)
    target = 'WetGrass'
    condition_list = ['Rain', 'Sprinkler']
    cond_prob_dict = {('rain',    'on'):  {'wet': .99, 'dry': .01},
                      ('no rain', 'on'):  {'wet': .9,  'dry': .1},
                      ('rain',    'off'): {'wet': .9,  'dry': .1},
                      ('no rain', 'off'): {'wet': 0,   'dry': 1}}
    """

    def __init__(self, target, condition_list, cond_prob_dict):
        self.target = str(target)
        self.condition_list = [str(rv) for rv in condition_list]
        self.cond_prob_dict = cond_prob_dict

        # check that each condition has probs which sum to 1
        for condition, prob in cond_prob_dict.items():
            assert np.isclose(1, sum(prob.values())), \
                f'probs dont sum to 1 in condition {condition}'

    def __str__(self):
        return f'ConditionalProb of {self.target} given {self.condition_list}'
