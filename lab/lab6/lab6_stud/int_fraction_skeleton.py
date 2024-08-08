class IntFraction:
    """ a fraction whose numerator / denominator are ints (rational fraction)

    Attributes:
        num (int): fraction numerator
        den (int): fraction denominator
    """
    @classmethod
    def from_int(cls, x):
        """ builds IntFraction from integer
        
        Args:
            x (int)
        """

    def __init__(self, num, den):
    	# check fraction is valid (num & den are ints, den != 0)

        # normalize +/- signs (extra credit)

        self.simplify()

    def simplify(self):
        """ removes any common factors from num and den """

    def __add__(self, other):
        """ adds to self with int or IntFraction
        
        Args:
            other (int or IntFraction)
            
        Returns:
            out (IntFraction)
        """

    def __mul__(self, other):
        """ multiplies self with int or IntFraction
        
        Args:
            other (int or IntFraction)
            
        Returns:
            out (IntFraction)
        """
        # cast other to IntFraction from int if need be

        # other is an IntFraction (cross multiply to get new fraction)

    def __eq__(self, other):
        """ returns True if other is equivilent to self
        
        Args:
            other (int or IntFraction)
            
        Returns:
            out (IntFraction)
        """
        # cast to IntFraction from int (if input is int)

    def __sub__(self, other):
        """ subtracts other from self
        
        Args:
            other (int or IntFraction)
            
        Returns:
            out (IntFraction)
        """
        # cast to IntFraction from int if need be

        # make other negative
        
        # call addition with negative other (subtraction)
        
    def __truediv__(self, other):
        """ divides self by other
        
        Args:
            other (int or IntFraction)
            
        Returns:
            out (IntFraction)
        """
        # cast to IntFraction from int if need be
        
        # make other reciprocal
        
        # call multiplication with reciprocal other (division)
