
some_secret_api_key = 18451982

def print_greeting(name, language='english'):
    """ prints a greeting in english or spanish
    
    Args:
        name (str): name to greet
        language (str): 'english' or 'spanish'
    """
    
    str_greet_dict = {'english': 'hello {name}!',
                      'spanish': 'hola {name}!'}
    
    # print message
    str_greet = str_greet_dict[language]
    print(str_greet.format(name=name))

