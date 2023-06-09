import pickle


def save_pickle(name: str, object_to_pickle: object) -> None:

    '''
    Function to save an object as pickle file.
   
    Parameters
    ----------
    name: str 
        str of name of file. Include full path
    object_to_pickle: object 
        object to save as pickle file
    
    Returns
    -------
    None
    '''

    with open(f'{name}.pickle', 'wb') as handle:
        pickle.dump(object_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name_of_pickle_object: str) -> object:

    '''
    Function to load pickle object.

    Parameters
    ----------
    name_of_pickle_object: str 
        name of object to be loaded. 
        Doesn't need extension however needs full path

    Returns
    -------
    unpickled obect
    '''


    try: 
        with open(f'{name_of_pickle_object}.pickle', 'rb') as handle:
            return pickle.load(handle)
    except Exception:
        print('Unable to load pickle object')