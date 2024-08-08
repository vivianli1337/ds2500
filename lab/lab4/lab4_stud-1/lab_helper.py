import numpy as np
import pandas as pd
import pathlib
import string
from scipy.io import wavfile


def get_df_wav(folder):
    """ gets a dataframe of all wave files

    Args
        folder (str or pathlib.Path): folder containing wav files

    Returns:
        df_wav (pd.DataFrame): voice, word, idx, file
            voice is user chosen pseudonym.  word is either "python" or "name"
            depending on if audio is of the word "python" or an arbitrary name.
            expected that all instances of a particular voice speaking a name
            use the same name
    """

    row_list = list()
    for file in pathlib.Path(folder).glob('*.wav'):
        # get word and idx from file name
        word = file.stem.rstrip(string.digits)
        idx = int(file.stem.replace(word, ''))

        # add a row to row_list
        row = {'word': word, 'idx': idx, 'file': file}
        row_list.append(pd.Series(row))

    df_wav = pd.concat(row_list, axis=1).transpose()
    df_wav.sort_values(['word', 'idx'], inplace=True)

    return df_wav


def load_wav(file, mono=True, rate=None, time_total=None, time_trim_start=None,
             zero_mean=True):
    """ loads a wav file, returns array

    Args:
        file (str): file to be loaded
        mono (bool): if True, will return a mono signal (a single vector).
        When False, will not process signal into mono (will return stereo, if
        that is what wav is stored as)
        rate (int): a sampling rate.  we ensure that the wav file has this
            given rate.  By default, wav files are not checked for any
            particular rate
        time (float): a time, in seconds.  signal is either truncated
            or extended to ensure its total runtime is this length (rounded
            down to nearest int)

    Returns:
        x (np.array): audio signal
    """

    # load signal
    rate_in, x = wavfile.read(str(file))
    
    # normalize scale
    x = x/x.std()
    
    # validate (or set) rate
    if rate is not None:
        assert rate_in == rate, f'invalid rate ({rate_in} Hz) in {file}'
    else:
        rate = rate_in

    if mono is not None and mono:
        if x.ndim == 2:
            # input is stereo but user requested mono signal
            x = x.mean(axis=1)

    if time_trim_start is not None:
        # discard start of trial
        n = int(rate * time_trim_start)
        x = x[n:, ...]
        
    # ensure all start time is begin at the same time
    n = np.where( x[:] > x[:].std())[0][0]
    x = x[n:,...]

    # truncate / extend as needed
    if time_total is not None:
        # compute number of samples needed in final sound
        n = int(rate * time_total)

        if n > x.shape[0]:
            # extend noise with silence at end
            diff = n - x.shape[0]
            if x.ndim == 2:
                # stereo
                silence = np.zeros((diff, 2))
            else:
                # mono
                silence = np.zeros(diff)
            x = np.concatenate((x, silence), axis=0)

        elif n < x.shape[0]:
            # truncate noise
            x = x[:n, ...]

    if zero_mean:
        # ensure signal is centered around zero
        x -= x.mean(axis=0)

    return x


def get_xy_array(df_wav, **kwargs):
    """ builds arrays of x, y for a particular "word" (python or name)

    Args:
        df_wav (pd.DataFrame): see get_df_wav() abovce
    Returns:
        x (np.array): each row is a sample (one sound)
        y (np.array): the voice associated with the noise
        file_list (list): a list of files, useful for validating noises
    """
    # init empty x, y, file_list
    x_list = list()
    y = list()
    file_list = list()

    for _, row in df_wav.iterrows():
        # add noise to x, y, file_list
        x_list.append(load_wav(row['file'], mono=True, **kwargs))
        y.append(row['word'])
        file_list.append(row['file'])

    # combine 1d vectors of x into a 2d array of shape (n_samples, n_features)
    x = np.vstack(x_list)

    return x, np.array(y), file_list
