def sort_int_nicely(l):
    """ Sort the given list in the way that humans expect and returns it
    as list of str usable together with reindex of pandas dataframes

    This function sorts integers not like 1, 10, 11, 12, 2, 20, 21, 22
    but 1, 2, 10, 11, 12, 20, 21, 22
    """
    l = list(l)
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

    l.sort(key=alphanum_key )

    l = [str(i) for i in l]
    return l