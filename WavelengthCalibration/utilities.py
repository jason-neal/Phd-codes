"""Utitlities for Phd-codes."""


def append_hdr(hdr, keys=None, values=None, item=0):
    """Apend/change parameters to fits hdr.
    
    Can take list or tuple as input of keywords
    and values to change in the header
    Defaults at changing the header in the 0th item
    unless the number the index is givien,
    If a key is not found it adds it to the header.
    """
    if keys is not None and values is not None:
        if isinstance(keys, str):           # To handle single value
            hdr[keys] = values
        else:
            assert len(keys) == len(values), 'Not the same number of keys as values'
            for i, key in enumerate(keys):
                hdr[key] = values[i]
                # print(repr(hdr[-2:10]))
    return hdr
