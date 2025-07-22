""" Helper functions to parse temperature from text """

from __future__ import print_function, unicode_literals

__plusminus = ['+/-', u'\u00b1', "+/\u2212"]

def is_unreadable(s):
    unreadable = [u"approx. \u2154\u00b0 C."]
    return s in unreadable

def special_rules(s):
    if s == u"133-7\u00b0C":
        s = u"133-137\u00b0C"
    return s.replace(' .. ', ' to ').replace('..', '.')

def convert2kelvin(_t, _unit):
    if _unit.lower() in ['k', 'kelvin']:
        if _t < 0:
            raise Exception('Negative kelvin: ' + str(_t))
        return _t
    elif _unit.lower() in ['c', 'celsius', 'centigrade']:
        return _t + 273.15
    elif _unit.lower() in ['f', 'fahrenheit']:
        return (_t+459.67)*5.0/9.0
    else:
        raise Exception('Unknown temperature unit: ' + str(_unit))

def find_next_nonspace(s, index, order='backward'):
    inc = 1
    if order == 'backward':
        inc = -1
    elif order == 'forward':
        inc = 1
    else:
        raise Exception('Unknown order: ' + str(order))

    i = index
    while(i >= 0 and i < len(s)):
        i += inc
        if not s[i].isspace():
            return i
        else:
            # is space
            pass
    return None

def find_first_strings(s, strings):
    idx = -1
    for i in strings:
        idx = s.find(i)
        if idx != -1:
            return idx, i
    return idx, None

def find_first_strings_minindex(s, strings):
    idx = {}
    for i in strings:
        s_i = s.find(i)
        if s_i != -1:
            idx[i] = s_i
    if len(idx) == 0:
        return -1, None
    else:
        idx = list(sorted(idx.items(), key=lambda x:x[1]))
        return idx[0][1], idx[0][0]


def detect_unit(s):
    all_unit_C = ['C.', 'C', '°C','degC', 'deg C', 'celsius','℃']
    all_unit_F = ['F.', 'F']
    all_unit_K = ['K']
    idx, _ = find_first_strings(s, all_unit_C)
    if idx != -1:
        return 'C', True
    idx, _ = find_first_strings(s, all_unit_F)
    if idx != -1:
        return 'F', True
    idx, _ = find_first_strings(s, all_unit_K)
    if idx != -1:
        return 'K', True
    if len(s.strip()) != 0 and s != '.':
        print('detect_unit(): error: ', s.__repr__())
    return 'C', False

def detect_groups_of_numbers(s):
    """number of continuous numberical groups of, excluding unicode
    also return indices of numbers
    """
    in_number_region = False
    ngroups = 0 # number of numberical groups
    group_start_index = []
    group_end_index = []
    i = 0
    while(i < len(s)):
        is_current_char_number = s[i].isdigit()
        if in_number_region and s[i] == '.':
            # . (decimal place)
            is_current_char_number = True

        if is_current_char_number:
            if not in_number_region:
                # entering number region
                in_number_region = True
                ngroups += 1
                group_start_index.append(i)
            else:
                # stay in number region
                pass
        else:
            if in_number_region:
                # leaving number region
                in_number_region = False
                group_end_index.append(i)
            else:
                # stay in non-number region
                pass
        i += 1
    if in_number_region:
        group_end_index.append(i)
    assert len(group_start_index) == ngroups
    assert len(group_end_index) == ngroups
    return ngroups, group_start_index, group_end_index

def replace_RT_20(s):
    w = ['room temperature', 'room Temperature', 'Room temperature', 'room temPerature', 'ROOM TEMPERATURE', 'ambient temperature', 'AMBIENT TEMPERATURE', 'Ambient temperature', 'ambient Temp.', 'ambient temp', 'Room Temp', 'room temp', 'ambient', 'room', 'r.t.', 'r.t', 'rt', 'RT', 'R.T.', 'R.T']
    for i in w:
        s = s.replace(i, ' 20 C')
    return s

def replace_number(s):
    w = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    for word, val in w.items():
        s = s.replace(word+' ', str(val))
    return s

def replace_celsius(s):
    w = ['celsius', 'Celsius', 'centigrade', 'Centigrade']
    for i in w:
        s = s.replace(i, ' C ')
    return s

def convert_betweenand_to(s):
    """convert (between .. and ..) to (.. to ..)"""
    between = 'between'
    idx = s.lower().find(between)
    if idx == -1:
        return s
    return s.replace(s[idx:idx+len(between)], '').replace('and', ' to ')

def remove_phrase(s, w):
    for i in w:
        s = s.replace(i, '')
    return s

def remove_approximate(s):
    w = ["about", "around", "approximately", "approx.", "approx", "ca."]
    return remove_phrase(s, w)

def remove_compare(s):
    w = ["or less", "or lower", "or below", "not more than", "above", "below", "less than", "greater than", "<", ">", u"\u2266", u"\u2264"]
    return remove_phrase(s, w)

def remove_degree(s):
    w = [u"\u00b0", u"\u00ba", "degrees", "degree", "deg.", "deg", u"\u00b2", u"\u2070"]
    s = remove_phrase(s, w)
    # oC
    idx = s.find('oC')
    if idx != -1:
        idx_nonspace = find_next_nonspace(s, idx, order='backward')
        if s[idx_nonspace].isdigit():
            s = s.replace('oC', ' C')
    return s

def parse_one_temperature(s, unit=None):
    s = s.replace(u'\u2212', '-').strip()
    ngroups, group_start_index, group_end_index = detect_groups_of_numbers(s)
    if ngroups != 1:
        raise Exception('parse_one_temperature(): ngroups: ' + str(ngroups))
    if group_start_index[0] != 0:
        if s[group_start_index[0] - 1] == '-':
            group_start_index[0] -= 1
    t = float(s[group_start_index[0]:group_end_index[0]])
    if unit is None:
        unit, has_unit = detect_unit(s[group_end_index[0]:].strip())
    else:
        has_unit = None
    return convert2kelvin(t, unit), has_unit, unit

def parse_temperature_range_average(strings):
    # need to determine unit using the second
    t1, t1_has_unit, _ = parse_one_temperature(strings[0])
    t2, t2_has_unit, t2_unit = parse_one_temperature(strings[1])
    if not t1_has_unit:
        if t2_has_unit:
            # let t1 use t2_unit
            t1, _, _ = parse_one_temperature(strings[0], t2_unit)

    # compare signs
    #if abs(t1 - t2) > 50:
    #    print(strings, ' :', t1, t2)

    return (t1+t2)/2

# tokens = "to", "+/-"
def parse_range_token(s, group_start_index, group_end_index):
    tokens = []
    strings = []

    all_range_middle_tokens = ['to', 'TO', 'up to', '~','～','-', u'\u02dc', u'\u2013', u'\u2014', u'\u2015', u'\u2212', u'\u00b1', u'\u223c']
    all_plusminus_tokens = __plusminus

    m = s[group_end_index[0]:group_start_index[1]]
    # find +/-
    idx, tok = find_first_strings_minindex(m, all_plusminus_tokens)
    if idx != -1:
        tokens = ['+/-']
        strings.append(s[0:group_end_index[0]+idx])
        strings.append(s[group_end_index[0]+idx+len(tok):])
        return tokens, strings

    idx, tok = find_first_strings_minindex(m, all_range_middle_tokens)
    if idx != -1:
        tokens = ['to']
        strings.append(s[0:group_end_index[0]+idx])
        strings.append(s[group_end_index[0]+idx+len(tok):])
        return tokens, strings
    else:
        print('parse_range_token(): unknown token: ' + s.__repr__())
        tokens = ['to']
        strings.append(s[0:group_end_index[0]])
        strings.append(s[group_end_index[0]:])
        return tokens, strings
    #raise Exception('parse_range_token(): unknown token: s=' + s.__repr__())

def parse_plusminus(strings):
    # need to determine unit using the second
    t1, t1_has_unit, _ = parse_one_temperature(strings[0])
    t2, t2_has_unit, t2_unit = parse_one_temperature(strings[1])
    if not t1_has_unit:
        if t2_has_unit:
            # let t1 use t2_unit
            t1, _, _ = parse_one_temperature(strings[0], t2_unit)
    return t1

def parse_temperature(s):
    if is_unreadable(s):
        return None
    s = s.replace(u'\xad', ' ')
    s = s.replace('+-', '+/-')
    s = special_rules(s)
    s = replace_number(s)
    s = remove_approximate(s)
    s = remove_degree(s)
    s = replace_celsius(s)
    s = remove_compare(s)
    s = convert_betweenand_to(s)
    s = replace_RT_20(s)
    s = s.replace(',', '.') # convert comma to decimal place
    #print('s=',s)
    ngroups, group_start_index, group_end_index = detect_groups_of_numbers(s)
    if ngroups == 1:
        return parse_one_temperature(s)[0]
    elif ngroups == 2:
        # parse range token
        tokens, strings = parse_range_token(s, group_start_index, group_end_index)
        if tokens[0] == '+/-':
            return parse_plusminus(strings)
        elif tokens[0] == 'to':
            return parse_temperature_range_average(strings)
        else:
            print(s.__repr__())
            raise Exception('unknown token: ' + tokens)
    else:
        print('parse_temperature error: ngroups=' + str(ngroups)+', s='+s.__repr__())
        return None
