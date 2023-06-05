from benchmark.utils.config_utils import *

# test with simple dictionary
config = {'a': {'b': 1}}
assert search_config(config, 'b') == (1, ['a', 'b'])

# test with list of dictionaries
config = {'a': [{'b': 1}, {'b': 2}]}
assert search_config(config, 'b') == (1, ['a', 0, 'b'])

# test with default value
config = {'a': {'c': 3}}
assert search_config(config, 'b', default=4) == 4

# test with empty dictionary
config = {}
assert search_config(config, 'b') is None

# test with argparse.Namespace object
from argparse import Namespace
config = Namespace(a=Namespace(b=1))
assert search_config(config, 'b') == (1, ['a', 'b'])

# test with nested Namespace object
config = Namespace(a=Namespace(b=Namespace(c=2)))
assert search_config(config, 'c') == (2, ['a', 'b', 'c'])


# Test case 1: Simple nested dictionary
config1 = {'a': {'b': 1}}
assert search_config(config1, 'b') == (1, ['a', 'b'])
assert search_config(config1, 'b', return_all=True) == [(1, ['a', 'b'])]

# Test case 2: Dictionary with list of dictionaries
config2 = {'a': [{'b': 1}, {'b': 2}]}
assert search_config(config2, 'b') == (1, ['a', 0, 'b'])
assert search_config(config2, 'b', return_all=True) == [(1, ['a', 0, 'b']), (2, ['a', 1, 'b'])]

# Test case 3: argparse.Namespace input
namespace_config = argparse.Namespace(a={'b': 1})
assert search_config(namespace_config, 'b') == (1, ['a', 'b'])
assert search_config(namespace_config, 'b', return_all=True) == [(1, ['a', 'b'])]

# Test case 4: Config with different data types
config3 = {'a': [{'b': 1}, {'b': "test"}, {'b': [1, 2, 3]}, {'b': {'c': 4}}]}
assert search_config(config3, 'b') == (1, ['a', 0, 'b'])
assert search_config(config3, 'b', return_all=True) == [(1, ['a', 0, 'b']), ("test", ['a', 1, 'b']), ([1, 2, 3], ['a', 2, 'b']), ({'c': 4}, ['a', 3, 'b'])]

# Test case 5: Config with no matching key
config4 = {'a': {'c': 1}}
assert search_config(config4, 'b') is None
assert search_config(config4, 'b', return_all=True) == None

# Test case 6: Default value when key not found
config5 = {'a': {'c': 1}}
assert search_config(config5, 'b', default="Not found") == "Not found"
assert search_config(config5, 'b', default="Not found", return_all=True) == "Not found"

# Test case 7: Multiple levels of nesting
config6 = {'a': {'b': {'c': {'d': 1}}}}
assert search_config(config6, 'd') == (1, ['a', 'b', 'c', 'd'])
assert search_config(config6, 'd', return_all=True) == [(1, ['a', 'b', 'c', 'd'])]

# Test case 8 (continued): Multiple occurrences of key at different levels
config7 = {'a': {'b': 1, 'c': {'b': 2}}}
assert search_config(config7, 'b') == (1, ['a', 'b'])
assert search_config(config7, 'b', return_all=True) == [(1, ['a', 'b']), (2, ['a', 'c', 'b'])]

# Test case 9: Key inside a list of dictionaries and other data types
config8 = {'a': [{'b': 1}, 2, {'b': 3}, 'test']}
assert search_config(config8, 'b') == (1, ['a', 0, 'b'])
assert search_config(config8, 'b', return_all=True) == [(1, ['a', 0, 'b']), (3, ['a', 2, 'b'])]

# Test case 10: Key in a dictionary inside a list inside a dictionary
config9 = {'a': [{'c': {'b': 1}}, {'c': {'b': 2}}]}
assert search_config(config9, 'b') == (1, ['a', 0, 'c', 'b'])
assert search_config(config9, 'b', return_all=True) == [(1, ['a', 0, 'c', 'b']), (2, ['a', 1, 'c', 'b'])]

# Test case 11: Complex config with multiple levels and different data types
config10 = {
    'a': {
        'b': [
            {'c': {'d': 1}},
            {'d': 2}
        ],
        'e': [
            {'d': 3}
        ],
        'f': {
            'g': {
                'h': [
                    {'d': 4}
                ]
            }
        }
    }
}
assert search_config(config10, 'd') == (1, ['a', 'b', 0, 'c', 'd'])
assert search_config(config10, 'd', return_all=True) == [(1, ['a', 'b', 0, 'c', 'd']), (2, ['a', 'b', 1, 'd']), (3, ['a', 'e', 0, 'd']), (4, ['a', 'f', 'g', 'h', 0, 'd'])]

# Test case 12: Large config
config11 = {f'level1_{i}': {f'level2_{j}': {f'level3_{k}': k for k in range(10)} for j in range(10)} for i in range(10)}
assert search_config(config11, 'level3_5') == (5, ['level1_0', 'level2_0', 'level3_5'])
assert len(search_config(config11, 'level3_5', return_all=True)) == 100

