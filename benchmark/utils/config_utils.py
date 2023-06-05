import collections
import pprint
from copy import deepcopy
import argparse

import yaml


###################################
# YAML Config Loader and Dumper
###################################
def include_constructor(loader, node):
    """this function build the "!include" constructor for yaml files

    "!include" allows to include a yaml file in another yaml file
    Usage:
    !include path/to/file.yaml 
    """
    # Get the path to the file to include
    filename = loader.construct_scalar(node)
    # Read the file and parse it as YAML
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def add_constructors():
    """Add all the constructors we need for yaml files.
    """
    yaml.add_constructor('!include', include_constructor, Loader=yaml.FullLoader)


def _import_handler(config):
    """this function handles the `_import` key in the config file

    `_import` allows to import and merge yaml files into the current yaml file

    Usage:
    ```current.yaml
        _import:
            - !include path/to/import1.yaml
            - !include path/to/import2.yaml
    ```
    """
    print('[Warning] Currently we do not support recursive `_import`. If the base file you are importing from also has `_import`, it will not be correctly imported. If not, you can safely ignore this warning.')
    imported_configs = config.pop('_import', [])
    new_config = config.copy()
    config = {}
    # Merge all the imported configs
    for imported_config in imported_configs:
        # assert no conflict between the imported configs
        assert set(imported_config.keys()).isdisjoint(set(config.keys())), \
            f'Conflict between imported config fields: {set(imported_config.keys()).intersection(set(config.keys()))}'
        config.update(imported_config)
    # Merge the imported configs with the original config
    config.update(new_config)
    # keep the _import, _new key in the config, might be used for conflict checking, and removed at the end
    config['_import'] = imported_configs
    config['_new'] = new_config
    return config


def _update_handler(config):
    """this function handles the `_update` key in the config file
    TODO: we haven't fully tested this function yet.

    `_update` allows to update specific leaves of the (nested) config file with a new value,
    without changing the rest of the config file

    Usage:
    ```current.yaml

        _import:
            - !include path/to/import.yaml

        _update:
            field: new_value
            nested_field1:
                nested_field2: new_value
            nested_field4:
                - name: name1
                    value: new_value
    ```

    ```import.yaml

        field: old_value
        nested_field1:
            nested_field2: old_value
            nested_field3: old_value
        nested_field4:
            - name: name1
                value: old_value
            - name: name2
                value: old_value
    ```

    Result:
    ```config

        field: new_value
        nested_field1:
            nested_field2: new_value
            nested_field3: old_value
        nested_field4:
            - name: name1
                value: new_value
            - name: name2
                value: old_value
    ```

    """
    new_config = config.get('_new', {})
    config_to_update = config.get('_update', {})
    # check for conflict between the updated config and the new config
    assert set(config_to_update.keys()).isdisjoint(set(new_config.keys())), \
        f'There should be no conflict between the updated config and the new config: {set(config_to_update.keys()).intersection(set(new_config.keys()))}'
    
    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update_nested_dict(d.get(k, {}), v)
            # handle list
            # TODO: list support is not tested yet
            elif isinstance(v, list):
                d[k] = []
                for item in v:
                    if isinstance(item, collections.abc.Mapping):
                        d[k].append(update_nested_dict({}, item))
                    else:
                        d[k].append(item)
            else:
                d[k] = v
        return d
    
    config = update_nested_dict(new_config, updated_config)


def _remove_handler(config):
    """remove all the keys starting with `_` in the config file
    """
    for key in list(config.keys()):
        if key.startswith('_'):
            config.pop(key)
    return config


def load_config(path, namespace=False):
    """Load a yaml file.
    """
    add_constructors()

    with open(path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    config = _import_handler(config)
    # config = _update_handler(config)
    config = _remove_handler(config)

    if namespace:
        config = to_namespace(config)
    
    return config


def save_config(config, path):
    """Save a config to a yaml file.
    """
    if isinstance(config, argparse.Namespace):
        config = to_dict(config)

    with open(path, 'w') as file:
        yaml.dump(config, file)


###################################
#      Config Manipulation
###################################
def merge_args_to_config(args, config):
    """merge args into config
    """
    setattr(config, 'args', args)
    return config


def override_config(string, config):
    """override the config with the string

    Example usgae:
        -o "optimizer.lr=1.0e-4,,model.downstream_structure.components[0].layer=0"
    """
    options = string.split(',,')
    for option in options:
        option = option.strip()
        key, value_str = option.split('=')
        key, value_str = key.strip(), value_str.strip()

        if value_str.lower() in ['null', 'none']:
            value_str = 'None'
        
        try:
            value = eval(value_str)
        except:
            value = value_str

        print(f'[Override] - {key} = {value}')

        # set the value
        if type(value) == str:
            exec(f"config.{key} = '{value}'")
        else:
            exec(f"config.{key} = {value}")


def to_namespace(config):
    """Convert a config dict to a namespace.
    """
    namespace = argparse.Namespace()
    _config = deepcopy(config)
    for key, value in _config.items():
        if isinstance(value, dict):
            value = to_namespace(value)
        # handle list of dict
        # turn into list of namespace
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    value[i] = to_namespace(item)
        setattr(namespace, key, value)
    return namespace


def to_dict(config):
    """Convert a namespace config to a dict.
    """
    config_dict = {}
    _config = deepcopy(config)
    for key, value in vars(_config).items():
        if isinstance(value, argparse.Namespace):
            value = to_dict(value)
        # handle list of namespace
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, argparse.Namespace):
                    value[i] = to_dict(item)
        config_dict[key] = value
    return config_dict


def print_config(config, placeholder='#', repeat=50):
    """Print a config.
    """
    if isinstance(config, argparse.Namespace):
        _config = to_dict(config)

    print(f'{placeholder * repeat} Start of Config {placeholder * repeat}')
    pprint.pprint(_config)
    print(f'{placeholder * repeat} End of Config {placeholder * repeat}\n')


def search_config(config, key, default=None, return_all=False):
    """
    Search a config for a key. Return the first found value & location, or all found values & locations.

    Args:
        config (dict or argparse.Namespace): The config to search.
        key (str): The key to search for.
        default (any, optional): The default value to return if the key is not found. Defaults to None.
        return_all (bool, optional): If True, return all found values & locations; if False, return only the first found value & location. Defaults to False.

    Returns:
        value (any): The value of the key if return_all is False.
        location (list): The location of the key if return_all is False.
        results (list of tuples): A list of tuples with values and their locations if return_all is True.

    Examples:
        >>> config1 = {'a': {'b': 1}}
        >>> search_config(config1, 'b')
        (1, ['a', 'b'])

        >>> config2 = {'a': [{'b': 1}, {'b': 2}]}
        >>> search_config(config2, 'b', return_all=True)
        [(1, ['a', 0, 'b']), (2, ['a', 1, 'b'])]

        >>> config3 = {'a': {'c': 1}}
        >>> search_config(config3, 'b', default="Not found")
        "Not found"
    """
    _config = deepcopy(config)  # avoid modifying the original config
    if isinstance(_config, argparse.Namespace):
        _config = to_dict(_config)

    def _search_config(config, key, location=[], results=None):
        if results is None:
            results = []

        for k, v in config.items():
            if k == key:
                new_location = location + [k]
                if return_all:
                    results.append((v, new_location))
                else:
                    return v, new_location
            elif isinstance(v, dict):
                result = _search_config(v, key, location + [k], results)
                if result and not return_all:
                    return result
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        result = _search_config(item, key, location + [k, i], results)
                        if result and not return_all:
                            return result

        if return_all:
            return results
        return None

    result = _search_config(_config, key)

    if result is None:
        return default
    elif return_all and result == []:
        return default
    else:
        return result


def search_enumerate(config_list, name, key):
    """Search a list of configs for a config with a specific name. Return the value of the given key in the config.

    Args:
        config_list (list of dict or argparse.Namespace): The list of configs to search.
        name (str): The name of the config to search for.
        key (str): The key to search for.

    Returns:
        value (any): The value of the key.

    Examples:
        >>> config_list = [{'name': 'config1', 'a': 1}, {'name': 'config2', 'b': 2}]
        >>> search_enumerate(config_list, 'config2', 'b')
        2
    """
    _config_list = deepcopy(config_list)  # avoid modifying the original config
    for config in _config_list:
        config = to_dict(config)
        if config['name'] == name:
            return config[key]

###################################
#      Deprecated Functions
###################################
def deprecated_search_config(config, key, default=None):
    """Search a config for a key. Return the first found value & location.

    Args:
        config (dict or argparse.Namespace): the config to search
        key (str): the key to search for

    Returns:
        value (any): the value of the key
        location (list): the location of the key

    Example:
        >>> config = {'a': {'b': 1}}
        >>> search_config(config, 'b')
        (1, ['a', 'b'])
        >>> config = {'a': [{'b': 1}, {'b': 2}]}
        >>> search_config(config, 'b')
        (1, ['a', 0, 'b'])
    """
    _config = deepcopy(config) # avoid modifying the original config
    if isinstance(_config, argparse.Namespace):
        _config = to_dict(_config)

    def _search_config(config, key, location=[]):
        for k, v in config.items():
            if k == key:
                location += [k]
                return v, location
            elif isinstance(v, dict):
                result = _search_config(v, key, location + [k])
                if result is not None:
                    return result
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        result = _search_config(item, key, location + [k, i])
                        if result is not None:
                            return result
        return None
    
    result = _search_config(_config, key)
    if result is None:
        return default
    else:
        return result
    
