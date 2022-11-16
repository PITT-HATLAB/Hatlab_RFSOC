import operator
from functools import reduce
import ruamel.yaml as yaml


def to_yaml_friendly(v):
    """convert possible numpy type to native python types"""
    if type(v) == str:
        vv = v
        return vv
    if type(v) == dict:
        vv = {}
        for k_, v_ in v.items():
            vv_ = to_yaml_friendly(v_)
            vv[k_] = vv_
        return vv
    try:
        if len(v) > 0:
            try:
                vv = v.tolist()
            except AttributeError:
                vv = v
            return vv
    except TypeError:
        vv = float(v)
        return vv



def update_yaml(yamlPath, newParamDict: dict):
    """
    update a yaml config file with updated parameters, and keep the original format.

    :param yamlPath: path to the yaml file to be updated
    :param newParamDict: dictionary that contains the updated parameters. For nested parameters, the key needs be the
        key of each layer jointed with '.'

    :Example:
        >>> old_config = {"config":{"relax_delay" : 100}} #to update relax_delay to 20, we do:
        >>> update_yaml(yamlPath, {"config.relax_delay": 20})

    :return:
    """
    def get_by_path(root, items):
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, items, root)

    def set_by_path(root, items, value):
        """Set a value in a nested object in root by item sequence."""
        data_type =  type(get_by_path(root, items))
        get_by_path(root, items[:-1])[items[-1]] = data_type(value)


    config, ind, bsi = yaml.util.load_yaml_guess_indent(open(yamlPath))
    for s, val in newParamDict.items():
        set_by_path(config, s.split("."), to_yaml_friendly(val))

    new_yaml = yaml.YAML()
    new_yaml.default_flow_style = None
    new_yaml.indent(mapping=ind, sequence=ind, offset=bsi)

    with open(yamlPath, 'w') as fp:
        new_yaml.dump(config, fp)



if __name__ == "__main__":
    yamlPath = r"M:\code\project\SNAIL_Pump_Limitation\RFSOC_phaseReset\config_files\\20221114Q3_10.87mA.yml"
    update_yaml(yamlPath, {"info.geLocation": [0,0,0], "config.test":1000})

    # alternatively:
    # config, ind, bsi = yaml.util.load_yaml_guess_indent(open(yamlPath))
    #
    # new_yaml = yaml.YAML()
    # new_yaml.default_flow_style = None
    # new_yaml.indent(mapping=ind, sequence=ind, offset=bsi)
    #
    # config["info"]["geLocation"] = [1,1]
    # config["config"]["snail_c_pulse"]["test"] = [1, 1]
    #
    # with open(yamlPath, 'w') as fp:
    #     new_yaml.dump(config, fp)
    #


