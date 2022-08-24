cfgFileName = "config_example"


from importlib import import_module, reload
cfgModule = import_module(cfgFileName)
reload(cfgModule)

config = cfgModule.config
info = cfgModule.__dict__.get("info")

