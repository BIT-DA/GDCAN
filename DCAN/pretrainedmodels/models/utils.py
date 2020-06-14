from __future__ import print_function, division, absolute_import

from .senet import pretrained_settings as senet_settings

all_settings = [
    senet_settings,
]

model_names = []
pretrained_settings = {}
for settings in all_settings:
    for model_name, model_settings in settings.items():
        pretrained_settings[model_name] = model_settings
        model_names.append(model_name)
