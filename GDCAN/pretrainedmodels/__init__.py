from .version import __version__

from . import models

from .models.utils import pretrained_settings
from .models.utils import model_names

# to support pretrainedmodels.__dict__['nasnetalarge']
# but depreciated
from .models.senet import se_resnet50
from .models.senet import se_resnet101
from .models.senet import se_resnet152
