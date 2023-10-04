from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# from .configuration_topt import TopTConfig
# from .modeling_topt import TopTForPreTraining
# from .modeling_topt import TopTForImageClassification

_import_structure = {"configuration_topt": ["TopTConfig"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_topt"] = [
        "TopTForPreTraining",
        "TopTLayer",
        "TopTModel",
        "TopTPreTrainedModel",
        "TopTForImageClassification",
        "TopTForJointClassificationRegression",
    ]


if TYPE_CHECKING:
    from .configuration_topt import TopTConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_topt import (
            TopTForPreTraining,
            TopTLayer,
            TopTModel,
            TopTPreTrainedModel,
            TopTForImageClassification,
            TopTForJointClassificationRegression,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
