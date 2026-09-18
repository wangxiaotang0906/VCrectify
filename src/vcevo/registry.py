"""Built-ins plus import-path factories for third-party backbone plugins."""
import importlib

from .backbones.reference import MemoryReasoner, RidgeBackbone


def numerical_backbone(config, genes, targets, control_cells=None):
    options = dict(config)
    name = options.pop("name")
    if name == "reference_ridge":
        return RidgeBackbone(genes, **options)
    if name == "txpert":
        from .backbones.txpert import TxPertBackbone
        return TxPertBackbone(genes=genes, targets=targets, control_cells=control_cells, **options)
    if ":" in name:
        module, attribute = name.split(":", 1)
        return getattr(importlib.import_module(module), attribute)(genes=genes, targets=targets,
                                                                  control_cells=control_cells, **options)
    raise ValueError("Unknown numerical backbone: " + name)


def reasoning_backbone(config):
    options = dict(config)
    name = options.pop("name")
    if name == "reference_memory":
        if options:
            raise ValueError("Reference memory reasoner has no configurable parameters")
        return MemoryReasoner()
    if name == "summer":
        from .backbones.summer import SummerReasoner
        return SummerReasoner.from_config(options)
    if ":" in name:
        module, attribute = name.split(":", 1)
        return getattr(importlib.import_module(module), attribute)(**options)
    raise ValueError("Unknown reasoning backbone: " + name)
