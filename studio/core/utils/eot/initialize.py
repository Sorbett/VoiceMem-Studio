"""Initialize semantic turn detection from verified local weights."""
from .component import EndOfTurn
from studio.paths import MODELS

def create():
    from studio.core.utils.component_config.component import selection, custom
    value = selection('eot')
    if value['provider'] == 'adapter':
        return custom('eot', value)
    return EndOfTurn(str(MODELS / "eot/smart-turn-v3.2-cpu.onnx"), threshold=0.5)
