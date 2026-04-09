import torch
import platform

def setup_quantization_engine():
    """
    Selecteert en configureert automatisch de beste kwantisatie-engine 
    gebaseerd op het besturingssysteem en de processor.
    """
    supported = torch.backends.quantized.supported_engines
    
    if platform.system() == "Darwin" and (platform.processor() == 'arm' or platform.machine().startswith('arm')):
        engine = 'qnnpack'
    elif 'x86' in supported:
        engine = 'x86'
    elif 'fbgemm' in supported:
        engine = 'fbgemm'
    else:
        engine = 'qnnpack'

    torch.backends.quantized.engine = engine
    return engine
