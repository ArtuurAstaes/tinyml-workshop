import torch
import platform

def setup_quantization_engine():
    """
    Automatically selects and configures the best quantization engine 
    based on the operating system and processor.
    """
    supported = torch.backends.quantized.supported_engines
    
    # Check what's actually supported on this machine
    if 'fbgemm' in supported:
        # FBGEMM is the default for x86 CPUs (Intel/AMD) on Windows/Linux
        engine = 'fbgemm'
    elif 'onednn' in supported:
        # OneDNN is the newer engine for Intel CPUs (replaces fbgemm in some builds)
        engine = 'onednn'
    elif 'x86' in supported:
        # Fallback to x86 if available
        engine = 'x86'
    elif 'qnnpack' in supported:
        # QNNPACK is for ARM processors (mobile, Apple Silicon)
        engine = 'qnnpack'
    else:
        # If nothing else works, raise an error
        raise RuntimeError(
            f"No supported quantization engine found. "
            f"Available engines: {supported}"
        )
    
    print(f"Using quantization engine: {engine}")
    print(f"Supported engines on this system: {supported}")
    
    torch.backends.quantized.engine = engine
    return engine
