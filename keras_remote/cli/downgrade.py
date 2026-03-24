"""Hardware fallback and downgrade logic for keras-remote."""

from keras_remote.core.accelerators import parse_accelerator, GpuConfig, TpuConfig

DOWNGRADE_PATHS = {
  # TPUs
  "v5p": ["v5litepod", "v4", "v3"],
  "v6e": ["v5p", "v5litepod", "v4", "v3"],
  "v5litepod": ["v4", "v3"],
  "v4": ["v3"],
  
  # GPUs
  "h100": ["a100-80gb", "a100", "l4", "v100", "t4", "p100", "p4"],
  "a100-80gb": ["a100", "l4", "v100", "t4", "p100", "p4"],
  "a100": ["l4", "v100", "t4", "p100", "p4"],
  "l4": ["v100", "t4", "p100", "p4"],
}

def get_fallback_configs(accel_config):
  """Yield fallback accelerator configs in order of preference."""
  if accel_config is None:
    return

  name = accel_config.name
  if name not in DOWNGRADE_PATHS:
    return

  for fallback_name in DOWNGRADE_PATHS[name]:
    try:
      # We try to keep the count similar or reset to 1 if not supported
      if isinstance(accel_config, GpuConfig):
        yield parse_accelerator(f"gpu:{fallback_name}x{accel_config.count}")
      elif isinstance(accel_config, TpuConfig):
        yield parse_accelerator(f"tpu:{fallback_name}-{accel_config.chips}")
    except ValueError:
       # if the specific count/chips topology is not supported for the fallback,
       # try the default minimum config for that fallback hardware.
       try:
           yield parse_accelerator(fallback_name)
       except ValueError:
           continue
