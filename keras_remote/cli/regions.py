"""Region and zone sorting logic for keras-remote."""

import os
import subprocess

from keras_remote.constants import CANDIDATE_ZONES, DEFAULT_ZONE, get_default_zone


def get_gcloud_zone():
  """Attempt to get the default zone from local gcloud config."""
  try:
    result = subprocess.run(
      ["gcloud", "config", "get-value", "compute/zone"],
      capture_output=True,
      text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
      return result.stdout.strip()
  except Exception:
    pass
  return None


def get_base_zone(override_zone=None):
  """Resolve a user's closest base zone."""
  # 1. explicit command line flag
  if override_zone:
    return override_zone
  # 2. environment variable
  if "KERAS_REMOTE_ZONE" in os.environ:
    return os.environ["KERAS_REMOTE_ZONE"]
  # 3. gcloud config
  gcloud_z = get_gcloud_zone()
  if gcloud_z:
    return gcloud_z
  # 4. default
  return DEFAULT_ZONE


def parse_zone(zone):
  """Parse a zone string into (continent, region). E.g. us-central1-a -> ('us', 'us-central1')"""
  parts = zone.split("-")
  if len(parts) >= 2:
    continent = parts[0]
    region = f"{parts[0]}-{parts[1]}"
    return continent, region
  return "", ""


def get_sorted_zones(override_zone=None):
  """Return a list of candidate zones sorted by proximity to the base zone."""
  base_zone = get_base_zone(override_zone)
  base_continent, base_region = parse_zone(base_zone)

  def sort_key(zone):
    continent, region = parse_zone(zone)
    # Tier 0: exact match
    if zone == base_zone:
      return 0
    # Tier 1: exact region match
    if region == base_region:
      return 1
    # Tier 2: continent match
    if continent == base_continent:
      return 2
    # Tier 3: rest of the world
    return 3

  # Using a stable sort to keep the default order of CANDIDATE_ZONES within each tier
  return sorted(CANDIDATE_ZONES, key=sort_key)
