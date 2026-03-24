"""keras-remote up command — provision infrastructure."""

import subprocess

import click

from keras_remote.cli.config import InfraConfig, NodePoolConfig
from keras_remote.cli.constants import DEFAULT_CLUSTER_NAME
from keras_remote.cli.infra.post_deploy import (
  configure_kubectl,
  install_gpu_drivers,
  install_lws,
)
from keras_remote.cli.infra.state import apply_update, apply_destroy, load_state
from keras_remote.cli.options import common_options
from keras_remote.cli.output import (
  LiveOutputPanel,
  banner,
  config_summary,
  console,
  warning,
)
from keras_remote.cli.prerequisites_check import check_all
from keras_remote.cli.prompts import prompt_accelerator, resolve_project
from keras_remote.core import accelerators
from keras_remote.core.accelerators import GpuConfig, generate_pool_name
from keras_remote.cli.regions import get_sorted_zones
from keras_remote.cli.downgrade import get_fallback_configs


@click.command()
@common_options
@click.option(
  "--accelerator",
  default=None,
  help="Accelerator spec: cpu, t4, l4, a100, a100-80gb, h100, "
  "v5litepod, v5p, v6e, v3",
)
@click.option(
  "--min-nodes",
  default=0,
  type=int,
  help="Minimum node count for accelerator node pools (default: 0, scale-to-zero)",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def up(project, zone, accelerator, cluster_name, min_nodes, yes):
  """Provision GCP infrastructure for keras-remote."""
  banner("keras-remote Setup")

  # Check prerequisites
  check_all()

  # Resolve configuration
  project = project or resolve_project()
  cluster_name = cluster_name or DEFAULT_CLUSTER_NAME

  # Resolve accelerator (interactive if not provided)
  if accelerator and accelerator.strip().lower() == "cpu":
    accel_config = None
  elif accelerator:
    try:
      accel_config = accelerators.parse_accelerator(accelerator)
    except ValueError as e:
      raise click.BadParameter(str(e), param_hint="--accelerator") from e
  else:
    accel_config = prompt_accelerator()

  zones = get_sorted_zones(zone)
  first_z = zones[0] if zones else "us-central1-a"

  # If a stack already exists in the first sorted zone, preserve its node pools.
  first_state = load_state(
    project,
    first_z,
    cluster_name,
    allow_missing=True,
    check_prerequisites=False,
  )

  first_config = InfraConfig(project=project, zone=first_z, cluster_name=cluster_name)
  if first_state.node_pools:
    first_config.node_pools = list(first_state.node_pools)
    console.print(
      f"\nFound {len(first_state.node_pools)} existing node pool(s)."
      "\nUse 'keras-remote pool add/remove/list' to manage node pools.\n"
    )
  elif accel_config is not None:
    first_config.node_pools.append(
      NodePoolConfig(
        generate_pool_name(accel_config), accel_config, min_nodes=min_nodes
      )
    )

  # Show summary and confirm based on the first candidate setup
  config_summary(first_config)
  if not yes:
    click.confirm("\nProceed with setup?", abort=True)

  console.print()

  current_accel = accel_config
  fallback_iterator = get_fallback_configs(accel_config)

  pulumi_ok = False
  successful_zone = None
  config = None

  while True:
    for z in zones:
      console.print(f"\n[bold]Attempting setup in zone {z}...[/bold]")
      state = load_state(project, z, cluster_name, allow_missing=True, check_prerequisites=False)
      config = InfraConfig(project=project, zone=z, cluster_name=cluster_name)
      if state.node_pools:
        config.node_pools = list(state.node_pools)
      elif current_accel is not None:
        # clamp min_nodes to 1 if we're hunting for capacity to ensure it is actually available
        test_min_nodes = max(1, min_nodes)
        config.node_pools.append(
          NodePoolConfig(generate_pool_name(current_accel), current_accel, min_nodes=test_min_nodes)
        )

      pulumi_ok = apply_update(config)
      if pulumi_ok:
        successful_zone = z
        break
      else:
        # cleanup failed attempt
        warning(f"Provisioning failed in {z}. Cleaning up...")
        if not state.node_pools: 
           # only destroy if cluster didn't exist prior to us trying
           apply_destroy(config)

    if pulumi_ok:
      if current_accel != accel_config:
        # It was a downgrade
        if not click.confirm(f"\nRequested hardware was unavailable. Successfully secured {current_accel.name} in {successful_zone}. Keep this hardware?"):
          apply_destroy(config)
          raise click.ClickException("Setup aborted by user.")
      
      zone = successful_zone
      break

    # try fallback
    try:
      current_accel = next(fallback_iterator)
    except StopIteration:
      raise click.ClickException("All hardware limits exhausted across candidate zones.")

    console.print(f"\n[bold yellow]Hardware severely constrained. Searching for fallback: {current_accel.name}[/bold yellow]")


  pulumi_failed = not pulumi_ok

  if pulumi_failed:
    warning("Attempting post-deploy configuration anyway...")

  # Post-deploy steps
  steps = [
    (
      "kubectl configuration",
      lambda: configure_kubectl(
        cluster_name,
        zone,
        project,
      ),
    ),
    ("LWS CRD installation", install_lws),
  ]
  if config and any(isinstance(np.accelerator, GpuConfig) for np in config.node_pools):
    steps.append(("GPU driver installation", install_gpu_drivers))

  failures = []
  with LiveOutputPanel("Post-deploy configuration", transient=True) as panel:
    for name, fn in steps:
      panel.on_output(f"{name}...")
      try:
        fn()
        panel.on_output(f"{name} complete.")
      except subprocess.CalledProcessError as e:
        failures.append(name)
        panel.on_output(f"{name} failed: {e}")
        if e.stderr:
          stderr_text = e.stderr.decode("utf-8", errors="replace").strip()
          if stderr_text:
            for line in stderr_text.splitlines():
              panel.on_output(f"  {line}")
        panel.mark_error()

  # Final summary
  console.print()
  if pulumi_failed or failures:
    banner("Setup Completed With Warnings")
    console.print()
    if pulumi_failed:
      warning("Pulumi provisioning encountered errors (see above).")
    if failures:
      warning(f"Post-deploy steps failed: {', '.join(failures)}")
    console.print()
    console.print(
      "You may re-run [bold]keras-remote up[/bold] to retry failed steps."
    )
  else:
    banner("Setup Complete")

  console.print()
  console.print("Add these environment variables to your shell config:")
  console.print(f"  export KERAS_REMOTE_PROJECT={project}")
  console.print(f"  export KERAS_REMOTE_ZONE={zone}")
  console.print(f"  export KERAS_REMOTE_CLUSTER={cluster_name}")
  console.print()
  console.print("View quotas:")
  console.print(
    f"  https://console.cloud.google.com/iam-admin/quotas?project={project}"
  )
  console.print()
