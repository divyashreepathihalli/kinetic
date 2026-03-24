"""keras-remote pool commands — add, remove, and list accelerator node pools."""

import click

from keras_remote.cli.config import InfraConfig, NodePoolConfig
from keras_remote.cli.infra.state import apply_update, load_state
from keras_remote.cli.options import common_options
from keras_remote.cli.output import (
  banner,
  console,
  infrastructure_state,
  warning,
)
from keras_remote.core import accelerators
from keras_remote.core.accelerators import generate_pool_name
from keras_remote.cli.regions import get_sorted_zones
from keras_remote.cli.downgrade import get_fallback_configs


@click.group()
def pool():
  """Manage accelerator node pools."""


@pool.command("add")
@common_options
@click.option(
  "--accelerator",
  required=True,
  help="Accelerator spec: t4, l4, a100, a100-80gb, h100, "
  "v5litepod, v5p, v6e, v3 (with optional count/topology)",
)
@click.option(
  "--min-nodes",
  default=0,
  type=int,
  help="Minimum node count for the accelerator node pool (default: 0)",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def pool_add(project, zone, cluster_name, accelerator, min_nodes, yes):
  """Add an accelerator node pool to the cluster."""
  banner("keras-remote Pool Add")

  # Parse the accelerator spec first to fail fast on bad input.
  try:
    accel_config = accelerators.parse_accelerator(accelerator)
  except ValueError as e:
    raise click.BadParameter(str(e), param_hint="--accelerator") from e

  if accel_config is None:
    raise click.BadParameter(
      "Cannot add a CPU node pool. Use 'keras-remote up' instead.",
      param_hint="--accelerator",
    )

  from keras_remote.cli.prompts import resolve_project
  project = project or resolve_project()

  zones = get_sorted_zones(zone)
  current_accel = accel_config
  fallback_iterator = get_fallback_configs(accel_config)

  if not yes:
    console.print(f"\nAdding pool for requested accelerator: {accelerator}")
    click.confirm("Proceed?", abort=True)

  update_succeeded = False
  successful_zone = None

  while True:
    for z in zones:
      state = load_state(project, z, cluster_name)
      if not state.stack:
          # don't try to add a pool to a zone without a stack
          continue
          
      console.print(f"\n[bold]Attempting pool setup in zone {z}...[/bold]")
      # to guarantee stock we set min_nodes to at least 1 for the test
      test_min_nodes = max(1, min_nodes)
      new_pool_name = generate_pool_name(current_accel)
      new_pool = NodePoolConfig(new_pool_name, current_accel, min_nodes=test_min_nodes)
      
      all_pools = state.node_pools + [new_pool]
      config = InfraConfig(
        project=state.project,
        zone=state.zone,
        cluster_name=state.cluster_name,
        node_pools=all_pools,
      )
      
      update_succeeded = apply_update(config)
      if update_succeeded:
        successful_zone = z
        break
      else:
        warning(f"Provisioning failed in {z}. Reverting cluster state...")
        # revert
        config.node_pools = state.node_pools
        apply_update(config)

    if update_succeeded:
      if current_accel != accel_config:
        # Downgrade successful
        if not click.confirm(f"\nRequested hardware was unavailable. Successfully secured {current_accel.name} in {successful_zone}. Keep this hardware?"):
           # revert
           config.node_pools = state.node_pools
           apply_update(config)
           raise click.ClickException("Pool addition aborted by user.")
      break

    # try fallback
    try:
      current_accel = next(fallback_iterator)
    except StopIteration:
      banner("Pool Update Failed")
      console.print("\nAll hardware limits exhausted across candidate zones.")
      return

    console.print(f"\n[bold yellow]Hardware severely constrained. Searching for fallback: {current_accel.name}[/bold yellow]")

  console.print()
  banner("Pool Added")
  console.print()


@pool.command("remove")
@common_options
@click.argument("pool_name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def pool_remove(project, zone, cluster_name, pool_name, yes):
  """Remove an accelerator node pool from the cluster."""
  banner("keras-remote Pool Remove")

  state = load_state(project, zone, cluster_name)

  remaining = [p for p in state.node_pools if p.name != pool_name]
  if len(remaining) == len(state.node_pools):
    existing_names = [p.name for p in state.node_pools]
    raise click.ClickException(
      f"Node pool '{pool_name}' not found. "
      f"Existing pools: {', '.join(existing_names) or '(none)'}"
    )

  console.print(f"\nRemoving pool [bold]{pool_name}[/bold]")
  console.print(f"Remaining pools after remove: {len(remaining)}\n")

  if not yes:
    click.confirm("Proceed?", abort=True)

  config = InfraConfig(
    project=state.project,
    zone=state.zone,
    cluster_name=state.cluster_name,
    node_pools=remaining,
  )
  update_succeeded = apply_update(config)

  console.print()
  if update_succeeded:
    banner("Pool Removed")
  else:
    banner("Pool Update Failed")
    console.print()
    console.print(
      "You may re-run the command to retry, or use"
      " [bold]keras-remote pool list[/bold] to check current state."
    )
  console.print()


@pool.command("list")
@common_options
def pool_list(project, zone, cluster_name):
  """List accelerator node pools on the cluster."""
  banner("keras-remote Node Pools")

  state = load_state(project, zone, cluster_name, allow_missing=True)

  if state.stack is None:
    warning("No Pulumi stack found.")
    console.print("Run 'keras-remote up' to provision infrastructure.")
    return

  outputs = state.stack.outputs()
  if not outputs:
    warning("No infrastructure found. Run 'keras-remote up' first.")
    return

  infrastructure_state(outputs)
