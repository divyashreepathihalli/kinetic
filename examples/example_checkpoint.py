import os

# Set backend to JAX before any keras imports
os.environ["KERAS_BACKEND"] = "jax"

import kinetic

@kinetic.run(accelerator="cpu")
def train_with_checkpoints():
  """Demo function showing Orbax checkpointing with Kinetic."""
  import jax
  import jax.numpy as jnp
  import orbax.checkpoint as ocp
  
  checkpoint_dir = os.environ.get("KINETIC_CHECKPOINT_DIR")
  print(f"--- Kinetic Checkpoint Dir: {checkpoint_dir} ---")

  if not checkpoint_dir:
    # Fallback for local testing if run without kinetic context
    checkpoint_dir = "/tmp/local_checkpoints"
    print(f"No KINETIC_CHECKPOINT_DIR found, using local: {checkpoint_dir}")

  # 1. Define dummy state (PyTree)
  state = {
    "step": 0,
    "weights": jnp.ones((10, 10)),
    "bias": jnp.zeros((10,)),
  }

  # 2. Initialize Orbax CheckpointManager
  # Orbax handles GCS paths transparently via etils.epath
  options = ocp.CheckpointManagerOptions(max_to_keep=2)
  mngr = ocp.CheckpointManager(
    checkpoint_dir,
    ocp.StandardCheckpointer(),
    options=options
  )

  print("Saving checkpoint at step 0...")
  mngr.save(0, state)
  mngr.wait_until_finished()
  print("Checkpoint saved successfully.")

  # 3. Restore to verify
  print("Restoring checkpoint from step 0...")
  restored_state = mngr.restore(0)
  
  # Assert equality for verification
  assert jnp.allclose(restored_state["weights"], state["weights"])
  print("Checkpoint restored successfully and verified!")

  return True


if __name__ == "__main__":
  print("Starting Orbax checkpointing demo...")
  success = train_with_checkpoints()
  print(f"Demo run success: {success}")
