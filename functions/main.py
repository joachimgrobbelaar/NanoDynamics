import typing
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from firebase_functions import https_fn
from firebase_admin import initialize_app

initialize_app()

from leo_simulator.experiment import run_single_simulation


@https_fn.on_call(memory=1024, timeout_sec=120)
def evaluate_sweep_point(req: https_fn.CallableRequest) -> typing.Any:
    """Serverless endpoint to evaluate a single trajectory and return metrics."""
    params = req.data["params"]
    include_traj = req.data.get("include_trajectory", False)

    metrics, traj = run_single_simulation(
        name=params["name"],
        mass=params["mass"],
        drag_area=params["drag_area"],
        cd=params["cd"],
        altitude_km=params["altitude_km"],
        eccentricity=params["eccentricity"],
        inclination_deg=params["inclination_deg"],
        raan_deg=params.get("raan_deg", 0.0),
        atmosphere_type=req.data.get("atmosphere_type", "piecewise"),
        density_scale=req.data.get("density_scale", 1.0),
        include_j2=params.get("include_j2", True),
        include_drag=params.get("include_drag", True),
        include_moon=params.get("include_moon", False),
        max_duration_seconds=req.data.get("max_duration_seconds", 30 * 86400.0),
        include_trajectory=include_traj,
    )

    clean_metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
    return {"metrics": clean_metrics, "trajectory": traj}


@https_fn.on_call(memory=2048, timeout_sec=540)
def trigger_pinn_training(req: https_fn.CallableRequest) -> typing.Any:
    """
    Serverless PyTorch PINN training trigger.
    Accepts: epochs (int), lr (float), batch_size (int), hidden (int), layers (int).
    Returns training loss history and final evaluation metrics.
    """
    import torch
    import numpy as np

    epochs = int(req.data.get("epochs", 500))
    lr = float(req.data.get("lr", 1e-3))
    batch_size = int(req.data.get("batch_size", 1024))
    hidden = int(req.data.get("hidden", 256))
    layers = int(req.data.get("layers", 4))

    # Load data from the bundled npz
    data_path = os.path.join(os.path.dirname(__file__), "leo_simulator", "ai", "data", "sim_pairs.npz")
    if not os.path.exists(data_path):
        return {"status": "error", "message": f"Training data not found at {data_path}"}

    d = np.load(data_path)
    X = torch.tensor(d["x"], dtype=torch.float32)
    Y = torch.tensor(d["y"], dtype=torch.float32)
    n_pairs = X.shape[0]

    # Build MLP surrogate
    layer_list = [torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh()]
    for _ in range(layers - 1):
        layer_list += [torch.nn.Linear(hidden, hidden), torch.nn.Tanh()]
    layer_list.append(torch.nn.Linear(hidden, Y.shape[1]))
    model = torch.nn.Sequential(*layer_list)

    torch.set_num_threads(os.cpu_count() or 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    loss_fn = torch.nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        scheduler.step(avg)
        if epoch % max(1, epochs // 20) == 0:
            loss_history.append({"epoch": epoch, "loss": avg})

    model.eval()
    with torch.no_grad():
        final_pred = model(X[:1000])
        final_loss = float(loss_fn(final_pred, Y[:1000]).item())

    return {
        "status": "success",
        "n_pairs": n_pairs,
        "epochs_run": epochs,
        "final_loss": final_loss,
        "loss_history": loss_history,
    }
