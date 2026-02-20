import numpy as np
import plotly.graph_objects as go


def plot_rotated_2d_objective(
    R,
    tau,
    a,
    c,
    b=0.0,
    solver_paths=None,   # dict name -> (T,2) array
    xlim=(-2, 2),
    ylim=(-2, 2),
    grid_points=200,
):
    """
    Interactive 3D plot of rotated exponential objective.

    Includes:
        - Surface
        - Feasible boundary
        - Optional solver trajectories
    """

    # ---------------------------------------------------------
    # Deterministic objective
    # ---------------------------------------------------------
    def f_det(z):
        z_rot = R @ z
        x_r, y_r = z_rot
        denom = tau**2 + a * y_r**2
        return -np.exp(-(y_r**2) - (x_r**2)/denom)

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------
    x = np.linspace(*xlim, grid_points)
    y = np.linspace(*ylim, grid_points)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(grid_points):
        for j in range(grid_points):
            Z[i, j] = f_det(np.array([X[i, j], Y[i, j]]))

    fig = go.Figure()

    # ---------------------------------------------------------
    # Surface
    # ---------------------------------------------------------
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        opacity=0.85,
        showscale=False,
        name="Objective surface"
    ))

    # ---------------------------------------------------------
    # Feasible boundary: c^T x = b
    # ---------------------------------------------------------
    if abs(c[1]) > 1e-12:
        x_line = np.linspace(*xlim, 500)
        y_line = (b - c[0] * x_line) / c[1]

        mask = (y_line >= ylim[0]) & (y_line <= ylim[1])
        x_line = x_line[mask]
        y_line = y_line[mask]

        z_line = np.array([
            f_det(np.array([xx, yy]))
            for xx, yy in zip(x_line, y_line)
        ])

        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode='lines',
            line=dict(color='red', width=6),
            name='Feasible boundary'
        ))

    # ---------------------------------------------------------
    # Solver trajectories (optional)
    # ---------------------------------------------------------
    if solver_paths is not None:
        for name, path in solver_paths.items():
            if path is None:
                continue

            xs = path[:, 0]
            ys = path[:, 1]
            zs = np.array([
                f_det(np.array([xx, yy]))
                for xx, yy in zip(xs, ys)
            ])

            fig.add_trace(go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(width=6),
                name=name
            ))

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    fig.update_layout(
        title="Rotated Exponential Objective (Interactive)",
        scene=dict(
            xaxis_title="x₁",
            yaxis_title="x₂",
            zaxis_title="f(x)",
        ),
        width=900,
        height=800,
    )

    fig.show(renderer="browser")

