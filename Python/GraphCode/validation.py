"""Compare exported five-point graph histories with CleanThermalImages FLIR results.

Inputs are the existing CSV outputs from DepositionTracker.export_tracking_histories
and Experimental_Processing/CleanThermalImages/ThermalValidation's
validation_results.csv. No fabricated measurements or extrapolation.

Run: python validation.py --simulation SIM.csv --experimental EXP.csv --output DIR
Optional: --block-length-m L --block-width-m W (for X/Y profiles and gradients)
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POINTS = ('center', 'x_minus', 'x_plus', 'y_minus', 'y_plus')


def load_tables(sim_path, exp_path):
    sim, exp = pd.read_csv(sim_path), pd.read_csv(exp_path)
    for title, df, suffix in [('Simulation', sim, 'sim'), ('FLIR', exp, 'exp')]:
        required = ['block', 'cooling_time_s'] + [f'{p}_T_{suffix}_C' for p in POINTS]
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f'{title} CSV missing {missing}. Available: {list(df.columns)}')
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['block', 'cooling_time_s'], inplace=True)
        df['block'] = df['block'].astype(int)
    if 'pass' not in exp:
        exp['pass'] = 1
    exp['pass'] = pd.to_numeric(exp['pass'], errors='coerce').fillna(1).astype(int)
    return sim, exp


def compare(sim, exp):
    records = []
    for (pass_no, block), observations in exp.groupby(['pass', 'block']):
        model = sim.loc[sim['block'] == block].sort_values('cooling_time_s')
        model = model.drop_duplicates('cooling_time_s', keep='last')
        if model.empty:
            print(f'WARNING: no simulation for pass {pass_no}, block {block}')
            continue
        for _, obs in observations.iterrows():
            t = float(obs['cooling_time_s'])
            record = {'pass': pass_no, 'block': block, 'cooling_time_s': t}
            for key in ('frame_count', 'validation_step', 'time_s'):
                if key in obs.index:
                    record[key] = obs[key]
            errors = []
            for point in POINTS:
                exp_col, sim_col = f'{point}_T_exp_C', f'{point}_T_sim_C'
                te = float(obs[exp_col])
                x = model['cooling_time_s'].to_numpy(float)
                y = model[sim_col].to_numpy(float)
                valid = np.isfinite(x) & np.isfinite(y)
                x, y = x[valid], y[valid]
                if len(x) >= 2 and x[0] <= t <= x[-1]:
                    ts = float(np.interp(t, x, y))
                elif len(x) == 1 and np.isclose(t, x[0], atol=1e-9):
                    ts = float(y[0])
                else:
                    ts = np.nan  # Never extrapolate beyond simulation coverage
                err = ts - te
                record.update({exp_col: te, sim_col: ts,
                               f'{point}_error_C': err,
                               f'{point}_abs_error_C': abs(err)})
                if np.isfinite(err):
                    errors.append(err)
            err = np.asarray(errors)
            record['valid_points'] = len(err)
            record['MAE_C'] = float(np.mean(abs(err))) if len(err) else np.nan
            record['RMSE_C'] = float(np.sqrt(np.mean(err**2))) if len(err) else np.nan
            record['max_abs_error_C'] = float(np.max(abs(err))) if len(err) else np.nan
            records.append(record)
    return pd.DataFrame(records)


def gradients(table, length, width):
    if length <= 0 or width <= 0:
        raise ValueError('Block dimensions must be positive')
    for axis, distance in [('x', length / 2), ('y', width / 2)]:
        for source in ('exp', 'sim'):
            table[f'dTd{axis}_{source}_C_per_m'] = (
                table[f'{axis}_plus_T_{source}_C'] -
                table[f'{axis}_minus_T_{source}_C']) / distance
        table[f'dTd{axis}_error_C_per_m'] = (
            table[f'dTd{axis}_sim_C_per_m'] - table[f'dTd{axis}_exp_C_per_m'])
    return table


def summary_table(table):
    rows = []
    for (pass_no, block), group in table.groupby(['pass', 'block']):
        e = np.concatenate([group[f'{p}_error_C'].to_numpy(float) for p in POINTS])
        e = e[np.isfinite(e)]
        rows.append({'pass': pass_no, 'block': block, 'matched_points': len(e),
                     'matched_frames': int(group['valid_points'].gt(0).sum()),
                     'MAE_C': np.mean(abs(e)) if len(e) else np.nan,
                     'RMSE_C': np.sqrt(np.mean(e**2)) if len(e) else np.nan,
                     'bias_C': np.mean(e) if len(e) else np.nan,
                     'max_abs_error_C': np.max(abs(e)) if len(e) else np.nan})
    return pd.DataFrame(rows)


def cooling_rate_table(table):
    rows = []
    for (pass_no, block), group in table.groupby(['pass', 'block']):
        group = group.sort_values('cooling_time_s').drop_duplicates('cooling_time_s')
        t = group['cooling_time_s'].to_numpy(float)
        for p in POINTS:
            for source in ('exp', 'sim'):
                values = group[f'{p}_T_{source}_C'].to_numpy(float)
                for i in range(1, len(t)):
                    if t[i] > t[i-1] and np.isfinite(values[i:i+1]).all() and np.isfinite(values[i-1]):
                        rows.append({'pass': pass_no, 'block': block, 'point': p,
                                     'source': source, 'time_start_s': t[i-1],
                                     'time_end_s': t[i],
                                     'temperature_rate_C_per_s': (values[i]-values[i-1])/(t[i]-t[i-1])})
    return pd.DataFrame(rows, columns=['pass', 'block', 'point', 'source', 'time_start_s',
                                        'time_end_s', 'temperature_rate_C_per_s'])


def plot_side_by_side(group, path):
    group = group.sort_values('cooling_time_s')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, source, title in zip(axes, ('exp', 'sim'), ('FLIR', 'Graph model')):
        for p in POINTS:
            ax.plot(group['cooling_time_s'], group[f'{p}_T_{source}_C'],
                    marker='.', label=p, linewidth=1.3)
        ax.set(xlabel='Cooling time (s)', title=title)
        ax.grid(alpha=.3)
    axes[0].set_ylabel('Temperature (°C)')
    axes[1].legend(fontsize=8)
    fig.suptitle(f'Pass {int(group["pass"].iloc[0])} | Block {int(group["block"].iloc[0])}')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_overlay(group, path):
    group = group.sort_values('cooling_time_s')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(group['cooling_time_s'], group['center_T_exp_C'], 'o-', label='FLIR', markersize=3)
    ax.plot(group['cooling_time_s'], group['center_T_sim_C'], 's--', label='Graph model', markersize=3)
    ax.set(xlabel='Cooling time (s)', ylabel='Center temperature (°C)', title='Center cooling comparison')
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_profiles(row, length, width, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, axis, size in zip(axes, ('x', 'y'), (length, width)):
        names = (f'{axis}_minus', 'center', f'{axis}_plus')
        offsets = np.array([-.25, 0, .25]) * size * 1000
        for source, label in [('exp', 'FLIR'), ('sim', 'Graph model')]:
            ax.plot(offsets, [row[f'{p}_T_{source}_C'] for p in names], 'o-', label=label)
        ax.set(xlabel=f'{axis.upper()} offset from block center (mm)',
               ylabel='Temperature (°C)', title=f'{axis.upper()} profile')
        ax.grid(alpha=.3); ax.legend()
    fig.suptitle(f'Pass {int(row["pass"])} | Block {int(row["block"])} | Cooling {row["cooling_time_s"]:.3f} s')
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_parity(table, path):
    x = np.concatenate([table[f'{p}_T_exp_C'].to_numpy(float) for p in POINTS])
    y = np.concatenate([table[f'{p}_T_sim_C'].to_numpy(float) for p in POINTS])
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return
    x, y = x[mask], y[mask]
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=12, alpha=.4)
    ax.plot([lo, hi], [lo, hi], '--', color='black')
    ax.set(xlabel='FLIR temperature (°C)', ylabel='Graph temperature (°C)', title='Five-point parity')
    ax.grid(alpha=.3); ax.set_aspect('equal', adjustable='box')
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def run(simulation_path, experimental_path, output_dir, block_length_m=None, block_width_m=None):
    output_dir = Path(output_dir)
    tables_dir, figures_dir = output_dir / 'Tables', output_dir / 'Figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sim, exp = load_tables(simulation_path, experimental_path)
    table = compare(sim, exp)
    if table.empty:
        raise ValueError('No matching block IDs between simulation and FLIR CSVs')
    if block_length_m is not None and block_width_m is not None:
        gradients(table, block_length_m, block_width_m)
    summary = summary_table(table)
    rates = cooling_rate_table(table)
    table.to_csv(tables_dir / 'validation_comparison.csv', index=False)
    summary.to_csv(tables_dir / 'validation_summary_by_block.csv', index=False)
    rates.to_csv(tables_dir / 'cooling_rates.csv', index=False)
    plot_parity(table, figures_dir / 'parity.png')
    for (pass_no, block), group in table.groupby(['pass', 'block']):
        folder = figures_dir / f'Pass_{pass_no}' / f'Block_{block}'
        folder.mkdir(parents=True, exist_ok=True)
        plot_side_by_side(group, folder / 'cooling_side_by_side.png')
        plot_overlay(group, folder / 'center_overlay.png')
        if block_length_m is not None and block_width_m is not None:
            for i, (_, row) in enumerate(group.iterrows()):
                plot_profiles(row, block_length_m, block_width_m, folder / f'profiles_{i:04d}.png')
    print(f'Validation saved: {output_dir}')
    print(f'Rows: {len(table)}, valid point comparisons: {table["valid_points"].sum()}')
    return table, summary, rates


def main():

    # ----- Base Directories -----
    BASE_DIR = Path(__file__).resolve().parent.parent.parent


    DOWNLOADS_DIR = Path.home() / "Downloads"

    # ----- Simulation Data -----
    SIMULATION_FILE = (
        BASE_DIR
        / "Validation"
        / "Results"
        / "Simulation"/"Validation"/"Tables"
        / "simulation_tracking_history.csv"
    )

    EXPERIMENTAL_FILE = (
        BASE_DIR
        / "Validation"
        / "Results"
        / "Experimental"
        / "experimental_tracking_history.csv"
    )


    OUTPUT_DIR = (
        BASE_DIR
        / "Validation"
        / "Results"
        / "Comparison"
    )

    # ----- Block Dimensions -----
    from config import BLOCK, BUILD

    # Use the same dimensions as your simulation
    BLOCK_LENGTH = BUILD["bead_length"] / BUILD["blocks_per_layer"]
    BLOCK_WIDTH = BLOCK["width"]

    # ----- Run Validation -----
    run(
        simulation_path=SIMULATION_FILE,
        experimental_path=EXPERIMENTAL_FILE,
        output_dir=OUTPUT_DIR,
        block_length_m=BLOCK_LENGTH,
        block_width_m=BLOCK_WIDTH
    )



if __name__ == '__main__':
    main()
