"""Compare graph-model five-point histories with FLIR tracking/cooling histories.

Requires simulation CSV with absolute time_s, block, and five *_T_sim_C columns.
Experimental CSVs require time_s, block, pass, and five *_T_exp_C columns.
Absolute time origins must match (deposition start = 0); no extrapolation.

Run: python validation_corrected.py --simulation SIM.csv --tracking TRACK.csv \
    --cooling COOL.csv --output OUTPUT_DIR
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POINTS = ('center', 'x_minus', 'x_plus', 'y_minus', 'y_plus')


def read_data(path, source):
    df = pd.read_csv(path)
    needed = ['block', 'time_s'] + [f'{p}_T_{source}_C' for p in POINTS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing {missing}. Columns: {list(df.columns)}. '
                         'Export absolute time_s from simulation; do not substitute cooling_time_s.')
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['block', 'time_s']).copy()
    df['block'] = df['block'].astype(int)
    if source == 'exp':
        if 'pass' not in df.columns:
            df['pass'] = 1
        df['pass'] = pd.to_numeric(df['pass'], errors='coerce').fillna(1).astype(int)
    return df


def compare(sim, exp):
    records = []
    for (pass_no, block), observations in exp.groupby(['pass', 'block']):
        model = sim[sim['block'] == block].sort_values('time_s')
        if model.empty:
            print(f'WARNING: no simulation history for block {block}')
            continue
        for _, obs in observations.iterrows():
            t = float(obs['time_s'])
            record = {'pass': int(pass_no), 'block': int(block), 'time_s': t}
            for key in ('frame_count', 'validation_step', 'cooling_time_s',
                        'actual_cooling_time_s', 'block_end_time_s'):
                if key in obs.index:
                    record[key] = obs[key]
            errors = []
            for p in POINTS:
                ec, sc = f'{p}_T_exp_C', f'{p}_T_sim_C'
                te = float(obs[ec])
                series = model[['time_s', sc]].dropna().sort_values('time_s')
                # At identical timestamps, keep the last saved simulation stage.
                series = series.drop_duplicates('time_s', keep='last')
                x = series['time_s'].to_numpy(float)
                y = series[sc].to_numpy(float)
                if len(x) >= 2 and x[0] <= t <= x[-1]:
                    ts = float(np.interp(t, x, y))
                elif len(x) == 1 and np.isclose(t, x[0], atol=1e-9):
                    ts = float(y[0])
                else:
                    ts = np.nan
                error = ts - te
                record.update({ec: te, sc: ts, f'{p}_error_C': error,
                               f'{p}_abs_error_C': abs(error)})
                for axis in ('x', 'y'):
                    coord = f'{p}_{axis}_m'
                    if coord in obs.index:
                        record[coord] = obs[coord]
                if np.isfinite(error):
                    errors.append(error)
            errors = np.asarray(errors, dtype=float)
            record['valid_points'] = len(errors)
            record['MAE_C'] = float(np.mean(np.abs(errors))) if len(errors) else np.nan
            record['RMSE_C'] = float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan
            record['max_abs_error_C'] = float(np.max(np.abs(errors))) if len(errors) else np.nan
            records.append(record)
    return pd.DataFrame(records)


def add_gradients(table, block_length_m=None, block_width_m=None):
    for axis, size in (('x', block_length_m), ('y', block_width_m)):
        lower, upper = f'{axis}_minus', f'{axis}_plus'
        coord_lower, coord_upper = f'{lower}_{axis}_m', f'{upper}_{axis}_m'
        if coord_lower in table and coord_upper in table:
            distance = pd.to_numeric(table[coord_upper], errors='coerce') - pd.to_numeric(table[coord_lower], errors='coerce')
        elif size is not None and size > 0:
            distance = size / 2.0
        else:
            continue
        distance = np.where(np.abs(distance) > 0, distance, np.nan)
        for source in ('exp', 'sim'):
            table[f'dTd{axis}_{source}_C_per_m'] = (
                table[f'{upper}_T_{source}_C'] - table[f'{lower}_T_{source}_C']) / distance
        table[f'dTd{axis}_error_C_per_m'] = (
            table[f'dTd{axis}_sim_C_per_m'] - table[f'dTd{axis}_exp_C_per_m'])
    return table


def summarize(table):
    rows = []
    for (pass_no, block), group in table.groupby(['pass', 'block']):
        e = np.concatenate([group[f'{p}_error_C'].to_numpy(float) for p in POINTS])
        e = e[np.isfinite(e)]
        rows.append({'pass': pass_no, 'block': block,
                     'matched_frames': int(group['valid_points'].gt(0).sum()),
                     'matched_points': len(e),
                     'MAE_C': float(np.mean(np.abs(e))) if len(e) else np.nan,
                     'RMSE_C': float(np.sqrt(np.mean(e**2))) if len(e) else np.nan,
                     'bias_C': float(np.mean(e)) if len(e) else np.nan,
                     'max_abs_error_C': float(np.max(np.abs(e))) if len(e) else np.nan})
    return pd.DataFrame(rows)


def rates(table):
    records = []
    for (pass_no, block), group in table.groupby(['pass', 'block']):
        group = group.sort_values('time_s').drop_duplicates('time_s', keep='last')
        t = group['time_s'].to_numpy(float)
        for point in POINTS:
            for source in ('exp', 'sim'):
                temp = group[f'{point}_T_{source}_C'].to_numpy(float)
                for i in range(1, len(t)):
                    if t[i] > t[i-1] and np.isfinite(temp[i]) and np.isfinite(temp[i-1]):
                        records.append({'pass': pass_no, 'block': block, 'point': point,
                                        'source': source, 'time_start_s': t[i-1],
                                        'time_end_s': t[i],
                                        'temperature_rate_C_per_s': (temp[i]-temp[i-1])/(t[i]-t[i-1])})
    return pd.DataFrame(records, columns=['pass', 'block', 'point', 'source',
                                          'time_start_s', 'time_end_s', 'temperature_rate_C_per_s'])


def plot_group(group, folder):
    group = group.sort_values('time_s')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, source, title in zip(axes, ('exp', 'sim'), ('FLIR', 'Graph model')):
        for p in POINTS:
            ax.plot(group['time_s'], group[f'{p}_T_{source}_C'], marker='.',
                    linewidth=1.2, label=p)
        ax.set(xlabel='Elapsed deposition time (s)', title=title)
        ax.grid(alpha=.3)
    axes[0].set_ylabel('Temperature (°C)')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / 'five_point_side_by_side.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(group['time_s'], group['center_T_exp_C'], 'o-', markersize=3, label='FLIR')
    ax.plot(group['time_s'], group['center_T_sim_C'], 's--', markersize=3, label='Graph model')
    ax.set(xlabel='Elapsed deposition time (s)', ylabel='Center temperature (°C)')
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(folder / 'center_overlay.png', dpi=200)
    plt.close(fig)


def plot_profiles(table, folder, length=None, width=None):
    for idx, (_, row) in enumerate(table.iterrows()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, axis, size in zip(axes, ('x', 'y'), (length, width)):
            names = (f'{axis}_minus', 'center', f'{axis}_plus')
            coords = [f'{p}_{axis}_m' for p in names]
            if all(c in row.index and pd.notna(row[c]) for c in coords):
                offsets = np.array([row[c] for c in coords], dtype=float)
                offsets = (offsets - offsets[1]) * 1000
            elif size is not None:
                offsets = np.array([-.25, 0, .25]) * size * 1000
            else:
                continue
            for source, label in (('exp', 'FLIR'), ('sim', 'Graph model')):
                ax.plot(offsets, [row[f'{p}_T_{source}_C'] for p in names], 'o-', label=label)
            ax.set(xlabel=f'{axis.upper()} offset (mm)', ylabel='Temperature (°C)',
                   title=f'{axis.upper()} profile')
            ax.grid(alpha=.3); ax.legend()
        fig.suptitle(f'Pass {int(row["pass"])} | Block {int(row["block"])} | t={row["time_s"]:.3f} s')
        fig.tight_layout(); fig.savefig(folder / f'profiles_{idx:04d}.png', dpi=200)
        plt.close(fig)


def plot_parity(table, path):
    x = np.concatenate([table[f'{p}_T_exp_C'].to_numpy(float) for p in POINTS])
    y = np.concatenate([table[f'{p}_T_sim_C'].to_numpy(float) for p in POINTS])
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return
    x, y = x[valid], y[valid]
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=12, alpha=.4)
    ax.plot([lo, hi], [lo, hi], '--', color='black')
    ax.set(xlabel='FLIR temperature (°C)', ylabel='Graph temperature (°C)',
           title='Five-point parity')
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def run(simulation_path, tracking_path, cooling_path, output_dir,
        block_length_m=None, block_width_m=None):
    sim = read_data(simulation_path, 'sim')
    out = Path(output_dir)
    for label, path in (('Selected', tracking_path), ('Continuous', cooling_path)):
        if path is None:
            continue
        exp = read_data(path, 'exp')
        result = compare(sim, exp)
        if result.empty:
            print(f'WARNING: no matching blocks for {label}')
            continue
        result = add_gradients(result, block_length_m, block_width_m)
        folder = out / label
        tables, figures = folder / 'Tables', folder / 'Figures'
        tables.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        result.to_csv(tables / 'validation_comparison.csv', index=False)
        summarize(result).to_csv(tables / 'validation_summary_by_block.csv', index=False)
        rates(result).to_csv(tables / 'cooling_rates.csv', index=False)
        plot_parity(result, figures / 'parity.png')
        for (pass_no, block), group in result.groupby(['pass', 'block']):
            dest = figures / f'Pass_{pass_no}' / f'Block_{block}'
            dest.mkdir(parents=True, exist_ok=True)
            plot_group(group, dest)
            if label == 'Selected':
                plot_profiles(group, dest, block_length_m, block_width_m)
        print(f'{label}: {len(result)} rows, {int(result["valid_points"].sum())} matched point measurements')
    print(f'Comparison outputs: {out}')


def main():

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent.parent.parent
    PYTHON_DIR = BASE_DIR.parent.parent
    GRAPH_CODE_DIR = PYTHON_DIR / "GraphCode"
    VALIDATION_DIR = PROJECT_DIR / "GraphThermalModel"/ "Validation" 
    DOWNLOADS_DIR = Path.home() / "Downloads"
    EXPERIMENTAL_DIR = VALIDATION_DIR / "Results" / "Experimental"
        
    #Validation 
    SIMULATION_FILE = (
       VALIDATION_DIR / "Results" 
        / "Simulation"
        / "Validation"
        / "Tables"
        / "simulation_tracking_history.csv")


    # Experimental data
    PROCESSED_DIR =  DOWNLOADS_DIR / "Thermal_Processed" / "Temperature_Data"

    TRACKING_FILE = (
        EXPERIMENTAL_DIR
        / "experimental_tracking_history.csv"
    )

    COOLING_FILE = (EXPERIMENTAL_DIR / "experimental_cooling_history.csv")

    
    #Comparison File directory
    OUTPUT_DIR = (VALIDATION_DIR / "Results" / "Comparison")
  
    #Block configuration
    from config import BLOCK, BUILD

    BLOCK_LENGTH = BUILD["bead_length"] / BUILD["blocks_per_layer"]
    BLOCK_WIDTH = BLOCK["width"]

    
    #verify files
    for file in [SIMULATION_FILE, TRACKING_FILE]:
        if not file.exists():
            raise FileNotFoundError(f"Required validation file not found: {file}")

    if not COOLING_FILE.exists():
        print(f"WARNING: Cooling history not found: {COOLING_FILE}" )
        COOLING_FILE = None

    #run validation
    run(simulation_path=SIMULATION_FILE,
        tracking_path=TRACKING_FILE,
        cooling_path=COOLING_FILE,
        output_dir=OUTPUT_DIR,
        block_length_m=BLOCK_LENGTH,
        block_width_m=BLOCK_WIDTH)


if __name__ == "__main__":
    main()