import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "savefig.dpi": 300,
    "savefig.format": 'png',
    "axes.linewidth": 1.5
})

def read_xvg(filename, return_sets=False):
    """Robust XVG reader handling multi-block datasets with '&'."""
    if not os.path.exists(filename):
        return None
    
    all_sets = []
    current_set = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('@', '#')):
                continue
            if line.startswith('&'):
                if current_set:
                    all_sets.append(np.array(current_set))
                    current_set = []
                continue
            try:
                numeric_parts = []
                for x in line.split():
                    try:
                        numeric_parts.append(float(x))
                    except ValueError:
                        continue
                if numeric_parts:
                    current_set.append(numeric_parts)
            except Exception:
                continue
                
    if current_set:
        all_sets.append(np.array(current_set))
    
    if return_sets:
        return all_sets
    return np.vstack(all_sets) if all_sets else np.array([])

def parse_xpm(filename):
    """Parses GROMACS XPM matrix files into a numpy array."""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = [l.strip().strip('",') for l in lines if l.startswith('"') and not l.startswith('" ')]
    
    if not data_lines: return None
    
    header = data_lines[0].split()
    if len(header) < 4: return None
    
    width, height, ncolors, cpp = map(int, header)
    
    color_map = {}
    for i in range(1, ncolors + 1):
        line = data_lines[i]
        symbol = line[:cpp]
        if 'c #' in line:
            color_hex = line.split('c #')[1].split()[0]
            color_map[symbol] = i
            
    matrix = []
    for i in range(ncolors + 1, len(data_lines)):
        row_str = data_lines[i]
        row = [color_map.get(row_str[j:j+cpp], 0) for j in range(0, len(row_str), cpp)]
        if len(row) == width:
            matrix.append(row)
            
    return np.array(matrix)

def plot_fes_2d(rmsd_file, rg_file, output='FES_2D.png'):
    """Calculates and plots Free Energy Surface (FES)."""
    rmsd = read_xvg(rmsd_file)
    rg = read_xvg(rg_file)
    
    if rmsd is None or rg is None:
        print(f"Skipping FES: Missing {rmsd_file} or {rg_file}")
        return

    min_len = min(len(rmsd), len(rg))
    x = rmsd[:min_len, 1]
    y = rg[:min_len, 1]

    k = gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    kT = 0.008314 * 300
    fes = -kT * np.log(zi.reshape(xi.shape))
    fes -= np.min(fes)

    plt.figure(figsize=(10, 8))
    cp = plt.contourf(xi, yi, fes, 20, cmap='viridis_r')
    cbar = plt.colorbar(cp)
    cbar.set_label('$\Delta$G (kJ/mol)', fontsize=14)
    
    plt.title('Free Energy Surface')
    plt.xlabel('RMSD (nm)')
    plt.ylabel('Rg (nm)')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_fes_3d(rmsd_file, rg_file, output='results_plot/FES_3D.png'):
    """3D visualization of the Free Energy Surface."""
    from matplotlib import cm
    rmsd = read_xvg(rmsd_file)
    rg = read_xvg(rg_file)
    
    if rmsd is None or rg is None: return

    min_len = min(len(rmsd), len(rg))
    x = rmsd[:min_len, 1]
    y = rg[:min_len, 1]

    k = gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    kT = 0.008314 * 300
    fes = -kT * np.log(zi.reshape(xi.shape))
    fes -= np.min(fes)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(xi, yi, fes, cmap='viridis_r', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='$\Delta$G (kJ/mol)')
    
    ax.set_title('Free Energy Surface')
    ax.set_xlabel('RMSD (nm)')
    ax.set_ylabel('Rg (nm)')
    ax.set_zlabel('$\Delta$G (kJ/mol)')
    
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_rmsf_perix(rmsf_file, output='RMSF.png'):
    """Plot RMSF per residue."""
    data = read_xvg(rmsf_file)
    if data is None: return

    plt.figure(figsize=(12, 5))
    plt.plot(data[:, 0], data[:, 1], color='#d62728', linewidth=2)
    plt.fill_between(data[:, 0], data[:, 1], color='#d62728', alpha=0.2)
    
    plt.title('RMSF')
    plt.xlabel('Residue number')
    plt.ylabel('RMSF (nm)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_sasa(sasa_file, output='results_plot/SASA.png'):
    """Plot Solvent Accessible Surface Area (SASA)."""
    data = read_xvg(sasa_file)
    if data is None: return

    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1], color='#2ca02c', linewidth=2, label='Total SASA')
    plt.fill_between(data[:, 0], data[:, 1], color='#2ca02c', alpha=0.1)
    
    plt.title('SASA')
    plt.xlabel('Time (ps)')
    plt.ylabel('SASA (nm$^2$)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_rg_anisotropy(rg_file, output='results_plot/Rg.png'):
    """Plot Radius of Gyration components."""
    data = read_xvg(rg_file)
    if data is None or data.shape[1] < 5: return

    plt.figure(figsize=(10, 6))
    time = data[:, 0]
    plt.plot(time, data[:, 1], label='Rg', color='black', linewidth=2)
    plt.plot(time, data[:, 2], label='Rg$_x$', alpha=0.7)
    plt.plot(time, data[:, 3], label='Rg$_y$', alpha=0.7)
    plt.plot(time, data[:, 4], label='Rg$_z$', alpha=0.7)
    
    plt.title('Radius of Gyration')
    plt.xlabel('Time (ps)')
    plt.ylabel('Rg (nm)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_hbond(hbond_file, output='HBond.png'):
    """Plot Hydrogen Bond Number over time."""
    data = read_xvg(hbond_file)
    if data is None: return

    plt.figure(figsize=(10, 5))
    plt.step(data[:, 0], data[:, 1], color='#9467bd', where='post', linewidth=1.5)
    
    window = max(1, len(data) // 20)
    avg = np.convolve(data[:, 1], np.ones(window)/window, mode='same')
    plt.plot(data[:, 0], avg, color='#4b0082', linewidth=2, label=f'Moving avg')

    plt.title('Hydrogen bonds')
    plt.xlabel('Time (ps)')
    plt.ylabel('Number of H-bonds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_pca(pca_file, output='results_plot/PCA.png'):
    """Plot PCA projection (PC1 vs PC2)."""
    sets = read_xvg(pca_file, return_sets=True)
    if not sets or len(sets) < 2:
        print(f"Skipping PCA: {pca_file} needs at least 2 eigenvectors.")
        return

    pc1 = sets[0][:, 1]
    pc2 = sets[1][:, 1]
    
    min_len = min(len(pc1), len(pc2))
    pc1 = pc1[:min_len]
    pc2 = pc2[:min_len]

    plt.figure(figsize=(8, 8))
    hb = plt.hexbin(pc1, pc2, gridsize=50, cmap='magma', mincnt=1)
    cb = plt.colorbar(hb)
    cb.set_label('Count', fontsize=12)

    plt.title('PCA')
    plt.xlabel('PC1 (nm)')
    plt.ylabel('PC2 (nm)')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_dccm(matrix_file, output='results_plot/DCCM.png'):
    """Plot Dynamic Cross-Correlation Matrix (DCCM)."""
    data = parse_xpm(matrix_file)
    if data is None: 
        print(f"Skipping DCCM: {matrix_file} not found or invalid.")
        return

    vmax = np.max(data)
    vmin = np.min(data)
    norm_data = 2.0 * (data - vmin) / (vmax - vmin) - 1.0

    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_data, cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True)
    
    plt.title('DCCM')
    plt.xlabel('Residue index')
    plt.ylabel('Residue index')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_interaction_heatmap(hbmap_file, output='results_plot/Hbond_Map.png'):
    """Plot Hydrogen Bond Residency Heatmap."""
    data = parse_xpm(hbmap_file)
    if data is None: 
        print(f"Skipping H-Bond Heatmap: {hbmap_file} not found or invalid.")
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(data, cmap='Blues', cbar_kws={'label': 'Existence'})
    
    plt.title('H-bond map')
    plt.xlabel('Frame')
    plt.ylabel('H-bond index')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_rmsd_matrix(matrix_file, output='results_plot/RMSD_Matrix.png'):
    """Plot 2D RMSD Matrix (All-vs-All)."""
    data = parse_xpm(matrix_file)
    if data is None: return

    plt.figure(figsize=(10, 9))
    sns.heatmap(data, cmap='magma', cbar_kws={'label': 'RMSD (nm)'})
    
    plt.title('RMSD matrix')
    plt.xlabel('Frame index')
    plt.ylabel('Frame index')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_clusters(log_file, output='results_plot/Clusters.png'):
    """Parses cluster.log and plots population distribution."""
    if not os.path.exists(log_file): return
    
    clusters = []
    with open(log_file, 'r') as f:
        found_section = False
        for line in f:
            if 'cl.' in line and '|' in line:
                found_section = True
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        size = int(parts[2].strip())
                        clusters.append(size)
                    except: continue
            elif found_section and not line.strip():
                break
    
    if not clusters: return
    
    clusters = clusters[:10]
    labels = [f'C{i+1}' for i in range(len(clusters))]
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(clusters))
    bars = plt.bar(labels, clusters, color=colors)
    
    total = sum(clusters)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{(height/total)*100:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.title('Cluster population')
    plt.xlabel('Cluster')
    plt.ylabel('Number of structures')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def create_summary_dashboard(output='results_plot/Summary.png'):
    """Assembles a 4-panel summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSD
    data = read_xvg('rmsd.xvg')
    if data is not None:
        axes[0,0].plot(data[:, 0], data[:, 1], color='#1f77b4', linewidth=1.5)
        axes[0,0].set_title('(A) RMSD', loc='left')
        axes[0,0].set_ylabel('RMSD (nm)')
    
    # RMSF
    data = read_xvg('rmsf.xvg')
    if data is not None:
        axes[0,1].plot(data[:, 0], data[:, 1], color='#d62728', linewidth=1.5)
        axes[0,1].set_title('(B) RMSF', loc='left')
        axes[0,1].set_ylabel('RMSF (nm)')

    # H-Bonds
    data = read_xvg('hbond.xvg')
    if data is not None:
        axes[1,0].step(data[:, 0], data[:, 1], color='#9467bd', alpha=0.5)
        window = max(1, len(data) // 20)
        avg = np.convolve(data[:, 1], np.ones(window)/window, mode='same')
        axes[1,0].plot(data[:, 0], avg, color='#4b0082', linewidth=2)
        axes[1,0].set_title('(C) H-bonds', loc='left')
        axes[1,0].set_ylabel('Count')

    # Radius of Gyration
    data = read_xvg('gyrate.xvg')
    if data is not None:
        axes[1,1].plot(data[:, 0], data[:, 1], color='#2ca02c', linewidth=1.5)
        axes[1,1].set_title('(D) Rg', loc='left')
        axes[1,1].set_ylabel('Rg (nm)')

    for ax in axes.flat:
        ax.set_xlabel('Time (ps)')
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Saved: {output}")

def plot_pca_fes(pca_file, output='results_plot/PCA_FES.png'):
    """Plot Free Energy Surface projected onto PCA space (PC1 vs PC2)."""
    sets = read_xvg(pca_file, return_sets=True)
    if not sets or len(sets) < 2: return

    pc1 = sets[0][:, 1]
    pc2 = sets[1][:, 1]
    
    min_len = min(len(pc1), len(pc2))
    pc1, pc2 = pc1[:min_len], pc2[:min_len]
    
    xy = np.vstack([pc1, pc2])
    kde = gaussian_kde(xy)
    
    nbins = 100
    xi, yi = np.mgrid[pc1.min():pc1.max():nbins*1j, pc2.min():pc2.max():nbins*1j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    R = 0.008314
    T = 300
    fes = -R * T * np.log(zi / zi.max())
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, fes, levels=20, cmap='viridis_r')
    plt.colorbar(label='$\Delta$G (kJ/mol)')
    
    plt.title('FES on PCA subspace')
    plt.xlabel('PC1 (nm)')
    plt.ylabel('PC2 (nm)')
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Saved: {output}")

def plot_hydration_rdf(rdf_file, output='results_plot/RDF.png'):
    """Plot Radial Distribution Function."""
    data = read_xvg(rdf_file)
    if data is None: return

    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1], color='#17becf', linewidth=2)
    plt.fill_between(data[:, 0], data[:, 1], color='#17becf', alpha=0.1)
    
    plt.title('Radial distribution function')
    plt.xlabel('r (nm)')
    plt.ylabel('g(r)')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_residue_sasa(res_sasa_file, output='results_plot/SASA_Residue.png'):
    """Plot Residue-wise SASA."""
    data = read_xvg(res_sasa_file)
    if data is None: return

    plt.figure(figsize=(15, 6))
    plt.bar(data[:, 0], data[:, 1], color='#ff7f0e', alpha=0.7)
    
    plt.title('SASA per residue')
    plt.xlabel('Residue number')
    plt.ylabel('SASA (nm$^2$)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_prot_lig_dist_map(matrix_file, output='results_plot/Distance_Map.png'):
    """Plot Mean Distance Matrix between Protein and Ligand."""
    data = parse_xpm(matrix_file)
    if data is None: return

    plt.figure(figsize=(12, 10))
    sns.heatmap(data, cmap='viridis_r', cbar_kws={'label': 'Distance (nm)'})
    
    plt.title('Protein-ligand distance map')
    plt.xlabel('Ligand atom index')
    plt.ylabel('Protein residue index')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_ramachandran(rama_file, output='results_plot/Ramachandran.png'):
    """Plot Ramachandran map (Phi vs Psi)."""
    data = read_xvg(rama_file)
    if data is None: return

    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], s=5, c='#1f77b4', alpha=0.5)
    
    plt.title('Ramachandran plot')
    plt.xlabel('$\Phi$ (deg)')
    plt.ylabel('$\Psi$ (deg)')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    plt.axvline(0, color='black', alpha=0.3, linestyle='--')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Saved: {output}")

def plot_pocket_sasa(pocket_file, output='results_plot/Pocket_SASA.png'):
    """Plot binding pocket SASA over time."""
    data = read_xvg(pocket_file)
    if data is None: return

    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1], color='#e377c2', linewidth=2)
    plt.fill_between(data[:, 0], data[:, 1], color='#e377c2', alpha=0.1)
    
    plt.title('Pocket SASA')
    plt.xlabel('Time (ps)')
    plt.ylabel('SASA (nm$^2$)')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")

def plot_protein_network(matrix_file, output='results_plot/Residue_Distance_Matrix.png'):
    """Plot full protein residue-residue distance matrix."""
    data = parse_xpm(matrix_file)
    if data is None: return

    plt.figure(figsize=(12, 10))
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Distance (nm)'})
    
    plt.title('Residue-residue distance matrix')
    plt.xlabel('Residue index')
    plt.ylabel('Residue index')
    plt.tight_layout()
    plt.savefig(output, dpi=600)
    print(f"Saved: {output}")

def plot_transition_path(cluster_log, output='results_plot/Cluster_Transition.png'):
    """Visualize structural transition between clusters over time."""
    if not os.path.exists(cluster_log): return
    
    times = []
    cluster_ids = []
    with open(cluster_log, 'r') as f:
        read_data = False
        for line in f:
            if 'Frame' in line and 'Cluster' in line:
                read_data = True
                continue
            if read_data and line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        times.append(float(parts[0]))
                        cluster_ids.append(int(parts[1]))
                    except: pass

    if not times: return

    plt.figure(figsize=(12, 4))
    plt.scatter(times, cluster_ids, s=10, c=cluster_ids, cmap='tab10', alpha=0.6)
    plt.plot(times, cluster_ids, alpha=0.2, color='black', linewidth=0.5)
    
    plt.title('Cluster transition')
    plt.xlabel('Frame index')
    plt.ylabel('Cluster ID')
    plt.yticks(range(min(cluster_ids), max(cluster_ids) + 1))
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


# ── New extended analyses ──────────────────────────────────────────────────────

def plot_rmsd(rmsd_file, output='results_plot/RMSD.png'):
    """Plot RMSD time series with moving average."""
    data = read_xvg(rmsd_file)
    if data is None: return

    plt.figure(figsize=(12, 5))
    t, r = data[:, 0], data[:, 1]
    plt.plot(t, r, color='#1f77b4', linewidth=1, alpha=0.6)
    window = max(1, len(t) // 50)
    avg = np.convolve(r, np.ones(window)/window, mode='same')
    plt.plot(t, avg, color='#d62728', linewidth=2, label='Moving avg')
    plt.axhline(np.mean(r), color='black', linestyle='--', linewidth=1, alpha=0.5, label=f'Mean = {np.mean(r):.3f} nm')

    plt.title('RMSD')
    plt.xlabel('Time (ns)')
    plt.ylabel('RMSD (nm)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


def plot_thermodynamics(temp_file='temperature.xvg', press_file='pressure.xvg',
                        dens_file='density.xvg', pot_file='potential_energy.xvg',
                        output='results_plot/Thermodynamics.png'):
    """4-panel thermodynamic convergence plot."""
    files = [
        (temp_file,  'Temperature (K)',   '#e377c2'),
        (press_file, 'Pressure (bar)',    '#17becf'),
        (dens_file,  'Density (kg/m$^3$)','#2ca02c'),
        (pot_file,   'Potential (kJ/mol)','#ff7f0e'),
    ]
    datasets = [(read_xvg(f), lbl, col) for f, lbl, col in files]
    available = [(d, lbl, col) for d, lbl, col in datasets if d is not None]
    if not available: return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)
    if n == 1: axes = [axes]

    for ax, (d, lbl, col) in zip(axes, available):
        ax.plot(d[:, 0], d[:, 1], color=col, linewidth=1)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.2)
        # running mean line
        w = max(1, len(d) // 50)
        avg = np.convolve(d[:, 1], np.ones(w)/w, mode='same')
        ax.plot(d[:, 0], avg, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

    axes[-1].set_xlabel('Time (ps)')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


def plot_dssp(ss_xpm='ss.xpm', scount_file='scount.xvg',
             output='results_plot/DSSP.png'):
    """Plot secondary structure evolution as heatmap + fraction time series."""
    # --- heatmap from xpm ---
    mat = parse_xpm(ss_xpm)
    sc  = read_xvg(scount_file)
    if mat is None and sc is None:
        print(f"Skipping DSSP: {ss_xpm} and {scount_file} missing.")
        return

    if mat is not None and sc is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                        gridspec_kw={'height_ratios': [3, 1]})
    elif mat is not None:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = None
    else:
        fig, ax2 = plt.subplots(figsize=(12, 4))
        ax1 = None

    if ax1 is not None:
        ax1.imshow(mat, aspect='auto', cmap='tab10', origin='upper')
        ax1.set_xlabel('Frame index')
        ax1.set_ylabel('Residue index')
        ax1.set_title('Secondary structure')

    if ax2 is not None:
        for i in range(1, sc.shape[1]):
            ax2.plot(sc[:, 0], sc[:, i], linewidth=1.2, alpha=0.8)
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Residue count')
        ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


def plot_contact_map(contact_xpm='contacts.xpm',
                    output='results_plot/Contact_Map.png'):
    """Plot residue-residue contact frequency (fraction of frames)."""
    data = parse_xpm(contact_xpm)
    if data is None:
        print(f"Skipping contact map: {contact_xpm} not found.")
        return

    # Normalise 0-1
    d_max = np.max(data)
    norm = data / d_max if d_max > 0 else data

    plt.figure(figsize=(10, 9))
    sns.heatmap(norm, cmap='Blues_r', square=True,
                cbar_kws={'label': 'Contact frequency'})
    plt.title('Residue contact map')
    plt.xlabel('Residue index')
    plt.ylabel('Residue index')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


def plot_eigenvalues(eigenval_file='eigenval.xvg',
                    output='results_plot/PCA_Eigenvalues.png'):
    """PCA scree plot — variance explained per eigenvector."""
    data = read_xvg(eigenval_file)
    if data is None: return

    # eigenval.xvg: col0 = index, col1 = eigenvalue (nm^2)
    idx = data[:, 0].astype(int)
    ev  = data[:, 1]
    pct = ev / ev.sum() * 100
    cumsum = np.cumsum(pct)

    n = min(20, len(idx))  # show top 20
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(idx[:n], pct[:n], color='#1f77b4', alpha=0.7, label='Individual')
    ax1.set_xlabel('Eigenvector')
    ax1.set_ylabel('Variance (%)')
    ax2 = ax1.twinx()
    ax2.plot(idx[:n], cumsum[:n], color='#d62728', marker='o',
             markersize=4, linewidth=2, label='Cumulative')
    ax2.set_ylabel('Cumulative variance (%)')
    ax2.axhline(90, color='grey', linestyle='--', linewidth=1, alpha=0.6)
    plt.title('PCA eigenvalues')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88))
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


def plot_hbond_autocorr(hbac_file='hbac.xvg',
                        output='results_plot/HBond_Autocorr.png'):
    """Plot H-bond autocorrelation function (lifetime indicator)."""
    data = read_xvg(hbac_file)
    if data is None: return

    plt.figure(figsize=(10, 5))
    t = data[:, 0]
    # hbac.xvg has multiple columns (continuous, intermittent, ...)
    labels = ['Continuous', 'Intermittent']
    colors = ['#1f77b4', '#ff7f0e']
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        if data.shape[1] > i + 1:
            plt.plot(t, data[:, i + 1], label=lbl, color=col, linewidth=1.5)

    plt.title('H-bond autocorrelation')
    plt.xlabel('Time (ps)')
    plt.ylabel('C(t)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


def plot_msd(msd_file='msd.xvg', output='results_plot/MSD.png'):
    """Plot Mean Square Displacement."""
    data = read_xvg(msd_file)
    if data is None: return

    plt.figure(figsize=(10, 6))
    t = data[:, 0]
    msd = data[:, 1]
    plt.plot(t, msd, color='#2ca02c', linewidth=2)

    # Estimate diffusion coefficient D from linear region (last 60%)
    n60 = int(len(t) * 0.4)
    if n60 > 2:
        coeffs = np.polyfit(t[n60:], msd[n60:], 1)
        D = coeffs[0] / 6.0  # 3D: MSD = 6Dt
        plt.plot(t[n60:], np.polyval(coeffs, t[n60:]),
                 '--', color='black', linewidth=1.5,
                 label=f'D ≈ {D:.3e} nm²/ps')
        plt.legend()

    plt.title('Mean square displacement')
    plt.xlabel('Time (ps)')
    plt.ylabel('MSD (nm$^2$)')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


def plot_saltbridges(saltbr_pattern='sb', output='results_plot/Salt_Bridges.png'):
    """Plot salt bridge distances from gmx saltbr output files."""
    import glob
    files = sorted(glob.glob(f'{saltbr_pattern}*.xvg'))
    if not files:
        print(f"Skipping salt bridges: no files matching '{saltbr_pattern}*.xvg'")
        return

    plt.figure(figsize=(12, 5))
    cmap = plt.get_cmap('tab10')
    for i, f in enumerate(files[:10]):
        d = read_xvg(f)
        if d is None: continue
        label = os.path.splitext(os.path.basename(f))[0]
        plt.plot(d[:, 0], d[:, 1], linewidth=1.2,
                 color=cmap(i % 10), alpha=0.8, label=label)

    plt.axhline(0.4, color='red', linestyle='--', linewidth=1,
                alpha=0.6, label='Cutoff 0.4 nm')
    plt.title('Salt bridge distances')
    plt.xlabel('Time (ps)')
    plt.ylabel('Distance (nm)')
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


def plot_cluster_rmsd_dist(dist_file='cluster_dist.xvg',
                           output='results_plot/Cluster_RMSD_Dist.png'):
    """Plot RMSD distribution histogram from clustering."""
    data = read_xvg(dist_file)
    if data is None: return

    plt.figure(figsize=(8, 5))
    rmsd_vals = data[:, 1] if data.shape[1] > 1 else data[:, 0]
    plt.hist(rmsd_vals, bins=40, color='#9467bd', edgecolor='white',
             linewidth=0.5, alpha=0.85)
    plt.axvline(np.mean(rmsd_vals), color='black', linestyle='--',
                linewidth=1.5, label=f'Mean = {np.mean(rmsd_vals):.3f} nm')
    plt.title('Pairwise RMSD distribution')
    plt.xlabel('RMSD (nm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


def plot_dihedral_dist(dih_file='dihedral.xvg',
                       output='results_plot/Dihedral_Dist.png'):
    """Plot dihedral angle distribution."""
    data = read_xvg(dih_file)
    if data is None: return

    angles = data[:, 1:].flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(angles, bins=72, range=(-180, 180), color='#8c564b',
             edgecolor='white', linewidth=0.4, alpha=0.85, density=True)
    plt.title('Dihedral angle distribution')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Probability density')
    plt.xlim(-180, 180)
    plt.xticks(range(-180, 181, 60))
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved: {output}")


def plot_equilibration(stage='nvt',
                       output_prefix='results_plot'):
    """Plot NVT or NPT equilibration convergence (T, P, potential)."""
    panels = [
        (f'{stage}_temperature.xvg', 'Temperature (K)',    '#e377c2'),
        (f'{stage}_pressure.xvg',    'Pressure (bar)',     '#17becf'),
        (f'{stage}_potential.xvg',   'Potential (kJ/mol)', '#ff7f0e'),
    ]
    available = [(read_xvg(f), lbl, col) for f, lbl, col in panels
                 if os.path.exists(f) and read_xvg(f) is not None]
    if not available: return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)
    if n == 1: axes = [axes]

    for ax, (d, lbl, col) in zip(axes, available):
        ax.plot(d[:, 0], d[:, 1], color=col, linewidth=1)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time (ps)')
    plt.suptitle(stage.upper() + ' equilibration', y=1.01)
    plt.tight_layout()
    out = f'{output_prefix}/{stage.upper()}_Equilibration.png'
    plt.savefig(out)
    print(f"Saved: {out}")

if __name__ == "__main__":
    
    output_dir = 'results_plot'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # 1. Structural stability
    plot_rmsd('rmsd_backbone.xvg',     output=f'{output_dir}/RMSD.png')
    plot_rmsf_perix('rmsf.xvg',        output=f'{output_dir}/RMSF.png')
    plot_rmsd_matrix('rmsd_matrix.xpm',output=f'{output_dir}/RMSD_Matrix.png')
    
    # 2. Energy landscape
    plot_fes_2d('rmsd_backbone.xvg', 'gyrate.xvg', output=f'{output_dir}/FES_2D.png')
    plot_fes_3d('rmsd_backbone.xvg', 'gyrate.xvg', output=f'{output_dir}/FES_3D.png')
    
    # 3. Interactions & surface
    plot_sasa('sasa.xvg',              output=f'{output_dir}/SASA.png')
    plot_hbond('hbond.xvg',            output=f'{output_dir}/HBond.png')
    plot_rg_anisotropy('gyrate.xvg',   output=f'{output_dir}/Rg.png')
    
    # 4. Essential dynamics
    plot_pca('proj.xvg',               output=f'{output_dir}/PCA.png')
    plot_dccm('dccm.xpm',              output=f'{output_dir}/DCCM.png')
    plot_eigenvalues('eigenval.xvg',   output=f'{output_dir}/PCA_Eigenvalues.png')
    plot_pca_fes('proj.xvg',           output=f'{output_dir}/PCA_FES.png')
    
    # 5. Interaction mapping
    plot_interaction_heatmap('hbond_matrix.xpm',    output=f'{output_dir}/Hbond_Map.png')
    plot_hbond_autocorr('hbac.xvg',                 output=f'{output_dir}/HBond_Autocorr.png')
    plot_contact_map('contacts.xpm',                output=f'{output_dir}/Contact_Map.png')
    
    # 6. Cluster analysis
    plot_clusters('cluster.log',             output=f'{output_dir}/Clusters.png')
    plot_cluster_rmsd_dist('cluster_dist.xvg', output=f'{output_dir}/Cluster_RMSD_Dist.png')
    create_summary_dashboard(            output=f'{output_dir}/Summary.png')

    # 7. Additional structure analyses
    plot_ramachandran('rama.xvg',            output=f'{output_dir}/Ramachandran.png')
    plot_dssp('ss.xpm', 'scount.xvg',       output=f'{output_dir}/DSSP.png')
    plot_dihedral_dist('dihedral.xvg',       output=f'{output_dir}/Dihedral_Dist.png')
    plot_hydration_rdf('hydration_rdf.xvg',  output=f'{output_dir}/RDF.png')
    plot_residue_sasa('residue_sasa.xvg',    output=f'{output_dir}/SASA_Residue.png')
    plot_prot_lig_dist_map('prot_lig_map.xpm', output=f'{output_dir}/Distance_Map.png')
    plot_pocket_sasa('pocket_sasa.xvg',      output=f'{output_dir}/Pocket_SASA.png')
    
    # 8. Protein network
    plot_protein_network('protein_matrix.xpm', output=f'{output_dir}/Residue_Distance_Matrix.png')
    plot_transition_path('cluster.log',      output=f'{output_dir}/Cluster_Transition.png')
    plot_saltbridges('sb',                   output=f'{output_dir}/Salt_Bridges.png')
    plot_msd('msd.xvg',                      output=f'{output_dir}/MSD.png')
    
    # 9. Thermodynamics (production MD)
    plot_thermodynamics(
        temp_file='temperature.xvg',
        press_file='pressure.xvg',
        dens_file='density.xvg',
        pot_file='potential_energy.xvg',
        output=f'{output_dir}/Thermodynamics.png'
    )
    
    # 10. Equilibration convergence (NVT & NPT)
    plot_equilibration(stage='nvt', output_prefix=output_dir)
    plot_equilibration(stage='npt', output_prefix=output_dir)

    print(f"\nDone. Results saved to: {output_dir}")
