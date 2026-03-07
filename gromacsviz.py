import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os

# Professional Styling for Q1 Journals
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
                # Handle gmx rama output which often has residue labels like "ALA-12 120 -30"
                # We only want the numeric parts (Phi and Psi)
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

    # Find where the data starts (usually after the header and color map)
    # The header looks like: "width height ncolors cpp"
    data_lines = [l.strip().strip('",') for l in lines if l.startswith('"') and not l.startswith('" ')]
    
    if not data_lines: return None
    
    # First line of data section is the dimensions: "width height ncolors cpp"
    header = data_lines[0].split()
    if len(header) < 4: return None
    
    width, height, ncolors, cpp = map(int, header)
    
    # Color map
    color_map = {}
    for i in range(1, ncolors + 1):
        line = data_lines[i]
        symbol = line[:cpp]
        # Usually: "symbol c #HEX_COLOR /* "label" */"
        if 'c #' in line:
            color_hex = line.split('c #')[1].split()[0]
            # Convert hex to a scalar (we'll normalize later)
            color_map[symbol] = i
            
    # Matrix data starts after ncolors lines
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

    # Sync data length
    min_len = min(len(rmsd), len(rg))
    x = rmsd[:min_len, 1]
    y = rg[:min_len, 1]

    # KDE for probability density
    k = gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Free Energy G = -kT ln(P)
    # T=300K, R=0.008314 kJ/molK
    kT = 0.008314 * 300
    fes = -kT * np.log(zi.reshape(xi.shape))
    fes -= np.min(fes) # Normalize to 0

    plt.figure(figsize=(10, 8))
    cp = plt.contourf(xi, yi, fes, 20, cmap='viridis_r')
    cbar = plt.colorbar(cp)
    cbar.set_label('Relative Free Energy (kJ/mol)', fontsize=14)
    
    plt.title('Free Energy Surface (FES)', fontweight='bold')
    plt.xlabel('RMSD (nm)', fontweight='bold')
    plt.ylabel('Radius of Gyration (nm)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready FES saved to {output}")

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
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Relative Free Energy (kJ/mol)')
    
    ax.set_title('3D Free Energy Surface', fontweight='bold', fontsize=18)
    ax.set_xlabel('RMSD (nm)', fontweight='bold')
    ax.set_ylabel('Radius of Gyration (nm)', fontweight='bold')
    ax.set_zlabel('Energy (kJ/mol)', fontweight='bold')
    
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready 3D FES saved to {output}")

def plot_rmsf_perix(rmsf_file, output='RMSF_Publication.png'):
    """Plot RMSF with residue highlighting."""
    data = read_xvg(rmsf_file)
    if data is None: return

    plt.figure(figsize=(12, 5))
    plt.plot(data[:, 0], data[:, 1], color='#d62728', linewidth=2)
    plt.fill_between(data[:, 0], data[:, 1], color='#d62728', alpha=0.2)
    
    plt.title('Residue-wise Fluctuation (RMSF)', fontweight='bold')
    plt.xlabel('Residue Number', fontweight='bold')
    plt.ylabel('RMSF (nm)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready RMSF saved to {output}")

def plot_sasa(sasa_file, output='results_plot/SASA_Publication.png'):
    """Plot Solvent Accessible Surface Area (SASA)."""
    data = read_xvg(sasa_file)
    if data is None: return

    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1], color='#2ca02c', linewidth=2, label='Total SASA')
    plt.fill_between(data[:, 0], data[:, 1], color='#2ca02c', alpha=0.1)
    
    plt.title('Solvent Accessible Surface Area (SASA)', fontweight='bold')
    plt.xlabel('Time (ps)', fontweight='bold')
    plt.ylabel('Surface Area (nm$^2$)', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready SASA saved to {output}")

def plot_rg_anisotropy(rg_file, output='results_plot/Rg_Anisotropy.png'):
    """Plot Radius of Gyration components to show shape anisotropy."""
    data = read_xvg(rg_file)
    if data is None or data.shape[1] < 5: return

    plt.figure(figsize=(10, 6))
    time = data[:, 0]
    plt.plot(time, data[:, 1], label='Total Rg', color='black', linewidth=2)
    plt.plot(time, data[:, 2], label='Rg$_x$', alpha=0.7)
    plt.plot(time, data[:, 3], label='Rg$_y$', alpha=0.7)
    plt.plot(time, data[:, 4], label='Rg$_z$', alpha=0.7)
    
    plt.title('Radius of Gyration: Shape Anisotropy', fontweight='bold')
    plt.xlabel('Time (ps)', fontweight='bold')
    plt.ylabel('Rg (nm)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready Rg Anisotropy saved to {output}")

def plot_hbond(hbond_file, output='HBond_Publication.png'):
    """Plot Hydrogen Bond Number over time."""
    data = read_xvg(hbond_file)
    if data is None: return

    plt.figure(figsize=(10, 5))
    plt.step(data[:, 0], data[:, 1], color='#9467bd', where='post', linewidth=1.5)
    
    # Moving average to show trend
    window = max(1, len(data) // 20)
    avg = np.convolve(data[:, 1], np.ones(window)/window, mode='same')
    plt.plot(data[:, 0], avg, color='#4b0082', linewidth=2, label=f'Avg (w={window})')

    plt.title('Protein-Ligand Hydrogen Bonds', fontweight='bold')
    plt.xlabel('Time (ps)', fontweight='bold')
    plt.ylabel('Number of H-Bonds', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready H-Bond plot saved to {output}")

def plot_pca(pca_file, output='results_plot/PCA_Publication.png'):
    """Plot PCA projection (PC1 vs PC2)."""
    sets = read_xvg(pca_file, return_sets=True)
    if not sets or len(sets) < 2:
        print(f"Skipping PCA: {pca_file} needs at least 2 eigenvectors.")
        return

    # PC1 is from set 0, PC2 is from set 1
    # Column 0 is time, Column 1 is projection
    pc1 = sets[0][:, 1]
    pc2 = sets[1][:, 1]
    
    # Ensure they have the same length
    min_len = min(len(pc1), len(pc2))
    pc1 = pc1[:min_len]
    pc2 = pc2[:min_len]

    plt.figure(figsize=(8, 8))
    # Use a density scatter or hexbin for large datasets
    hb = plt.hexbin(pc1, pc2, gridsize=50, cmap='magma', mincnt=1)
    cb = plt.colorbar(hb)
    cb.set_label('Density of Conformations', fontsize=12)

    plt.title('Principal Component Analysis (PCA)', fontweight='bold')
    plt.xlabel('Principal Component 1 (nm)', fontweight='bold')
    plt.ylabel('Principal Component 2 (nm)', fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready PCA plot saved to {output}")

def plot_dccm(matrix_file, output='results_plot/DCCM_Publication.png'):
    """Plot Dynamic Cross-Correlation Matrix (DCCM)."""
    data = parse_xpm(matrix_file)
    if data is None: 
        print(f"Skipping DCCM: {matrix_file} not found or invalid.")
        return

    # Normalize data to [-1, 1] range for correlation
    # GROMACS -xpma usually scales colors from -1 to 1
    vmax = np.max(data)
    vmin = np.min(data)
    norm_data = 2.0 * (data - vmin) / (vmax - vmin) - 1.0

    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_data, cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True)
    
    plt.title('Dynamic Cross-Correlation Map (DCCM)', fontweight='bold')
    plt.xlabel('Residue Index', fontweight='bold')
    plt.ylabel('Residue Index', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready DCCM saved to {output}")

def plot_interaction_heatmap(hbmap_file, output='results_plot/Interaction_Heatmap.png'):
    """Plot Hydrogen Bond Residency Heatmap."""
    data = parse_xpm(hbmap_file)
    if data is None: 
        print(f"Skipping H-Bond Heatmap: {hbmap_file} not found or invalid.")
        return

    plt.figure(figsize=(12, 8))
    # Simple binary/scaled heatmap for interactions
    sns.heatmap(data, cmap='Blues', cbar_kws={'label': 'Interaction Existence'})
    
    plt.title('Protein-Ligand Interaction Persistence', fontweight='bold')
    plt.xlabel('Time/Frame', fontweight='bold')
    plt.ylabel('H-Bond Index / Residue Pair', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready Interaction Heatmap saved to {output}")

def plot_rmsd_matrix(matrix_file, output='results_plot/RMSD_Matrix.png'):
    """Plot 2D RMSD Matrix (All-vs-All)."""
    data = parse_xpm(matrix_file)
    if data is None: return

    plt.figure(figsize=(10, 9))
    # Reverse viridis or different map to show transitions
    sns.heatmap(data, cmap='magma', cbar_kws={'label': 'RMSD (nm)'})
    
    plt.title('2D All-vs-All RMSD Matrix', fontweight='bold')
    plt.xlabel('Frame Index', fontweight='bold')
    plt.ylabel('Frame Index', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready RMSD Matrix saved to {output}")

def plot_clusters(log_file, output='results_plot/Cluster_Distribution.png'):
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
    
    # Take top 10 clusters
    clusters = clusters[:10]
    labels = [f'C{i+1}' for i in range(len(clusters))]
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(clusters))
    bars = plt.bar(labels, clusters, color=colors)
    
    # Add percentages
    total = sum(clusters)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{(height/total)*100:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.title('Structural Cluster Population (Top 10)', fontweight='bold')
    plt.xlabel('Cluster ID', fontweight='bold')
    plt.ylabel('Number of Structures', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Publication-ready Cluster Plot saved to {output}")

def create_summary_dashboard(output='results_plot/Figure1_Master_Panel.png'):
    """Assembles a high-impact 4-panel summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSD
    data = read_xvg('rmsd.xvg')
    if data is not None:
        axes[0,0].plot(data[:, 0], data[:, 1], color='#1f77b4', linewidth=1.5)
        axes[0,0].set_title('A. Structural Stability (RMSD)', loc='left', fontweight='bold')
        axes[0,0].set_ylabel('RMSD (nm)')
    
    # RMSF
    data = read_xvg('rmsf.xvg')
    if data is not None:
        axes[0,1].plot(data[:, 0], data[:, 1], color='#d62728', linewidth=1.5)
        axes[0,1].set_title('B. Local Flexibility (RMSF)', loc='left', fontweight='bold')
        axes[0,1].set_ylabel('RMSF (nm)')

    # H-Bonds
    data = read_xvg('hbond.xvg')
    if data is not None:
        axes[1,0].step(data[:, 0], data[:, 1], color='#9467bd', alpha=0.5)
        window = max(1, len(data) // 20)
        avg = np.convolve(data[:, 1], np.ones(window)/window, mode='same')
        axes[1,0].plot(data[:, 0], avg, color='#4b0082', linewidth=2)
        axes[1,0].set_title('C. Binding Interactions (H-Bonds)', loc='left', fontweight='bold')
        axes[1,0].set_ylabel('Count')

    # Radius of Gyration
    data = read_xvg('gyrate.xvg')
    if data is not None:
        axes[1,1].plot(data[:, 0], data[:, 1], color='#2ca02c', linewidth=1.5)
        axes[1,1].set_title('D. Protein Compactness (Rg)', loc='left', fontweight='bold')
        axes[1,1].set_ylabel('Rg (nm)')

    for ax in axes.flat:
        ax.set_xlabel('Time / Index')
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('Molecular Dynamics Simulation Summary Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output, dpi=400)
    print(f"Summary Dashboard saved to {output}")

def plot_pca_fes(pca_file, output='results_plot/PCA_FES_2D.png'):
    """Plot Free Energy Surface projected onto PCA space (PC1 vs PC2)."""
    sets = read_xvg(pca_file, return_sets=True)
    if not sets or len(sets) < 2: return

    pc1 = sets[0][:, 1]
    pc2 = sets[1][:, 1]
    
    # Calculate FES
    min_len = min(len(pc1), len(pc2))
    pc1, pc2 = pc1[:min_len], pc2[:min_len]
    
    # Use gaussian KDE to estimate density
    xy = np.vstack([pc1, pc2])
    kde = gaussian_kde(xy)
    
    # Create grid
    nbins = 100
    xi, yi = np.mgrid[pc1.min():pc1.max():nbins*1j, pc2.min():pc2.max():nbins*1j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    # Convert density to free energy (kJ/mol)
    # G = -RT ln(P/Pmax)
    R = 0.008314
    T = 300
    fes = -R * T * np.log(zi / zi.max())
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, fes, levels=20, cmap='viridis_r')
    plt.colorbar(label='Free Energy (kJ/mol)')
    
    plt.title('Free Energy Surface on PCA Subspace', fontweight='bold')
    plt.xlabel('Principal Component 1 (Projection)', fontweight='bold')
    plt.ylabel('Principal Component 2 (Projection)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Post-Processing: PCA-FES Plot saved to {output}")

def plot_hydration_rdf(rdf_file, output='results_plot/Hydration_RDF.png'):
    """Plot Radial Distribution Function for Ligand Hydration."""
    data = read_xvg(rdf_file)
    if data is None: return

    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1], color='#17becf', linewidth=2)
    plt.fill_between(data[:, 0], data[:, 1], color='#17becf', alpha=0.1)
    
    plt.title('Ligand Hydration: Radial Distribution Function (RDF)', fontweight='bold')
    plt.xlabel('Distance r (nm)', fontweight='bold')
    plt.ylabel('g(r)', fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Post-Processing: Hydration RDF saved to {output}")

def plot_residue_sasa(res_sasa_file, output='results_plot/Residue_SASA.png'):
    """Plot Residue-wise SASA for structural integrity analysis."""
    data = read_xvg(res_sasa_file)
    if data is None: return

    plt.figure(figsize=(15, 6))
    # data[:,0] is residue index, data[:,1] is average SASA over trajectory
    plt.bar(data[:, 0], data[:, 1], color='#ff7f0e', alpha=0.7)
    
    plt.title('Residue-wise Solvent Accessibility (Average)', fontweight='bold')
    plt.xlabel('Residue Number', fontweight='bold')
    plt.ylabel('Average SASA (nm$^2$)', fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Post-Processing: Residue SASA Plot saved to {output}")

def plot_prot_lig_dist_map(matrix_file, output='results_plot/Protein_Ligand_Distance_Map.png'):
    """Plot Mean Distance Matrix between Protein and Ligand."""
    data = parse_xpm(matrix_file)
    if data is None: return

    plt.figure(figsize=(12, 10))
    # Reverse viridis to show small distances (contacts) in bright colors
    sns.heatmap(data, cmap='viridis_r', cbar_kws={'label': 'Mean Distance (nm)'})
    
    plt.title('Protein-Ligand Mean Distance Map', fontweight='bold')
    plt.xlabel('Ligand Atom / Residue Index', fontweight='bold')
    plt.ylabel('Protein Residue Index', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Post-Processing: Protein-Ligand Distance Map saved to {output}")

def plot_ramachandran(rama_file, output='results_plot/Ramachandran_Map.png'):
    """Plot Ramachandran map (Phi vs Psi) to show structural quality."""
    data = read_xvg(rama_file)
    if data is None: return

    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], s=5, c='#1f77b4', alpha=0.5)
    
    plt.title('Ramachandran Map: Backbone Integrity', fontweight='bold')
    plt.xlabel('$\Phi$ (deg)', fontweight='bold')
    plt.ylabel('$\Psi$ (deg)', fontweight='bold')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    plt.axvline(0, color='black', alpha=0.3, linestyle='--')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Post-Processing: Ramachandran Map saved to {output}")

def plot_pocket_sasa(pocket_file, output='results_plot/Pocket_SASA_Evolution.png'):
    """Plot binding pocket exposure evolution."""
    data = read_xvg(pocket_file)
    if data is None: return

    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1], color='#e377c2', linewidth=2)
    plt.fill_between(data[:, 0], data[:, 1], color='#e377c2', alpha=0.1)
    
    plt.title('Binding Pocket Exposure: Breathing Evolution', fontweight='bold')
    plt.xlabel('Time (ps)', fontweight='bold')
    plt.ylabel('Exposure Area (nm$^2$)', fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Post-Processing: Pocket Evolution saved to {output}")

def plot_protein_network(matrix_file, output='results_plot/Protein_Network_Heatmap.png'):
    """Plot full protein residue-residue distance matrix for network analysis."""
    data = parse_xpm(matrix_file)
    if data is None: return

    plt.figure(figsize=(12, 10))
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Mean Distance (nm)'})
    
    plt.title('Protein Residue Network: All-to-All Distance', fontweight='bold')
    plt.xlabel('Residue Index', fontweight='bold')
    plt.ylabel('Residue Index', fontweight='bold')
    plt.tight_layout()
    # Use high DPI for detailed matrices
    plt.savefig(output, dpi=600)
    print(f"Post-Processing: Protein Network Matrix saved to {output}")

def plot_transition_path(cluster_log, output='results_plot/Structural_Transition_Path.png'):
    """Vizualize structural transition between dominant clusters over time."""
    # This is a bit complex as it requires parsing time-series cluster data
    # cluster.log usually contains frames and their cluster IDs
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
    
    plt.title('Structural Transition Path: Cluster State Evolution', fontweight='bold')
    plt.xlabel('Time / Frame Index', fontweight='bold')
    plt.ylabel('Cluster ID', fontweight='bold')
    plt.yticks(range(min(cluster_ids), max(cluster_ids) + 1))
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Post-Processing: Transition Path saved to {output}")

if __name__ == "__main__":
    
    # Create output directory
    output_dir = 'results_plot'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # 1. Structural Stability
    plot_rmsf_perix('rmsf.xvg', output=f'{output_dir}/RMSF_Publication.png')
    plot_rmsd_matrix('rmsd_matrix.xpm', output=f'{output_dir}/RMSD_Matrix.png')
    
    # 2. Energy Landscape
    plot_fes_2d('rmsd.xvg', 'gyrate.xvg', output=f'{output_dir}/FES_2D.png')
    plot_fes_3d('rmsd.xvg', 'gyrate.xvg', output=f'{output_dir}/FES_3D.png')
    
    # 3. Interactions & Surface
    plot_sasa('sasa.xvg', output=f'{output_dir}/SASA_Publication.png')
    plot_hbond('hbond.xvg', output=f'{output_dir}/HBond_Publication.png')
    plot_rg_anisotropy('gyrate.xvg', output=f'{output_dir}/Rg_Anisotropy.png')
    
    # 4. Essential Dynamics
    plot_pca('proj.xvg', output=f'{output_dir}/PCA_Publication.png')
    plot_dccm('dccm.xpm', output=f'{output_dir}/DCCM_Publication.png')
    
    # 5. Interaction Mapping
    plot_interaction_heatmap('hbond_matrix.xpm', output=f'{output_dir}/Interaction_Heatmap.png')
    
    # 6. Elite Plus High-Impact Visuals
    plot_clusters('cluster.log', output=f'{output_dir}/Cluster_Distribution.png')
    create_summary_dashboard(output=f'{output_dir}/Figure1_Master_Panel.png')

    # 7. Zenith Research Tier (Nature/Science Level)
    plot_pca_fes('proj.xvg', output=f'{output_dir}/PCA_FES_2D.png')
    plot_hydration_rdf('hydration_rdf.xvg', output=f'{output_dir}/Hydration_RDF.png')
    plot_residue_sasa('residue_sasa.xvg', output=f'{output_dir}/Residue_SASA_Stability.png')
    plot_prot_lig_dist_map('prot_lig_map.xpm', output=f'{output_dir}/Protein_Ligand_Distance_Map.png')

    # 8. Apex Research Tier (Highest Impact - Cell/Nature/Science)
    plot_ramachandran('rama.xvg', output=f'{output_dir}/Ramachandran_Integrity.png')
    plot_pocket_sasa('pocket_sasa.xvg', output=f'{output_dir}/Binding_Pocket_Breathing.png')
    plot_protein_network('protein_matrix.xpm', output=f'{output_dir}/Protein_Network_Matrix.png')
    plot_transition_path('cluster.log', output=f'{output_dir}/Cluster_Transition_Path.png')

    print(f"\n[ANALYSIS COMPLETE] Quantitative analytics available in: {output_dir}")
