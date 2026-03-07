"""
GROMACS Molecular Dynamics Simulation Workflow
Automated pipeline for Protein-Ligand and Protein-only systems.
Requirements: GROMACS, ACPYPE, OpenBabel.
"""

import os
import sys
import subprocess
import shutil
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict


def get_cpu_cores() -> int:
    """Get number of CPU cores available"""
    try:
        # For WSL/Linux
        return int(subprocess.check_output(['nproc']).decode().strip())
    except:
        return os.cpu_count() or 4


def check_gpu_support() -> bool:
    """Check if GROMACS binary has GPU support enabled"""
    try:
        result = subprocess.run("gmx -version", shell=True, capture_output=True, text=True)
        return "GPU support:        enabled" in result.stdout
    except:
        return False


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.upper().center(60)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_step(step_num: int, text: str):
    print(f"{Colors.CYAN}[STEP {step_num}]{Colors.END} {Colors.BOLD}{text}{Colors.END}")


def print_success(text: str):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.WARNING}[WARN] {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.FAIL}[ERROR] {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")


def run_cmd(cmd: str, input_text: Optional[str] = None, check: bool = True, 
            extra_paths: Optional[List[str]] = None, 
            extra_envs: Optional[Dict[str, str]] = None) -> bool:
    """Run shell command"""
    print(f"{Colors.BLUE}$ {cmd}{Colors.END}")
    
    # Setup environment
    my_env = os.environ.copy()
    if extra_paths:
        p_sep = ":" if os.name != "nt" else ";"
        my_env["PATH"] = p_sep.join(extra_paths) + p_sep + my_env.get("PATH", "")
    
    if extra_envs:
        for k, v in extra_envs.items():
            if k in my_env and (k.endswith("PATH") or k == "PYTHONPATH"):
                p_sep = ":" if os.name != "nt" else ";"
                my_env[k] = v + p_sep + my_env[k]
            else:
                my_env[k] = v

    try:
        result = subprocess.run(
            cmd, shell=True,
            input=input_text.encode() if input_text else None,
            capture_output=False,
            check=check,
            env=my_env
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def file_exists(path: str) -> bool:
    return os.path.exists(path)


class PDBHandler:
    """Handle PDB file operations"""
    
    PROTEIN_RESIDUES = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'ACE', 'NME', 'NMA', 'NH2', 'HID', 'HIE', 'HIP', 'HSE', 'HSD', 'HSP',
        'CYX', 'CYM', 'HOH', 'WAT', 'SOL', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN'
    }
    
    @staticmethod
    def analyze(pdb_file: str) -> Dict:
        """Analyze PDB to find protein chains and ligand"""
        protein_atoms = []
        ligand_atoms = []
        ligand_name = None
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res_name = line[17:20].strip()
                    if res_name.upper() in PDBHandler.PROTEIN_RESIDUES:
                        protein_atoms.append(line)
                    else:
                        ligand_atoms.append(line)
                        if ligand_name is None:
                            ligand_name = res_name
        
        return {
            'has_ligand': len(ligand_atoms) > 0,
            'ligand_name': ligand_name if ligand_name and ligand_name != '*' else 'LIG',
            'ligand_atoms': len(ligand_atoms),
            'protein_atoms': len(protein_atoms)
        }
    
    @staticmethod
    def extract_protein(pdb_file: str, output: str):
        """Extract only protein atoms"""
        lines = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res = line[17:20].strip()
                    if res.upper() in PDBHandler.PROTEIN_RESIDUES and res.upper() not in ('ACE', 'NME', 'NMA'):
                        lines.append(line)
                elif not line.startswith('CONECT'):
                    lines.append(line)
        with open(output, 'w') as f:
            f.writelines(lines)
    
    @staticmethod
    def extract_ligand(pdb_file: str, output: str, lig_name: str):
        """Extract ligand atoms"""
        lines = ["HEADER    LIGAND\n"]
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res = line[17:20].strip()
                    if res.upper() == lig_name.upper() or res == '*':
                        # Rename * to standardized lig_name
                        new_line = line[:17] + f"{lig_name:>3}" + line[20:]
                        lines.append(new_line)
        lines.append("END\n")
        with open(output, 'w') as f:
            f.writelines(lines)


class GromacsInstaller:
    """Install GROMACS and dependencies"""
    
    @staticmethod
    def install_all():
        print_header("INSTALLATION OF GROMACS AND DEPENDENCIES")
        
        cmds = [
            "sudo apt update",
            "sudo apt install -y gromacs",
            "sudo apt install -y python3-pip",
            "sudo apt install -y openbabel",
            "sudo apt install -y libhdf4-0-alt",
            "pip3 install acpype",
            "sudo apt install -y grace"
        ]
        
        for i, cmd in enumerate(cmds, 1):
            print_step(i, cmd)
            run_cmd(cmd, check=False)
        
        print_success("Installation completed.")
    
    @staticmethod
    def check():
        print_header("SYSTEM VERIFICATION")
        
        # Check GROMACS
        result = subprocess.run("gmx --version", shell=True, capture_output=True)
        if result.returncode == 0:
            print_success("GROMACS: OK")
        else:
            print_error("GROMACS: Not found")
        
        # Check ACPYPE
        result = subprocess.run("acpype -h", shell=True, capture_output=True)
        if result.returncode == 0:
            print_success("ACPYPE: OK")
        else:
            print_warning("ACPYPE: Not found (required for ligand topology)")
            print_info("Install with: pip3 install acpype")
        
        # Check OpenBabel
        result = subprocess.run("obabel -V", shell=True, capture_output=True)
        if result.returncode == 0:
            print_success("OpenBabel: OK")
        else:
            print_warning("OpenBabel: Not found (required for ligand topology)")
            print_info("Install with: sudo apt install openbabel")


class MDSimulation:
    """GROMACS MD Simulation for Protein-Ligand Complex"""
    
    def __init__(self, workdir: str, pdb_file: str):
        self.workdir = Path(workdir).resolve()
        self.pdb_file = Path(pdb_file).resolve()
        self.has_ligand = False
        self.ligand_name = "LIG"
        self.ligand_moleculetype = "ligand" # Default moleculetype for ligand
        self.cpu_cores = get_cpu_cores()
        self.nt_threads = self.cpu_cores
        self.gpu_support = check_gpu_support()
        
        os.makedirs(self.workdir, exist_ok=True)
        os.chdir(self.workdir)
        
        # State discovery: check if ligand files already exist
        if file_exists("ligand.itp"):
            self.has_ligand = True
            # Try to detect actual moleculetype name from itp
            with open("ligand.itp", 'r') as f:
                for line in f:
                    if line.strip().startswith('[ moleculetype ]'):
                        next_line = next(f, "").strip()
                        while next_line.startswith(';') or not next_line:
                            next_line = next(f, "").strip()
                        if next_line:
                            self.ligand_moleculetype = next_line.split()[0]
                            break
            print(f"{Colors.BLUE}Detected existing ligand: {self.ligand_moleculetype}{Colors.END}")

        print(f"{Colors.CYAN}Working dir: {self.workdir}{Colors.END}")
        print(f"{Colors.CYAN}PDB file: {self.pdb_file}{Colors.END}")
        print(f"{Colors.CYAN}Detected CPU cores: {self.cpu_cores}{Colors.END}")
        if self.gpu_support:
            print(f"{Colors.GREEN}GPU support: Enabled{Colors.END}")
        else:
            print(f"{Colors.YELLOW}GPU support: Disabled (using CPU mode){Colors.END}")
    
    def step0_preprocess(self) -> bool:
        """Analyze and split PDB if needed"""
        print_header("STEP 0: PREPROCESSING")
        
        # Copy PDB to workdir
        local_pdb = self.workdir / self.pdb_file.name
        if self.pdb_file != local_pdb:
            shutil.copy(self.pdb_file, local_pdb)
        
        # Analyze
        print_step(1, "Input processing and analysis")
        info = PDBHandler.analyze(str(local_pdb))
        
        print(f"  Protein atoms: {info['protein_atoms']}")
        print(f"  Ligand atoms: {info['ligand_atoms']}")
        if info['has_ligand']:
            print(f"  Ligand name: {info['ligand_name']}")
        
        self.has_ligand = info['has_ligand']
        self.ligand_name = info['ligand_name'] or "LIG"
        
        # Extract protein
        print_step(2, "Extracting protein component")
        PDBHandler.extract_protein(str(local_pdb), "protein.pdb")
        print_success("Generated: protein.pdb")
        
        if self.has_ligand:
            print_step(3, "Extracting ligand component")
            PDBHandler.extract_ligand(str(local_pdb), "ligand.pdb", self.ligand_name)
            print_success(f"Generated: ligand.pdb (Residue: {self.ligand_name})")
        
        return True
    
    def step1_protein_topology(self, ff: int = 6) -> bool:
        """Generate protein topology"""
        print_header("STEP 1: PROTEIN TOPOLOGY")
        
        if not file_exists("protein.pdb"):
            print_error("Không tìm thấy protein.pdb")
            return False
        
        print_step(1, f"Executing pdb2gmx (Force Field: {ff})")
        
        success = run_cmd(
            'gmx pdb2gmx -f protein.pdb -o protein.gro -p topol.top -ignh',
            input_text=f"{ff}\n1\n"
        )
        
        if success and file_exists("protein.gro"):
            print_success("Topology successfully generated.")
            return True
        return False
    
    def find_acpype(self) -> str:
        """Find acpype executable"""
        # 1. Try system PATH
        if shutil.which("acpype"):
            return "acpype"
        
        # 2. Try common environment paths
        paths = [
            "acpype-env/bin/acpype",
            "../acpype-env/bin/acpype",
            "acpype-env/Scripts/acpype.exe",
            "../acpype-env/Scripts/acpype.exe"
        ]
        
        for p in paths:
            if os.path.exists(p):
                return os.path.abspath(p)
                
        return "acpype"  # Fallback to default

    def fix_acpype_libs(self, acpype_cmd: str) -> Optional[str]:
        """Fix missing symlinks in acpype internal lib directory"""
        if not os.path.isabs(acpype_cmd):
            return None
            
        # Try to find the lib directory relative to acpype/teLeap
        # Error log says: .../acpype/amber_linux/bin/teLeap
        # We want: .../acpype/amber_linux/lib
        
        # acpype_cmd is often a script in .../bin/acpype
        # The package is in .../lib/python3.x/site-packages/acpype
        
        acpype_dir = None
        # Try finding by looking for site-packages
        base_dir = os.path.dirname(os.path.dirname(acpype_cmd)) # acpype-env/
        for root, dirs, files in os.walk(base_dir):
            if 'acpype' in dirs and 'amber_linux' in os.listdir(os.path.join(root, 'acpype')):
                acpype_dir = os.path.join(root, 'acpype')
                break
        
        if not lib_dir:
            return None
            
        print_info(f"Checking internal library dependencies in: {lib_dir}")
        
        # Map expected names to existing versioned names
        links = {
            'libhdf5.so.310': 'libhdf5.so.310.2.0',
            'libhdf5_hl.so.310': 'libhdf5_hl.so.310.0.2',
            'libmfhdf.so.0': 'libmfhdf.so.0.0.0',
            'libdf.so.0': 'libdf.so.0.0.0',
            'libsz.so.2': 'libsz.so.2.0.1',
            'libzip.so.5': 'libzip.so.5.5'
        }
        
        for link, target in links.items():
            link_path = os.path.join(lib_dir, link)
            target_path = os.path.join(lib_dir, target)
            if os.path.exists(target_path) and not os.path.exists(link_path):
                try:
                    os.symlink(target, link_path)
                    print_info(f"Established symlink: {link} -> {target}")
                except Exception as e:
                    print_warning(f"Failed to create symlink {link}: {e}")
                    
        return lib_dir

    def step2_ligand_topology(self) -> bool:
        """Generate ligand topology using ACPYPE"""
        if not self.has_ligand:
            print_info("Không có ligand, bỏ qua bước này")
            return True
        
        print_header("STEP 2: LIGAND TOPOLOGY (ACPYPE)")
        
        if not file_exists("ligand.pdb"):
            print_error("Không tìm thấy ligand.pdb")
            return False
        
        acpype_cmd = self.find_acpype()
        print_step(1, "Chạy ACPYPE để tạo ligand topology")
        
        # Add acpype dir to PATH and check for internal libs
        extra_paths = []
        extra_envs = {}
        if os.path.isabs(acpype_cmd):
            acpype_bin_dir = os.path.dirname(acpype_cmd)
            extra_paths.append(acpype_bin_dir)
            print_info(f"Using: {acpype_cmd}")
            
            # Fix internal libraries
            internal_lib_dir = self.fix_acpype_libs(acpype_cmd)
            if internal_lib_dir:
                extra_envs["LD_LIBRARY_PATH"] = internal_lib_dir
                print_info(f"Configured LD_LIBRARY_PATH: {internal_lib_dir}")
            
        print_info("Processing ligand topology... this may take several minutes.")
        
        success = run_cmd(f'{acpype_cmd} -i ligand.pdb -c bcc -n 0 -a gaff2', 
                         check=False, extra_paths=extra_paths, extra_envs=extra_envs)
        
        # Find ACPYPE output directory
        acpype_dir = None
        for d in os.listdir('.'):
            if d.startswith('ligand.acpype'):
                acpype_dir = d
                break
        
        if acpype_dir and file_exists(f"{acpype_dir}/ligand_GMX.gro"):
            print_step(2, "Importing ligand topology files")
            shutil.copy(f"{acpype_dir}/ligand_GMX.gro", "ligand.gro")
            
            # Process ITP to separate atomtypes and find moleculetype name
            itp_path = f"{acpype_dir}/ligand_GMX.itp"
            with open(itp_path, 'r') as f:
                itp_lines = f.readlines()
            
            atomtypes_lines = []
            itp_only_lines = []
            in_atomtypes = False
            
            for line in itp_lines:
                if line.strip().startswith('[ atomtypes ]'):
                    in_atomtypes = True
                elif line.strip().startswith('[') and in_atomtypes:
                    in_atomtypes = False
                
                if in_atomtypes:
                    atomtypes_lines.append(line)
                else:
                    itp_only_lines.append(line)

            # Find actual moleculetype name
            mol_name = self.ligand_name
            for i, line in enumerate(itp_only_lines):
                if line.strip().startswith('[ moleculetype ]'):
                    for j in range(i+1, len(itp_only_lines)):
                        l = itp_only_lines[j].strip()
                        if l and not l.startswith(';'):
                            mol_name = l.split()[0]
                            break
                    break
            
            self.ligand_moleculetype = mol_name
            print_info(f"Phát hiện moleculetype: {self.ligand_moleculetype}")
            
            # Write files
            if atomtypes_lines:
                with open("ligand_atomtypes.itp", 'w') as f:
                    f.writelines(atomtypes_lines)
                print_info("Đã tách: ligand_atomtypes.itp")
            
            with open("ligand.itp", 'w') as f:
                f.writelines(itp_only_lines)

            # Fallback for separate atomtypes file
            if not atomtypes_lines and file_exists(f"{acpype_dir}/ligand_GMX_atomtypes.itp"):
                shutil.copy(f"{acpype_dir}/ligand_GMX_atomtypes.itp", "ligand_atomtypes.itp")
            
            print_success("Đã tạo: ligand.gro, ligand.itp")
            return True
        else:
            print_error("ACPYPE thất bại!")
            print_info("Thử cài đặt: pip3 install acpype")
            print_info("Hoặc dùng SwissParam (https://swissparam.ch)")
            return False
    
    def step3_merge_system(self) -> bool:
        """Merge protein and ligand geometries"""
        print_header("STEP 3: COMPLEX ASSEMBLY")
        
        if not file_exists("protein.gro"):
            print_error("protein.gro not found.")
            return False
        
        if not self.has_ligand:
            # Just copy protein
            shutil.copy("protein.gro", "complex.gro")
            print_success("Generated: complex.gro (Protein-only system)")
            return True
        
        if not file_exists("ligand.gro"):
            print_error("ligand.gro not found.")
            return False
        
        print_step(1, "Merging coordinates (Gro)")
        
        # Read protein
        with open("protein.gro", 'r') as f:
            prot_lines = f.readlines()
        
        # Read ligand
        with open("ligand.gro", 'r') as f:
            lig_lines = f.readlines()
        
        # Merge
        title = prot_lines[0]
        prot_atoms = int(prot_lines[1].strip())
        lig_atoms = int(lig_lines[1].strip())
        total_atoms = prot_atoms + lig_atoms
        
        # Get protein atoms (skip first 2 lines and last line)
        prot_atom_lines = prot_lines[2:-1]
        lig_atom_lines = lig_lines[2:-1]
        box_line = prot_lines[-1]
        
        with open("complex.gro", 'w') as f:
            f.write("System Complex Generated by Simulation Engine\n")
            f.write(f"{total_atoms}\n")
            f.writelines(prot_atom_lines)
            f.writelines(lig_atom_lines)
            f.write(box_line)
        
        print_success(f"Generated: complex.gro ({total_atoms} atoms integrated)")
        
        # Update topology
        print_step(2, "Integrating topologies in topol.top")
        
        with open("topol.top", 'r') as f:
            top_lines = f.readlines()
        
        new_top_lines = []
        atomtypes_inserted = False
        itp_inserted = False
        
        for line in top_lines:
            # Add to new lines
            new_top_lines.append(line)
            
            # Insert atomtypes after forcefield
            if not atomtypes_inserted and 'forcefield.itp' in line:
                if file_exists("ligand_atomtypes.itp"):
                    new_top_lines.append('\n; Include ligand atomtypes\n#include "ligand_atomtypes.itp"\n')
                atomtypes_inserted = True
                
            # Insert ligand itp before water or ions or system
            if not itp_inserted:
                l = line.strip()
                if l.startswith('[ system ]') or 'tip3p.itp' in l or 'ions.itp' in l:
                    # Insert BEFORE current line
                    new_top_lines.insert(-1, '\n; Include ligand topology\n#include "ligand.itp"\n')
                    itp_inserted = True

        # If system not found yet (unlikely), append at end before molecules
        if not itp_inserted:
             new_top_lines.append('\n; Include ligand topology\n#include "ligand.itp"\n')

        # Clean up molecules section and add ligand
        final_lines = []
        in_molecules = False
        for line in new_top_lines:
            if line.strip().startswith('[ molecules ]'):
                in_molecules = True
            
            # Remove any existing LIG or ligand lines to avoid duplicates
            if in_molecules:
                l = line.strip().split()
                if l and (l[0].upper() == self.ligand_name.upper() or l[0].lower() == self.ligand_moleculetype.lower()):
                    continue
            
            final_lines.append(line)
            
        # Add the correct ligand moleculetype at the very end
        if final_lines[-1].strip() != "":
            final_lines.append(f"{self.ligand_moleculetype}     1\n")
        else:
            final_lines[-1] = f"{self.ligand_moleculetype}     1\n"
            
        with open("topol.top", 'w') as f:
            f.writelines(final_lines)
        
        print_success("Updated: topol.top (System topology assembled)")
        return True
    
    def step4_box(self, dist: float = 1.0) -> bool:
        """Create simulation box"""
        print_header("STEP 4: CREATE BOX")
        
        input_file = "complex.gro" if file_exists("complex.gro") else "protein.gro"
        if not file_exists(input_file):
            print_error(f"{input_file} not found.")
            return False
        
        print_step(1, f"Defining cubic boundary conditions (distance: {dist} nm)")
        return run_cmd(f"gmx editconf -f {input_file} -d {dist} -bt cubic -o box.gro")
    
    def step5_solvate(self) -> bool:
        """Add water"""
        print_header("STEP 5: SOLVATE")
        
        if not file_exists("box.gro"):
            return False
        
        print_step(1, "Solvating system with SPC/E water molecules")
        return run_cmd("gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o solv.gro")
    
    def step6_ions(self, conc: float = 0.15) -> bool:
        """Add neutralizing ions and set salinity"""
        print_header("STEP 6: IONIZATION AND SOLVATION CHEMISTRY")
        
        if not file_exists("solv.gro") or not file_exists("ions.mdp"):
            print_error("solv.gro or ions.mdp missing.")
            return False
        
        print_step(1, "Generating preprocessing run file (TPR)")
        if not run_cmd("gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 5"):
            return False
        
        print_step(2, f"Neutralizing system and adding counter-ions ({conc} M)")
        return run_cmd(
            f"gmx genion -s ions.tpr -p topol.top -conc {conc} -neutral -o ions.gro",
            input_text="SOL\n"
        )
    
    def step_generate_index(self) -> bool:
        """Generate custom index file for coupling groups"""
        print_header("INDEX GENERATION")
        
        input_file = "em.gro" if file_exists("em.gro") else "protein.gro"
        if not file_exists(input_file):
            print_error(f"{input_file} required for index generation not found.")
            return False
            
        print_step(1, "Identifying default group indices")
        # Run make_ndx briefly just to see the groups or generate a base ndx
        run_cmd(f'echo "q" | gmx make_ndx -f {input_file} -o index_base.ndx', check=False)
        
        if not file_exists("index_base.ndx"):
            print_error("index_base.ndx generation failed.")
            return False
            
        # Parse index_base.ndx to find Protein and Ligand
        # Index file format: [ GroupName ] followed by atom indices
        groups = {}
        current_group = None
        group_idx = 0
        with open("index_base.ndx", 'r') as f:
            for line in f:
                if line.startswith('[') and ']' in line:
                    name = line.replace('[', '').replace(']', '').strip()
                    groups[name] = group_idx
                    group_idx += 1
        
        prot_idx = groups.get("Protein")
        lig_idx = groups.get(self.ligand_moleculetype) or groups.get("LIG") or groups.get("Other")
        
        if prot_idx is None:
            print_warning("Protein group not detected in index file.")
            return False
            
        if not self.has_ligand or lig_idx is None:
            print_info("Using default index map.")
            shutil.copy("index_base.ndx", "index.ndx")
            return True
            
        print_step(2, f"Merging groups: {prot_idx} (Protein) + {lig_idx} ({self.ligand_moleculetype})")
        
        # Commands for make_ndx
        # 1. Merge groups: "1 | 13" (example)
        # 2. Name the new group: "name 21 Protein_LIG"
        # 3. Save and quit: "q"
        # Since we added one group, the new group index is exactly 'group_idx' (total groups)
        new_group_idx = group_idx
        
        commands = [
            f"{prot_idx} | {lig_idx}",
            f"name {new_group_idx} Protein_LIG",
            "q"
        ]
        
        run_cmd(f'gmx make_ndx -f {input_file} -n index_base.ndx -o index.ndx', 
                input_text="\n".join(commands) + "\n", check=False)
        
        if file_exists("index.ndx"):
            # Final check
            with open("index.ndx", 'r') as f:
                if "Protein_LIG" in f.read():
                    print_success("Index successfully updated with Protein_LIG group.")
                    return True
        
        print_error("Protein_LIG group integration failed.")
        return False

    def step7_em(self) -> bool:
        """Energy minimization"""
        print_header("STEP 7: ENERGY MINIMIZATION")
        
        if not file_exists("ions.gro") or not file_exists("EM.mdp"):
            print_error("Input parameters (ions.gro/EM.mdp) not found.")
            return False
        
        print_step(1, "Generating energy minimization run binary (TPR)")
        if not run_cmd("gmx grompp -f EM.mdp -c ions.gro -p topol.top -o em.tpr -maxwarn 5"):
            return False
        
        print_step(2, "Executing steepest descent energy minimization")
        # Optimization: Use more threads. GPU for EM is limited but we'll try to use all CPU cores.
        return run_cmd(f"gmx mdrun -v -deffnm em -nt {self.nt_threads}")
    
    def step8_nvt(self) -> bool:
        """NVT equilibration"""
        print_header("STEP 8: NVT EQUILIBRATION")
        
        if not file_exists("em.gro") or not file_exists("NVT.mdp"):
            return False
            
        # Ensure index exists
        if not file_exists("index.ndx"):
            self.step_generate_index()
        
        print_step(1, "Compiling NVT equilibration binary")
        idx_flag = "-n index.ndx" if file_exists("index.ndx") else ""
        if not run_cmd(f"gmx grompp -f NVT.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr {idx_flag} -maxwarn 5"):
            return False
        
        print_step(2, "Executing NVT thermal equilibration")
        # Optimization: Use GPU and optimal threads
        gpu_flag = "-nb gpu" if self.gpu_support else ""
        return run_cmd(f"gmx mdrun -deffnm nvt {gpu_flag} -nt {self.nt_threads}")
    
    def step9_npt(self) -> bool:
        """NPT equilibration"""
        print_header("STEP 9: NPT EQUILIBRATION")
        
        if not file_exists("nvt.gro") or not file_exists("NPT.mdp"):
            return False
            
        print_step(1, "Compiling NPT equilibration binary")
        idx_flag = "-n index.ndx" if file_exists("index.ndx") else ""
        if not run_cmd(f"gmx grompp -f NPT.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr {idx_flag} -maxwarn 5"):
            return False
        
        print_step(2, "Executing NPT pressure equilibration")
        # Optimization: Use GPU and optimal threads
        gpu_flag = "-nb gpu" if self.gpu_support else ""
        return run_cmd(f"gmx mdrun -deffnm npt {gpu_flag} -nt {self.nt_threads}")
    
    def step10_md(self) -> bool:
        """Production MD"""
        print_header("STEP 10: PRODUCTION MD")
        
        if not file_exists("npt.gro") or not file_exists("MD.mdp"):
            return False
        
        print_step(1, "Assembling production MD run binary")
        idx_flag = "-n index.ndx" if file_exists("index.ndx") else ""
        if not run_cmd(f"gmx grompp -f MD.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr {idx_flag} -maxwarn 5"):
            return False
        
        print_step(2, "Executing production molecular dynamics")
        # Final MD run - maximize performance with GPU for both non-bonded and PME
        gpu_flags = "-nb gpu -pme gpu" if self.gpu_support else ""
        return run_cmd(f"gmx mdrun -deffnm md {gpu_flags} -nt {self.nt_threads}")
    
    def step11_analysis(self) -> bool:
        """Post-simulation analysis for Q1 publications"""
        print_header("STEP 11: ADVANCED ANALYSIS")
        
        if not file_exists("md.tpr") or not file_exists("md.xtc"):
            print_warning("Analysis skipped: md.tpr or md.xtc not found.")
            return False
        
        # 1. Structural Stability
        print_step(1, "Computing RMSD trajectory")
        run_cmd("gmx rms -s md.tpr -f md.xtc -o rmsd.xvg -tu ns", input_text="Backbone\nBackbone\n", check=False)
        # RMSD Matrix for 2D comparison
        print_step(1, "Generating 2D RMSD map")
        run_cmd("gmx rms -s md.tpr -f md.xtc -f md.xtc -m rmsd_matrix.xpm", input_text="Backbone\nBackbone\n", check=False)
        
        print_step(2, "Computing Residue Fluctuations (RMSF)")
        run_cmd("gmx rmsf -s md.tpr -f md.xtc -o rmsf.xvg", input_text="Backbone\n", check=False)
        
        print_step(3, "Evaluating System Compactness (Rg)")
        # -p gives components
        run_cmd("gmx gyrate -s md.tpr -f md.xtc -o gyrate.xvg -p", input_text="Protein\n", check=False)
        
        # 2. Interactions & Surface
        print_step(4, "SASA (Solvent Accessible Surface Area)")
        run_cmd("gmx sasa -s md.tpr -f md.xtc -o sasa.xvg", input_text="System\n", check=False)
        
        if self.has_ligand:
            print_step(5, f"Hydrogen Bond Profiling (Protein - {self.ligand_name})")
            idx_flag = "-n index.ndx" if file_exists("index.ndx") else ""
            input_groups = f"Protein\n{self.ligand_name}\n"
            # -num for number vs time, -hbm for residue-residue matrix
            run_cmd(f"gmx hbond -s md.tpr -f md.xtc {idx_flag} -num hbond.xvg -hbm hbond_matrix.xpm", input_text=input_groups, check=False)
            
        # 3. Essential Dynamics (PCA) & DCCM
        # Use C-alpha instead of Backbone for speed (especially for large proteins)
        print_step(6, "Essential Dynamics: Covariance Analysis (PCA) and DCCM generation")
        # -xpma outputs the correlation matrix (DCCM) in XPM format
        run_cmd("gmx covar -s md.tpr -f md.xtc -o eigenval.xvg -xpma dccm.xpm", input_text="C-alpha\nC-alpha\n", check=False)
        run_cmd("gmx anaeig -s md.tpr -f md.xtc -v eigenvec.trr -proj proj.xvg -last 2", input_text="C-alpha\nC-alpha\n", check=False)
        
        # 4. Structural Census (Clustering)
        print_step(7, "Structural Clustering: Gromos method (RMSD Cutoff: 0.2nm)")
        # -cutoff 0.2 is common, output cluster.xvg gives population data
        run_cmd("gmx cluster -s md.tpr -f md.xtc -dist cluster_dist.xvg -cl cluster_main.gro -cutoff 0.2 -method gromos -g cluster.log", input_text="C-alpha\nC-alpha\n", check=False)

        # 5. Zenith Research Tier (Optional but High Impact)
        print_step(8, "Zenith: Hydration RDF & Residue-wise SASA")
        # RDF of water around the ligand
        if self.has_ligand:
            run_cmd(f"gmx rdf -s md.tpr -f md.xtc -o hydration_rdf.xvg -sel '{self.ligand_name}' -ref 'SOL'", check=False)
            
        # Residue-wise SASA (Structural stability proxy)
        run_cmd("gmx sasa -s md.tpr -f md.xtc -or residue_sasa.xvg", input_text="Protein\n", check=False)
        
        # Distance Matrix for Protein-Ligand
        if self.has_ligand:
            run_cmd(f"gmx mdmat -s md.tpr -f md.xtc -n index.ndx -mean -t 0.5 -o prot_lig_map.xpm", input_text=f"Protein\n{self.ligand_name}\n", check=False)

        # 6. Apex Research Tier (World Class Output)
        print_step(9, "Apex: Ramachandran Map & Protein Network Matrix")
        # Ramachandran Plot
        run_cmd("gmx rama -s md.tpr -f md.xtc -o rama.xvg", check=False)
        
        # Full Protein Distance Matrix (for network/allostery analysis)
        run_cmd("gmx mdmat -s md.tpr -f md.xtc -mean -o protein_matrix.xpm", input_text="C-alpha\n", check=False)
        
        # Pocket exposure monitoring (SASA of residues around ligand)
        if self.has_ligand:
            run_cmd(f"gmx sasa -s md.tpr -f md.xtc -surface 'Protein' -output '{self.ligand_name}' -o pocket_sasa.xvg", check=False)
        
        print_success("Advanced post-processing protocol completed.")
        print_info("Quantitive analytics data ready. Execute: python3 gromacsviz.py")
        return True
    
    def run_all(self, start: int = 0, end: int = 11):
        """Run complete workflow from start to end step"""
        print_header("PROTEIN-LIGAND MD SIMULATION")
        print(f"{Colors.CYAN}Chạy từ Step {start} đến Step {end}{Colors.END}\n")
        
        steps = [
            (0, "Preprocess", self.step0_preprocess),
            (1, "Protein Topology", self.step1_protein_topology),
            (2, "Ligand Topology", self.step2_ligand_topology),
            (3, "Merge System", self.step3_merge_system),
            (4, "Create Box", self.step4_box),
            (5, "Solvate", self.step5_solvate),
            (6, "Add Ions", self.step6_ions),
            (7, "Energy Minimization", self.step7_em),
            (8, "NVT", self.step8_nvt),
            (9, "NPT", self.step9_npt),
            (10, "Production MD", self.step10_md),
            (11, "Analysis", self.step11_analysis),
        ]
        
        for num, name, func in steps:
            if num < start:
                continue
            if num > end:
                break
            
            if not func():
                print_error(f"Step {num} ({name}) failed.")
                resp = input("Continue execution? [y/N]: ").strip().lower()
                if resp != 'y':
                    return
        
        print_header("SIMULATION WORKFLOW COMPLETED")


def run_batch(list_file: str, start: int = 0, end: int = 11):
    """Batch run: đọc file .txt, mỗi dòng là 1 complex PDB, tạo folder calculations/<tên_pdb>/"""
    list_path = Path(list_file).resolve()
    if not list_path.exists():
        print_error(f"Không tìm thấy file danh sách: {list_file}")
        return

    # Thư mục gốc chứa script (để copy các file .mdp)
    script_dir = Path(__file__).resolve().parent
    calc_root = script_dir / "calculations"
    calc_root.mkdir(exist_ok=True)

    # Đọc danh sách complex
    with open(list_path, 'r') as f:
        entries = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not entries:
        print_error("File danh sách rỗng hoặc không hợp lệ.")
        return

    print_header(f"BATCH RUN: {len(entries)} complex(es) từ '{list_path.name}'")
    print(f"{Colors.CYAN}Kết quả sẽ được lưu vào: {calc_root}{Colors.END}\n")

    results = []

    for idx, entry in enumerate(entries, 1):
        p = Path(entry)
        # Nếu đường dẫn tương đối, resolve từ thư mục chứa file list
        pdb_path = (list_path.parent / p).resolve() if not p.is_absolute() else p.resolve()

        if not pdb_path.exists():
            print_error(f"[{idx}/{len(entries)}] Không tìm thấy file: {pdb_path}")
            results.append((str(pdb_path), False, "File not found"))
            continue

        pdb_stem = pdb_path.stem  # Tên file không có extension
        workdir = calc_root / pdb_stem
        workdir.mkdir(exist_ok=True)

        print_header(f"[{idx}/{len(entries)}] {pdb_stem}")
        print(f"{Colors.CYAN}Workdir: {workdir}{Colors.END}")

        # Copy .mdp files và file hỗ trợ từ thư mục script sang workdir
        mdp_files = list(script_dir.glob("*.mdp"))
        for mdp in mdp_files:
            dest = workdir / mdp.name
            if not dest.exists():
                shutil.copy(mdp, dest)

        # Copy file PDB sang workdir
        dest_pdb = workdir / pdb_path.name
        if not dest_pdb.exists():
            shutil.copy(pdb_path, dest_pdb)

        try:
            sim = MDSimulation(str(workdir), str(dest_pdb))
            sim.run_all(start=start, end=end)
            results.append((pdb_stem, True, "OK"))
        except Exception as e:
            print_error(f"Lỗi khi chạy {pdb_stem}: {e}")
            results.append((pdb_stem, False, str(e)))

    # Tóm tắt kết quả
    print_header("BATCH SUMMARY")
    ok_count = sum(1 for _, s, _ in results if s)
    print(f"{Colors.CYAN}Hoàn thành: {ok_count}/{len(results)}{Colors.END}")
    for name, status, msg in results:
        if status:
            print_success(f"  ✓ {name}")
        else:
            print_error(f"  ✗ {name}: {msg}")


def main():
    print_header("MD SIMULATION ENGINE")
    print("Supports Protein-only or Protein-Ligand Complex systems\n")
    print("1. Initialize GROMACS and Protocol Dependencies")
    print("2. Execute MD Simulation Pipeline")
    print("3. Verify System Configuration")
    print("4. Batch Run (từ file danh sách .txt)")
    print("0. Exit\n")
    
    choice = input("Select command (0-4): ").strip()
    
    if choice == '1':
        GromacsInstaller.install_all()
    
    elif choice == '2':
        pdb = input("Path to PDB file: ").strip()
        if not file_exists(pdb):
            print_error(f"File not found: {pdb}")
            return
        
        workdir = input("Operational directory [Default: PDB directory]: ").strip()
        if not workdir:
            workdir = os.path.dirname(os.path.abspath(pdb)) or '.'
        
        step_start = input("Initial step [0-11, Default: 0]: ").strip()
        step_start = int(step_start) if step_start.isdigit() else 0
        
        step_end = input("Final step [0-11, Default: 11]: ").strip()
        step_end = int(step_end) if step_end.isdigit() else 11
        
        sim = MDSimulation(workdir, pdb)
        sim.run_all(start=step_start, end=step_end)
    
    elif choice == '3':
        GromacsInstaller.check()

    elif choice == '4':
        list_file = input("Đường dẫn file .txt danh sách complex: ").strip()
        step_start = input("Initial step [0-11, Default: 0]: ").strip()
        step_start = int(step_start) if step_start.isdigit() else 0
        step_end = input("Final step [0-11, Default: 11]: ").strip()
        step_end = int(step_end) if step_end.isdigit() else 11
        run_batch(list_file, start=step_start, end=step_end)
    
    elif choice == '0':
        print("Exit.")


if __name__ == "__main__":
    # Command line support
    if len(sys.argv) > 1:
        if sys.argv[1] == 'install':
            GromacsInstaller.install_all()
        elif sys.argv[1] == 'check':
            GromacsInstaller.check()
        elif sys.argv[1] == 'run' and len(sys.argv) > 2:
            pdb = sys.argv[2]
            workdir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(pdb)) or '.'
            start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            end = int(sys.argv[5]) if len(sys.argv) > 5 else 11
            sim = MDSimulation(workdir, pdb)
            sim.run_all(start=start, end=end)
        elif sys.argv[1] == 'batch' and len(sys.argv) > 2:
            list_txt = sys.argv[2]
            start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
            end = int(sys.argv[4]) if len(sys.argv) > 4 else 11
            run_batch(list_txt, start=start, end=end)
        else:
            print("Usage:")
            print("  python gromacsmda.py                          # Interactive mode")
            print("  python gromacsmda.py install                  # Hardware setup")
            print("  python gromacsmda.py check                    # System check")
            print("  python gromacsmda.py run <pdb> [workdir] [start] [end]")
            print("  python gromacsmda.py batch <list.txt> [start] [end]")
            print("\nExample:")
            print("  python gromacsmda.py run complex.pdb . 0 7")
            print("  python gromacsmda.py batch complexes.txt 0 11")
    else:
        main()
