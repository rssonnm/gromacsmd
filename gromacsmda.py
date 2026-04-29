"""
GROMACS Molecular Dynamics Simulation Workflow
Automated pipeline for Protein-Ligand and Protein-only systems.
GPU-accelerated (NVIDIA CUDA) — optimized for Ubuntu + GROMACS 2022/2023.

Requirements: GROMACS (CUDA build), ACPYPE, OpenBabel.
"""

import os
import sys
import subprocess
import shutil
import re
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict


# ─── Utility: Hardware detection ────────────────────────────────────────────

def get_cpu_cores() -> int:
    """Get number of physical CPU cores available."""
    try:
        return int(subprocess.check_output(['nproc']).decode().strip())
    except Exception:
        return os.cpu_count() or 4


def get_gpu_info() -> Dict:
    """
    Detect NVIDIA GPUs via nvidia-smi.
    Returns a dict with keys:
        available  : bool
        count      : int
        ids        : str  (e.g. "0" or "0123")
        names      : list of str
        vram_mb    : list of int
        cuda_version: str
    """
    info = {
        "available": False, "count": 0, "ids": "",
        "names": [], "vram_mb": [], "cuda_version": "N/A"
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        if not out:
            return info

        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                info["ids"] += parts[0]
                info["names"].append(parts[1])
                try:
                    info["vram_mb"].append(int(parts[2]))
                except ValueError:
                    info["vram_mb"].append(0)

        info["count"] = len(info["names"])
        info["available"] = info["count"] > 0

        # CUDA version
        try:
            cv = subprocess.check_output(
                ["nvidia-smi", "--query", "--display=COMPUTE"],
                stderr=subprocess.DEVNULL
            ).decode()
            m = re.search(r"CUDA Version\s*:\s*([\d.]+)", cv)
            if m:
                info["cuda_version"] = m.group(1)
        except Exception:
            pass

    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return info


def check_gromacs_gpu_support() -> bool:
    """Verify that the installed GROMACS binary was compiled with GPU/CUDA."""
    try:
        result = subprocess.run(
            "gmx -version", shell=True, capture_output=True, text=True
        )
        return "GPU support:        enabled" in result.stdout
    except Exception:
        return False


def get_optimal_mdrun_args(gpu_info: Dict, cpu_cores: int,
                           stage: str = "md") -> str:
    """
    Build the optimal mdrun performance arguments for a given stage.

    stage: "em" | "nvt" | "npt" | "md"

    Strategy (GROMACS best-practices for single-node NVIDIA GPU):
      - 1 MPI rank per GPU  (-ntmpi <n_gpu>)
      - CPU cores split among ranks  (-ntomp <cores_per_rank>)
      - Non-bonded forces on GPU  (-nb gpu)
      - PME on GPU (md/npt/nvt only, single GPU recommended)  (-pme gpu)
      - Bonded forces on GPU (GROMACS ≥ 2021)  (-bonded gpu)
      - Update (integration) on GPU (GROMACS ≥ 2021, single rank)  (-update gpu)
    """
    if not gpu_info["available"]:
        # CPU-only fallback
        return f"-ntmpi 1 -ntomp {cpu_cores} -nb cpu"

    n_gpu = gpu_info["count"]
    # Threads per rank — leave at least 1 logical core headroom
    n_omp = max(1, (cpu_cores // n_gpu))

    if stage == "em":
        # EM: GPU for non-bonded only (PME/update offload not supported in EM)
        return (
            f"-ntmpi {n_gpu} -ntomp {n_omp} "
            f"-gpu_id {gpu_info['ids']} "
            f"-nb gpu -pme cpu"
        )
    elif stage in ("nvt", "npt"):
        # Equilibration: NB + PME on GPU; bonded on GPU if single rank
        args = (
            f"-ntmpi {n_gpu} -ntomp {n_omp} "
            f"-gpu_id {gpu_info['ids']} "
            f"-nb gpu -pme gpu -bonded gpu"
        )
        return args
    else:
        # Production MD: maximum GPU offload
        # -update gpu requires single MPI rank
        if n_gpu == 1:
            return (
                f"-ntmpi 1 -ntomp {cpu_cores} "
                f"-gpu_id {gpu_info['ids']} "
                f"-nb gpu -pme gpu -bonded gpu -update gpu"
            )
        else:
            return (
                f"-ntmpi {n_gpu} -ntomp {n_omp} "
                f"-gpu_id {gpu_info['ids']} "
                f"-nb gpu -pme gpu -bonded gpu"
            )


# ─── Console colors ─────────────────────────────────────────────────────────

class Colors:
    HEADER  = '\033[95m'
    BLUE    = '\033[94m'
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    END     = '\033[0m'
    BOLD    = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*65}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.upper().center(65)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*65}{Colors.END}\n")


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


def print_gpu_info(gpu_info: Dict):
    """Pretty-print GPU detection results."""
    if not gpu_info["available"]:
        print_warning("No NVIDIA GPU detected — running in CPU-only mode.")
        return
    print_success(f"GPU acceleration enabled  ({gpu_info['count']} device(s) found)")
    for i, (name, vram) in enumerate(zip(gpu_info["names"], gpu_info["vram_mb"])):
        print_info(f"  GPU {i}: {name}  |  VRAM: {vram} MB")
    print_info(f"  CUDA Version : {gpu_info['cuda_version']}")
    print_info(f"  GPU IDs      : {gpu_info['ids']}")


# ─── Shell runner ────────────────────────────────────────────────────────────

def run_cmd(cmd: str,
            input_text: Optional[str] = None,
            check: bool = True,
            extra_paths: Optional[List[str]] = None,
            extra_envs: Optional[Dict[str, str]] = None) -> bool:
    """Execute a shell command and return True on success."""
    print(f"{Colors.BLUE}$ {cmd}{Colors.END}")

    my_env = os.environ.copy()
    if extra_paths:
        sep = ":"
        my_env["PATH"] = sep.join(extra_paths) + sep + my_env.get("PATH", "")
    if extra_envs:
        for k, v in extra_envs.items():
            if k in my_env and (k.endswith("PATH") or k == "PYTHONPATH"):
                my_env[k] = v + ":" + my_env[k]
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


# ─── PDB handling ────────────────────────────────────────────────────────────

class PDBHandler:
    """Handle PDB file splitting and inspection."""

    PROTEIN_RESIDUES = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'ACE', 'NME', 'NMA', 'NH2', 'HID', 'HIE', 'HIP', 'HSE', 'HSD', 'HSP',
        'CYX', 'CYM', 'HOH', 'WAT', 'SOL', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN'
    }

    @staticmethod
    def analyze(pdb_file: str) -> Dict:
        """Analyze PDB to find protein chains and detect ligand."""
        protein_atoms, ligand_atoms, ligand_name = [], [], None
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
            'has_ligand':    len(ligand_atoms) > 0,
            'ligand_name':   ligand_name if ligand_name and ligand_name != '*' else 'LIG',
            'ligand_atoms':  len(ligand_atoms),
            'protein_atoms': len(protein_atoms),
        }

    @staticmethod
    def extract_protein(pdb_file: str, output: str):
        lines = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res = line[17:20].strip()
                    if res.upper() in PDBHandler.PROTEIN_RESIDUES and \
                       res.upper() not in ('ACE', 'NME', 'NMA'):
                        lines.append(line)
                elif not line.startswith('CONECT'):
                    lines.append(line)
        with open(output, 'w') as f:
            f.writelines(lines)

    @staticmethod
    def extract_ligand(pdb_file: str, output: str, lig_name: str):
        lines = ["HEADER    LIGAND\n"]
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res = line[17:20].strip()
                    if res.upper() == lig_name.upper() or res == '*':
                        lines.append(line[:17] + f"{lig_name:>3}" + line[20:])
        lines.append("END\n")
        with open(output, 'w') as f:
            f.writelines(lines)


# ─── Installation helpers ────────────────────────────────────────────────────

class GromacsInstaller:
    """Install GROMACS (CPU or GPU build) and all dependencies."""

    @staticmethod
    def install_cpu():
        """Standard apt GROMACS (CPU only)."""
        print_header("INSTALLATION — CPU build")
        cmds = [
            "sudo apt update",
            "sudo apt install -y gromacs",
            "sudo apt install -y python3-pip openbabel libhdf4-0-alt grace",
            "pip3 install acpype",
        ]
        for i, cmd in enumerate(cmds, 1):
            print_step(i, cmd)
            run_cmd(cmd, check=False)
        print_success("CPU installation completed.")

    @staticmethod
    def install_gpu():
        """
        Install GROMACS compiled with CUDA support from the GROMACS PPA
        (Ubuntu 20.04 / 22.04).  Requires NVIDIA driver + CUDA already installed.
        """
        print_header("INSTALLATION — GPU/CUDA build")
        print_info("This installs GROMACS 2023 with CUDA from source (recommended).")
        print_info("Prerequisites: NVIDIA driver ≥ 520, CUDA ≥ 11.8, cmake ≥ 3.18")

        cmds = [
            # System dependencies
            "sudo apt update",
            "sudo apt install -y build-essential cmake libfftw3-dev libopenmpi-dev "
            "openmpi-bin python3-pip openbabel libhdf4-0-alt grace",
            # Download GROMACS 2023.3 source
            "wget -q https://ftp.gromacs.org/gromacs/gromacs-2023.3.tar.gz "
            "-O /tmp/gromacs.tar.gz",
            "tar xf /tmp/gromacs.tar.gz -C /tmp/",
            # Configure with CUDA
            "mkdir -p /tmp/gromacs-2023.3/build && "
            "cmake /tmp/gromacs-2023.3 "
            "-S /tmp/gromacs-2023.3 "
            "-B /tmp/gromacs-2023.3/build "
            "-DGMX_GPU=CUDA "
            "-DGMX_CUDA_TARGET_COMPUTE='75;80;86;89' "
            "-DGMX_MPI=OFF "
            "-DGMX_OPENMP=ON "
            "-DGMX_BUILD_OWN_FFTW=ON "
            "-DREGRESSIONTEST_DOWNLOAD=OFF "
            "-DCMAKE_INSTALL_PREFIX=/usr/local/gromacs",
            # Build & install (uses all cores)
            f"cmake --build /tmp/gromacs-2023.3/build -j$(nproc) "
            f"&& sudo cmake --install /tmp/gromacs-2023.3/build",
            # Source GMXRC
            "echo 'source /usr/local/gromacs/bin/GMXRC' >> ~/.bashrc",
            "source /usr/local/gromacs/bin/GMXRC",
            # Python tools
            "pip3 install acpype",
        ]
        for i, cmd in enumerate(cmds, 1):
            print_step(i, cmd[:80] + ("..." if len(cmd) > 80 else ""))
            run_cmd(cmd, check=False)
        print_success("GPU installation completed. Open a new terminal or run:")
        print_info("  source /usr/local/gromacs/bin/GMXRC")

    @staticmethod
    def install_all():
        """Interactive installer."""
        print_header("GROMACS INSTALLER")
        print("1. CPU-only build (apt, quick)")
        print("2. GPU/CUDA build  (compile from source, recommended for 500 ns)")
        choice = input("Select [1/2]: ").strip()
        if choice == '2':
            GromacsInstaller.install_gpu()
        else:
            GromacsInstaller.install_cpu()

    @staticmethod
    def check():
        print_header("SYSTEM VERIFICATION")

        # GROMACS
        res = subprocess.run("gmx --version", shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            ver_line = next((l for l in res.stdout.splitlines()
                             if "GROMACS version" in l), "")
            print_success(f"GROMACS: OK  — {ver_line.strip()}")
            gpu_ok = "GPU support:        enabled" in res.stdout
            if gpu_ok:
                print_success("GROMACS GPU support: ENABLED (CUDA)")
            else:
                print_warning("GROMACS GPU support: DISABLED (CPU-only build)")
        else:
            print_error("GROMACS: Not found")

        # NVIDIA GPU
        gpu_info = get_gpu_info()
        print_gpu_info(gpu_info)

        # ACPYPE
        res = subprocess.run("acpype -h", shell=True, capture_output=True)
        if res.returncode == 0:
            print_success("ACPYPE: OK")
        else:
            print_warning("ACPYPE: Not found — install with: pip3 install acpype")

        # OpenBabel
        res = subprocess.run("obabel -V", shell=True, capture_output=True)
        if res.returncode == 0:
            print_success("OpenBabel: OK")
        else:
            print_warning("OpenBabel: Not found — install with: sudo apt install openbabel")

        # CUDA toolkit
        res = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            ver_line = next((l for l in res.stdout.splitlines()
                             if "release" in l), "").strip()
            print_success(f"CUDA toolkit: OK  — {ver_line}")
        else:
            print_warning("CUDA toolkit (nvcc): Not found (not required if GROMACS "
                          "already compiled with CUDA)")


# ─── Main simulation class ───────────────────────────────────────────────────

class MDSimulation:
    """
    GROMACS MD Simulation for Protein-Ligand Complex.
    GPU-accelerated when NVIDIA hardware is detected.
    """

    def __init__(self, workdir: str, pdb_file: str):
        self.workdir   = Path(workdir).resolve()
        self.pdb_file  = Path(pdb_file).resolve()
        self.has_ligand = False
        self.ligand_name = "LIG"
        self.ligand_moleculetype = "ligand"

        # Hardware detection
        self.cpu_cores  = get_cpu_cores()
        self.gpu_info   = get_gpu_info()
        self.gmx_gpu_ok = check_gromacs_gpu_support()

        # Effective GPU availability: hardware AND GROMACS compiled with CUDA
        self.use_gpu = self.gpu_info["available"] and self.gmx_gpu_ok

        os.makedirs(self.workdir, exist_ok=True)
        os.chdir(self.workdir)

        # State discovery: check if ligand files already exist
        if file_exists("ligand.itp"):
            self.has_ligand = True
            with open("ligand.itp", 'r') as f:
                for line in f:
                    if line.strip().startswith('[ moleculetype ]'):
                        nl = next(f, "").strip()
                        while nl.startswith(';') or not nl:
                            nl = next(f, "").strip()
                        if nl:
                            self.ligand_moleculetype = nl.split()[0]
                            break
            print_info(f"Detected existing ligand: {self.ligand_moleculetype}")

        # Print hardware summary
        print_header("HARDWARE CONFIGURATION")
        print_info(f"Working directory : {self.workdir}")
        print_info(f"PDB file          : {self.pdb_file}")
        print_info(f"CPU cores         : {self.cpu_cores}")
        print_gpu_info(self.gpu_info)
        if self.gpu_info["available"] and not self.gmx_gpu_ok:
            print_warning(
                "GPU found but GROMACS was NOT compiled with CUDA support.\n"
                "  → Install GPU GROMACS: choose option 2 in the installer menu.\n"
                "  → Falling back to CPU-only mode."
            )
        if self.use_gpu:
            # Print performance estimate
            self._print_performance_estimate()

    # ── Performance info ───────────────────────────────────────────────────

    def _print_performance_estimate(self):
        """Print a rough ns/day estimate based on GPU model."""
        estimates = {
            "H100": (400, 600), "A100": (200, 400), "A6000": (150, 300),
            "3090":  (80, 130), "4090": (120, 180), "3080":  (60, 100),
            "2080":  (40,  70), "V100": (100, 180), "T4":    (30,  60),
        }
        for model, (lo, hi) in estimates.items():
            if any(model in name for name in self.gpu_info["names"]):
                print_info(
                    f"Estimated performance : ~{lo}–{hi} ns/day  "
                    f"→  500 ns ≈ {500//hi}–{500//lo} days"
                )
                break

    # ── Build mdrun command ────────────────────────────────────────────────

    def _mdrun(self, deffnm: str, stage: str = "md",
               extra_flags: str = "") -> str:
        """
        Construct a full gmx mdrun command with optimal GPU flags.

        stage: "em" | "nvt" | "npt" | "md"
        """
        perf = get_optimal_mdrun_args(self.gpu_info if self.use_gpu
                                      else {"available": False},
                                      self.cpu_cores, stage)
        return f"gmx mdrun -v -deffnm {deffnm} {perf} {extra_flags}".strip()

    # ── Pipeline steps ─────────────────────────────────────────────────────

    def step0_preprocess(self) -> bool:
        """Analyze and split PDB if needed."""
        print_header("STEP 0: PREPROCESSING")

        local_pdb = self.workdir / self.pdb_file.name
        if self.pdb_file != local_pdb:
            shutil.copy(self.pdb_file, local_pdb)

        print_step(1, "Input processing and analysis")
        info = PDBHandler.analyze(str(local_pdb))
        print_info(f"  Protein atoms : {info['protein_atoms']}")
        print_info(f"  Ligand atoms  : {info['ligand_atoms']}")
        if info['has_ligand']:
            print_info(f"  Ligand name   : {info['ligand_name']}")

        self.has_ligand  = info['has_ligand']
        self.ligand_name = info['ligand_name'] or "LIG"

        print_step(2, "Extracting protein component")
        PDBHandler.extract_protein(str(local_pdb), "protein.pdb")
        print_success("Generated: protein.pdb")

        if self.has_ligand:
            print_step(3, "Extracting ligand component")
            PDBHandler.extract_ligand(str(local_pdb), "ligand.pdb", self.ligand_name)
            print_success(f"Generated: ligand.pdb  (residue: {self.ligand_name})")

        return True

    def step1_protein_topology(self, ff: int = 6) -> bool:
        """Generate protein topology via pdb2gmx."""
        print_header("STEP 1: PROTEIN TOPOLOGY")

        if not file_exists("protein.pdb"):
            print_error("protein.pdb not found.")
            return False

        print_step(1, f"Running pdb2gmx  (Force Field option: {ff})")
        success = run_cmd(
            'gmx pdb2gmx -f protein.pdb -o protein.gro -p topol.top -ignh',
            input_text=f"{ff}\n1\n"
        )
        if success and file_exists("protein.gro"):
            print_success("Topology successfully generated.")
            return True
        return False

    # ── ACPYPE helpers ──────────────────────────────────────────────────────

    def find_acpype(self) -> str:
        if shutil.which("acpype"):
            return "acpype"
        for p in ["acpype-env/bin/acpype", "../acpype-env/bin/acpype"]:
            if os.path.exists(p):
                return os.path.abspath(p)
        return "acpype"

    def fix_acpype_libs(self, acpype_cmd: str) -> Optional[str]:
        if not os.path.isabs(acpype_cmd):
            return None
        base_dir = os.path.dirname(os.path.dirname(acpype_cmd))
        acpype_dir = None
        for root, dirs, _ in os.walk(base_dir):
            if 'acpype' in dirs and 'amber_linux' in os.listdir(
                    os.path.join(root, 'acpype')):
                acpype_dir = os.path.join(root, 'acpype')
                break
        if not acpype_dir:
            return None
        lib_dir = os.path.join(acpype_dir, 'amber_linux', 'lib')
        if not os.path.isdir(lib_dir):
            return None
        links = {
            'libhdf5.so.310':    'libhdf5.so.310.2.0',
            'libhdf5_hl.so.310': 'libhdf5_hl.so.310.0.2',
            'libmfhdf.so.0':     'libmfhdf.so.0.0.0',
            'libdf.so.0':        'libdf.so.0.0.0',
            'libsz.so.2':        'libsz.so.2.0.1',
            'libzip.so.5':       'libzip.so.5.5',
        }
        for link, target in links.items():
            lp, tp = os.path.join(lib_dir, link), os.path.join(lib_dir, target)
            if os.path.exists(tp) and not os.path.exists(lp):
                try:
                    os.symlink(target, lp)
                    print_info(f"Symlink: {link} → {target}")
                except Exception as e:
                    print_warning(f"Cannot create symlink {link}: {e}")
        return lib_dir

    def step2_ligand_topology(self) -> bool:
        """Generate ligand topology using ACPYPE."""
        if not self.has_ligand:
            print_info("No ligand — skipping.")
            return True

        print_header("STEP 2: LIGAND TOPOLOGY (ACPYPE)")
        if not file_exists("ligand.pdb"):
            print_error("ligand.pdb not found.")
            return False

        acpype_cmd = self.find_acpype()
        extra_paths, extra_envs = [], {}
        if os.path.isabs(acpype_cmd):
            extra_paths.append(os.path.dirname(acpype_cmd))
            lib_dir = self.fix_acpype_libs(acpype_cmd)
            if lib_dir:
                extra_envs["LD_LIBRARY_PATH"] = lib_dir

        print_step(1, "Running ACPYPE (GAFF2 + BCC charges) — may take several minutes")
        run_cmd(f'{acpype_cmd} -i ligand.pdb -c bcc -n 0 -a gaff2',
                check=False, extra_paths=extra_paths, extra_envs=extra_envs)

        acpype_dir = next(
            (d for d in os.listdir('.') if d.startswith('ligand.acpype')), None
        )
        if not acpype_dir or not file_exists(f"{acpype_dir}/ligand_GMX.gro"):
            print_error("ACPYPE failed — install: pip3 install acpype")
            return False

        print_step(2, "Importing ligand topology files")
        shutil.copy(f"{acpype_dir}/ligand_GMX.gro", "ligand.gro")

        itp_path = f"{acpype_dir}/ligand_GMX.itp"
        with open(itp_path, 'r') as f:
            itp_lines = f.readlines()

        atomtypes_lines, itp_only_lines, in_atomtypes = [], [], False
        for line in itp_lines:
            if line.strip().startswith('[ atomtypes ]'):
                in_atomtypes = True
            elif line.strip().startswith('[') and in_atomtypes:
                in_atomtypes = False
            (atomtypes_lines if in_atomtypes else itp_only_lines).append(line)

        mol_name = self.ligand_name
        for i, line in enumerate(itp_only_lines):
            if line.strip().startswith('[ moleculetype ]'):
                for j in range(i + 1, len(itp_only_lines)):
                    lj = itp_only_lines[j].strip()
                    if lj and not lj.startswith(';'):
                        mol_name = lj.split()[0]
                        break
                break
        self.ligand_moleculetype = mol_name
        print_info(f"Detected moleculetype: {self.ligand_moleculetype}")

        if atomtypes_lines:
            with open("ligand_atomtypes.itp", 'w') as f:
                f.writelines(atomtypes_lines)
        with open("ligand.itp", 'w') as f:
            f.writelines(itp_only_lines)

        if not atomtypes_lines and file_exists(
                f"{acpype_dir}/ligand_GMX_atomtypes.itp"):
            shutil.copy(f"{acpype_dir}/ligand_GMX_atomtypes.itp",
                        "ligand_atomtypes.itp")

        print_success("Created: ligand.gro, ligand.itp")
        return True

    def step3_merge_system(self) -> bool:
        """Merge protein and ligand coordinates + topology."""
        print_header("STEP 3: COMPLEX ASSEMBLY")

        if not file_exists("protein.gro"):
            print_error("protein.gro not found.")
            return False

        if not self.has_ligand:
            shutil.copy("protein.gro", "complex.gro")
            print_success("Generated: complex.gro (protein-only)")
            return True

        if not file_exists("ligand.gro"):
            print_error("ligand.gro not found.")
            return False

        print_step(1, "Merging .gro coordinates")
        with open("protein.gro") as f:
            prot = f.readlines()
        with open("ligand.gro") as f:
            lig = f.readlines()

        total = int(prot[1].strip()) + int(lig[1].strip())
        with open("complex.gro", 'w') as f:
            f.write("System Complex Generated by Simulation Engine\n")
            f.write(f"{total}\n")
            f.writelines(prot[2:-1])
            f.writelines(lig[2:-1])
            f.write(prot[-1])
        print_success(f"Generated: complex.gro  ({total} atoms total)")

        print_step(2, "Integrating topology (topol.top)")
        with open("topol.top") as f:
            top_lines = f.readlines()

        new_lines, at_done, itp_done = [], False, False
        for line in top_lines:
            new_lines.append(line)
            if not at_done and 'forcefield.itp' in line:
                if file_exists("ligand_atomtypes.itp"):
                    new_lines.append(
                        '\n; Ligand atomtypes\n#include "ligand_atomtypes.itp"\n'
                    )
                at_done = True
            if not itp_done:
                if line.strip().startswith('[ system ]') or \
                   'tip3p.itp' in line or 'ions.itp' in line:
                    new_lines.insert(-1,
                        '\n; Ligand topology\n#include "ligand.itp"\n')
                    itp_done = True
        if not itp_done:
            new_lines.append('\n; Ligand topology\n#include "ligand.itp"\n')

        final, in_mol = [], False
        for line in new_lines:
            if line.strip().startswith('[ molecules ]'):
                in_mol = True
            if in_mol:
                parts = line.strip().split()
                if parts and (
                    parts[0].upper() == self.ligand_name.upper() or
                    parts[0].lower() == self.ligand_moleculetype.lower()
                ):
                    continue
            final.append(line)

        tail = f"{self.ligand_moleculetype}     1\n"
        if final[-1].strip():
            final.append(tail)
        else:
            final[-1] = tail

        with open("topol.top", 'w') as f:
            f.writelines(final)
        print_success("Updated: topol.top")
        return True

    def step4_box(self, dist: float = 1.2) -> bool:
        """Create cubic simulation box (1.2 nm from solute edge — Q1 standard)."""
        print_header("STEP 4: CREATE SIMULATION BOX")
        gro = "complex.gro" if file_exists("complex.gro") else "protein.gro"
        if not file_exists(gro):
            print_error(f"{gro} not found.")
            return False
        return run_cmd(f"gmx editconf -f {gro} -d {dist} -bt cubic -o box.gro")

    def step5_solvate(self) -> bool:
        """Solvate with TIP3P water."""
        print_header("STEP 5: SOLVATE")
        if not file_exists("box.gro"):
            return False
        return run_cmd("gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o solv.gro")

    def step6_ions(self, conc: float = 0.15) -> bool:
        """Add neutralizing ions at physiological concentration (0.15 M NaCl)."""
        print_header("STEP 6: ION PLACEMENT")
        if not file_exists("solv.gro") or not file_exists("ions.mdp"):
            print_error("solv.gro or ions.mdp missing.")
            return False

        print_step(1, "Generating TPR for genion")
        if not run_cmd(
            "gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 5"
        ):
            return False

        print_step(2, f"Adding ions ({conc} M NaCl, charge-neutral)")
        return run_cmd(
            f"gmx genion -s ions.tpr -p topol.top -conc {conc} -neutral -o ions.gro",
            input_text="SOL\n"
        )

    def step_generate_index(self) -> bool:
        """Create index.ndx with a merged Protein_LIG group."""
        print_header("INDEX GENERATION")
        src = "em.gro" if file_exists("em.gro") else "ions.gro"
        if not file_exists(src):
            print_error(f"Source for index ({src}) not found.")
            return False

        run_cmd(f'echo "q" | gmx make_ndx -f {src} -o index_base.ndx', check=False)
        if not file_exists("index_base.ndx"):
            print_error("index_base.ndx generation failed.")
            return False

        groups, idx = {}, 0
        with open("index_base.ndx") as f:
            for line in f:
                if line.startswith('[') and ']' in line:
                    name = line.replace('[', '').replace(']', '').strip()
                    groups[name] = idx
                    idx += 1

        prot_idx = groups.get("Protein")
        lig_idx  = (groups.get(self.ligand_moleculetype) or
                    groups.get("LIG") or groups.get("Other"))

        if prot_idx is None:
            print_warning("Protein group not found in index file.")
            return False

        if not self.has_ligand or lig_idx is None:
            shutil.copy("index_base.ndx", "index.ndx")
            return True

        new_grp = idx
        cmds = [f"{prot_idx} | {lig_idx}",
                f"name {new_grp} Protein_LIG", "q"]
        run_cmd(f'gmx make_ndx -f {src} -n index_base.ndx -o index.ndx',
                input_text="\n".join(cmds) + "\n", check=False)

        if file_exists("index.ndx"):
            with open("index.ndx") as f:
                if "Protein_LIG" in f.read():
                    print_success("index.ndx created with Protein_LIG group.")
                    return True
        print_error("Protein_LIG group creation failed.")
        return False

    def step7_em(self) -> bool:
        """Energy minimization — steepest descent."""
        print_header("STEP 7: ENERGY MINIMIZATION")
        if not file_exists("ions.gro") or not file_exists("EM.mdp"):
            print_error("ions.gro or EM.mdp not found.")
            return False

        print_step(1, "Generating TPR for energy minimization")
        if not run_cmd(
            "gmx grompp -f EM.mdp -c ions.gro -p topol.top -o em.tpr -maxwarn 5"
        ):
            return False

        print_step(2, "Running steepest-descent energy minimization")
        # EM: GPU for NB only; PME on CPU (GROMACS limitation for EM)
        cmd = self._mdrun("em", stage="em")
        print_info(f"mdrun command: {cmd}")
        return run_cmd(cmd)

    def step8_nvt(self) -> bool:
        """NVT equilibration — 1 ns."""
        print_header("STEP 8: NVT EQUILIBRATION (1 ns)")
        if not file_exists("em.gro") or not file_exists("NVT.mdp"):
            return False

        if not file_exists("index.ndx"):
            self.step_generate_index()

        print_step(1, "Compiling NVT TPR")
        idx = "-n index.ndx" if file_exists("index.ndx") else ""
        if not run_cmd(
            f"gmx grompp -f NVT.mdp -c em.gro -r em.gro "
            f"-p topol.top -o nvt.tpr {idx} -maxwarn 5"
        ):
            return False

        print_step(2, "Running NVT thermal equilibration (GPU-accelerated)")
        cmd = self._mdrun("nvt", stage="nvt")
        print_info(f"mdrun command: {cmd}")
        return run_cmd(cmd)

    def step9_npt(self) -> bool:
        """NPT equilibration — 5 ns."""
        print_header("STEP 9: NPT EQUILIBRATION (5 ns)")
        if not file_exists("nvt.gro") or not file_exists("NPT.mdp"):
            return False

        print_step(1, "Compiling NPT TPR")
        idx = "-n index.ndx" if file_exists("index.ndx") else ""
        if not run_cmd(
            f"gmx grompp -f NPT.mdp -c nvt.gro -r nvt.gro "
            f"-t nvt.cpt -p topol.top -o npt.tpr {idx} -maxwarn 5"
        ):
            return False

        print_step(2, "Running NPT pressure equilibration (GPU-accelerated)")
        cmd = self._mdrun("npt", stage="npt")
        print_info(f"mdrun command: {cmd}")
        return run_cmd(cmd)

    def step10_md(self) -> bool:
        """Production MD — 500 ns, maximum GPU offload."""
        print_header("STEP 10: PRODUCTION MD — 500 ns")
        if not file_exists("npt.gro") or not file_exists("MD.mdp"):
            return False

        print_step(1, "Compiling production MD TPR")
        idx = "-n index.ndx" if file_exists("index.ndx") else ""
        if not run_cmd(
            f"gmx grompp -f MD.mdp -c npt.gro -t npt.cpt "
            f"-p topol.top -o md.tpr {idx} -maxwarn 5"
        ):
            return False

        print_step(2, "Running production MD — 500 ns (maximum GPU offload)")
        if self.use_gpu:
            n_gpu = self.gpu_info["count"]
            ids   = self.gpu_info["ids"]
            if n_gpu == 1:
                print_info(
                    "Single GPU: using -update gpu for maximum performance "
                    "(all forces + integration on GPU)"
                )
            else:
                print_info(
                    f"{n_gpu} GPUs detected: NB + PME + bonded on GPU "
                    "(update on CPU — multi-rank limitation)"
                )
        cmd = self._mdrun("md", stage="md")
        print_info(f"mdrun command: {cmd}")
        return run_cmd(cmd)

    def step10_md_restart(self) -> bool:
        """Resume an interrupted production MD from checkpoint."""
        print_header("STEP 10: PRODUCTION MD — RESTART FROM CHECKPOINT")
        if not file_exists("md.cpt"):
            print_error("md.cpt not found — cannot restart.")
            return False

        print_step(1, "Extending TPR if needed")
        # Extend by remaining steps (GROMACS will compute delta from .cpt)
        run_cmd(
            "gmx convert-tpr -s md.tpr -extend 500000 -o md_ext.tpr",
            check=False
        )
        tpr = "md_ext.tpr" if file_exists("md_ext.tpr") else "md.tpr"

        print_step(2, "Resuming MD from checkpoint")
        perf = get_optimal_mdrun_args(
            self.gpu_info if self.use_gpu else {"available": False},
            self.cpu_cores, "md"
        )
        cmd = f"gmx mdrun -v -s {tpr} -deffnm md -cpi md.cpt {perf}"
        print_info(f"mdrun command: {cmd}")
        return run_cmd(cmd)

    def step11_analysis(self) -> bool:
        """Post-simulation analysis for Q1 publications."""
        print_header("STEP 11: ADVANCED ANALYSIS")

        if not file_exists("md.tpr") or not file_exists("md.xtc"):
            print_warning("md.tpr or md.xtc not found — analysis skipped.")
            return False

        idx = "-n index.ndx" if file_exists("index.ndx") else ""

        # 1. Structural stability
        print_step(1, "RMSD — backbone (vs. reference frame 0)")
        run_cmd("gmx rms -s md.tpr -f md.xtc -o rmsd_backbone.xvg -tu ns",
                input_text="Backbone\nBackbone\n", check=False)

        print_step(2, "RMSD 2-D matrix")
        run_cmd("gmx rms -s md.tpr -f md.xtc -m rmsd_matrix.xpm",
                input_text="Backbone\nBackbone\n", check=False)

        print_step(3, "RMSF — per-residue fluctuations")
        run_cmd("gmx rmsf -s md.tpr -f md.xtc -o rmsf.xvg -res",
                input_text="Backbone\n", check=False)

        print_step(4, "Radius of gyration (Rg)")
        run_cmd("gmx gyrate -s md.tpr -f md.xtc -o gyrate.xvg -p",
                input_text="Protein\n", check=False)

        # 2. Surface & interactions
        print_step(5, "SASA — total")
        run_cmd("gmx sasa -s md.tpr -f md.xtc -o sasa.xvg",
                input_text="System\n", check=False)

        print_step(6, "SASA — per residue")
        run_cmd("gmx sasa -s md.tpr -f md.xtc -or residue_sasa.xvg",
                input_text="Protein\n", check=False)

        if self.has_ligand:
            print_step(7, f"Hydrogen bonds (Protein ↔ {self.ligand_name})")
            run_cmd(
                f"gmx hbond -s md.tpr -f md.xtc {idx} "
                f"-num hbond.xvg -hbm hbond_matrix.xpm",
                input_text=f"Protein\n{self.ligand_name}\n", check=False
            )
            print_step(8, f"Protein-ligand distance map")
            run_cmd(
                f"gmx mdmat -s md.tpr -f md.xtc {idx} "
                f"-mean -t 0.5 -o prot_lig_map.xpm",
                input_text=f"Protein\n{self.ligand_name}\n", check=False
            )

        # 3. Essential dynamics / PCA
        print_step(9, "PCA — covariance diagonalization (DCCM)")
        run_cmd(
            "gmx covar -s md.tpr -f md.xtc -o eigenval.xvg -xpma dccm.xpm",
            input_text="C-alpha\nC-alpha\n", check=False
        )
        run_cmd(
            "gmx anaeig -s md.tpr -f md.xtc -v eigenvec.trr "
            "-proj proj.xvg -last 2",
            input_text="C-alpha\nC-alpha\n", check=False
        )

        # 4. Clustering
        print_step(10, "Gromos clustering (cutoff 0.2 nm)")
        run_cmd(
            "gmx cluster -s md.tpr -f md.xtc -dist cluster_dist.xvg "
            "-cl cluster_main.gro -cutoff 0.2 -method gromos -g cluster.log",
            input_text="C-alpha\nC-alpha\n", check=False
        )

        # 5. Thermodynamic properties
        print_step(11, "Energy / Temperature / Pressure / Density")
        for prop, out in [
            ("Temperature", "temperature.xvg"),
            ("Pressure",    "pressure.xvg"),
            ("Density",     "density.xvg"),
            ("Potential",   "potential_energy.xvg"),
        ]:
            run_cmd(f"gmx energy -f md.edr -o {out}",
                    input_text=f"{prop}\n\n", check=False)

        # 6. Ramachandran
        print_step(12, "Ramachandran map")
        run_cmd("gmx rama -s md.tpr -f md.xtc -o rama.xvg", check=False)

        # 7. Protein distance matrix
        print_step(13, "Full protein distance matrix (network / allostery)")
        run_cmd("gmx mdmat -s md.tpr -f md.xtc -mean -o protein_matrix.xpm",
                input_text="C-alpha\n", check=False)

        # 8. Hydration RDF (ligand systems)
        if self.has_ligand:
            print_step(14, "Hydration RDF around ligand")
            run_cmd(
                f"gmx rdf -s md.tpr -f md.xtc -o hydration_rdf.xvg "
                f"-sel '{self.ligand_name}' -ref 'SOL'",
                check=False
            )
            print_step(15, "Pocket SASA (protein surface around ligand)")
            run_cmd(
                f"gmx sasa -s md.tpr -f md.xtc "
                f"-surface 'Protein' -output '{self.ligand_name}' "
                f"-o pocket_sasa.xvg",
                check=False
            )

        # 9. Secondary structure (DSSP) — time evolution
        print_step(16, "Secondary structure (DSSP)")
        run_cmd(
            "gmx dssp -s md.tpr -f md.xtc -o ss.xpm -sc scount.xvg",
            input_text="Protein\n", check=False
        )

        # 10. Residue-residue contact map (fraction of frames)
        print_step(17, "Residue contact map")
        run_cmd(
            "gmx mdmat -s md.tpr -f md.xtc -mean -t 0.35 -o contacts.xpm",
            input_text="C-alpha\n", check=False
        )

        # 11. H-bond autocorrelation (lifetime)
        if self.has_ligand:
            print_step(18, "H-bond autocorrelation (lifetime)")
            run_cmd(
                f"gmx hbond -s md.tpr -f md.xtc {idx} "
                f"-ac hbac.xvg -num hbond.xvg",
                input_text=f"Protein\n{self.ligand_name}\n", check=False
            )

        # 12. Mean Square Displacement (protein diffusion)
        print_step(19, "MSD — protein diffusion")
        run_cmd(
            "gmx msd -s md.tpr -f md.xtc -o msd.xvg -lateral z",
            input_text="Protein\n", check=False
        )

        # 13. Salt bridges
        print_step(20, "Salt bridge distances")
        run_cmd(
            "gmx saltbr -s md.tpr -f md.xtc -t 0.4 -sep",
            check=False
        )

        # 14. Backbone dihedral angle (Phi/Psi) distribution over time
        print_step(21, "Backbone dihedral angles")
        run_cmd(
            "gmx angle -s md.tpr -f md.xtc -type dihedral -ov dihedral.xvg",
            check=False
        )

        # 15. Cluster RMSD distribution (already from step 10, verify output)
        print_step(22, "Cluster RMSD distribution histogram")
        run_cmd(
            "gmx cluster -s md.tpr -f md.xtc -dist cluster_dist.xvg "
            "-cl cluster_main.gro -cutoff 0.2 -method gromos -g cluster.log",
            input_text="C-alpha\nC-alpha\n", check=False
        )

        # 16. Equilibration energy terms (NVT + NPT)
        for stage, edr in [("nvt", "nvt.edr"), ("npt", "npt.edr")]:
            if file_exists(edr):
                print_step(23, f"Equilibration energy terms ({stage.upper()})")
                for prop, out in [
                    ("Temperature", f"{stage}_temperature.xvg"),
                    ("Pressure",    f"{stage}_pressure.xvg"),
                    ("Potential",   f"{stage}_potential.xvg"),
                ]:
                    run_cmd(f"gmx energy -f {edr} -o {out}",
                            input_text=f"{prop}\n\n", check=False)

        print_success("Post-processing analysis complete.")
        print_info("Run 'python3 gromacsviz.py' to generate publication figures.")
        return True

    # ── Orchestration ──────────────────────────────────────────────────────

    def run_all(self, start: int = 0, end: int = 11):
        """Execute the complete workflow from step `start` to step `end`."""
        print_header("PROTEIN-LIGAND MD SIMULATION WORKFLOW")
        print_info(f"Steps: {start} → {end}")

        steps = [
            (0,  "Preprocess",          self.step0_preprocess),
            (1,  "Protein Topology",    self.step1_protein_topology),
            (2,  "Ligand Topology",     self.step2_ligand_topology),
            (3,  "Merge System",        self.step3_merge_system),
            (4,  "Create Box",          self.step4_box),
            (5,  "Solvate",             self.step5_solvate),
            (6,  "Add Ions",            self.step6_ions),
            (7,  "Energy Minimization", self.step7_em),
            (8,  "NVT Equilibration",   self.step8_nvt),
            (9,  "NPT Equilibration",   self.step9_npt),
            (10, "Production MD",       self.step10_md),
            (11, "Analysis",            self.step11_analysis),
        ]

        for num, name, func in steps:
            if num < start:
                continue
            if num > end:
                break
            if not func():
                print_error(f"Step {num} ({name}) FAILED.")
                resp = input("Continue anyway? [y/N]: ").strip().lower()
                if resp != 'y':
                    return

        print_header("SIMULATION WORKFLOW COMPLETED")


# ─── Batch runner ────────────────────────────────────────────────────────────

def run_batch(list_file: str, start: int = 0, end: int = 11):
    """Batch mode: read a .txt file (one PDB per line) and run each sequentially."""
    list_path = Path(list_file).resolve()
    if not list_path.exists():
        print_error(f"List file not found: {list_file}")
        return

    script_dir = Path(__file__).resolve().parent
    calc_root  = script_dir / "calculations"
    calc_root.mkdir(exist_ok=True)

    with open(list_path) as f:
        entries = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    if not entries:
        print_error("List file is empty.")
        return

    print_header(f"BATCH RUN: {len(entries)} complex(es)")
    results = []

    for idx, entry in enumerate(entries, 1):
        p = Path(entry)
        pdb_path = (list_path.parent / p).resolve() if not p.is_absolute() else p.resolve()
        if not pdb_path.exists():
            print_error(f"[{idx}] Not found: {pdb_path}")
            results.append((str(pdb_path), False, "File not found"))
            continue

        pdb_stem = pdb_path.stem
        workdir  = calc_root / pdb_stem
        workdir.mkdir(exist_ok=True)

        print_header(f"[{idx}/{len(entries)}] {pdb_stem}")

        for mdp in script_dir.glob("*.mdp"):
            dest = workdir / mdp.name
            if not dest.exists():
                shutil.copy(mdp, dest)

        dest_pdb = workdir / pdb_path.name
        if not dest_pdb.exists():
            shutil.copy(pdb_path, dest_pdb)

        try:
            sim = MDSimulation(str(workdir), str(dest_pdb))
            sim.run_all(start=start, end=end)
            results.append((pdb_stem, True, "OK"))
        except Exception as e:
            print_error(f"Error running {pdb_stem}: {e}")
            results.append((pdb_stem, False, str(e)))

    print_header("BATCH SUMMARY")
    ok = sum(1 for _, s, _ in results if s)
    print_info(f"Completed: {ok}/{len(results)}")
    for name, status, msg in results:
        (print_success if status else print_error)(
            f"  {'✓' if status else '✗'} {name}" + (f": {msg}" if not status else "")
        )


# ─── Interactive menu ────────────────────────────────────────────────────────

def main():
    print_header("MD SIMULATION ENGINE")
    gpu_info = get_gpu_info()
    gmx_gpu  = check_gromacs_gpu_support()
    print_gpu_info(gpu_info)
    if gpu_info["available"] and not gmx_gpu:
        print_warning("GPU found but GROMACS not compiled with CUDA — install GPU build!")
    print()
    print("1. Install dependencies (CPU or GPU GROMACS)")
    print("2. Run MD simulation pipeline")
    print("3. System & GPU verification")
    print("4. Batch run (from .txt list)")
    print("5. Restart production MD from checkpoint")
    print("0. Exit\n")

    choice = input("Select command (0-5): ").strip()

    if choice == '1':
        GromacsInstaller.install_all()

    elif choice == '2':
        pdb = input("Path to PDB file: ").strip()
        if not file_exists(pdb):
            print_error(f"File not found: {pdb}")
            return
        workdir = input(
            "Working directory [Default: PDB directory]: "
        ).strip() or os.path.dirname(os.path.abspath(pdb)) or '.'
        s = input("Start step [0–11, default 0]: ").strip()
        e = input("End step   [0–11, default 11]: ").strip()
        sim = MDSimulation(workdir, pdb)
        sim.run_all(int(s) if s.isdigit() else 0,
                    int(e) if e.isdigit() else 11)

    elif choice == '3':
        GromacsInstaller.check()

    elif choice == '4':
        lst = input("Path to .txt list file: ").strip()
        s   = input("Start step [0–11, default 0]: ").strip()
        e   = input("End step   [0–11, default 11]: ").strip()
        run_batch(lst,
                  int(s) if s.isdigit() else 0,
                  int(e) if e.isdigit() else 11)

    elif choice == '5':
        pdb = input("Path to PDB file (same as original): ").strip()
        workdir = input(
            "Working directory (same as original): "
        ).strip() or os.path.dirname(os.path.abspath(pdb)) or '.'
        sim = MDSimulation(workdir, pdb)
        sim.step10_md_restart()

    elif choice == '0':
        print("Goodbye.")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'install':
            GromacsInstaller.install_all()
        elif cmd == 'check':
            GromacsInstaller.check()
        elif cmd == 'run' and len(sys.argv) > 2:
            pdb     = sys.argv[2]
            workdir = sys.argv[3] if len(sys.argv) > 3 else \
                      os.path.dirname(os.path.abspath(pdb)) or '.'
            start   = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            end     = int(sys.argv[5]) if len(sys.argv) > 5 else 11
            MDSimulation(workdir, pdb).run_all(start, end)
        elif cmd == 'batch' and len(sys.argv) > 2:
            run_batch(
                sys.argv[2],
                int(sys.argv[3]) if len(sys.argv) > 3 else 0,
                int(sys.argv[4]) if len(sys.argv) > 4 else 11,
            )
        elif cmd == 'restart' and len(sys.argv) > 2:
            pdb     = sys.argv[2]
            workdir = sys.argv[3] if len(sys.argv) > 3 else \
                      os.path.dirname(os.path.abspath(pdb)) or '.'
            MDSimulation(workdir, pdb).step10_md_restart()
        else:
            print("Usage:")
            print("  python gromacsmda.py                             # Interactive")
            print("  python gromacsmda.py install                     # Install deps")
            print("  python gromacsmda.py check                       # System check")
            print("  python gromacsmda.py run <pdb> [workdir] [s] [e]")
            print("  python gromacsmda.py batch <list.txt> [s] [e]")
            print("  python gromacsmda.py restart <pdb> [workdir]     # Resume 500 ns")
            print("\nExamples:")
            print("  python gromacsmda.py run complex.pdb . 0 11")
            print("  python gromacsmda.py restart complex.pdb ./calculations/complex")
    else:
        main()
