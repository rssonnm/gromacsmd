#!/usr/bin/env python3
"""
GROMACS MD Automation Tool - Protein-Ligand Complex
Công cụ tự động hóa mô phỏng Dynamics phân tử với GROMACS
Hỗ trợ: Protein đơn thuần hoặc Protein-Ligand complex
"""

import os
import sys
import subprocess
import shutil
import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List


class Colors:
    """Terminal colors for better visualization"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_step(step_num: int, text: str):
    print(f"{Colors.CYAN}[Step {step_num}]{Colors.END} {Colors.BOLD}{text}{Colors.END}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.WARNING}⚠ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.FAIL}✗ {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def run_command(cmd: str, input_text: Optional[str] = None, cwd: Optional[str] = None,
                check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command with optional input"""
    print(f"{Colors.BLUE}$ {cmd}{Colors.END}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            input=input_text.encode() if input_text else None,
            cwd=cwd,
            capture_output=capture,
            check=check,
            text=False if input_text else True
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with exit code {e.returncode}")
        if check:
            raise
        return e


def check_file_exists(filepath: str, description: str = "") -> bool:
    """Check if file exists and print appropriate message"""
    if os.path.exists(filepath):
        return True
    else:
        print_error(f"File không tồn tại: {filepath} {description}")
        return False


class PDBProcessor:
    """Process PDB files to separate protein and ligand"""
    
    # Standard amino acid residue names
    STANDARD_RESIDUES = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        # Modified/capped residues
        'ACE', 'NME', 'NMA', 'NH2',
        # Histidine variants
        'HID', 'HIE', 'HIP', 'HSE', 'HSD', 'HSP',
        # Cysteine variants
        'CYX', 'CYM',
        # Water and ions
        'HOH', 'WAT', 'SOL', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN',
        # DNA/RNA
        'DA', 'DT', 'DG', 'DC', 'A', 'U', 'G', 'C'
    }
    
    @staticmethod
    def analyze_pdb(pdb_file: str) -> dict:
        """Analyze PDB file to identify protein chains and ligands"""
        chains = {}
        ligand_residues = set()
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip() or '_'
                    res_num = line[22:26].strip()
                    
                    if chain_id not in chains:
                        chains[chain_id] = {'residues': set(), 'atoms': 0, 'is_protein': True}
                    
                    chains[chain_id]['residues'].add((res_name, res_num))
                    chains[chain_id]['atoms'] += 1
                    
                    # Check if non-standard residue (potential ligand)
                    if res_name.upper() not in PDBProcessor.STANDARD_RESIDUES:
                        ligand_residues.add(res_name)
                        chains[chain_id]['is_protein'] = False
        
        return {
            'chains': chains,
            'ligand_residues': ligand_residues,
            'has_ligand': len(ligand_residues) > 0
        }
    
    @staticmethod
    def extract_protein_only(pdb_file: str, output_file: str) -> bool:
        """Extract only protein atoms from PDB file (remove ligands)"""
        protein_lines = []
        removed_residues = set()
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res_name = line[17:20].strip()
                    if res_name.upper() in PDBProcessor.STANDARD_RESIDUES:
                        protein_lines.append(line)
                    else:
                        removed_residues.add(res_name)
                elif line.startswith(('TER', 'END', 'HEADER', 'TITLE', 'REMARK', 
                                     'HELIX', 'SHEET', 'SSBOND', 'CRYST')):
                    protein_lines.append(line)
        
        with open(output_file, 'w') as f:
            f.writelines(protein_lines)
        
        if removed_residues:
            print_info(f"Đã loại bỏ các residue: {', '.join(removed_residues)}")
        
        return True
    
    @staticmethod
    def remove_nme_ace_caps(pdb_file: str, output_file: str) -> bool:
        """Remove NME and ACE capping groups from PDB"""
        clean_lines = []
        removed = {'ACE': 0, 'NME': 0}
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res_name = line[17:20].strip()
                    if res_name in ('ACE', 'NME', 'NMA'):
                        removed[res_name] = removed.get(res_name, 0) + 1
                        continue
                clean_lines.append(line)
        
        with open(output_file, 'w') as f:
            f.writelines(clean_lines)
        
        if any(removed.values()):
            print_info(f"Đã loại bỏ: {sum(removed.values())} atoms từ ACE/NME caps")
        
        return True


class GromacsInstaller:
    """Handle GROMACS installation"""
    
    @staticmethod
    def install_dependencies():
        print_header("CÀI ĐẶT CÁC GÓI PHỤ THUỘC")
        
        commands = [
            ("Cập nhật package list", "sudo apt update"),
            ("Nâng cấp hệ thống", "sudo apt upgrade -y"),
            ("Cài đặt GCC", "sudo apt install -y gcc"),
            ("Cài đặt CMake", "sudo apt install -y cmake"),
            ("Cài đặt build-essential", "sudo apt install -y build-essential"),
            ("Cài đặt FFTW3", "sudo apt-get install -y libfftw3-dev"),
        ]
        
        for i, (desc, cmd) in enumerate(commands, 1):
            print_step(i, desc)
            try:
                run_command(cmd, check=False)
                print_success(f"{desc} - Hoàn thành!")
            except Exception:
                print_warning(f"{desc} - Có thể đã cài đặt hoặc gặp lỗi")
    
    @staticmethod
    def install_gromacs_quick():
        print_header("CÀI ĐẶT GROMACS")
        print_step(1, "Cài đặt GROMACS từ apt repository")
        try:
            run_command("sudo apt install -y gromacs", check=False)
            print_success("GROMACS đã được cài đặt!")
        except Exception:
            print_error("Không thể cài đặt GROMACS.")
    
    @staticmethod
    def install_additional_tools():
        print_header("CÀI ĐẶT CÔNG CỤ BỔ SUNG")
        tools = [("PyMOL", "sudo apt-get install -y pymol"), ("Grace", "sudo apt-get install -y grace")]
        for i, (name, cmd) in enumerate(tools, 1):
            print_step(i, f"Cài đặt {name}")
            try:
                run_command(cmd, check=False)
                print_success(f"{name} đã được cài đặt!")
            except Exception:
                print_warning(f"Không thể cài đặt {name}")
    
    @staticmethod
    def check_gromacs() -> bool:
        print_header("KIỂM TRA GROMACS")
        try:
            result = subprocess.run("gmx --version", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print_success("GROMACS đã được cài đặt!")
                return True
        except Exception:
            pass
        print_error("GROMACS chưa được cài đặt!")
        return False


class GromacsMDSimulation:
    """Handle GROMACS MD Simulation workflow for protein-only systems"""
    
    def __init__(self, working_dir: str, pdb_file: str):
        self.working_dir = Path(working_dir).resolve()
        self.pdb_file = Path(pdb_file).resolve()
        self.protein_pdb = None  # Will be set after preprocessing
        
        if not self.pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {self.pdb_file}")
        
        os.makedirs(self.working_dir, exist_ok=True)
        os.chdir(self.working_dir)
        
        print(f"{Colors.CYAN}Working directory: {self.working_dir}{Colors.END}")
        print(f"{Colors.CYAN}Input PDB file: {self.pdb_file}{Colors.END}")
    
    def preprocess_pdb(self) -> str:
        """Preprocess PDB: analyze, remove ligands and caps if needed"""
        print_header("PREPROCESSING - Phân tích và chuẩn bị PDB")
        
        # Analyze PDB
        print_step(1, "Phân tích file PDB")
        analysis = PDBProcessor.analyze_pdb(str(self.pdb_file))
        
        print(f"\n{Colors.CYAN}Thông tin file PDB:{Colors.END}")
        for chain_id, info in analysis['chains'].items():
            chain_type = "Protein" if info['is_protein'] else "Ligand/Unknown"
            print(f"  Chain '{chain_id}': {info['atoms']} atoms, {len(info['residues'])} residues [{chain_type}]")
        
        if analysis['ligand_residues']:
            print(f"\n{Colors.WARNING}Phát hiện ligand/residue không chuẩn: {', '.join(analysis['ligand_residues'])}{Colors.END}")
        
        # Copy original PDB to working directory
        local_pdb = self.working_dir / self.pdb_file.name
        if self.pdb_file != local_pdb:
            shutil.copy(self.pdb_file, local_pdb)
        
        # Process PDB based on content
        clean_pdb = self.working_dir / "protein_clean.pdb"
        
        if analysis['has_ligand']:
            print_step(2, "Loại bỏ ligand - chỉ giữ protein")
            PDBProcessor.extract_protein_only(str(local_pdb), str(clean_pdb))
            print_success("Đã tạo: protein_clean.pdb (chỉ chứa protein)")
            
            print(f"\n{Colors.WARNING}LƯU Ý: Ligand đã được loại bỏ!{Colors.END}")
            print("Để mô phỏng protein-ligand complex, bạn cần:")
            print("  1. Tạo topology cho ligand riêng (dùng SwissParam hoặc ACPYPE)")
            print("  2. Merge thủ công vào hệ thống")
            print("  Xem hướng dẫn: https://manual.gromacs.org/\n")
        else:
            # Check for ACE/NME caps
            with open(str(local_pdb), 'r') as f:
                content = f.read()
                has_nme = 'NME' in content or 'NMA' in content
                has_ace = ' ACE ' in content
            
            if has_nme or has_ace:
                print_step(2, "Loại bỏ ACE/NME capping groups")
                PDBProcessor.remove_nme_ace_caps(str(local_pdb), str(clean_pdb))
                print_success("Đã tạo: protein_clean.pdb")
            else:
                shutil.copy(local_pdb, clean_pdb)
                print_success("PDB không cần xử lý thêm")
        
        self.protein_pdb = clean_pdb
        return str(clean_pdb)
    
    def step_1_pdb2gmx(self, force_field: int = 6, water_model: int = 1) -> bool:
        """Step 1: Convert PDB to GROMACS format"""
        print_header("STEP 1: PDB2GMX - Tạo topology")
        
        if self.protein_pdb is None:
            self.preprocess_pdb()
        
        print("Force fields: 1-7: AMBER, 8: CHARMM27, 9-14: GROMOS, 15: OPLS-AA")
        print(f"{Colors.CYAN}Sử dụng: FF={force_field} (AMBER99SB-ILDN), Water={water_model} (TIP3P){Colors.END}\n")
        
        print_step(1, "Chạy pdb2gmx")
        pdb_name = self.protein_pdb.name
        
        try:
            run_command(
                f'gmx pdb2gmx -f "{pdb_name}" -o processed.gro -water tip3p -ignh',
                input_text=f"{force_field}\n{water_model}\n"
            )
            print_success("Đã tạo: processed.gro, topol.top, posre.itp")
            return True
        except subprocess.CalledProcessError:
            print_error("pdb2gmx thất bại!")
            return False
    
    def step_2_editconf(self, distance: float = 1.0, box_type: str = "cubic") -> bool:
        """Step 2: Define simulation box"""
        print_header("STEP 2: EDITCONF - Tạo hộp mô phỏng")
        
        if not check_file_exists("processed.gro", "(từ Step 1)"):
            return False
        
        print_step(1, f"Tạo hộp {box_type} với khoảng cách {distance} nm")
        try:
            run_command(f"gmx editconf -f processed.gro -d {distance} -bt {box_type} -o box.gro")
            print_success("Đã tạo: box.gro")
            return True
        except subprocess.CalledProcessError:
            return False
    
    def step_3_solvate(self) -> bool:
        """Step 3: Add solvent"""
        print_header("STEP 3: SOLVATE - Thêm nước")
        
        if not check_file_exists("box.gro", "(từ Step 2)"):
            return False
        
        print_step(1, "Thêm phân tử nước")
        try:
            run_command("gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o box_sol.gro")
            print_success("Đã tạo: box_sol.gro")
            return True
        except subprocess.CalledProcessError:
            return False
    
    def step_4_add_ions(self, concentration: float = 0.15) -> bool:
        """Step 4: Add ions"""
        print_header("STEP 4: GENION - Thêm ion trung hòa")
        
        if not check_file_exists("box_sol.gro", "(từ Step 3)"):
            return False
        if not check_file_exists("ions.mdp", "(file parameter)"):
            return False
        
        print_step(1, "Tạo TPR file")
        try:
            run_command("gmx grompp -f ions.mdp -c box_sol.gro -p topol.top -o ions.tpr -maxwarn 5")
        except subprocess.CalledProcessError:
            return False
        
        print_step(2, f"Thêm ion với nồng độ {concentration} M")
        try:
            # Group 13 is usually SOL
            run_command(
                f"gmx genion -s ions.tpr -p topol.top -conc {concentration} -neutral -o box_ion.gro",
                input_text="SOL\n"
            )
            print_success("Đã tạo: box_ion.gro")
            return True
        except subprocess.CalledProcessError:
            return False
    
    def step_5_energy_minimization(self) -> bool:
        """Step 5: Energy minimization"""
        print_header("STEP 5: ENERGY MINIMIZATION")
        
        if not check_file_exists("box_ion.gro", "(từ Step 4)"):
            return False
        if not check_file_exists("EM.mdp", "(file parameter)"):
            return False
        
        print_step(1, "Tạo TPR file")
        try:
            run_command("gmx grompp -f EM.mdp -c box_ion.gro -p topol.top -o em.tpr -maxwarn 5")
        except subprocess.CalledProcessError:
            return False
        
        print_step(2, "Chạy energy minimization")
        try:
            run_command("gmx mdrun -v -deffnm em")
            print_success("Energy minimization hoàn thành!")
            return True
        except subprocess.CalledProcessError:
            return False
    
    def step_6_nvt_equilibration(self) -> bool:
        """Step 6: NVT equilibration"""
        print_header("STEP 6: NVT EQUILIBRATION")
        
        if not check_file_exists("em.gro", "(từ Step 5)"):
            return False
        if not check_file_exists("NVT.mdp", "(file parameter)"):
            return False
        
        print_step(1, "Tạo TPR file")
        try:
            run_command("gmx grompp -f NVT.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 5")
        except subprocess.CalledProcessError:
            return False
        
        print_step(2, "Chạy NVT equilibration")
        try:
            run_command("gmx mdrun -deffnm nvt")
            print_success("NVT hoàn thành!")
            return True
        except subprocess.CalledProcessError:
            return False
    
    def step_7_npt_equilibration(self) -> bool:
        """Step 7: NPT equilibration"""
        print_header("STEP 7: NPT EQUILIBRATION")
        
        if not check_file_exists("nvt.gro", "(từ Step 6)"):
            return False
        if not check_file_exists("NPT.mdp", "(file parameter)"):
            return False
        
        print_step(1, "Tạo TPR file")
        try:
            run_command("gmx grompp -f NPT.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 5")
        except subprocess.CalledProcessError:
            return False
        
        print_step(2, "Chạy NPT equilibration")
        try:
            run_command("gmx mdrun -deffnm npt")
            print_success("NPT hoàn thành!")
            return True
        except subprocess.CalledProcessError:
            return False
    
    def step_8_production_md(self) -> bool:
        """Step 8: Production MD"""
        print_header("STEP 8: PRODUCTION MD")
        
        if not check_file_exists("npt.gro", "(từ Step 7)"):
            return False
        if not check_file_exists("MD.mdp", "(file parameter)"):
            return False
        
        print_step(1, "Tạo TPR file")
        try:
            run_command("gmx grompp -f MD.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 5")
        except subprocess.CalledProcessError:
            return False
        
        print_step(2, "Chạy Production MD")
        try:
            run_command("gmx mdrun -deffnm md")
            print_success("Production MD hoàn thành!")
            return True
        except subprocess.CalledProcessError:
            return False
    
    def step_9_analysis(self) -> bool:
        """Step 9: Analysis"""
        print_header("STEP 9: ANALYSIS")
        
        if not check_file_exists("md.tpr"):
            return False
        if not check_file_exists("md.xtc"):
            print_warning("md.xtc không tồn tại - có thể MD chưa hoàn thành")
            return False
        
        print_step(1, "Recentering trajectory")
        try:
            run_command(
                "gmx trjconv -s md.tpr -f md.xtc -o md_center.xtc -center -pbc mol -ur compact",
                input_text="Protein\nSystem\n"
            )
        except subprocess.CalledProcessError:
            print_warning("Không thể recenter trajectory")
        
        print_step(2, "Tính RMSD")
        try:
            run_command("gmx rms -s md.tpr -f md_center.xtc -o rmsd.xvg -tu ns", input_text="Backbone\nBackbone\n")
        except subprocess.CalledProcessError:
            print_warning("Không thể tính RMSD")
        
        print_step(3, "Tính RMSF")
        try:
            run_command("gmx rmsf -s md.tpr -f md_center.xtc -o rmsf.xvg", input_text="Backbone\n")
        except subprocess.CalledProcessError:
            print_warning("Không thể tính RMSF")
        
        print_step(4, "Tính Radius of Gyration")
        try:
            run_command("gmx gyrate -s md.tpr -f md_center.xtc -o gyrate.xvg", input_text="Protein\n")
        except subprocess.CalledProcessError:
            print_warning("Không thể tính Rg")
        
        print_success("Phân tích hoàn thành!")
        print(f"\n{Colors.GREEN}Output files:{Colors.END}")
        print("  - rmsd.xvg, rmsf.xvg, gyrate.xvg")
        print(f"{Colors.CYAN}Xem đồ thị: xmgrace <file.xvg>{Colors.END}")
        return True
    
    def run_full_simulation(self, start_step: int = 0) -> bool:
        """Run complete MD simulation"""
        print_header("BẮT ĐẦU MÔ PHỎNG MD")
        
        steps = [
            (0, "PREPROCESS", self.preprocess_pdb),
            (1, "PDB2GMX", self.step_1_pdb2gmx),
            (2, "EDITCONF", self.step_2_editconf),
            (3, "SOLVATE", self.step_3_solvate),
            (4, "GENION", self.step_4_add_ions),
            (5, "EM", self.step_5_energy_minimization),
            (6, "NVT", self.step_6_nvt_equilibration),
            (7, "NPT", self.step_7_npt_equilibration),
            (8, "MD", self.step_8_production_md),
            (9, "ANALYSIS", self.step_9_analysis),
        ]
        
        for step_num, step_name, step_func in steps:
            if step_num < start_step:
                continue
            
            try:
                result = step_func()
                if result is False:
                    print_error(f"Bước {step_num} ({step_name}) thất bại!")
                    resp = input("Tiếp tục bước tiếp theo? (y/n): ").strip().lower()
                    if resp != 'y':
                        print("Đã dừng mô phỏng.")
                        return False
            except Exception as e:
                print_error(f"Lỗi tại bước {step_num}: {e}")
                resp = input("Tiếp tục? (y/n): ").strip().lower()
                if resp != 'y':
                    return False
        
        print_header("MÔ PHỎNG HOÀN THÀNH!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="GROMACS MD Automation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python gromacs_md_auto.py install              # Cài đặt GROMACS
  python gromacs_md_auto.py run complex.pdb      # Chạy MD
  python gromacs_md_auto.py run complex.pdb -s 5 # Từ bước 5 (EM)
"""
    )
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Install
    install_parser = subparsers.add_parser('install', help='Cài đặt GROMACS')
    install_parser.add_argument('--check', action='store_true', help='Chỉ kiểm tra')
    
    # Run
    run_parser = subparsers.add_parser('run', help='Chạy MD simulation')
    run_parser.add_argument('pdb_file', help='File PDB đầu vào')
    run_parser.add_argument('-w', '--workdir', default='.', help='Thư mục làm việc')
    run_parser.add_argument('-s', '--step', type=int, default=0, help='Bắt đầu từ bước (0-9)')
    
    # Preprocess only
    prep_parser = subparsers.add_parser('prep', help='Chỉ preprocessing PDB')
    prep_parser.add_argument('pdb_file', help='File PDB')
    
    args = parser.parse_args()
    
    if args.command is None:
        # Interactive mode
        print_header("GROMACS MD AUTOMATION TOOL")
        print("1. Cài đặt GROMACS")
        print("2. Chạy MD simulation")
        print("3. Preprocessing PDB")
        print("4. Kiểm tra GROMACS")
        print("0. Thoát\n")
        
        choice = input("Chọn (0-4): ").strip()
        
        if choice == '1':
            GromacsInstaller.install_dependencies()
            GromacsInstaller.install_gromacs_quick()
            GromacsInstaller.install_additional_tools()
        elif choice == '2':
            pdb = input("Đường dẫn file PDB: ").strip()
            workdir = input("Thư mục làm việc (Enter = thư mục chứa PDB): ").strip()
            if not workdir:
                workdir = os.path.dirname(os.path.abspath(pdb)) or '.'
            step = input("Bắt đầu từ bước (0-9, Enter=0): ").strip()
            step = int(step) if step else 0
            
            sim = GromacsMDSimulation(workdir, pdb)
            sim.run_full_simulation(start_step=step)
        elif choice == '3':
            pdb = input("Đường dẫn file PDB: ").strip()
            workdir = os.path.dirname(os.path.abspath(pdb)) or '.'
            sim = GromacsMDSimulation(workdir, pdb)
            sim.preprocess_pdb()
        elif choice == '4':
            GromacsInstaller.check_gromacs()
        elif choice == '0':
            sys.exit(0)
    
    elif args.command == 'install':
        if args.check:
            GromacsInstaller.check_gromacs()
        else:
            GromacsInstaller.install_dependencies()
            GromacsInstaller.install_gromacs_quick()
            GromacsInstaller.install_additional_tools()
    
    elif args.command == 'run':
        workdir = os.path.abspath(args.workdir)
        sim = GromacsMDSimulation(workdir, args.pdb_file)
        sim.run_full_simulation(start_step=args.step)
    
    elif args.command == 'prep':
        workdir = os.path.dirname(os.path.abspath(args.pdb_file)) or '.'
        sim = GromacsMDSimulation(workdir, args.pdb_file)
        sim.preprocess_pdb()


if __name__ == "__main__":
    main()
