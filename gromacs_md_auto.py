#!/usr/bin/env python3
"""
GROMACS MD Automation Tool - Protein-Ligand Complex
Công cụ tự động hóa mô phỏng MD với GROMACS
Hỗ trợ: Protein đơn thuần HOẶC Protein-Ligand complex
Yêu cầu: GROMACS, ACPYPE (cho ligand topology)
"""

import os
import sys
import subprocess
import shutil
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict


class Colors:
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


def run_cmd(cmd: str, input_text: Optional[str] = None, check: bool = True) -> bool:
    """Run shell command"""
    print(f"{Colors.BLUE}$ {cmd}{Colors.END}")
    try:
        result = subprocess.run(
            cmd, shell=True,
            input=input_text.encode() if input_text else None,
            capture_output=False,
            check=check
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
            'ligand_name': ligand_name,
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
                    if res.upper() == lig_name.upper():
                        lines.append(line)
        lines.append("END\n")
        with open(output, 'w') as f:
            f.writelines(lines)


class GromacsInstaller:
    """Install GROMACS and dependencies"""
    
    @staticmethod
    def install_all():
        print_header("CÀI ĐẶT GROMACS VÀ DEPENDENCIES")
        
        cmds = [
            "sudo apt update",
            "sudo apt install -y gromacs",
            "sudo apt install -y python3-pip",
            "pip3 install acpype",
            "sudo apt install -y grace"
        ]
        
        for i, cmd in enumerate(cmds, 1):
            print_step(i, cmd)
            run_cmd(cmd, check=False)
        
        print_success("Hoàn thành cài đặt!")
    
    @staticmethod
    def check():
        print_header("KIỂM TRA CÀI ĐẶT")
        
        # Check GROMACS
        result = subprocess.run("gmx --version", shell=True, capture_output=True)
        if result.returncode == 0:
            print_success("GROMACS: OK")
        else:
            print_error("GROMACS: Chưa cài đặt")
        
        # Check ACPYPE
        result = subprocess.run("acpype -h", shell=True, capture_output=True)
        if result.returncode == 0:
            print_success("ACPYPE: OK")
        else:
            print_warning("ACPYPE: Chưa cài (cần cho ligand topology)")
            print_info("Cài đặt: pip3 install acpype")


class MDSimulation:
    """GROMACS MD Simulation for Protein-Ligand Complex"""
    
    def __init__(self, workdir: str, pdb_file: str):
        self.workdir = Path(workdir).resolve()
        self.pdb_file = Path(pdb_file).resolve()
        self.has_ligand = False
        self.ligand_name = "LIG"
        
        os.makedirs(self.workdir, exist_ok=True)
        os.chdir(self.workdir)
        
        print(f"{Colors.CYAN}Working dir: {self.workdir}{Colors.END}")
        print(f"{Colors.CYAN}PDB file: {self.pdb_file}{Colors.END}")
    
    def step0_preprocess(self) -> bool:
        """Analyze and split PDB if needed"""
        print_header("STEP 0: PREPROCESSING")
        
        # Copy PDB to workdir
        local_pdb = self.workdir / self.pdb_file.name
        if self.pdb_file != local_pdb:
            shutil.copy(self.pdb_file, local_pdb)
        
        # Analyze
        print_step(1, "Phân tích PDB file")
        info = PDBHandler.analyze(str(local_pdb))
        
        print(f"  Protein atoms: {info['protein_atoms']}")
        print(f"  Ligand atoms: {info['ligand_atoms']}")
        if info['has_ligand']:
            print(f"  Ligand name: {info['ligand_name']}")
        
        self.has_ligand = info['has_ligand']
        self.ligand_name = info['ligand_name'] or "LIG"
        
        # Extract protein
        print_step(2, "Tách protein")
        PDBHandler.extract_protein(str(local_pdb), "protein.pdb")
        print_success("Đã tạo: protein.pdb")
        
        if self.has_ligand:
            print_step(3, "Tách ligand")
            PDBHandler.extract_ligand(str(local_pdb), "ligand.pdb", self.ligand_name)
            print_success(f"Đã tạo: ligand.pdb (residue: {self.ligand_name})")
        
        return True
    
    def step1_protein_topology(self, ff: int = 6) -> bool:
        """Generate protein topology"""
        print_header("STEP 1: PROTEIN TOPOLOGY")
        
        if not file_exists("protein.pdb"):
            print_error("Không tìm thấy protein.pdb")
            return False
        
        print_step(1, f"Chạy pdb2gmx (force field {ff})")
        
        success = run_cmd(
            'gmx pdb2gmx -f protein.pdb -o protein.gro -p topol.top -ignh',
            input_text=f"{ff}\n1\n"
        )
        
        if success and file_exists("protein.gro"):
            print_success("Đã tạo: protein.gro, topol.top")
            return True
        return False
    
    def step2_ligand_topology(self) -> bool:
        """Generate ligand topology using ACPYPE"""
        if not self.has_ligand:
            print_info("Không có ligand, bỏ qua bước này")
            return True
        
        print_header("STEP 2: LIGAND TOPOLOGY (ACPYPE)")
        
        if not file_exists("ligand.pdb"):
            print_error("Không tìm thấy ligand.pdb")
            return False
        
        print_step(1, "Chạy ACPYPE để tạo ligand topology")
        print_info("Quá trình này có thể mất vài phút...")
        
        success = run_cmd(f'acpype -i ligand.pdb -c bcc -n 0 -a gaff2', check=False)
        
        # Find ACPYPE output directory
        acpype_dir = None
        for d in os.listdir('.'):
            if d.startswith('ligand.acpype'):
                acpype_dir = d
                break
        
        if acpype_dir and file_exists(f"{acpype_dir}/ligand_GMX.gro"):
            print_step(2, "Copy ligand files")
            shutil.copy(f"{acpype_dir}/ligand_GMX.gro", "ligand.gro")
            shutil.copy(f"{acpype_dir}/ligand_GMX.itp", "ligand.itp")
            
            # Get ligand atomtypes
            if file_exists(f"{acpype_dir}/ligand_GMX_atomtypes.itp"):
                shutil.copy(f"{acpype_dir}/ligand_GMX_atomtypes.itp", "ligand_atomtypes.itp")
            
            print_success("Đã tạo: ligand.gro, ligand.itp")
            return True
        else:
            print_error("ACPYPE thất bại!")
            print_info("Thử cài đặt: pip3 install acpype")
            print_info("Hoặc dùng SwissParam (https://swissparam.ch)")
            return False
    
    def step3_merge_system(self) -> bool:
        """Merge protein and ligand"""
        print_header("STEP 3: MERGE SYSTEM")
        
        if not file_exists("protein.gro"):
            print_error("Không tìm thấy protein.gro")
            return False
        
        if not self.has_ligand:
            # Just copy protein
            shutil.copy("protein.gro", "complex.gro")
            print_success("Đã tạo: complex.gro (protein only)")
            return True
        
        if not file_exists("ligand.gro"):
            print_error("Không tìm thấy ligand.gro")
            return False
        
        print_step(1, "Merge coordinates")
        
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
            f.write("Protein-Ligand Complex\n")
            f.write(f"{total_atoms}\n")
            f.writelines(prot_atom_lines)
            f.writelines(lig_atom_lines)
            f.write(box_line)
        
        print_success(f"Đã tạo: complex.gro ({total_atoms} atoms)")
        
        # Update topology
        print_step(2, "Cập nhật topology")
        
        with open("topol.top", 'r') as f:
            top_content = f.read()
        
        # Add ligand atomtypes after forcefield include
        if file_exists("ligand_atomtypes.itp"):
            atomtypes_include = '\n; Include ligand atomtypes\n#include "ligand_atomtypes.itp"\n'
            top_content = re.sub(
                r'(#include.*forcefield\.itp.*\n)',
                r'\1' + atomtypes_include,
                top_content
            )
        
        # Add ligand itp before position restraints
        lig_include = '\n; Include ligand topology\n#include "ligand.itp"\n'
        top_content = re.sub(
            r'(; Include Position restraint file)',
            lig_include + r'\n\1',
            top_content
        )
        
        # Add ligand to molecules section
        top_content = top_content.rstrip() + f"\n{self.ligand_name.upper()}     1\n"
        
        with open("topol.top", 'w') as f:
            f.write(top_content)
        
        print_success("Đã cập nhật: topol.top")
        return True
    
    def step4_box(self, dist: float = 1.0) -> bool:
        """Create simulation box"""
        print_header("STEP 4: CREATE BOX")
        
        input_file = "complex.gro" if file_exists("complex.gro") else "protein.gro"
        if not file_exists(input_file):
            print_error(f"Không tìm thấy {input_file}")
            return False
        
        print_step(1, f"Tạo hộp cubic, khoảng cách {dist} nm")
        return run_cmd(f"gmx editconf -f {input_file} -d {dist} -bt cubic -o box.gro")
    
    def step5_solvate(self) -> bool:
        """Add water"""
        print_header("STEP 5: SOLVATE")
        
        if not file_exists("box.gro"):
            return False
        
        print_step(1, "Thêm phân tử nước")
        return run_cmd("gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o solv.gro")
    
    def step6_ions(self, conc: float = 0.15) -> bool:
        """Add ions"""
        print_header("STEP 6: ADD IONS")
        
        if not file_exists("solv.gro") or not file_exists("ions.mdp"):
            print_error("Cần solv.gro và ions.mdp")
            return False
        
        print_step(1, "Tạo TPR")
        if not run_cmd("gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 5"):
            return False
        
        print_step(2, f"Thêm ion ({conc} M)")
        return run_cmd(
            f"gmx genion -s ions.tpr -p topol.top -conc {conc} -neutral -o ions.gro",
            input_text="SOL\n"
        )
    
    def step7_em(self) -> bool:
        """Energy minimization"""
        print_header("STEP 7: ENERGY MINIMIZATION")
        
        if not file_exists("ions.gro") or not file_exists("EM.mdp"):
            print_error("Cần ions.gro và EM.mdp")
            return False
        
        print_step(1, "Tạo TPR")
        if not run_cmd("gmx grompp -f EM.mdp -c ions.gro -p topol.top -o em.tpr -maxwarn 5"):
            return False
        
        print_step(2, "Chạy EM")
        return run_cmd("gmx mdrun -v -deffnm em")
    
    def step8_nvt(self) -> bool:
        """NVT equilibration"""
        print_header("STEP 8: NVT EQUILIBRATION")
        
        if not file_exists("em.gro") or not file_exists("NVT.mdp"):
            return False
        
        print_step(1, "Tạo TPR")
        if not run_cmd("gmx grompp -f NVT.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 5"):
            return False
        
        print_step(2, "Chạy NVT")
        return run_cmd("gmx mdrun -deffnm nvt")
    
    def step9_npt(self) -> bool:
        """NPT equilibration"""
        print_header("STEP 9: NPT EQUILIBRATION")
        
        if not file_exists("nvt.gro") or not file_exists("NPT.mdp"):
            return False
        
        print_step(1, "Tạo TPR")
        if not run_cmd("gmx grompp -f NPT.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 5"):
            return False
        
        print_step(2, "Chạy NPT")
        return run_cmd("gmx mdrun -deffnm npt")
    
    def step10_md(self) -> bool:
        """Production MD"""
        print_header("STEP 10: PRODUCTION MD")
        
        if not file_exists("npt.gro") or not file_exists("MD.mdp"):
            return False
        
        print_step(1, "Tạo TPR")
        if not run_cmd("gmx grompp -f MD.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 5"):
            return False
        
        print_step(2, "Chạy Production MD")
        return run_cmd("gmx mdrun -deffnm md")
    
    def step11_analysis(self) -> bool:
        """Post-simulation analysis"""
        print_header("STEP 11: ANALYSIS")
        
        if not file_exists("md.tpr") or not file_exists("md.xtc"):
            print_warning("MD chưa hoàn thành")
            return False
        
        print_step(1, "RMSD")
        run_cmd("gmx rms -s md.tpr -f md.xtc -o rmsd.xvg -tu ns", input_text="Backbone\nBackbone\n", check=False)
        
        print_step(2, "RMSF")
        run_cmd("gmx rmsf -s md.tpr -f md.xtc -o rmsf.xvg", input_text="Backbone\n", check=False)
        
        print_step(3, "Radius of Gyration")
        run_cmd("gmx gyrate -s md.tpr -f md.xtc -o gyrate.xvg", input_text="Protein\n", check=False)
        
        print_success("Hoàn thành phân tích!")
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
                print_error(f"Step {num} ({name}) thất bại!")
                resp = input("Tiếp tục? (y/n): ").strip().lower()
                if resp != 'y':
                    return
        
        print_header("MÔ PHỎNG HOÀN THÀNH!")


def main():
    print_header("GROMACS MD AUTOMATION TOOL")
    print("Hỗ trợ Protein đơn thuần hoặc Protein-Ligand Complex\n")
    print("1. Cài đặt GROMACS + ACPYPE")
    print("2. Chạy MD Simulation")
    print("3. Kiểm tra cài đặt")
    print("0. Thoát\n")
    
    choice = input("Chọn (0-3): ").strip()
    
    if choice == '1':
        GromacsInstaller.install_all()
    
    elif choice == '2':
        pdb = input("Đường dẫn file PDB: ").strip()
        if not file_exists(pdb):
            print_error(f"File không tồn tại: {pdb}")
            return
        
        workdir = input("Thư mục làm việc (Enter = thư mục chứa PDB): ").strip()
        if not workdir:
            workdir = os.path.dirname(os.path.abspath(pdb)) or '.'
        
        step_start = input("Bắt đầu từ bước (0-11, Enter=0): ").strip()
        step_start = int(step_start) if step_start.isdigit() else 0
        
        step_end = input("Kết thúc tại bước (0-11, Enter=11): ").strip()
        step_end = int(step_end) if step_end.isdigit() else 11
        
        sim = MDSimulation(workdir, pdb)
        sim.run_all(start=step_start, end=step_end)
    
    elif choice == '3':
        GromacsInstaller.check()
    
    elif choice == '0':
        print("Tạm biệt!")


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
        else:
            print("Usage:")
            print("  python gromacs_md_auto.py                      # Interactive mode")
            print("  python gromacs_md_auto.py install              # Install dependencies")
            print("  python gromacs_md_auto.py check                # Check installation")
            print("  python gromacs_md_auto.py run <pdb> [workdir] [start] [end]")
            print("\nVí dụ:")
            print("  python gromacs_md_auto.py run complex.pdb . 0 7   # Chạy step 0-7")
            print("  python gromacs_md_auto.py run complex.pdb . 3     # Chạy từ step 3 đến 11")
    else:
        main()
