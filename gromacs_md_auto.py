#!/usr/bin/env python3
"""
GROMACS MD Automation Tool
Công cụ tự động hóa mô phỏng Dynamics phân tử với GROMACS
Author: Auto-generated
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path
from typing import Optional


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
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_step(step_num: int, text: str):
    """Print step information"""
    print(f"{Colors.CYAN}[Step {step_num}]{Colors.END} {Colors.BOLD}{text}{Colors.END}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.END}")


def run_command(cmd: str, input_text: Optional[str] = None, cwd: Optional[str] = None, 
                check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """
    Run a shell command with optional input
    
    Args:
        cmd: Command to execute
        input_text: Optional input to send to the command
        cwd: Working directory
        check: Whether to raise exception on failure
        capture: Whether to capture output
    """
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


class GromacsInstaller:
    """Handle GROMACS installation"""
    
    @staticmethod
    def install_dependencies():
        """Install required system dependencies"""
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
                run_command(cmd)
                print_success(f"{desc} - Hoàn thành!")
            except subprocess.CalledProcessError:
                print_warning(f"{desc} - Có thể đã cài đặt hoặc gặp lỗi")
    
    @staticmethod
    def install_gromacs_quick():
        """Quick GROMACS installation via apt"""
        print_header("CÀI ĐẶT GROMACS (QUICK INSTALL)")
        
        print_step(1, "Cài đặt GROMACS từ apt repository")
        try:
            run_command("sudo apt install -y gromacs")
            print_success("GROMACS đã được cài đặt thành công!")
        except subprocess.CalledProcessError:
            print_error("Không thể cài đặt GROMACS. Vui lòng thử cài đặt thủ công.")
    
    @staticmethod
    def install_additional_tools():
        """Install additional visualization tools"""
        print_header("CÀI ĐẶT CÔNG CỤ BỔ SUNG")
        
        tools = [
            ("PyMOL", "sudo apt-get install -y pymol"),
            ("Grace", "sudo apt-get install -y grace"),
        ]
        
        for i, (name, cmd) in enumerate(tools, 1):
            print_step(i, f"Cài đặt {name}")
            try:
                run_command(cmd)
                print_success(f"{name} đã được cài đặt!")
            except subprocess.CalledProcessError:
                print_warning(f"Không thể cài đặt {name}")
    
    @staticmethod
    def check_gromacs():
        """Check if GROMACS is installed"""
        print_header("KIỂM TRA CÀI ĐẶT GROMACS")
        
        try:
            result = run_command("gmx --version", capture=True, check=False)
            if result.returncode == 0:
                print_success("GROMACS đã được cài đặt!")
                return True
            else:
                print_warning("GROMACS chưa được cài đặt hoặc chưa được thêm vào PATH")
                return False
        except FileNotFoundError:
            print_error("Không tìm thấy lệnh gmx")
            return False


class GromacsMDSimulation:
    """Handle GROMACS MD Simulation workflow"""
    
    def __init__(self, working_dir: str, pdb_file: str):
        """
        Initialize MD simulation
        
        Args:
            working_dir: Working directory path
            pdb_file: Path to the input PDB file (protein-ligand complex)
        """
        self.working_dir = Path(working_dir).resolve()
        self.pdb_file = Path(pdb_file).resolve()
        
        # Check if files exist
        if not self.pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {self.pdb_file}")
        
        os.chdir(self.working_dir)
        print(f"{Colors.CYAN}Working directory: {self.working_dir}{Colors.END}")
        print(f"{Colors.CYAN}Input PDB file: {self.pdb_file}{Colors.END}")
    
    def source_gromacs(self):
        """Source GROMACS environment (for manual installation)"""
        gmxrc_path = "/usr/local/gromacs/bin/GMXRC"
        if os.path.exists(gmxrc_path):
            print_step(0, "Sourcing GROMACS environment")
            # Note: This won't work in subprocess, user needs to source manually
            print_warning(f"Please run: source {gmxrc_path}")
    
    def step_1_pdb2gmx(self, force_field: int = 8, water_model: int = 1):
        """
        Step 1: Convert PDB to GROMACS format
        
        Args:
            force_field: Force field selection (default: 8 = CHARMM27)
            water_model: Water model selection (default: 1 = TIP3P)
        """
        print_header("STEP 1: PDB2GMX - Chuyển đổi cấu trúc")
        
        print_step(1, "Chạy pdb2gmx để tạo topology")
        
        # Copy PDB file to working directory if not already there
        if self.pdb_file.parent != self.working_dir:
            shutil.copy(self.pdb_file, self.working_dir)
        
        pdb_name = self.pdb_file.name
        input_text = f"{force_field}\n{water_model}\n"
        
        run_command(
            f"gmx pdb2gmx -f \"{pdb_name}\" -o processed.gro -ignh",
            input_text=input_text
        )
        print_success("Đã tạo: conf.gro, topol.top, posre.itp")
    
    def step_2_editconf(self, distance: float = 1.0, box_type: str = "triclinic"):
        """
        Step 2: Define simulation box
        
        Args:
            distance: Distance from solute to box edge (nm)
            box_type: Box type (triclinic, cubic, dodecahedron)
        """
        print_header("STEP 2: EDITCONF - Tạo hộp mô phỏng")
        
        print_step(1, f"Tạo hộp {box_type} với khoảng cách {distance} nm")
        run_command(f"gmx editconf -f processed.gro -d {distance} -bt {box_type} -o box.gro")
        print_success("Đã tạo: box.gro")
    
    def step_3_solvate(self):
        """Step 3: Add solvent (water)"""
        print_header("STEP 3: SOLVATE - Thêm dung môi")
        
        print_step(1, "Thêm phân tử nước vào hộp")
        run_command("gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o box_sol.gro")
        print_success("Đã tạo: box_sol.gro")
    
    def step_4_add_ions(self, concentration: float = 0.1):
        """
        Step 4: Add ions to neutralize system
        
        Args:
            concentration: Ion concentration (M)
        """
        print_header("STEP 4: ADD IONS - Thêm ion trung hòa")
        
        print_step(1, "Tạo file TPR cho genion")
        run_command("gmx grompp -f ions.mdp -c box_sol.gro -maxwarn 2 -p topol.top -o ION.tpr")
        
        print_step(2, f"Thêm ion với nồng độ {concentration} M")
        # Select SOL group (usually group 15 or similar)
        input_text = "SOL\n"
        run_command(
            f"gmx genion -s ION.tpr -p topol.top -conc {concentration} -neutral -o box_sol_ion.gro",
            input_text=input_text
        )
        print_success("Đã tạo: box_sol_ion.gro")
    
    def step_5_energy_minimization(self):
        """Step 5: Energy minimization"""
        print_header("STEP 5: ENERGY MINIMIZATION - Tối ưu hóa năng lượng")
        
        print_step(1, "Tạo file TPR cho EM")
        run_command("gmx grompp -f EM.mdp -c box_sol_ion.gro -maxwarn 2 -p topol.top -o EM.tpr")
        
        print_step(2, "Chạy energy minimization")
        run_command("gmx mdrun -v -deffnm EM")
        print_success("Energy minimization hoàn thành!")
    
    def step_6_make_index(self):
        """Step 6: Create index file"""
        print_header("STEP 6: MAKE INDEX - Tạo file index")
        
        print_step(1, "Tạo index file cho hệ thống")
        # Create a simple index file combining protein and other groups
        input_text = "1 | 13\nq\n"
        run_command("gmx make_ndx -f EM.gro -o index.ndx", input_text=input_text)
        print_success("Đã tạo: index.ndx")
    
    def step_7_nvt_equilibration(self):
        """Step 7: NVT equilibration"""
        print_header("STEP 7: NVT EQUILIBRATION - Cân bằng nhiệt độ")
        
        print_step(1, "Tạo file TPR cho NVT")
        run_command("gmx grompp -f NVT.mdp -c EM.gro -r EM.gro -p topol.top -n index.ndx -maxwarn 2 -o NVT.tpr")
        
        print_step(2, "Chạy NVT equilibration")
        run_command("gmx mdrun -deffnm NVT")
        print_success("NVT equilibration hoàn thành!")
    
    def step_8_npt_equilibration(self):
        """Step 8: NPT equilibration"""
        print_header("STEP 8: NPT EQUILIBRATION - Cân bằng áp suất")
        
        print_step(1, "Tạo file TPR cho NPT")
        run_command("gmx grompp -f NPT.mdp -c NVT.gro -r NVT.gro -p topol.top -n index.ndx -maxwarn 2 -o NPT.tpr")
        
        print_step(2, "Chạy NPT equilibration")
        run_command("gmx mdrun -deffnm NPT")
        print_success("NPT equilibration hoàn thành!")
    
    def step_9_production_md(self):
        """Step 9: Production MD run"""
        print_header("STEP 9: PRODUCTION MD - Chạy mô phỏng chính")
        
        print_step(1, "Tạo file TPR cho MD")
        run_command("gmx grompp -f MD.mdp -c NPT.gro -t NPT.cpt -p topol.top -n index.ndx -maxwarn 2 -o MD.tpr")
        
        print_step(2, "Chạy Production MD")
        run_command("gmx mdrun -deffnm MD")
        print_success("Production MD hoàn thành!")
    
    def step_10_analysis(self):
        """Step 10: Post-simulation analysis"""
        print_header("STEP 10: ANALYSIS - Phân tích kết quả")
        
        # Recentering and rewrapping
        print_step(1, "Recentering và rewrapping trajectory")
        input_text = "Protein\nSystem\n"
        run_command(
            "gmx trjconv -s MD.tpr -f MD.xtc -o MD_center.xtc -center -pbc mol -ur compact",
            input_text=input_text
        )
        
        # Extract first frame
        print_step(2, "Trích xuất frame đầu tiên")
        input_text = "System\n"
        run_command(
            "gmx trjconv -s MD.tpr -f MD_center.xtc -o start.pdb -dump 0",
            input_text=input_text
        )
        
        # RMSD calculation
        print_step(3, "Tính toán RMSD")
        input_text = "4\n4\n"  # Backbone for both
        run_command(
            "gmx rms -s MD.tpr -f MD_center.xtc -o rmsd.xvg -tu ns",
            input_text=input_text
        )
        
        # RMSF calculation
        print_step(4, "Tính toán RMSF")
        input_text = "4\n"  # Backbone
        run_command(
            "gmx rmsf -s MD.tpr -f MD_center.xtc -o rmsf.xvg",
            input_text=input_text
        )
        
        # Radius of gyration
        print_step(5, "Tính toán bán kính quay (Gyration)")
        input_text = "1\n"  # Protein
        run_command(
            "gmx gyrate -s MD.tpr -f MD_center.xtc -o gyrate.xvg",
            input_text=input_text
        )
        
        # Energy analysis
        print_step(6, "Phân tích năng lượng")
        input_text = "10 11 12\n\n"  # Potential, Kinetic, Total
        run_command(
            "gmx energy -f MD.edr -o energy.xvg",
            input_text=input_text
        )
        
        print_success("Phân tích hoàn thành!")
        print(f"\n{Colors.GREEN}Các file kết quả:{Colors.END}")
        print("  - rmsd.xvg: RMSD theo thời gian")
        print("  - rmsf.xvg: RMSF theo residue")
        print("  - gyrate.xvg: Bán kính quay")
        print("  - energy.xvg: Năng lượng hệ thống")
        print(f"\n{Colors.CYAN}Sử dụng 'xmgrace <file.xvg>' để xem đồ thị{Colors.END}")
    
    def run_full_simulation(self, skip_to_step: int = 1):
        """
        Run the complete MD simulation workflow
        
        Args:
            skip_to_step: Start from this step (1-10)
        """
        print_header("BẮT ĐẦU MÔ PHỎNG DYNAMICS PHÂN TỬ")
        
        steps = [
            (1, "PDB2GMX", self.step_1_pdb2gmx),
            (2, "EDITCONF", self.step_2_editconf),
            (3, "SOLVATE", self.step_3_solvate),
            (4, "ADD IONS", self.step_4_add_ions),
            (5, "ENERGY MINIMIZATION", self.step_5_energy_minimization),
            (6, "MAKE INDEX", self.step_6_make_index),
            (7, "NVT EQUILIBRATION", self.step_7_nvt_equilibration),
            (8, "NPT EQUILIBRATION", self.step_8_npt_equilibration),
            (9, "PRODUCTION MD", self.step_9_production_md),
            (10, "ANALYSIS", self.step_10_analysis),
        ]
        
        for step_num, step_name, step_func in steps:
            if step_num >= skip_to_step:
                try:
                    step_func()
                except subprocess.CalledProcessError as e:
                    print_error(f"Lỗi tại bước {step_num}: {step_name}")
                    print_error(f"Exit code: {e.returncode}")
                    response = input(f"Bạn có muốn tiếp tục? (y/n): ")
                    if response.lower() != 'y':
                        print("Đã dừng mô phỏng.")
                        return
        
        print_header("MÔ PHỎNG HOÀN THÀNH!")
        print_success("Tất cả các bước đã hoàn thành thành công!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="GROMACS MD Automation Tool - Công cụ tự động hóa mô phỏng MD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python gromacs_md_auto.py install          # Cài đặt GROMACS và dependencies
  python gromacs_md_auto.py run complex.pdb  # Chạy mô phỏng MD
  python gromacs_md_auto.py run complex.pdb --step 5  # Tiếp tục từ bước 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Các lệnh có sẵn')
    
    # Install command
    install_parser = subparsers.add_parser('install', help='Cài đặt GROMACS và các công cụ')
    install_parser.add_argument('--deps-only', action='store_true', 
                               help='Chỉ cài đặt dependencies, không cài GROMACS')
    install_parser.add_argument('--check', action='store_true',
                               help='Kiểm tra GROMACS đã được cài đặt chưa')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Chạy mô phỏng MD')
    run_parser.add_argument('pdb_file', type=str, help='File PDB đầu vào (protein-ligand complex)')
    run_parser.add_argument('--workdir', '-w', type=str, default='.',
                           help='Thư mục làm việc (mặc định: thư mục hiện tại)')
    run_parser.add_argument('--step', '-s', type=int, default=1,
                           help='Bắt đầu từ bước (1-10)')
    run_parser.add_argument('--ff', type=int, default=8,
                           help='Force field (mặc định: 8 = CHARMM27)')
    run_parser.add_argument('--water', type=int, default=1,
                           help='Water model (mặc định: 1 = TIP3P)')
    
    args = parser.parse_args()
    
    if args.command is None:
        # Interactive mode
        print_header("GROMACS MD AUTOMATION TOOL")
        print("Chọn một tùy chọn:")
        print("  1. Cài đặt GROMACS và dependencies")
        print("  2. Chạy mô phỏng MD")
        print("  3. Kiểm tra cài đặt GROMACS")
        print("  0. Thoát")
        print()
        
        choice = input("Nhập lựa chọn (0-3): ").strip()
        
        if choice == '1':
            installer = GromacsInstaller()
            installer.install_dependencies()
            installer.install_gromacs_quick()
            installer.install_additional_tools()
            
        elif choice == '2':
            pdb_file = input("Nhập đường dẫn file PDB: ").strip()
            workdir = input("Nhập thư mục làm việc (Enter để dùng thư mục hiện tại): ").strip()
            if not workdir:
                workdir = os.path.dirname(os.path.abspath(pdb_file)) or '.'
            
            step = input("Bắt đầu từ bước (1-10, Enter cho bước 1): ").strip()
            step = int(step) if step else 1
            
            sim = GromacsMDSimulation(workdir, pdb_file)
            sim.run_full_simulation(skip_to_step=step)
            
        elif choice == '3':
            installer = GromacsInstaller()
            installer.check_gromacs()
            
        elif choice == '0':
            print("Tạm biệt!")
            sys.exit(0)
        else:
            print_error("Lựa chọn không hợp lệ!")
            
    elif args.command == 'install':
        installer = GromacsInstaller()
        if args.check:
            installer.check_gromacs()
        elif args.deps_only:
            installer.install_dependencies()
        else:
            installer.install_dependencies()
            installer.install_gromacs_quick()
            installer.install_additional_tools()
            
    elif args.command == 'run':
        # Resolve paths
        pdb_path = os.path.abspath(args.pdb_file)
        work_dir = os.path.abspath(args.workdir)
        
        if not os.path.exists(pdb_path):
            print_error(f"File không tồn tại: {pdb_path}")
            sys.exit(1)
        
        sim = GromacsMDSimulation(work_dir, pdb_path)
        sim.run_full_simulation(skip_to_step=args.step)


if __name__ == "__main__":
    main()
