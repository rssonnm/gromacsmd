# gromacsmd

Công cụ Python tự động hóa mô phỏng Molecular Dynamics với GROMACS.  
Hỗ trợ: **Protein đơn thuần** hoặc **Protein-Ligand Complex**

---

## 📋 Yêu cầu hệ thống

- Ubuntu/Debian (WSL hoặc native Linux)
- Python 3.8+
- GROMACS 2020+
- ACPYPE (cho ligand topology)

---

## 🔧 Cài đặt

### 1. Cài đặt tự động
```bash
python3 gromacs_md_auto.py install
```

### 2. Cài đặt thủ công
```bash
# GROMACS
sudo apt update
sudo apt install -y gromacs

# ACPYPE (cho protein-ligand)
pipx install acpype
# hoặc
python3 -m venv ~/gromacs_venv
source ~/gromacs_venv/bin/activate
pip install acpype
```

### 3. Kiểm tra cài đặt
```bash
python3 gromacs_md_auto.py check
```

---

## 🚀 Sử dụng

### Chế độ Interactive
```bash
python3 gromacs_md_auto.py
```

### Chế độ Command-line
```bash
# Chạy toàn bộ workflow
python3 gromacs_md_auto.py run complex.pdb

# Chạy từ step X đến step Y
python3 gromacs_md_auto.py run complex.pdb . 0 7

# Chỉ định thư mục làm việc
python3 gromacs_md_auto.py run complex.pdb /path/to/workdir 0 11
```

---

## 📊 Các bước (Steps)

| Step | Tên | Mô tả |
|------|-----|-------|
| 0 | Preprocess | Phân tích PDB, tách protein/ligand |
| 1 | Protein Topology | Tạo topology protein (pdb2gmx) |
| 2 | Ligand Topology | Tạo topology ligand (ACPYPE) |
| 3 | Merge System | Gộp protein + ligand |
| 4 | Box | Tạo hộp mô phỏng |
| 5 | Solvate | Thêm phân tử nước |
| 6 | Ions | Thêm ion trung hòa |
| 7 | Energy Min | Tối ưu hóa năng lượng |
| 8 | NVT | Cân bằng nhiệt độ |
| 9 | NPT | Cân bằng áp suất |
| 10 | Production MD | Chạy mô phỏng chính |
| 11 | Analysis | Phân tích RMSD, RMSF, Rg |

---

## 📁 File cần có

Đặt các file `.mdp` trong cùng thư mục với PDB:

```
workdir/
├── complex.pdb      # File PDB đầu vào
├── ions.mdp         # Parameter cho ion
├── EM.mdp           # Energy minimization
├── NVT.mdp          # NVT equilibration
├── NPT.mdp          # NPT equilibration
└── MD.mdp           # Production MD
```

---

## 📈 Kết quả & Phân tích

### File output chính
| File | Mô tả |
|------|-------|
| `md.xtc` | Trajectory |
| `md.edr` | Năng lượng |
| `md.gro` | Cấu trúc cuối |
| `rmsd.xvg` | RMSD theo thời gian |
| `rmsf.xvg` | RMSF theo residue |
| `gyrate.xvg` | Bán kính quay |

### Xem kết quả
```bash
# Đồ thị
xmgrace rmsd.xvg

# Trajectory (cần VMD)
vmd md.gro md.xtc
```

### Ý nghĩa các chỉ số
- **RMSD < 0.3 nm**: Cấu trúc ổn định
- **RMSF cao**: Vùng linh động (loop, terminal)
- **Rg ổn định**: Protein không unfold

---

## ⚠️ Xử lý lỗi thường gặp

### 1. Lỗi "Residue not found"
```
File PDB có residue lạ. Tool sẽ tự xử lý, nhưng nếu vẫn lỗi:
- Kiểm tra file PDB có đúng format không
- Thử loại bỏ HETATM thủ công
```

### 2. ACPYPE chạy lâu
```
Bình thường! Thời gian phụ thuộc kích thước ligand:
- 30 atoms: ~5 phút
- 67 atoms: ~20 phút
- 100+ atoms: 1+ giờ
```

### 3. Lỗi pip install acpype
```bash
# Dùng pipx thay thế
pipx install acpype

# Hoặc dùng virtual environment
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install acpype
```

---

## 📚 Tham khảo

- [GROMACS Manual](https://manual.gromacs.org/)
- [ACPYPE](https://github.com/alanwilter/acpype)
- [GROMACS Tutorials](http://www.mdtutorials.com/gmx/)

---
