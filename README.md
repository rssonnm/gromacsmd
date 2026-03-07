# GROMACS MD Automation Pipeline (gromacsmda)

Công cụ này tự động hóa toàn bộ quy trình mô phỏng Molecular Dynamics (MD) sử dụng GROMACS, hỗ trợ cả hệ Protein đơn thuần và phức hợp Protein-Ligand.

## Yêu cầu hệ thống (Prerequisites)

- **Hệ điều hành**: Linux (Ubuntu/Debian recommended) hoặc WSL trên Windows.
- **Phần cứng**: CPU đa nhân (Khuyến nghị có GPU NVIDIA để tăng tốc mô phỏng).
- **Phần mềm**: Python 3.8+

## Cài đặt (Installation)

### Cách 1: Cài đặt tự động qua script (Ubuntu/Debian)
Chạy script để tự động cài đặt các dependency cần thiết:
```bash
python3 gromacsmda.py install
```
*Lưu ý: Bạn có thể cần nhập mật khẩu sudo để cài đặt các gói hệ thống.*

### Cách 2: Cài đặt script và Conda (Khuyến nghị cho Ligand)
Để tránh lỗi tương thích thư viện (đặc biệt là với Antechamber/ACPYPE), bạn nên cài đặt môi trường thông qua Conda:
```bash
# 1. Cài đặt GROMACS và OpenBabel qua apt
sudo apt update
sudo apt install -y gromacs openbabel

# 2. Tạo môi trường Conda cho ACPYPE (xử lý topology của Ligand)
conda create -n md_env python=3.12 -y
conda activate md_env
conda install -c conda-forge acpype -y
```

### Kiểm tra cài đặt
Bạn có thể kiểm tra xem hệ thống đã cài đặt đầy đủ công cụ chưa:
```bash
python3 gromacsmda.py check
```

---

## Hướng dẫn sử dụng (Usage)

Script hỗ trợ 2 chế độ: **Interactive Mode** (Tương tác qua Menu) và **Batch Mode** (Chạy hàng loạt từ file list).

### Chế độ tương tác (Interactive Mode)
Chạy lệnh sau và làm theo hướng dẫn trên màn hình:
```bash
python3 gromacsmda.py
```
Chọn `2` để chạy mô phỏng. Script sẽ hỏi đường dẫn đến file `complex.pdb` (hoặc `protein.pdb`).

### Chế độ chạy đơn dòng lệnh (CLI - Single Run)
```bash
python3 gromacsmda.py run <đường_dẫn_pdb> [thư_mục_làm_việc] [bước_bắt_đầu] [bước_kết_thúc]
```
Ví dụ:
```bash
python3 gromacsmda.py run complex.pdb . 0 11
```

### Chế độ chạy hàng loạt (Batch Run)
Rất hữu ích khi bạn có danh sách nhiều phức hợp cần chạy screening.
1. Tạo một file `complexes.txt`, mỗi dòng là một đường dẫn đến file PDB:
```txt
# Danh sách phức hợp
/path/to/project/complex_A.pdb
/path/to/project/complex_B.pdb
```
2. Chạy lệnh batch:
```bash
python3 gromacsmda.py batch complexes.txt
```
Script sẽ tự động tạo các thư mục con `calculations/complex_A/`, `calculations/complex_B/`, copy các tham số (`.mdp`) vào và tiến hành mô phỏng tự động toàn bộ.

---

## Cấu trúc dữ liệu đầu vào

Bạn chỉ cần cung cấp file `.pdb` chứa cấu trúc (Protein hoặc Protein-Ligand).
- Nếu có Ligand, đảm bảo Ligand được đặt tên hợp lý trong PDB (ví dụ: `LIG` thay vì `*` hoặc `UNL`).
- Script sẽ **tự động phân tách** Protein và Ligand, tạo topology riêng biệt, sau đó ghép lại.

Bạn cũng cần đảm bảo các file thông số `.mdp` nằm trong cùng thư mục với `gromacsmda.py`:
- `EM.mdp`: Energy Minimization
- `NVT.mdp`: NVT Equilibration
- `NPT.mdp`: NPT Equilibration
- `MD.mdp`: Production MD
- `ions.mdp`: Cấu hình thêm Ion

---

## Xử lý các lỗi thường gặp (Troubleshooting)

### 1. Lỗi ACPYPE: `libhdf5_hl.so.310: cannot open shared object file` hoặc `H5FDperform_init`
- **Nguyên nhân**: Phiên bản ACPYPE cài qua `pip` bị xung đột thư viện (ABI mismatch) với hệ điều hành.
- **Cách sửa**: Chuyển sang cài ACPYPE bằng Conda.
  ```bash
  pip uninstall acpype -y
  conda install -c conda-forge acpype -y
  ```

### 2. GROMACS báo lỗi thiếu bộ nhớ / Treo máy (OOM / Freeze)
- **Nguyên nhân**: MD tốn nhiều tài nguyên, đặc biệt ở bước NVT/NPT/Production.
- **Cách sửa**: Giảm số CPU threads bằng cách chỉnh sửa biến `self.nt_threads` trong file `gromacsmda.py` hoặc tắt bớt các chương trình đang chạy.

### 3. Lỗi không tìm thấy file `protein.gro` hoặc `ligand.itp`
- **Nguyên nhân**: Quá trình chuyển đổi pdb2gmx hoặc acpype thất bại ở bước trước.
- **Cách sửa**:
  - Kiểm tra xem tên residue của Ligand trong file PDB có chuẩn không (ví dụ đừng chứa ký tự lạ).
  - PDB của Protein có chứa nguyên tử không chuẩn không? (GROMACS có thể cảnh báo thiếu parameters).

### 4. Lỗi Index Generation (`Protein_LIG group integration failed`)
- **Nguyên nhân**: GROMACS không nhận diện được tên Ligand mặc định.
- **Cách sửa**: Script đã cố gắng auto-detect, nhưng nếu lỗi, hãy mở thủ công bằng `gmx make_ndx -f em.gro -o index.ndx` và ghép group Protein với Ligand.
