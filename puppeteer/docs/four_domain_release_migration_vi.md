# Migration release bốn domain RoutingBench v4-beta

Ngày: 2026-11-02

## Kết quả

Nguồn chạy chính thức cho database, code debugging, formal proof và structured
data processing là:

`data/RoutingBench/v4_beta_4d_82_20261102`

Release gồm 82 packet:

- `database_migration_integrity`: 12 packet;
- `code_debugging`: 20 packet;
- `formal_proof`: 30 packet;
- `structured_data_processing`: 20 packet.

Mỗi domain vẫn dùng adapter chuyên biệt. Việc hợp nhất chỉ chuẩn hóa packet
root, agent-pool snapshot, release metadata, validation và freeze; không thay
đổi task contract hoặc role-card semantics đã review.

## Nguồn được hợp nhất

| Domain | Source release | Trạng thái sau migration |
| --- | --- | --- |
| Database | `v4_domain_packets_rc1` | 12 packet được chọn; 10 formal packet cũ trong cùng source không được đưa vào release mới |
| Code debugging | `v4_code_debugging_rc1` | 20 packet được giữ nguyên byte |
| Formal proof | `v4_formal_proof_30_20261101` | 30 packet hiện hành |
| Structured data | `v4_five_domains_100_20261101` | Chỉ 20 packet structured-data được chọn |

Các source release trên là provenance đầu vào, không còn là packet root của
live runner. Không sửa hoặc ghi đè release hợp nhất; mọi thay đổi contract,
packet, role, adapter hay checker phải tạo release ID mới.

## Adapter authority

Project chính là implementation authority. Worktree `4381` không chứa adapter
database hoặc formal-proof live hiện hành. Adapter code-debugging ở project
chính có thêm các invariant về checkout readiness, protected paths, rejected
attempt preservation, cumulative patch và fresh validation evidence so với
worktree.

Live entry point chỉ nhận adapter được đánh dấu live trong execution registry.
Các adapter mô phỏng của track lịch sử vẫn có thể tồn tại trong registry nhưng
không được chạy nhầm bằng shared live runner.

## Cấu trúc và công cụ

- Release specification:
  `config/shared_environment/releases/v4_beta_4d_82_20261102.json`
- Assembler/validator:
  `scripts/assemble_benchmark_release.py`
- Live runner:
  `scripts/run_shared_environment.py`
- Generic freezer/verifier:
  `scripts/freeze_benchmark_release.py`
- Immutable freeze:
  `benchmarks/four_domain_v1_20261102`

Assembler từ chối ghi đè output đã tồn tại. `--verify` kiểm tra file set,
checksum, packet count, manifest, domain và adapter ID. Freezer không đóng gói
credential, episode workspace, repository checkout, virtual environment hoặc
Lean cache.

## Evidence cũ

Historical trajectory được đóng gói kèm `resolved_config` và `source_hashes`
gốc. Chúng không được gán lại là trajectory đã chạy trên đường dẫn release mới.
So sánh router chính thức trong tương lai phải chạy các router trên cùng release
ID và cùng runtime/config freeze.

## Cleanup policy

Các source root cũ và worktree chỉ được xóa sau khi:

1. release validation pass;
2. benchmark archive verify pass;
3. release, config, source và manifest đã được commit hoặc backup;
4. mọi evidence độc quyền cần giữ đã được đưa vào archive.

Worktree phải được gỡ bằng Git worktree management, không xóa thư mục thủ công.
