# Đóng băng ba domain — 2026-11-01

Release: **three_domain_v1_20261101**. Đã chốt source/config/packet, role/tool contracts, artifact/checker implementation và bằng chứng live hiện có. Không chạy thêm model, không sửa adapter hay chấm lại lịch sử trong bước đóng băng.

| Domain | Packet | Nguồn authoritative |
|---|---:|---|
| Database | 12 | database_v1_20261031 nguyên trạng, được nhúng vào release mới |
| Code debugging | 20 | Current source/config + v4_code_debugging_rc1 |
| Formal proof | 30 | Current source/config + v4_formal_proof_30_20261101 (10 APRIL/Workbook, 20 miniF2F) |

Gói `benchmarks/three_domain_v1_20261101/three_domain_v1.zip` chứa **1.280 entry**, SHA256 **f2b6315356fd08ac24aa7c4584e4999ec311252eb162e0b63a9f3e16ff298184**. Manifest bên cạnh ghi hash/size từng entry, packet inventory và nguồn run lịch sử. Chỉ mục machine-readable: `benchmarks/FROZEN_DOMAINS.json`.

Database v1 vẫn có SHA256 `be3891b46e0065855af3a29a2ea86346c9d84a91085fa2202584b90e66c97fb4`; phần `database_reference/` trong archive mới là authority để tái dựng database v1. Current shared source ở `project/` được đóng băng cho code/formal, không thay thế ngược lại engine của DB v1.

Archive lưu root reports, raw state/decision/trajectory, runtime metadata và source snapshot có sẵn cho bốn run canonical: database_shared_jev_16k_full_20261031, code_shared_jev_gemini_v3_20261101, formal_shared_jev_prover_v4_20261031, formal_minif2f_20_jev_live_20261101. Mỗi run giữ hash/version riêng. Đây không phải thí nghiệm mới chạy 62 task trên cùng một engine version; human labels vẫn theo trạng thái đã có, không tự chuyển pending thành verified.

## Kiểm tra đã thực hiện

- Validate manifest/checksum/path/openability cho toàn bộ 62 packet.
- Verify archive hash, tập entry và từng file hash, kể cả archive database nhúng.
- Thử tạo lại release: script từ chối trước khi ghi; checksum archive không đổi.
- Khôi phục engine Python source từ archive, import registry/runner/actor/checker/router client trong Python isolated mode, xác nhận tất cả module được import từ source khôi phục.
- Lưu evidence ở `runs/three_domain_freeze_verification_20261101/report.json`; verification gốc trong release ở `verification_report.json`.

Không chạy lại toàn bộ live suite chỉ để tạo archive. Release đính kèm các gate/test evidence trước đó với đúng nguồn/thời điểm; restore import không thay thế runtime preflight. Python/SQLite/package inventory và Lean toolchain/lake manifest/head được lưu trong runtime_inventory. Code venv profiles, repository mirrors, Lean/Mathlib binaries và provider service không đóng gói thành runtime image. Không có credential file được thêm vào release.

## Tài liệu chuẩn để thêm domain

Tài liệu chính: `docs/domain_extension_packet_adapter_artifact_standard_vi.md`.

Bao gồm mục tiêu trung lập với decision model; task/manifest/provenance; public-private-fixture visibility; adapter interface và role/tool identity; immutable artifact/version/dependency/freshness; handoff, bounded reads và normalization; failure/rollback/recovery; STOP/budget; independent checker/human review; các điểm registry/runtime/actor/live allowlist/evaluator phải tích hợp; gate nghiệm thu và versioning.

Hai biểu mẫu: `docs/templates/domain_extension/packet_spec.template.json` và `ACCEPTANCE.md`. Đây là worksheet thiết kế/nghiệm thu, không phải packet có hash hợp lệ để chạy ngay. Builder phải sinh manifest thật theo schema hiện có.

Khác biệt fixture hiện tại được ghi rõ: code materialize public + fixture, database/formal chỉ public. Domain mới phải dùng allowlist và test visibility thật, không dựa vào tên thư mục. Tài liệu phân biệt yêu cầu cho extension mới với những gì framework hiện kiểm tra tự động.

## Quy tắc mở rộng

Giữ nguyên archive và manifest của release này. Làm domain mới/thay đổi trong working tree, tạo phiên bản environment mới nếu đổi task/adapter/visibility/tool/role/budget/checker. Không ghi đè packet hoặc run cũ; nhãn human quản lý bằng artifact/version riêng. Decision model mới dùng experiment config riêng và lưu overlay khi so sánh trên cùng environment baseline.

Từ thư mục puppeteer, kiểm tra release bằng:

```powershell
python scripts/freeze_three_domain_environment.py --verify benchmarks/three_domain_v1_20261101
```

Trên Windows nên giải nén vào đường dẫn ngắn hoặc bật hỗ trợ long path vì cấu trúc packet có đường dẫn dài. Khi chuyển máy, provision runtime rồi preflight trước khi chạy live; không coi checksum pass là runtime-ready trên máy mới.
