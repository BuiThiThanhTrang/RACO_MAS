# Formal proof chung: 30 packet và live cohort miniF2F

Ngày: 2026-11-01. Theo yêu cầu gộp domain, task bank canonical là `data/RoutingBench/v4_formal_proof_30_20261101`. Domain/adapter không đổi: `formal_proof` / `formal_proof.lean4.v1`.

## Tổ chức task bank

| Cohort | Số packet | Nguồn packet |
|---|---:|---|
| `legacy_formal_10` | 10 | 5 APRIL + 5 Lean Workbook |
| `minif2f_20` | 20 | 10 easy + 10 medium miniF2F |

`cohorts.json` trong bank ghi danh sách ID cụ thể và nguồn copy. Toàn bộ file trong 30 packet được copy nguyên trạng, manifest và checksum được validate lại. Các bank lịch sử và run cũ không bị thay thế. Database vẫn ở archive đóng băng riêng.

`config/shared_environment/formal_jev_16k.yaml` là entrypoint canonical cho cả 30 packet. Lần live mới truyền **20 `--packet` miniF2F cụ thể** vào runner, không chạy lại 10 bài cũ. `resolved_config.json:selected_packets` là bằng chứng selection thực tế; không suy selection từ tên config/run.

Config miniF2F preparation cũ vẫn trỏ đến bank 20 riêng để giữ khả năng tái dựng bước chuẩn bị. Lần live đã gộp dùng config formal canonical, không sửa source chuẩn bị cũ.

## Gate trước khi chạy

Gate và launcher: `runs/formal_minif2f_live_gate_20261101`.

- Kiểm tra lock của source/bank/config preparation trước migration; không thay statement, proof input hay private/public boundary.
- Validate 30 manifest, resolve đúng runtime và kiểm tra binary/project có trên máy cho cả APRIL, Workbook và miniF2F.
- Tái sử dụng bằng chứng preflight 20/20 miniF2F và 20 context/private-witness/axiom gates; hai smoke adapter + independent checker đều pass. Các file nội dung packet/runtime được giữ nguyên.
- Chạy lại **28/28** kiểm tra common runner, raw routing, STOP/budget, human scorer và toàn bộ formal integration; không hosted model call trong gate.
- Kiểm tra 7 role + STOP, guard false, 8 lượt non-STOP, 16384 output cap/request, tối đa 3 tool calls/lượt actor. Giữ Jev 1.13 structured và Goedel-Prover-V2-8B.
- Kiểm tra credential tồn tại mà không ghi giá trị secret vào báo cáo.
- Trước launch xác minh lại hash packet, đóng gói source snapshot, lưu launch gate và selection chính xác.

`tests.json`, `tests.log`, `readiness.json`, `launch.json` chứa bằng chứng chi tiết. Snapshot giữ cả engine và packet có private reference để tái dựng, nhưng private reference không materialize vào workspace agent.

## Những quy tắc giữ giống môi trường đã chốt

Router chọn trên full pool, không masking theo precondition và không sửa action giúp router. Role chỉ thực thi đúng contract; không ép pipeline. Compiler/kernel/lemma lookup là công cụ thật. Candidate sai bị giữ hậu quả và feedback; không tự sửa proof hoặc rollback ngầm.

Chỉ proof trong context gốc được sửa. Chuẩn hóa tên declaration theo bản adapter đã review; không chuẩn hóa giả thiết/kết luận. Compile pass không đủ nếu còn sorry/sorryAx. Evidence phải fresh. STOP và final task checker được chấm riêng. Decision cuối non-STOP ở budget 0 được lưu nhưng không tính là routing đã thực thi.

Model/tool transport error, checker unavailable và lỗi toán/code được phân biệt. Nếu phải chạy lại sample lỗi hạ tầng, tạo attempt có provenance riêng; không chọn attempt theo task pass. Không chuyển nhãn của cohort cũ sang state mới. Nhãn human của run mới vẫn pending.

## Live run và kiểm chứng sau chạy

Run mới: `runs/formal_minif2f_20_jev_live_20261101`.

Lệnh đã chạy qua launcher có gate: `python runs/formal_minif2f_live_gate_20261101/launch_live.py`. Launcher gọi runner canonical với từng ID miniF2F; snapshot và launch receipt được lưu trong run. Nó từ chối overwrite run đã tồn tại.

Sau completion phải kiểm tra `freeze_verification.json`, validator trace/source, execution health, checker cuối độc lập và export human review. Kết quả và registry domain được ghi khi các bước này hoàn tất. Đây là run dynamic routing, không dùng sequence role định nghĩa sẵn hoặc proof mẫu preparation để giải task.

10 trajectory APRIL/Workbook cũ vẫn có provenance/source snapshot của phiên bản trước sửa tên theorem. Domain chung có thể tra cứu cả 30 task, nhưng không gộp kết quả cũ/mới thành một thí nghiệm cùng source version. So sánh decision model trên toàn bộ 30 bài cần chạy cùng phiên bản môi trường nếu muốn loại yếu tố phiên bản.

Tài liệu nền: `code_formal_environment_design_vi.md`, `minif2f_environment_preparation_vi.md`, `formal_rejected_output_audit_vi.md`.

## Kết quả hoàn tất

20/20 episode hoàn tất và đủ điều kiện đánh giá; 16/20 final task pass, 15/20 workflow + STOP pass. Có 112 state, 107 quyết định routing chính, không masking hay lỗi hạ tầng. Chi tiết ở `formal_minif2f_live_report_vi.md`; human labels vẫn pending.
