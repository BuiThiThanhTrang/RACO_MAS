# Review output formal proof bị từ chối — 2026-11-01

Đã review 25 lần write_source bị từ chối của cohort formal_shared_jev_prover_v4_20261031, đối chiếu raw tool-call content trong trajectory, không chỉ file rejected đã qua chọn code block. Tìm thấy 1 proof đúng bị từ chối nhầm; đã sửa adapter.

## Kết quả

| Nhóm | Output |
|---|---:|
| Phân tích, không có candidate Lean độc lập | 21 |
| Có code fence chưa đóng; phần code cũng compile fail | 1 |
| Header/proof sai; không có proof hợp lệ | 2 |
| Proof đúng, chỉ khác tên theorem | 1 |

Đã kiểm tra cả 6 code block hoàn chỉnh trong 3 raw output và phần Lean trong 1 fence chưa đóng (7 probe). Hai block compile được nhưng dùng sorry/sorryAx, không phải proof hợp lệ. Một block đúng trong output đổi tên theorem; những block còn lại compile fail. Các probe khôi phục header chỉ dùng chẩn đoán, không cấp quyền chấp nhận statement đã bị đổi.

## Lỗi và bản sửa

`proof-workbook-lean-workbook-plus-27829:dynamic-004`: block cuối khai báo `lean_workbook_plus_100000` thay vì `lean_workbook_plus_27829`. Mọi token còn lại trong header đều khớp. Giữ prefix gốc và lấy nguyên proof body đã compile thành công, #print axioms chỉ có propext, Classical.choice, Quot.sound.

FormalProofAdapter nay chuẩn hóa riêng tên declaration về tên gốc. Giả thiết, kết luận, loại declaration và context vẫn so sánh token như trước. Không sửa proof body, không thử nhiều block rồi chọn block compile tốt nhất. Thứ tự chọn block cuối được giữ nguyên. Metadata ghi submitted_declaration_name, original_declaration_name, declaration_name_normalized. Raw output được lưu trong trajectory như trước.

Đã replay nguyên raw tool call của decision lỗi qua adapter mới trong workspace riêng. Adapter nhận candidate và source tạo ra khớp chính xác với source đã được kernel kiểm tra (`adapter_replay.json`).

## Kiểm chứng và phạm vi

8/8 kiểm tra formal proof pass, gồm chuẩn hóa tên + compiler/kernel thật, từ chối đổi kết luận/kiểu biến/context và thêm axiom. Bộ test chung đạt 11/12; test code checkout fail do Git không clone được repository local, trước khi patch code. Trace lỗi lưu riêng dưới code_regression_diagnostic.

Không gọi model mới, không đổi router, budget, role pool, database hay code adapter. Hash các trajectory nguồn không đổi. Kết quả live lịch sử vẫn là 2/10; packet có proof bị từ chối nhầm vốn đã pass ở bước sau. Không được suy ra bản sửa tự động nâng số task pass; cần live rerun để đo ảnh hưởng đến số bước/routing.

Chi tiết machine-readable, hash raw output và compiler diagnostics: `../runs/formal_rejected_output_audit_20261101/audit.json`.

## Từng output

| Decision | Kết luận |
|---|---|
| proof-april-lme-0008eb23463d:dynamic-004 | Phân tích, không có candidate Lean độc lập |
| proof-april-mlme-006213febdb2:dynamic-001 | Phân tích, không có candidate Lean độc lập |
| proof-april-mlme-006213febdb2:dynamic-005 | Header/proof sai; không có proof hợp lệ |
| proof-april-thme-0014a26fbc97:dynamic-002 | Phân tích, không có candidate Lean độc lập |
| proof-april-thme-0014a26fbc97:dynamic-006 | Phân tích, không có candidate Lean độc lập |
| proof-april-thme-0014a26fbc97:dynamic-007 | Phân tích, không có candidate Lean độc lập |
| proof-april-tme-00338b98d4ff:dynamic-003 | Phân tích, không có candidate Lean độc lập |
| proof-april-tme-00338b98d4ff:dynamic-005 | Phân tích, không có candidate Lean độc lập |
| proof-april-tme-00338b98d4ff:dynamic-006 | Phân tích, không có candidate Lean độc lập |
| proof-april-tme-0072686794ba:dynamic-001 | Phân tích, không có candidate Lean độc lập |
| proof-april-tme-0072686794ba:dynamic-002 | Header/proof sai; không có proof hợp lệ |
| proof-april-tme-0072686794ba:dynamic-003 | Phân tích, không có candidate Lean độc lập |
| proof-april-tme-0072686794ba:dynamic-005 | Phân tích, không có candidate Lean độc lập |
| proof-workbook-lean-workbook-plus-1002:dynamic-003 | Phân tích, không có candidate Lean độc lập |
| proof-workbook-lean-workbook-plus-1002:dynamic-007 | Phân tích, không có candidate Lean độc lập |
| proof-workbook-lean-workbook-plus-1003:dynamic-004 | Phân tích, không có candidate Lean độc lập |
| proof-workbook-lean-workbook-plus-2536:dynamic-000 | Candidate Lean chưa hoàn chỉnh, compile fail |
| proof-workbook-lean-workbook-plus-27829:dynamic-004 | Proof đúng, bị từ chối do đổi tên theorem |
| proof-workbook-lean-workbook-plus-66384:dynamic-000 | Phân tích, không có candidate Lean độc lập |
| proof-workbook-lean-workbook-plus-66384:dynamic-001 | Phân tích, không có candidate Lean độc lập |
| proof-workbook-lean-workbook-plus-66384:dynamic-002 | Phân tích, không có candidate Lean độc lập |
| proof-workbook-lean-workbook-plus-66384:dynamic-003 | Phân tích, không có candidate Lean độc lập |
| proof-workbook-lean-workbook-plus-66384:dynamic-005 | Phân tích, không có candidate Lean độc lập |
| proof-workbook-lean-workbook-plus-66384:dynamic-006 | Phân tích, không có candidate Lean độc lập |
| proof-workbook-lean-workbook-plus-66384:dynamic-007 | Phân tích, không có candidate Lean độc lập |
