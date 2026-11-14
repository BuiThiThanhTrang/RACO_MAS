# Chuẩn bị môi trường cho 20 sample miniF2F

Ngày: 2026-11-01. Môi trường tuân theo `code_formal_environment_design_vi.md`, dùng formal adapter sau bản sửa chuẩn hóa tên theorem. Đây là bước chuẩn bị task/runtime, chưa phải một run đánh giá decision model.

## Nguồn và nguyên tắc lựa chọn

Nguồn trực tiếp là [miniF2F JSONL trong repository Goedel-Prover-V2](https://github.com/Goedel-LM/Goedel-Prover-V2/blob/2e9036e118464aa96a8bebaf9f5b9d091aa3585c/dataset/minif2f.jsonl), ghim commit `2e9036e118464aa96a8bebaf9f5b9d091aa3585c`. File chứa 244 bài thuộc split **test** của bản export này. Snapshot được lưu ở `data/RoutingBench/minif2f_sources_20261101/minif2f.jsonl`, checksum và commit ở task bank `source_lock.json`.

miniF2F gốc gồm nhiều mức từ toán trung học đến olympiad; tên dataset không tự xác định độ khó. License MIT của miniF2F và attribution Meta được lưu kèm packet. [Nguồn miniF2F](https://github.com/facebookresearch/miniF2F), [license Lean4 port](https://github.com/yangky11/miniF2F-lean4/blob/main/LICENSE).

Danh sách 20 ID và strata được chốt trước khi gọi model hoặc kiểm tra thành tích model. Dễ/vừa là nhãn project theo nội dung toán, chưa được human xác nhận, không phải annotation chính thức. Không lọc bài theo kết quả Jev/Goedel; không có hosted model call trong preparation. Private witness dùng để xác nhận task giải được, không dùng để phân loại lại độ khó hoặc tạo nhãn routing. Một proof mẫu được sửa cách biểu diễn equivalence sau compiler feedback; danh sách, statement và strata không thay đổi.

- **Dễ:** tính toán trực tiếp, thay định nghĩa, hệ tuyến tính nhỏ hoặc đồng nhất thức đa thức; ít phụ thuộc lemma chuyên biệt.
- **Vừa:** cần chọn phép biến đổi/lemma và điều kiện áp dụng, hoặc nối nhiều giả thiết/nhánh/sai phân. Tactic automation vẫn có thể giải một số bài bằng ít dòng; đây không phải yêu cầu phải gọi nhiều role.

Giữ split test để bảo toàn provenance. Tập 20 bài là **custom pilot subset**, không phải điểm benchmark chính thức miniF2F và không được coi là tập chưa từng xuất hiện trong huấn luyện model. Khi so decision model, dùng cùng tập đã chốt và cùng environment version.

## Môi trường tái sử dụng

- Task bank: `data/RoutingBench/v4_minif2f_20_20261101`.
- Config live tương lai: `config/shared_environment/formal_minif2f_20_jev_16k.yaml`.
- Adapter: `formal_proof.lean4.v1`; 7 role formal + STOP, không đổi pool.
- Executor: `Goedel-LM/Goedel-Prover-V2-8B:featherless-ai`.
- Baseline router đã cấu hình: Jev 1.13 structured, guard tắt.
- Budget: 8 lượt non-STOP, 16384 token/request, max 3 tool calls/lượt actor.
- Runtime thực tế: Lean **4.8.0-rc1**, Mathlib revision **db651742f2c631e5b8525e9aabcf3d61ed094a4a**, tái sử dụng `.routingbench_lean_cache/mathlib-4.8`.

Goedel công bố evaluation setup Lean 4.9; cohort này dùng runtime local đã pin và kiểm tra tương thích, vì vậy không tuyên bố tái lập nguyên setup công bố. [Thiết lập Goedel](https://github.com/Goedel-LM/Goedel-Prover-V2). Không lấy HEAD của Lean4 port hiện tại rồi mặc định tương thích. Cả 20 `public/Main.lean` giữ nguyên nội dung upstream `lean4_code` ngoài chuẩn hóa newline cuối file: imports/options, giả thiết, kết luận và `by sorry` ban đầu không đổi. Không bổ sung helper lemma, tactic hint hoặc sửa bài để compile.

## Cấu trúc packet và thông tin agent

Mỗi packet `proof-minif2f-<source-id với dấu gạch ngang>` có:

| File | Nội dung |
|---|---|
| `public/Main.lean` | Source theorem upstream, proof ban đầu `by sorry` |
| `public/task.md` | Yêu cầu hoàn thành và kiểm chứng theorem |
| `public/proof_state.json` | Goal/hypotheses lấy bằng Lean `trace_state` thật |
| `public/execution_contract.json` | Toolchain, visibility, edit/terminal contract |
| `private/Reference.lean` | Proof mẫu locally authored chỉ phục vụ kiểm tra chuẩn bị |
| `private/runtime_gate.json` | Bằng chứng context/reference/axiom gate |
| `fixtures/upstream_provenance.json` | ID/split/commit/hash, strata và lý do chọn |
| `fixtures/LICENSE.miniF2F` | License và attribution |

Adapter chỉ materialize public files. Source snapshot, selection có proof mẫu, private reference, preparation diagnostics và fixture không tự đưa vào prompt. Router tiếp tục không thấy raw source theo formal visibility contract; executor thấy source nguyên văn và có artifact reads. Goal ban đầu được công bố, không công bố proof mẫu hay hướng giải của người chuẩn bị.

`human_review_status` và `packet_review_status` giữ pending. Runtime readiness được kiểm tra bằng công cụ không đồng nghĩa human đã duyệt bài, không đồng nghĩa executor đã giải được task.

Cả 20 task mới là **proof construction** với `by sorry` ban đầu, không phải task sửa một proof lỗi có sẵn như APRIL. Không cài lỗi giả để buộc gọi Repair/Diagnostician. Những role này chỉ có cơ hội hữu ích khi trajectory tự sinh lỗi hoặc thiếu evidence. Khi phân tích phối hợp, phải phân tầng theo loại task và độ khó, không coi việc ít dùng role repair trên bài dễ là lỗi routing mặc định.

## Kiểm chứng

Preparation chính: `runs/minif2f_environment_preparation_v2_20261101/report.json`.

1. Kiểm tra context của cả 20 statement bằng Lean, giữ statement gốc và `sorry` chỉ ở proof input.
2. Compile 20 private witness và kiểm tra `#print axioms`: không `sorryAx`, chỉ các axiom được phép. Gate này dùng batch với context upstream đã xác minh giống nhau; không phải model trajectory hay điểm performance.
3. Validate checksum/shape của 20 artifact manifest và kiểm tra initial goal không rỗng.
4. Preflight từng packet trong workspace riêng: `runs/minif2f_preflight_20261101/report.json`.
5. Smoke adapter trên một bài dễ (`mathd_algebra_24`) và một bài vừa (`mathd_algebra_188`): public/private isolation, reject đổi statement, compile/kernel pass, evidence stale sau thay candidate, checker cuối độc lập. Bằng chứng ở `runs/minif2f_adapter_smoke_20261101/report.json`.

Lần preparation đầu được giữ ở `runs/minif2f_environment_preparation_20261101`: context 20/20 pass, một private witness có rewrite chưa đúng dạng biểu diễn equivalence. Đã sửa riêng witness, không thay sample; lần v2 là gate chính. Không chọn task khác để bỏ lỗi.

Preflight script được siết đúng contract: formal context chỉ ready khi Lean trả mã 0; không suy ready từ việc thiếu một vài chuỗi lỗi module. Đây là sửa kiểm tra readiness, không đổi routing hay scorer.

## Sử dụng và khả năng tái dựng

Chạy từ thư mục `puppeteer`. Preparation builder không ghi đè thư mục đã có; muốn tái dựng phải dùng output/checks directory mới:

```powershell
python scripts/prepare_minif2f_environment.py --output data/RoutingBench/<new-bank> --checks runs/<new-preparation>
python scripts/preflight_shared_domains.py --config config/shared_environment/formal_minif2f_20_jev_16k.yaml --output runs/<new-preflight>
python scripts/check_minif2f_environment.py --output runs/<new-smoke>
```

Nếu đổi tên task bank khi rebuild, tạo config trỏ tới bank mới trước preflight/live. Máy khác phải provision đúng Lean/Mathlib và library paths; task-bank archive không đóng gói binary/cache đầy đủ. `runtime_lock.json` trong bank ghi runtime và hash những file định danh quan trọng.

Lệnh sau dành cho bước thu live trajectory tiếp theo; **chưa được thực thi trong preparation**:

```powershell
python scripts/run_shared_environment.py --config config/shared_environment/formal_minif2f_20_jev_16k.yaml --run-id formal_minif2f_20_jev_<date>
python scripts/validate_shared_environment.py runs/formal_minif2f_20_jev_<date>
python scripts/evaluate_shared_environment.py runs/formal_minif2f_20_jev_<date>
python scripts/export_shared_review.py runs/formal_minif2f_20_jev_<date>
```

Trước khi so model, cố định bank/config/source snapshot. Báo riêng easy/medium và lỗi hạ tầng; giữ final task pass, workflow+STOP, routing labels và coverage tách biệt. Không gán human valid_actions từ private proof hoặc từ role sequence của smoke.

## Danh sách sample

| Mức | Source ID | Cơ sở chọn |
|---|---|---|
| easy | mathd_algebra_24 | Phương trình tuyến tính một ẩn với mẫu số hằng. |
| easy | mathd_algebra_359 | Thu gọn tổng tuyến tính một ẩn. |
| easy | mathd_algebra_412 | Khử hai phương trình tuyến tính. |
| easy | mathd_algebra_160 | Tổ hợp tuyến tính hai giả thiết. |
| easy | mathd_algebra_419 | Thay giá trị cụ thể rồi tính đa thức. |
| easy | mathd_algebra_171 | Áp dụng định nghĩa hàm tại một điểm. |
| easy | mathd_algebra_346 | Tính hợp hàm với hai định nghĩa tường minh. |
| easy | mathd_algebra_176 | Đồng nhất thức đa thức. |
| easy | mathd_algebra_432 | Khai triển tích đa thức bậc hai. |
| easy | mathd_numbertheory_299 | Tích số cụ thể rồi lấy phần dư. |
| medium | mathd_algebra_141 | Từ tổng và tích suy ra tổng bình phương; cần quan hệ bậc hai. |
| medium | mathd_algebra_33 | Khử biến và xử lý phép chia với điều kiện khác không. |
| medium | mathd_algebra_80 | Khử mẫu phụ thuộc biến rồi giải phương trình. |
| medium | mathd_algebra_113 | Chứng minh cận dưới bằng bình phương không âm. |
| medium | mathd_algebra_263 | Chuyển căn bậc hai sang bình phương với điều kiện miền xác định. |
| medium | mathd_algebra_188 | Dùng tính chất nghịch đảo của equivalence và rewrite. |
| medium | mathd_algebra_76 | Chọn nhánh chẵn của định nghĩa hàm rồi tính trên số nguyên. |
| medium | mathd_algebra_452 | Liên kết các đẳng thức sai phân để suy ra phần tử giữa dãy. |
| medium | mathd_numbertheory_185 | Suy luận modulo với biến không bị chặn. |
| medium | mathd_numbertheory_1124 | Kết hợp chia hết và miền chữ số hữu hạn. |

## Bằng chứng đã hoàn tất

- Context 20/20, private witness 20/20 và allowed-axiom gate 20/20 pass.
- Source upstream được đối chiếu nguyên văn cho cả 20 packet; manifest/checksum và goal đều hợp lệ.
- Hai smoke adapter và hai checker cuối độc lập đều PASS; không có model call hoặc benchmark trajectory.
- Preflight từng packet và bộ lock cuối được tổng hợp trong `runs/minif2f_preparation_cohort_20261101.json`.
