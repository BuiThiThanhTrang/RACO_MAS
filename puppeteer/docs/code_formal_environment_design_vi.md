# Thiết kế môi trường đánh giá điều phối — code debugging và formal proof

Ngày ghi nhận: 2026-11-01. Tài liệu mô tả môi trường đã tạo 30 trajectory gần nhất, cùng bản sửa formal adapter sau review. Đây là đặc tả để chuẩn bị task mới như miniF2F; không phải hướng dẫn tối ưu riêng cho Jev.

## 1. Mục tiêu và đơn vị đánh giá

Môi trường cho phép cắm một decision model để chọn agent tiếp theo, quan sát hậu quả và giải task nhiều bước. Đơn vị task là **packet**; một lần chạy từ trạng thái ban đầu đến STOP/hết budget là **episode/trajectory**; mỗi lần router được hỏi tạo một **decision state** bất biến.

Môi trường cung cấp đủ task, state, role contract, artifact và feedback công cụ. Không ép thứ tự role, không tự sửa output toán/code, không chọn lại action giúp router, không yêu cầu mọi role phải được sử dụng. Task đơn giản có thể giải bằng ít role; mức độ đa dạng phối hợp là kết quả đo, không phải điều kiện cưỡng chế.

## 2. Cohort gốc và phiên bản

| Domain | Packet | Nguồn | Executor | Final task pass | State | Routing chính |
|---|---:|---|---|---:|---:|---:|
| Code debugging | 20 | 10 BugsInPy, 6 QuixBugs, 4 SWE-bench | Gemini 2.5 Flash | 12/20 | 151 | 143 |
| Formal proof | 10 | 5 APRIL, 5 Lean Workbook | Goedel-Prover-V2-8B | 2/10 | 86 | 78 |

Run gốc: `runs/code_shared_jev_gemini_v3_20261101` và `runs/formal_shared_jev_prover_v4_20261031`. Registry: `runs/non_database_cohort_20261031.json`. Có 237 decision state; 16 lựa chọn non-STOP tại budget 0 không thực thi, nên routing denominator chính là 221. Nhãn human vẫn pending; số pass không phải action accuracy.

Cả hai run dùng Jev 1.13 structured, seed 42, 8 lượt hành động non-STOP, output cap 16384 token/request, tối đa 3 tool calls trong lượt actor. Role deterministic không cần gọi LLM vẫn tiêu thụ một lượt hành động khi được dispatch. Precondition rejection cũng tiêu thụ lượt. Retry transport là xử lý request lỗi, không phải routing mới.

Run lưu `resolved_config.json`, `role_cards.json`, `source_hashes.json`, `source_snapshot.zip`, runtime từng episode và báo cáo freeze/validation/evaluation. Các snapshot gốc là nguồn xác định chính xác phiên bản lịch sử; không được suy ra source hiện tại luôn khớp lịch sử.

Sau cohort gốc, review phát hiện một output có proof đúng bị từ chối vì đổi riêng tên declaration. Adapter hiện chuẩn hóa tên về tên gốc khi mọi token header/context khác khớp; metadata ghi nhận phép chuẩn hóa. Tham khảo `formal_rejected_output_audit_vi.md`. Kết quả lịch sử 2/10 không bị tính lại; miniF2F dùng bản adapter sau sửa. Khi so sánh nhân quả giữa cohort/decision model cần cố định cùng phiên bản adapter và chạy lại nếu cần.

Database vẫn theo archive đóng băng riêng, không nhập vào thay đổi hay cohort miniF2F.

## 3. Vòng lặp chung và interface decision model

`SharedTrajectoryRunner` nhận `router`, `actor`, `adapter`, `max_calls`; router cung cấp:

```python
reset(seed)
select(observation, candidates) -> RoutingDecision
```

CLI `scripts/run_shared_environment.py` hỗ trợ decision-service và random; decision model API khác cần adapter mỏng theo interface trên. Chỉ thay router khi so sánh điều phối, giữ task bank, role cards, executor, công cụ, budget và checker cố định.

Mỗi vòng: lấy public state → chụp snapshot → hỏi router với **toàn bộ role pool + STOP** → xử lý STOP hoặc kiểm tra budget → kiểm tra precondition → dispatch role → ghi tool/result và state delta → vòng kế tiếp. Guard tắt (`observable_rejection_guard_enabled: false`); precondition không dùng để masking action trước lựa chọn.

Không auto-STOP khi task đã đúng. Router có thể tiếp tục và làm mất trạng thái đã đúng. STOP sớm được ghi `PREMATURE_STOP`. Khi đã dùng hết 8 lượt, runner vẫn hỏi router theo flow đã thống nhất: STOP được xử lý; non-STOP ghi `suppressed_by_budget`, không dispatch và loại khỏi denominator routing chính. Không đổi quyết định cuối thành một action đã thực thi.

## 4. Packet, visibility và thông tin quan sát

Manifest schema `4.2-artifact-rc1`, benchmark observation version `4.4-shared-environment-v1`. `manifest.json` xác định task, adapter, file path/hash/visibility, provenance, readiness và trạng thái human review. Mỗi packet có:

| Vùng | Nội dung | Quyền truy cập |
|---|---|---|
| `public/` | Task, source/contract, thông tin ban đầu và dữ liệu thực thi công khai | Materialize vào workspace agent |
| `private/` | Reference, oracle và bằng chứng gate | Checker/preparation; không materialize cho agent |
| `fixtures/` | Nguồn, commit, license, bằng chứng chuẩn bị | Provenance; không tự đưa vào prompt |
| `_runtime/` | Artifact có version/hash, state và event sinh trong episode | Public artifact được đọc theo contract |

Router nhận task, `current_state`, role cards, lịch sử public, `calls_remaining`, allowed actions. Với code/formal, raw role output không được tự nhúng vào lịch sử router; router thấy trạng thái/evidence mà adapter công bố. Formal router không nhận raw Lean source theo contract. Đây là cấu hình visibility phải giữ ổn định khi so sánh, không phải toàn bộ nội dung executor đều được sao chép cho router.

Executor chỉ chạy role đã chọn, nhận source và feedback phù hợp domain, không được chọn role tiếp theo. Formal source được render nguyên văn để giữ indentation. Artifact dài giữ nguyên file và hash, có `read_artifact` đọc chính xác theo offset, mỗi đoạn tối đa 8000 ký tự. Source/diagnostic được giới hạn phần hiển thị với tổng độ dài và ref; không xóa dữ liệu gốc hoặc dùng model phụ tóm tắt.

## 5. Role và implementation code debugging

| Role | Công việc thực thi |
|---|---|
| Repository Test Reproducer | Checkout revision đã pin và chạy baseline/reproducer thật |
| Failure Diagnostician | Đọc source/test feedback, ghi diagnosis |
| Patch Author | Tạo và áp dụng patch trong edit scope |
| Regression Verifier | Chạy regression thật trên candidate hiện tại |
| Verified Patch Consumer | Xử lý deliverable theo role card hiện tại |
| Patch Finalizer | Đóng gói kết quả theo role card hiện tại |

Tên đầy đủ và tool schema authoritative ở `config/shared_environment/code_roles.json`; bảng trên chỉ tóm tắt. Map tool theo identity của role, không suy tool từ từ khóa trong mô tả.

Runtime profile theo packet ghim repository/revision, Python và test command. Một số profile gốc tái sử dụng virtualenv trong worktree cũ, vì vậy chuyển máy cần provision và preflight lại; archive source không phải image runtime độc lập.

Patch chỉ sửa implementation `.py/.pyi` trong scope cho phép. Tests, harness, config và reference được bảo vệ. Patch bị reject có artifact riêng, không ghi đè candidate đã áp dụng. Nhiều patch thành công tạo diff tích lũy và history. Reproducer không tự reset checkout đã có patch. Patch Author trước checkout là action rejection.

Validation gắn với source/repository/contract/runtime hiện tại; evidence cũ không chứng minh candidate mới đúng. Workflow terminal cần validation pass còn fresh. Consumer/finalizer không phải hai bước metadata bắt buộc. Checker cuối dựng checkout sạch, áp dụng implementation cuối và chạy trusted tests độc lập; đây là điểm theo task contract của packet, không phải điểm chính thức toàn bộ SWE-bench.

## 6. Role và implementation formal proof

| Role | Operation chính | Điều kiện/công việc |
|---|---|---|
| Proof State Inspector | `inspect_proof` | Đọc source và phân biệt state ban đầu với feedback mới |
| Premise and Lemma Retriever | `write_premises` | Đề xuất 1–8 symbol; xác minh bằng `#check` thật |
| Tactic Synthesizer | `write_source` | Sinh candidate trong theorem gốc |
| Proof Repair Specialist | `write_source` | Sửa candidate từ evidence hiện có |
| Lean Compiler Executor | `compile_proof` | Cần candidate; gọi Lean đã pin |
| Kernel and Assumption Verifier | `verify_proof` | Cần compilation thành công còn fresh |
| Formal Failure Diagnostician | `write_diagnosis` | Ghi phân tích lỗi; không tự sửa source |

Các role ghi work product có bounded artifact read. Việc write source thành công chỉ có nghĩa candidate được nhận, chưa có nghĩa proof đúng.

Packet có đúng một theorem/lemma editable. Imports, definitions, giả thiết và kết luận bất biến. Adapter nhận proof term, declaration lặp lại hoặc full source có header/context phù hợp; luôn giữ prefix gốc. Khi có nhiều Lean fence hoàn chỉnh, chọn block cuối theo quy tắc cố định, không compile nhiều block rồi chọn block tốt nhất. Tên declaration khác được chuẩn hóa về tên gốc và ghi metadata; không sửa body hay tự sửa toán.

Compilation thật có timeout 180 giây, runtime Lean/Mathlib theo contract. Compile pass vẫn có thể chứa `sorry`; verifier kiểm tra source, statement preservation, freshness, compile lại và `#print axioms`. Chỉ cho phép `propext`, `Classical.choice`, `Quot.sound`; `sorryAx` và axiom khác làm fail. Context gốc compile bằng `by sorry` chỉ là gate runtime, không phải proof pass.

Không tự phục hồi candidate sau hành động bất lợi. Candidate mới được chấp nhận làm compilation/verification cũ stale; candidate bị reject giữ source cũ và lưu rejected output/error riêng. Formal không có rollback ngầm.

## 7. Log, evaluator và human review

Mỗi episode lưu `decision_states.jsonl`, `decisions.jsonl`, `trajectory.jsonl`, `runtime.json`, `result.json`, workspace và evaluator output. Decision ID nối snapshot → action → dispatch → execution. Log giữ raw responses, requests, usage/model/finish_reason, tool arguments/results, normalization, artifact hashes và semantic delta.

`scripts/validate_shared_environment.py` kiểm tra liên kết trace, visibility, output cap, context và source freeze. `scripts/evaluate_shared_environment.py` kiểm tra trạng thái cuối trên workspace riêng. Tách:

- **Final task pass:** trạng thái cuối đạt contract theo checker độc lập.
- **Workflow + STOP pass:** có evidence fresh theo workflow và router chọn STOP.
- **Routing quality:** human labels theo state; `valid_actions` có thể có nhiều role hợp lệ.
- **Infrastructure validity:** transport/tool/runtime/checker unavailable không gộp vào lỗi toán/code thông thường.

Exporter tạo queue/trang review với lựa chọn hiện tại được ẩn, giữ lịch sử đã quan sát. Không chuyển nhãn Luna cũ sang state mới. Human scorer yêu cầu đúng run/decision, `human_verified`, reviewer và action trong pool; loại budget-suppressed khỏi denominator chính, báo coverage cùng accuracy. Nhãn pending không tạo accuracy.

Nếu trace thiếu, chạy lại sample thành attempt có provenance riêng theo quy ước vận hành; không ghép tùy ý action/state của các attempt hoặc chỉ chọn attempt pass.

## 8. Áp dụng cho miniF2F

Giữ nguyên runner, role pool, executor Goedel, routing interface, visibility, 8 lượt, 16k cap, guard-off và checker. Chỉ bổ sung task bank + provenance/runtime contract. Không đổi statement để làm dễ, không đưa private witness/hướng giải vào public state, không thêm role hoặc ép pipeline.

Quy trình chuẩn bị: chốt IDs/độ khó trước khi chạy model → ghim source commit/hash/license/split → giữ source statement → xác minh context/toolchain → lấy initial goal bằng Lean thật → kiểm tra private witness + allowed axioms → tạo/check manifest → chạy smoke qua adapter → lưu source/runtime/config lock. Human packet review vẫn pending nếu người dùng chưa duyệt nội dung.

Nhãn dễ/vừa là đánh giá theo cấu trúc toán do project đặt, không phải nhãn chính thức miniF2F hay lời hứa về success rate executor. Bài có thể được tactic automation giải nhanh dù toán có nhiều bước; không được ép nhiều role để tăng bề ngoài phối hợp. Giữ kết quả theo từng stratum và xem trajectory thực tế để đánh giá khả năng phân biệt decision model.

Tập miniF2F mới là cohort bổ sung, không thay 10 task cũ và không cộng lẫn score thành điểm miniF2F chính thức. Chi tiết IDs, nguồn, bằng chứng chuẩn bị và lệnh chạy ở `minif2f_environment_preparation_vi.md`.

## 9. Mã nguồn và bằng chứng authoritative

- Engine: `routing_benchmark/v4/execution/shared_runner.py`, `shared_actor.py`.
- Adapter: `execution/adapters/external_code_debugging.py`, `formal_proof.py`.
- Runtime/checker: `execution/runtime.py`, `final_evaluation.py`.
- Role cards: `config/shared_environment/code_roles.json`, `v4/domain_packets/roles.py`.
- Công cụ: `scripts/run_shared_environment.py`, `preflight_shared_domains.py`, `validate_shared_environment.py`, `evaluate_shared_environment.py`, `export_shared_review.py`, `score_shared_reviews.py`.
- Cohort gốc và kiểm chứng: `code_formal_environment_implementation_report_vi.md`; snapshot từng run dùng để tái dựng source lịch sử.

## Phụ lục: 30 packet thuộc cohort gốc

| Domain | Packet | Workflow pass |
|---|---|---|
| code | code-bugsinpy-black-3 | FAIL |
| code | code-bugsinpy-cookiecutter-1 | PASS |
| code | code-bugsinpy-fastapi-10 | FAIL |
| code | code-bugsinpy-httpie-4 | PASS |
| code | code-bugsinpy-pysnooper-3 | PASS |
| code | code-bugsinpy-scrapy-37 | FAIL |
| code | code-bugsinpy-thefuck-2 | PASS |
| code | code-bugsinpy-tornado-9 | FAIL |
| code | code-bugsinpy-tqdm-8 | FAIL |
| code | code-bugsinpy-youtube-dl-43 | FAIL |
| code | code-quixbugs-bucketsort | PASS |
| code | code-quixbugs-flatten | PASS |
| code | code-quixbugs-next-palindrome | FAIL |
| code | code-quixbugs-quicksort | FAIL |
| code | code-quixbugs-sieve | PASS |
| code | code-quixbugs-to-base | PASS |
| code | code-swebench-django-16485 | PASS |
| code | code-swebench-pylint-4970 | PASS |
| code | code-swebench-sphinx-9698 | PASS |
| code | code-swebench-sympy-14711 | PASS |
| formal | proof-april-lme-0008eb23463d | FAIL |
| formal | proof-april-mlme-006213febdb2 | FAIL |
| formal | proof-april-thme-0014a26fbc97 | FAIL |
| formal | proof-april-tme-00338b98d4ff | FAIL |
| formal | proof-april-tme-0072686794ba | FAIL |
| formal | proof-workbook-lean-workbook-plus-1002 | FAIL |
| formal | proof-workbook-lean-workbook-plus-1003 | FAIL |
| formal | proof-workbook-lean-workbook-plus-2536 | PASS |
| formal | proof-workbook-lean-workbook-plus-27829 | PASS |
| formal | proof-workbook-lean-workbook-plus-66384 | FAIL |
