# Kiểm tra code debugging và formal proof theo database v1

Ngày: 2026-10-31. Mốc tham chiếu: `benchmarks/database_v1_20261031`. Database đã được lưu thành bundle có checksum; không thay đổi thiết kế database trong lần kiểm tra này. Hai domain còn lại chưa benchmark-ready cho đánh giá chính thức. Có 20 packet code debugging và 10 packet formal proof; số lượng này không đồng nghĩa đã có 30 live trajectory hợp lệ dưới protocol chung.

## Thiết kế giữ nguyên

- Cùng interface reset/select, observation trước quyết định, role cards công khai, toàn bộ agent pool và STOP; không masking hay tự thay lựa chọn của decision model.
- Cùng ngân sách 16k token/lượt executor và 8 lượt executor; giữ quyết định cuối khi budget bằng 0, chỉ loại action bị budget chặn khi tính routing metric.
- Agent nhận đủ task contract, artifact hiện tại, version và feedback thật. Router nhận bản tóm tắt theo visibility contract; không nhận gold patch/proof hay đáp án evaluator.
- Không tự repair, retry đến khi đúng, hoặc tự rollback mà không ghi nhận. Recovery thuộc quyền orchestrator; transaction/runtime behavior phải hiện trong trace.
- Tách final task correctness, workflow completion/STOP và chất lượng routing. Không dùng tỷ lệ gọi nhiều role làm thước đo chất lượng phối hợp.
- Giữ chính sách rerun lỗi hạ tầng/mất trace đã thống nhất ở database: episode mới, lưu attempt cũ, cùng quy tắc giữa các model. Không bắt buộc sửa trace-loss trước khi tiếp tục.

## Những phần còn thiếu

| Ưu tiên | Phần | Kết quả kiểm tra | Thay đổi tối thiểu |
|---|---|---|---|
| P0 | Chấm cuối cho cả hai domain | `evaluate_shared_environment.py` chỉ gọi `audit_database`; không có SQLite thì trả null và tổng hợp thành 0 pass. Probe formal hợp lệ đã cho workflow pass nhưng end-task pass bằng 0. | Thêm evaluator dispatch theo domain. Trạng thái chưa chấm/lỗi hạ tầng phải tách khỏi task fail. Không thay checker database v1. |
| P0 | Formal: bảo toàn bài toán | Verifier so header theorem/lemma nhưng cho sửa cả file. Probe đổi `IntendedGoal n := n = 37` thành `True` vẫn compile, kernel-check và báo statement preserved. | Công khai vùng proof được sửa; khóa context, imports, definitions, theorem type/binders. Dùng template chèn proof body hoặc kiểm tra declaration trong context gốc; không dựa riêng regex header. |
| P0 | Code: phạm vi patch và test | `_repair_candidate` dùng `git apply --check` rồi apply; chưa kiểm tra allowlist file/hunk. Patch repository-relative có thể đụng test/config mà không bị contract checker chặn tại đây. | Công khai phạm vi được sửa; bảo vệ regression harness, fixtures và evaluator. Chạy final checker với tests/runtime tin cậy trên bản sao checkout cuối. Không dùng test do patch sửa làm bằng chứng duy nhất. |
| P0 | Code: lượt nhiều tool | Diagnostician/Patch Author có nhiều tool. Shared actor chưa có action kết thúc lượt; sau tool thành công, thông điệp hoàn tất có thể thành invalid_output hoặc tiếp tục gọi đến cap. | Thêm kết thúc lượt có schema cho các role nhiều tool, hoặc thiết kế một tool tạo work product với nguồn đọc đủ thông tin. Giữ nguyên đường một-tool của database. Ghi rõ số tool calls và completion. |
| P1 | Code: terminal contract | Test pass chưa terminal: bắt buộc gọi Verified Patch Consumer rồi Patch Finalizer để tạo hai JSON metadata. | Đối chiếu public task: nếu yêu cầu chỉ verified patch thì dùng fresh passing validation + STOP như database. Chỉ bắt buộc promotion/commit khi chính task yêu cầu; không ép pipeline role để tạo phối hợp giả. |
| P1 | Code: runtime từng packet | Bộ 20 packet gồm QuixBugs, BugsInPy và SWE-bench; runner hiện nhận runtime config chung. Chưa xác nhận toàn bộ môi trường hiện tại chạy được baseline/checker tương ứng. | Runtime profile theo packet/revision, dependency lock và preflight. Dùng baseline lỗi đúng kỳ vọng để phân biệt task với infrastructure failure; không coi báo cáo runtime lịch sử là kiểm tra hiện tại. |
| P1 | Formal: thông tin trạng thái | Inspector đọc `proof_state.json` ban đầu và gắn source_ref; chưa tạo active goals tương ứng source mới. | Ghi rõ đây là initial task state; cung cấp source version và compiler feedback mới nhất. Chỉ cần thêm goal extraction nếu task/role contract thật sự cần; không bắt buộc LSP. |
| P1 | Formal: retriever đúng contract | Premise and Lemma Retriever hiện chỉ viết text qua `write_premises`, chưa tra cứu thư viện. | Cấp tra cứu readonly symbol/type/source trên Mathlib đã pin, hoặc thu hẹp role card đúng khả năng thực tế. Ưu tiên implementation đúng contract như lựa chọn của người dùng. |
| P1 | Freshness và môi trường chấm | Code đã xóa test/downstream/final cũ khi patch mới được apply; formal kiểm tra fresh compilation. Chưa có bằng chứng đầy đủ cho mọi dependency mutation/repository change. | Chuẩn hóa lineage candidate → execution → validation theo hash source/repo/tests/runtime; test stale evidence bị từ chối. Dùng checker độc lập cho kết quả cuối ngay cả khi orchestrator chưa gọi validator. |
| P1 | Formal: toolchain và assumptions | Có Lean/Mathlib local và test thật; chưa preflight đủ 10 packet hiện tại. | Pin version theo packet, công khai allowed assumptions, kiểm tra proof trên context gốc với toolchain đúng version; evaluator không thay đổi workspace. |
| P1 | Dữ liệu đánh giá | Router outage có thể tạo state không có decision; labels còn pending; CLI chưa nhập nhãn human. | Lập accepted-attempt/cohort rõ ràng, đánh dấu unmeasurable và denominator; nối nhãn đã review vào action metrics. Có thể vận hành bằng manifest ngoài trước khi tự động hóa. |

Code template đang dùng Gemini 2.5 Flash, khác Qwen 3.5 9B của database; formal dùng Goedel Prover 8B. Nếu mục tiêu là đánh giá phối hợp với SLM, cần chốt executor phù hợp trước pilot và cố định giữa các decision model. Có thể dùng Qwen 9B cho code để giảm khác biệt; formal giữ prover chuyên biệt nếu đó là thiết kế thí nghiệm. Không cần thay provider chỉ vì khác tên cấu hình.

## Trình tự hoàn thiện và tiêu chí nghiệm thu

1. Chốt public task/edit scope/terminal contract cho code và formal; giữ protocol database và version riêng cho adapter/evaluator mới.
2. Sửa P0: final checker từng domain, chống đổi đề formal/chỉnh test code, completion protocol của role nhiều tool.
3. Hoàn thiện thông tin/lineage/runtime ở P1. Kiểm tra âm có ý nghĩa: đổi definition hoặc theorem type, sorry/admit/axiom, sửa protected test, validation cũ sau patch/proof mới, infrastructure failure. Kiểm tra dương bằng patch/proof đúng với tool thật.
4. Preflight 20 code + 10 formal: manifest, checkout/toolchain, baseline và checker. Báo riêng packet chưa đo được; không tự loại packet khó vì model fail.
5. Chạy live pilot với cấu hình cố định, không kịch bản role cố định. Kiểm tra trace/state/usage/dispatch, task correctness độc lập và STOP; human review các quyết định khi đủ state. Không yêu cầu pass rate cao để công nhận môi trường hợp lệ.
6. Chỉ freeze từng domain khi không còn false positive đã biết, checker phân biệt fail với unavailable, không có hidden routing intervention và cohort có thể tái lập. Sau đó mới dùng số liệu so sánh decision model.

Đa dạng trajectory nên đến từ cấu trúc task: lỗi cần tái hiện/định vị/sửa/kiểm tra nhiều lần; proof cần tìm lemma hoặc sửa theo compiler feedback. Không đặt quota phải gọi đủ role, không thưởng đường đi dài, không tự gợi ý next role từ đáp án. Bộ sample nhỏ phù hợp pilot/debug môi trường; chưa đủ kết luận rộng về performance.

## Bằng chứng và giới hạn

- `runs/shared_environment_capability_audit_20261031/probe-001/findings.json`: formal context bypass, evaluator null, multi-tool completion và partial trace probes; không dùng hosted-model calls.
- `routing_benchmark/v4/execution/adapters/external_code_debugging.py`: patch application, invalidation sau patch, promotion/finalization.
- `routing_benchmark/v4/execution/adapters/formal_proof.py`: whole-file source replacement, inspector, compile/assumption checks.
- `scripts/evaluate_shared_environment.py`, `execution/shared_actor.py`: checker dispatch hiện tại và protocol nhiều tool.
- Đây là audit mã nguồn và các probe local đã có, không phải kết quả chạy live mới của 30 packet. Lần này chỉ tạo freeze bundle và tài liệu audit; chưa sửa implementation hai domain.
