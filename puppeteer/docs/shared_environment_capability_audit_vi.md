# Audit khả năng đánh giá của shared environment

Ngày 2026-10-31. Đây là audit độc lập đối với implementation hiện tại; không thay runtime code, role cards, token budget, task hoặc các trajectory đã chạy.

## Kết luận

Chưa đạt đầy đủ tiêu chí benchmark chung để cắm decision model vào và đo khả năng điều phối nhiều agent. Đã có nền tảng runtime và database pilot sử dụng được, nhưng task checker, evaluator giữa các domain và tính đầy đủ của trace còn lỗi cần sửa.

25 kiểm thử trước xác nhận các đường chức năng đã kiểm tra hoạt động. Chúng chưa bao phủ các negative cases và failure paths trong audit này, nên không đủ để kết luận benchmark-ready.

## Phương pháp và evidence

Script tái hiện: `tmp/audit_shared_environment_capability.py`.

Evidence: `runs/shared_environment_capability_audit_20261031/probe-001/findings.json`, cùng các workspace/trace của probes. Không gọi hosted decision model/executor. Router/client được stub để kiểm soát đường lỗi; formal probes chạy Lean compiler/kernel thật.

Source freeze của database run hiện tại được đọc lại: 289 files, không có thay đổi. 90 quyết định của run đó có 0 lần guard lọc action, 0 precondition rejection, 0 role transport errors và 0 multi-tool invocations. Các issue dưới đây không tự động phủ nhận 12 database trajectories; phải phân biệt phạm vi đã xảy ra với khả năng benchmark chưa được bảo đảm.

## Đối chiếu tiêu chí

| Tiêu chí | Đánh giá |
|---|---|
| Thay decision model qua cùng `reset/select`, environment không rẽ nhánh theo tên Jev | Đạt nền tảng; CLI hiện hỗ trợ typed-decision adapter và random, API khác cần adapter tương ứng |
| Task/tool thật, input công khai, không tự chọn chuỗi role hoặc tự viết lời giải | Đạt cho database pilot và các smoke paths đã kiểm tra |
| Cung cấp execution feedback và đúng version cho router/executor | Đạt cho 37 LLM contexts của database run; chưa bảo đảm khi actor lỗi giữa multi-tool invocation |
| Để model tự quyết định trên cùng action space | Cần chốt neutral mode: router wrapper có guard lọc action mặc định |
| Đánh giá đúng task ban đầu | Chưa đạt cho formal proof vì có false positive khi đổi định nghĩa goal |
| Cùng semantics end-task evaluation cho ba domain | Chưa đạt: final evaluator chỉ có SQLite checker |
| Budget accounting và giữ raw decisions | Đạt cho run hiện tại: 90 raw, 84 primary, 6 budget-suppressed |
| Phân biệt hỏng dịch vụ với khả năng điều phối | Chưa hoàn thiện episode eligibility/denominators |

## F1 — Formal verifier có false positive khi context của theorem đổi [P1]

Vị trí: `routing_benchmark/v4/execution/adapters/formal_proof.py:143–158`.

Checker so sánh chuỗi declaration header và kiểm tra axioms của candidate. Nó không bảo vệ các định nghĩa/context mà header tham chiếu.

Probe dùng task nguồn:

```lean
def IntendedGoal (n : Nat) : Prop := n = 37
theorem audit_goal (n : Nat) (h : n = 37) : IntendedGoal n := by sorry
```

Candidate đổi định nghĩa goal:

```lean
def IntendedGoal (n : Nat) : Prop := True
theorem audit_goal (n : Nat) (h : n = 37) : IntendedGoal n := by trivial
```

Kết quả thực tế: compile success; `statement_preserved=true`, `no_forbidden_placeholders=true`, `kernel_assumptions=true`, `passed=true`. Theorem header có cùng text nhưng nghĩa của conclusion đã đổi. Kernel chỉ xác nhận theorem trong context candidate, không xác nhận candidate giữ nguyên bài toán nguồn.

Tác động: có thể ghi task success cho việc giải bài toán đã bị làm dễ đi. Đây là lỗi correctness của evaluator, không phải vấn đề pass rate hay số sample.

Cần sửa: bảo vệ context/definitions và theorem identity của task nguồn; kiểm tra candidate trong cùng context, với edit scope công khai. Nếu cho phép helper lemmas/imports thì phải định nghĩa rõ vùng được thêm/sửa và ngăn shadowing/redefinition. Bổ sung negative cases đổi definition, namespace/identity và assumptions.

## F2 — Final-task evaluator chưa chung cho ba domain [P1]

Vị trí: `scripts/evaluate_shared_environment.py:17–20,81`.

`audit_database()` trả `None` nếu workspace không có `working.sqlite`; aggregator tính `bool(None)` thành false khi cộng `end_task_passed`.

Probe formal riêng giữ đúng context nguồn, dùng proof `by exact h`, compiler/kernel thật và SharedTrajectoryRunner. Kết quả:

- episode `passed=true`;
- evaluator `completed=1`, workflow passed=1;
- `end_task_audit=null` nhưng `end_task_passed=0`.

Code-debugging workspace cũng không có working SQLite nên đi vào cùng nhánh chưa có checker. Chưa có verdict không được đánh đồng với task fail.

Cần sửa: domain-specific final checker qua một interface chung; verdict tách `pass`, `fail`, `not_evaluated` và lỗi checker. Dùng mẫu số có checker hợp lệ; giữ evaluator-only probe ngoài router state. Performance contract của database, nếu có, cũng phải nằm trong final checker thay vì chỉ kiểm tra integrity.

## F3 — Lỗi API giữa invocation làm mất trace của tool đã chạy [P1]

Vị trí: `shared_actor.py:23–36,58–94` và `shared_runner.py:89–97` trong execution package.

Probe multi-tool role:

1. response đầu chọn tool và tool thay state;
2. request model tiếp theo phát sinh ConnectionError;
3. `RoleTransportError` thoát ra mà không mang partial invocation record;
4. runner tạo result mới với `tool_events=[]`, `responses=[]`.

Kết quả thực tế: một tác động tool đã xảy ra, hai model requests đã được thử; trace role ghi 0 tool events, 0 model responses và `semantic_delta=state_changed`.

Tác động: thiếu bằng chứng để review recovery, sai cost accounting và khó quy attribution. Các role nhiều tool của code debugging có thể đi qua đường này. Database run hiện tại không có multi-tool invocation/transport error nên chưa bị issue này.

Cần sửa: exception/outcome giữ partial requests, responses và tool events; persist mỗi tool effect; kết thúc invocation bằng failure event mà không xóa lịch sử đã thành công. Regression test phải mô phỏng thành công một tool rồi lỗi API.

## F4 — Action-filtering guard là policy bổ sung ngoài raw decision model [P2 / quyết định protocol]

Vị trí: `routing_benchmark/routers.py:95–110,229–271`; factory ở `scripts/run_shared_environment.py:99`.

Guard structured mặc định bỏ các action vừa bị precondition rejection nếu state chưa đổi. Đây là rule-based action masking từ evidence công khai, không dùng oracle; tuy nhiên model không còn tự chọn trên nguyên action space mà environment ban đầu công bố.

Probe: caller đưa `[A,B,STOP]`, history có rejection của A; API nhận choices `[B,STOP]`. Thêm key `observable_rejection_guard_enabled=false` vào router config không tắt guard vì CLI không truyền key đó vào constructor argument tương ứng.

Tác động: số lỗi routing có thể bị giảm bởi wrapper; khác biệt adapter/guard có thể bị hiểu thành chất lượng decision model. Canonical decision-state snapshot cũng chưa ghi effective masked API input. Trong 90 decisions hiện có, guard không kích hoạt nên không có bằng chứng nó làm thay đổi kết quả run đó.

Cần chốt: primary benchmark raw-model nên không có action masking bổ sung; hoặc định nghĩa rõ benchmark đo model + cùng guard cố định cho mọi adapter. Expose config thực sự và ghi effective model input nếu dùng guard. Không thay public executable preconditions của environment.

## F5 — Service-error episodes chưa có evaluation eligibility riêng [P2]

Vị trí: `shared_runner.py:58–64,105–114` và aggregator trong `evaluate_shared_environment.py`.

Probe router outage: 1 pre-decision state, 0 decisions, 0 executor calls, reason `ROUTER_SERVICE_ERROR`, `passed=false`. Aggregator vẫn đếm result file này như completed episode khi báo task success. Validator lại coi state/decision join đó là mismatch, chưa có một semantics lỗi dịch vụ thống nhất.

Cần sửa: phân loại run/episode validity và lỗi service/tool/setup; báo rõ denominator dùng để đo coordination. Vẫn giữ counts của mọi sự cố để tránh âm thầm bỏ task khó. Response sai choice/schema và lỗi mạng cần được phân biệt, không gom mọi exception thành router transport.

## Hai hạn chế bổ sung

1. Multi-tool actor chỉ có protocol trả tool-call; không có thông điệp finish hợp lệ. Nếu tool luôn success thì nó tiếp tục đến hết cap. Probe gửi completion sau một tool thành công bị gán invalid_output. Cần cho phép role kết thúc invocation khi đã hoàn thành, đồng thời giữ quyền chọn next agent thuộc router. Smoke code hiện dùng actor fixture cho Patch Author nên chưa kiểm tra đường này.
2. `episode_metrics()` nhận được labels nhưng CLI evaluator chưa có đường nạp human labels và luôn đặt global action accuracy là null. Khi muốn chấm action-level cần nối thêm input labels/coverage. Đây là công việc evaluation wiring, khác với việc chưa có human annotations; end-task evaluation không bắt buộc phải đợi chỉ số này.

## Phần đã đạt và dữ liệu có thể dùng

- Shared runner và adapter interface là nền tảng phù hợp cho mục tiêu plug decision model.
- Database có các tool thật và context đầy đủ ở đường đã chạy; task/result sources giữ nguyên.
- Budget boundary được ghi nhận và lọc đúng phạm vi đã chốt. STOP tại budget 0 vẫn được xử lý, không bị loại máy móc.
- End-task audit của database được chạy ngoài router context trên bản sao, phân biệt đúng 6 workflow success với 7 final-artifact success trong run hiện tại.
- Các 12 database trajectories vẫn dùng làm pilot và human review theo protocol đã lưu. Chưa dùng chúng để khẳng định benchmark chung đã hoàn thiện hoặc kết luận tổng quát về decision model.

## Thứ tự khắc phục để đạt mục tiêu

1. Sửa F1 và F2: chấm đúng task/context, có final-checker interface chung và verdict không chấm được.
2. Sửa F3 và hoàn thiện completion protocol: đảm bảo agent/tool trace đầy đủ ở cả thành công và lỗi.
3. Chốt F4: neutral primary mode hoặc công khai cùng policy wrapper; expose config và actual model input.
4. Sửa F5: thống nhất service-error semantics và evaluation denominators; bổ sung human-label ingestion khi cần action-level scoring.
5. Chạy contract/negative-case probes, rồi smoke lại ba domain. Giữ model, 16k cap, task contracts và raw runs hiện tại; việc tăng sample/diversity là bước sau.

Điều kiện đạt là evaluator đúng và protocol trung thực, nhất quán, thay model được; không có điều kiện ép Jev/MAS phải tăng tỷ lệ giải thành công.
