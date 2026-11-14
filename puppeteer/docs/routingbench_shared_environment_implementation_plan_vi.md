# Implementation plan: môi trường đánh giá điều phối multi-agent

Ngày: 2026-10-31. Trạng thái: đã triển khai và chạy xác nhận phạm vi P0–P6. Xem `shared_environment_implementation_report_vi.md` cho kết quả thực tế, giới hạn và các artifacts. Human review vẫn pending; diversity/sample expansion là đợt sau.

## 1. Mục tiêu và các quyết định đã chốt

Xây dựng một môi trường chung để cắm decision model vào và đánh giá khả năng chọn agent tiếp theo nhằm giải task nhiều bước. Môi trường cung cấp state và bằng chứng thực thi đúng contract; model tự quyết định thứ tự, retry, recovery và STOP.

Các quyết định áp dụng cho implementation:

- Cung cấp implementation thực sự cho mọi role theo role card hiện có; không thu hẹp contract để hợp thức hóa placeholder.
- Giữ flow ra quyết định hiện tại, bao gồm việc có thể hỏi router một lần cuối khi executor budget bằng 0. Phân loại và lọc quyết định đó ở evaluator khi nó bị chặn chỉ vì hết budget.
- Cố định `max_tokens=16384` cho các LLM executor trong cấu hình thí nghiệm mới; giữ model `qwen/qwen3.5-9b`. Model của formal proof hoặc code debugging được giữ theo cấu hình domain đã chốt, không tự thay backbone.
- Lưu `finish_reason`, usage và metadata response để phân tích hậu kỳ. Không thêm state hay logic routing riêng cho token exhaustion; không tự tăng token budget theo kết quả task.
- Provider giữ theo cấu hình hiện tại; không bổ sung công việc pin/tối ưu provider.
- Giữ nguyên 12 database packets, task contract, dữ liệu nguồn và evaluator/reference. Không viết lại task hoặc thêm sample trong đợt hoàn thiện này.
- Chạy ngân sách 8 executor calls cho đợt database tiếp theo, giữ cách tính call/rejection hiện có. Các action và cost được ghi riêng để kiểm chứng.
- Giữ các run cũ bất biến. Không chuyển human-review pending thành approved bằng chương trình.

Không đặt ngưỡng pass rate của Jev hoặc MAS làm điều kiện nghiệm thu môi trường. Không thêm preferred action, lời giải, tự sửa SQL, tự chọn role tiếp theo hoặc bắt buộc một trình tự role.

## 2. Căn cứ từ implementation hiện tại

- `tmp/run_database_dynamic_jev.py` là bridge thử nghiệm, có đường dẫn tuyệt đối tới worktree `4381` và nhiều nhánh dispatch trong cùng một hàm.
- Router nhận `latest_migration_execution`, nhưng LLM role nhận `shared` không có đầy đủ execution feedback tương ứng.
- Schema inspection, candidate và validation chưa có lineage/version rõ ràng; thông tin cũ có thể còn trong shared state sau khi artifact đổi.
- Data Profiler gọi cùng lệnh inspect; Query Performance Auditor chưa thực hiện probe.
- Bridge có thể tự restore `source.sqlite` khi execution lỗi và DB thay đổi; đây không tương đương rollback riêng transaction vừa thất bại.
- Đã có các abstractions trong worktree nguồn: `ExecutionAdapter`, `ToolDefinition`, `ToolResult`, `EnvironmentSnapshot`, `SnapshotStore`, `EmpiricalTrajectoryRunner` và router adapter. Cần tái sử dụng, đối chiếu phiên bản trước khi bổ sung abstractions mới.
- Registry hiện chưa đăng ký adapter cho `database_migration.sqlite.v1` của domain packet.

## 3. Kiến trúc đích

Luồng chung:

`public task/artifacts → environment observation → decision-model adapter → selected role → role executor/tools → execution result/artifacts → state update`

Các ranh giới trách nhiệm:

| Thành phần | Trách nhiệm | Không làm |
|---|---|---|
| Environment/runner | Lifecycle, workspace, artifact versions, budget, dispatch và checkpoint | Chọn agent hộ model hoặc sửa output của agent |
| Domain adapter | Public preconditions, công cụ thật, tác động artifact và kiểm tra domain | Đưa private reference hoặc nhãn safe action vào public observation |
| Role executor | Thực hiện role được chọn, đọc input công khai, trả output hoặc tool call trong quyền hạn | Tự gọi role khác hoặc tuyên bố thành công chưa được kiểm tra |
| Decision adapter | Chuyển cùng public observation/action space sang API của model; trả lựa chọn và metadata | Đọc evaluator label hoặc thay đổi environment theo model identity |
| Evaluator/review exporter | Chấm outcome, tổng hợp cost, chọn mẫu số metrics và xuất queue | Viết lại lịch sử thực thi hoặc coi hành vi router là gold label |

Decision interface giữ `reset(seed)` và `select(observation, candidate_actions)`. Agent chỉ nhận bằng chứng công khai trong phạm vi role; có quyền đọc lại artifact công khai cần thiết. Context projection phải cố định, có kiểm thử và công bố được. Không cắt bỏ lỗi thực thi hoặc candidate liên quan mà không ghi rõ.

STOP và final task checker tiếp tục theo contract hiện hành. Phân biệt rõ task contract đạt, router đã chọn STOP, và episode bị chặn do budget.

## 4. P0 — Đóng gói baseline và cấu hình chạy

Đầu ra:

1. Chọn một canonical checkout; đối chiếu các module v4 mới trong worktree `4381` và chuyển phần cần thiết bằng thay đổi có thể review. Không để production runner import code từ đường dẫn worktree cá nhân.
2. Đưa dataset root, output root, model, token cap, call budget, timeout, seed và retry policy vào config/CLI.
3. Chốt config mới: SQL/analysis/planning LLM executor có cùng trần 16384 tokens; timeout đủ cho trần này, ban đầu 300 giây/request. Ghi timeout thành tài nguyên execution, không coi timeout là lỗi reasoning.
4. Lưu manifest/checksum cho packets, role cards, adapter/tool code, prompts và resolved config. Giữ nguyên các run 4096-token và probe 16384-token để đối chiếu.

Nghiệm thu: chạy được từ canonical checkout mà không phụ thuộc `C:/Users/.../worktrees/4381`; không thay task/reference; mọi tham số thí nghiệm có trong resolved config.

## 5. P1 — Public state và artifact lineage

Bổ sung trên state/history và snapshot hiện có:

- Artifact có ID, version/hash, producer event, input references và visibility.
- Execution result gắn với candidate hash, database version trước/sau và tool invocation cụ thể.
- Inspection, profile, analysis, plan và validation lưu input versions; đánh dấu stale khi input thay đổi. Giữ output cũ trong history, không trình bày như bằng chứng mới nhất.
- Mỗi role invocation nhận task/contract, artifact liên quan và execution/validation feedback mới nhất. Với SQL Author, lỗi SQL trước và candidate gây lỗi phải truy cập được.
- Một nguồn state chuẩn tạo cả router observation và role context. Hai bên có projection khác nhau theo contract nhưng không dùng hai bản state cập nhật độc lập.
- Chuẩn hóa history thành các event có bằng chứng, không tự sinh khuyến nghị sửa hoặc next-agent.
- Lưu metadata response đầy đủ cần cho phân tích: model, request/generation ID, finish_reason, usage, latency và lỗi dịch vụ. Không thêm trường token-exhaustion riêng vào task state.
- Output rỗng được lưu nguyên văn; không quảng bá file chỉ có khoảng trắng thành candidate SQL có nội dung. Đây là kiểm tra artifact, không phải suy diễn nguyên nhân hết token. Không tự retry model ngoài chính sách transport đã chốt.

Nghiệm thu: fixture với candidate A lỗi và candidate B mới phải cho thấy B nhận được lỗi A khi liên quan, validation A không được gán cho B, và inspection của DB cũ không bị hiểu là snapshot mới.

## 6. P2 — Database adapter và implementation đủ 10 roles

Đăng ký adapter cho packet contract hiện có; dùng SQLite/tool runner thật, không chuyển sang simulator. Chỉ chạy trên working copies.

| Role | Implementation cần có | Bằng chứng đầu ra |
|---|---|---|
| Schema Inspector | Introspection bảng/cột, PK, NOT NULL, index, trigger, view, FK và dependency khai báo | Schema snapshot, dependency map, DB version |
| Data Profiler | Read-only profiling theo public schema/task: row counts, nulls, duplicate groups, value types và orphan candidates khi quan hệ mục tiêu công khai | Queries đã chạy, kết quả/statistics, phạm vi khảo sát, DB version |
| Constraint Analyzer | Bounded LLM đọc schema/profile, public invariants và lỗi execution/validation; trả phân tích có evidence refs | Analysis artifact, input refs; không giả lập tool execution |
| Migration Planner | Bounded LLM lập ordering, dependency, transaction và rollback boundaries từ dữ liệu quan sát được | Plan artifact với assumptions và input refs |
| Data Repair Planner | Bounded LLM lập backfill/dedup/remap/orphan repair theo policy trong task; nêu rõ nếu public policy còn thiếu | Repair plan có mapping/rule và bằng chứng, không tự bịa chính sách nghiệp vụ |
| SQL Migration Author | Bounded LLM viết candidate SQLite SQL từ artifact/context hiện tại | Candidate SQL và input refs; không tự chạy hoặc tuyên bố thành công |
| Transactional Dry-run Executor | Chạy candidate hiện tại trên bản sao cô lập theo transaction contract đã công bố; ghi kết quả và artifact thực tế | SQL hash, stdout/stderr, return code, DB versions, transaction outcome |
| Integrity Validator | Chạy các public postcondition/regression/integrity checks đã được phép, trên đúng executed artifact; không để probe làm đổi working DB | Từng check/query, kết quả, candidate/DB version được kiểm tra |
| Query Performance Auditor | Thực thi EXPLAIN QUERY PLAN và phép đo/threshold nếu task công khai yêu cầu; ghi điều kiện đo | Query plans/measurements thực tế; không tự thêm tiêu chí performance |
| Snapshot & Rollback Operator | Tạo snapshot có hash; liệt kê snapshot hợp lệ; restore snapshot được xác định rõ qua tool invocation | Snapshot/restore ID, DB versions trước/sau, tác động có thể kiểm tra |

Quy tắc triển khai:

- Tool thực hiện thao tác; LLM được dùng khi role cần chọn query, phân tích hay lập kế hoạch. Không bắt buộc mọi tool role trở thành một LLM call.
- Với role cần chọn tham số như snapshot target hoặc profiling query, dùng typed tool calls trong phạm vi role. Không suy ra thao tác restore chỉ từ một cờ fail mơ hồ.
- Quyền read/write, tool schemas, giới hạn số tool calls trong một role invocation và cách tính cost được chốt trong executor config; giống nhau cho mọi decision model.
- Profiling không gọi evaluator/private reference để phát hiện anomaly. Audit performance không tự chế query hoặc threshold thành terminal requirement.
- Task không có performance contract thì public precondition của Performance Auditor phản ánh điều đó; chọn role không áp dụng được trả rejection có lý do, không giả lập audit thành công.
- Không bắt SQL Author phải chờ Planner nếu input công khai đã đủ. Các role có thể có vùng áp dụng chồng lấn.

Nghiệm thu: mỗi role có ít nhất một fixture chức năng chứng minh output đúng contract và một fixture precondition/failure cần thiết. Performance fixture có public query/contract riêng cho kiểm thử công cụ, không sửa 12 packet đang đánh giá.

## 7. P3 — Transaction, snapshot và recovery

1. Giữ atomic transaction behavior do SQLite runner quy định. Khi DB tự rollback, ghi đây là transaction outcome của invocation hiện tại.
2. Không gán transaction rollback thành một action recovery do router chọn.
3. Bỏ việc driver ngầm restore DB nguồn ban đầu sau bất kỳ failure có thay đổi dữ liệu. Đó là thay đổi state ngoài transaction cần contract riêng, có thể xóa tiến triển của các bước trước.
4. Khi môi trường có state partial sau failure, giữ và công khai state/evidence đủ để router quyết định kiểm tra hoặc gọi Snapshot & Rollback Operator. Không tự chọn recovery hộ router.
5. Nếu cần containment do lỗi hạ tầng, ghi event riêng với actor=`environment`, origin và DB versions trước/sau; không tính là agent call hoặc recovery success. Lỗi vi phạm atomic contract được đánh dấu environment failure, không âm thầm sửa thành actor success.
6. Snapshot restore phải đúng snapshot và cập nhật lineage/freshness. Không tự dùng source ban đầu thay cho snapshot đã chọn.

Nghiệm thu: scenario A commit rồi B fail phải giữ A sau transaction rollback B. Chỉ thao tác restore được yêu cầu rõ mới được quay về snapshot trước A. Mọi thay đổi tự động đều có event tương ứng.

## 8. P4 — Budget, evaluator và human review

Giữ nguyên thứ tự hiện tại: hỏi router → xử lý STOP → kiểm tra executor budget → dispatch hoặc kết thúc. Không bỏ lần hỏi cuối trong runner.

Evaluator tạo các nhóm record dựa trên execution linkage, không dựa riêng vào vị trí cuối file:

- `dispatched`: action được dispatch tới role/tool.
- `precondition_rejected`: action không chạy do precondition công khai; vẫn là quyết định cần đánh giá.
- `stop_handled`: STOP thực sự được runner xử lý; giữ terminal/early-stop semantics.
- `suppressed_by_budget`: action không phải STOP, không được dispatch chỉ vì budget đã bằng 0; giữ trong raw trace, loại khỏi primary routing-action denominator.

Không loại mọi quyết định có `executed=false`: như vậy sẽ xóa cả lỗi precondition của router. Không tự loại mọi quyết định cuối: STOP hợp lệ vẫn là bằng chứng có ý nghĩa.

Đầu ra metrics tách biệt:

- Workflow task success, terminal-contract success và STOP behavior.
- Applicable-action accuracy/unsafe routing khi có human labels; nếu chưa có thì để pending, không dùng behavior action làm gold.
- Calls, tokens, latency; số decisions được tính/loại và lý do.
- Executor/tool/infrastructure failure từ evidence; trường hợp chưa đủ căn cứ giữ unresolved hoặc mixed.
- Role/transition coverage để mô tả trajectory diversity, không cộng điểm vì gọi nhiều role.

Review exporter hiển thị task, public state, role cards, artifact lineage, actual outputs và execution linkage. Đề xuất nhãn/hành vi router tách khỏi bước blind human review. Metadata token dùng cho thống kê hậu kỳ.

Nghiệm thu: trace cũ gồm 96 decisions phải phân biệt được 84 role calls, 3 STOP và 9 budget-suppressed decisions; không làm thay đổi raw trace hoặc loại nhầm precondition rejection.

## 9. P5 — Runner chung và tương thích ba domain

Database là implementation đầu tiên. Dùng lại cùng runner, state/history, snapshot và evaluator khi tích hợp formal proof và code debugging.

- Database: đăng ký adapter mới và role/tool implementation ở P2.
- Code debugging: tái sử dụng external code adapter và tool-aware actor hiện có; kiểm tra việc truyền patch/test feedback và artifact freshness theo contract chung.
- Formal proof: bọc compiler/toolchain hiện có thành domain adapter; compile diagnostics, source candidate, theorem/assumption verification có lineage. Không đưa private reference proof vào executor context.
- Giữ role cards và runtime/toolchain theo từng domain; không ép ba domain có cùng số role hoặc cùng kiểu artifact.
- Decision model được thay qua config/adapter. Environment không rẽ nhánh theo tên Jev hoặc model khác.
- Mọi guard thuộc policy/router adapter phải được công bố và giữ nhất quán trong so sánh; environment chỉ thực thi public executable preconditions.

Không bắt buộc chạy lại toàn bộ code/formal packets trong đợt database này. Điều kiện hoàn thiện môi trường chung là có adapter và contract tests/smoke thực tế đại diện cho cả ba domain; đánh giá live diện rộng thực hiện theo từng run riêng.

## 10. Phân chia code dự kiến

Đường dẫn dưới đây là cấu trúc dự kiến; ưu tiên mở rộng module hiện có nếu đã đáp ứng trách nhiệm tương ứng.

| Khu vực | Công việc |
|---|---|
| `routing_benchmark/v4/execution/base.py`, `schema.py` | Bổ sung execution linkage và artifact/input version contracts |
| `state_history.py`, `snapshot_store.py` | Freshness, lineage, snapshot/restore và event origin |
| `trajectory_runner.py` | Runner dùng chung, public state/context nhất quán; giữ budget decision flow |
| `execution/adapters/` và `registry.py` | Database adapter, formal adapter; tái sử dụng code adapter |
| `v4/domain_packets/roles.py` và role-executor modules | Registry role → implementation, typed tools và context projections |
| `routing_benchmark/routers.py` | Giữ pluggable router interface; không chứa domain solution logic |
| `scripts/collect_routingbench_v4_trajectories.py` hoặc entry point chung tương đương | CLI/config, resume có kiểm tra run identity; thay bridge trong `tmp/` |
| Evaluator/review exporter | Decision denominator, execution linkage, queue và báo cáo metrics |
| `config/` | Cấu hình frozen cho database 16k, Jev, 8 calls |
| `tests/` | Functional adapter/tool tests, state/visibility, transaction và accounting regression |

## 11. P6 — Kiểm thử và chạy xác nhận

Thứ tự thực hiện:

1. Contract tests không cần API: version/freshness, context projection, public/private boundary, dispatch, metadata retention, budget filtering.
2. SQLite integration: profile query thật; FK/constraint inspection; validation không làm đổi DB; transaction rollback; snapshot restore; EXPLAIN/measurement thật.
3. Integration runner với router/executor stub: cho nhiều thứ tự action khác nhau, retry, rejection, STOP và budget boundary; không khẳng định một đường routing duy nhất là đúng.
4. Smoke bằng adapter thật cho code debugging và formal proof, có toolchain/runtime thực tế và status môi trường rõ ràng.
5. Live smoke với Jev và SLM trên một packet trước để kiểm tra kết nối, lưu artifacts và protocol. Không chỉnh prompt/role design theo pass/fail của smoke.
6. Chạy toàn bộ 12 database packets trong namespace mới, cùng config 16k/8 calls. Lưu đầy đủ resolved config, checksums, model metadata và evaluator report. Không trộn kết quả probe một SQL request với dynamic episode.
7. Xuất human-review queue mới và đối chiếu từng state/role contract. Giữ pending cho tới khi reviewer xác nhận.

Live success rate không phải gate. Gate là state faithful, role/tool implementation đúng contract, visibility đúng, log/accounting đầy đủ và nguồn không bị sửa.

## 12. Diversity sau khi môi trường đạt contract

Đợt tiếp theo bổ sung task có dependency giữa nhiều bảng/artifacts, data repair với policy công khai, semantic validation failure, recovery từ trạng thái thực và performance contract rõ ràng. Giữ task đơn giản làm đối chứng.

Không ép mỗi trajectory gọi nhiều role, không giấu input một cách tùy ý để bắt gọi Inspector/Planner, không tạo fault chỉ để làm Jev chọn role mong muốn. Nếu thêm controlled fault track thì tách khỏi natural live track, công bố intervention và giữ cùng public observability contract.

## 13. Thứ tự và điều kiện hoàn thành

Critical path: P0 → P1 → P2/P3 → P4 → database live validation ở P6. Tương thích ba domain ở P5 là gate cho tuyên bố môi trường dùng chung; mở rộng mẫu/diversity không chặn việc xác nhận 12 database packets hiện tại.

Hoàn thành khi:

- Có một runner/config flow dùng chung, không phụ thuộc worktree tuyệt đối hoặc model identity.
- Đủ 10 database roles thực thi được đúng contract; các role của code/formal được kiểm tra qua cùng protocol.
- Router và agent nhận bằng chứng cần thiết, đúng artifact version, không nhận gold/reference.
- Transaction rollback và orchestrator recovery có provenance chính xác.
- Flow quyết định cuối giữ nguyên; evaluator công bố rõ các record bị loại theo budget.
- Các LLM executor của run mới dùng 16k như config; chỉ tổng hợp token exhaustion từ metadata hậu kỳ.
- 12 database episodes chạy lại được và human-review artifacts đầy đủ, bất kể tỷ lệ thành công.

