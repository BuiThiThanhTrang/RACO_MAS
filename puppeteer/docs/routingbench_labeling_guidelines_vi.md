# Hướng dẫn gán nhãn RoutingBench V3-beta

## 1. Mục đích của tài liệu

Tài liệu này quy định cách đánh giá một **quyết định routing tại một trạng thái cụ
thể**. Đối tượng được đánh giá là orchestrator: từ task, state quan sát được,
execution memory, budget và role card, orchestrator cần chọn agent tiếp theo hoặc
`STOP`.

Đây là quy tắc gán nhãn và từ điển trường dữ liệu. Tài liệu không mô tả cách cài
đặt, mở hoặc thao tác với phần mềm gán nhãn.

Nguyên tắc trung tâm:

- chỉ dùng thông tin mà router được phép quan sát tại decision state hiện tại;
- đánh giá **action có thể biện minh**, không đoán action mà một baseline sẽ chọn;
- một state có thể có nhiều action hợp lệ;
- không ưu tiên agent vì backbone mạnh hơn, nổi tiếng hơn hoặc từng cho accuracy cao;
- không dùng đáp án gold, transition ẩn hoặc kết quả baseline trong blind review;
- lỗi reasoning ngẫu nhiên của executor không tự động làm role contract sai.

## 2. Đơn vị gán nhãn

### 2.1 Workflow

Một workflow mô tả toàn bộ quá trình từ state ban đầu tới goal. Workflow có task,
agent pool, các state có thể quan sát, transition contract, budget và terminal
contract.

### 2.2 Decision state

Một decision state là ảnh chụp của workflow tại đúng một thời điểm. Nhãn cần trả
lời:

> Với những gì đang quan sát được ngay lúc này, mỗi candidate action có phải là
> một bước tiếp theo hợp lệ không?

Không đánh giá toàn bộ trajectory chỉ từ một state. Không giả định một action đã
xảy ra nếu action đó chưa xuất hiện trong state hoặc execution memory.

### 2.3 Candidate action

Candidate action là một `agent_id` hoặc `STOP`. Mỗi candidate phải nhận một trong
ba nhãn `valid`, `invalid`, `uncertain`.

## 3. Quy trình suy luận chuẩn cho một candidate action

Đánh giá lần lượt bảy câu hỏi sau:

1. State có một nhu cầu chưa hoàn tất mà role này giải quyết được không?
2. Input mà role yêu cầu đã có trong state/artifact, hoặc sẽ được cấp trực tiếp
   trong lời gọi tiếp theo không?
3. Có điều kiện nào trong `avoid_when` đang đúng không?
4. Expected output có giải quyết một open requirement, tạo artifact cần thiết,
   xử lý failure hoặc tiến gần goal không?
5. Action có khả thi với `calls_remaining` không?
6. Nếu role đã được gọi trước đó, state/evidence/query có thay đổi thực chất để
   biện minh cho lần gọi lại không?
7. Lý do chọn action có hoàn toàn dựa trên public state, không cần hidden gold
   hay transition ẩn không?

Gán nhãn:

- `valid`: cả bảy điều kiện đều thỏa mãn.
- `invalid`: có ít nhất một vi phạm rõ ràng và thông tin công khai đủ để kết luận.
- `uncertain`: role card hoặc public state thiếu/không rõ đến mức chưa thể kết
  luận hợp lý; phải ghi `uncertainty_reason`.

`Uncertain` không phải nhãn cho trường hợp reviewer chưa đọc kỹ, và cũng không
phải lựa chọn trung gian khi phân vân giữa hai action đều có thể hợp lệ.

## 4. Set-valued ground truth

RoutingBench cho phép nhiều action hợp lệ tại cùng một state. Ví dụ specialist và
generalist đều có thể tạo đúng output contract. Khi đó phải gán `valid` cho cả
hai nếu cả hai đều thỏa rubric.

Không chọn một action “tốt nhất” chỉ vì:

- action đó rẻ hơn nhưng action còn lại vẫn nằm trong budget;
- role chuyên biệt hơn nhưng generalist vẫn có đủ input và output contract;
- một backbone thường mạnh hơn backbone khác;
- current oracle chỉ chứa một action.

Chi phí chỉ làm action invalid khi gọi action đó khiến workflow không còn đường
hoàn tất trong budget. Dataset V3-beta không dùng capability score hoặc utility
score làm nhãn ưu tiên.

## 5. Quy tắc cho `STOP`

`STOP` chỉ `valid` khi đồng thời đúng:

1. goal công khai đã hoàn tất;
2. final artifact đã được commit;
3. không còn dependency, conflict, verification hoặc finalization bắt buộc;
4. terminal contract của task đã thỏa mãn.

Các tình huống sau **không đủ** để `STOP`:

- đã có candidate answer nhưng chưa commit final artifact;
- answer có vẻ hợp lý nhưng evidence hoặc dependency còn thiếu;
- agent trước nói “final answer” trong text nhưng state chưa ghi nhận final artifact;
- hết actor budget trong khi goal chưa hoàn tất;
- verifier/retriever tạo intermediate answer nhưng chưa hoàn tất output contract.

Trong V3-beta, terminal `STOP` là bước zero-cost dành riêng và không tiêu actor
execution budget. `reserved_terminal_stop=true` không có nghĩa là `STOP` luôn hợp
lệ; nó chỉ bảo đảm còn một bước terminal sau khi final artifact được commit.

## 6. Gọi lại cùng một role và semantic loop

Không cấm lặp role theo tên. Gọi lại cùng role có thể `valid` nếu state thay đổi
thực chất, ví dụ:

- Web Researcher dùng một query mới sau khi query trước thất bại;
- verifier kiểm tra candidate mới sau khi conflict đã được sửa;
- retriever tìm dependency thứ hai sau khi dependency thứ nhất đã giải xong;
- parser xử lý artifact khác với lần trước.

Gọi lại thường `invalid` khi input, query, evidence, artifact và mục tiêu con đều
không đổi, đặc biệt sau một lần thất bại không tạo tiến triển. Đây là semantic
loop, ngay cả khi chuỗi text hoặc wording hơi khác.

## 7. Phân biệt rejection và actor failure

### 7.1 Pre-execution rejection

Action bị harness từ chối trước khi agent thực thi, thường vì thiếu input,
precondition không thỏa hoặc action không được phép. Dấu hiệu nằm trong
`execution_memory.rejected_actions` và `routing_rejection_count`.

Sau rejection, lặp lại action không đổi thường `invalid`. Một agent khác có khả
năng tạo input còn thiếu có thể `valid`.

### 7.2 Post-execution actor failure

Agent đã được gọi nhưng execution không tạo output hợp lệ, được phản ánh trong
`attempts_by_agent`, `actor_failure_count` hoặc `last_event.status`. Đây không
nhất thiết là lỗi routing ban đầu.

Retry có thể `valid` nếu contract cho phép và có thay đổi thực chất về input,
evidence, query hoặc recovery state. Nếu không có thay đổi, retry thường tạo loop.

## 8. `state_sufficient`

Gán `true` khi public state và role cards đủ để một reviewer độc lập xác định
validity của **tất cả** candidate actions.

Gán `false` khi ít nhất một quyết định cần thông tin bị ẩn hoặc thiếu, ví dụ:

- không biết artifact cần thiết có tồn tại hay không;
- không biết source scope nhưng validity phụ thuộc vào local corpus hay live web;
- không phân biệt được rejection với execution failure;
- role card không nói rõ input/output boundary;
- goal status mâu thuẫn với facts/artifacts.

`state_sufficient=false` không bắt buộc mọi action đều `uncertain`. Những action
chắc chắn đúng hoặc sai vẫn phải được gán nhãn tương ứng.

## 9. Từ điển trường: decision state công khai

### 9.1 Identity và grouping

| Trường | Ý nghĩa khi gán nhãn |
|---|---|
| `decision_id` | ID duy nhất của decision state. |
| `scenario_id` | Workflow chứa state này. |
| `template_id` | Template sinh workflow; dùng khi kiểm tra lỗi có hệ thống. |
| `instance_id` | Instance cụ thể của template. |
| `split` | Split dữ liệu như development/validation/test. Không ảnh hưởng validity. |
| `split_group_id` | Nhóm phải được giữ cùng phía khi chia split để tránh leakage. |
| `counterfactual_group_id` | Nhóm các biến thể phản thực; dùng để kiểm tra consistency. |
| `provenance_id` | Khóa nối tới hồ sơ nguồn ở phase audit. |
| `workflow_family` | Loại workflow, không phải đáp án cho next action. |
| `state_provenance` | Cách state được tạo, ví dụ initial hay từ một transition. |
| `schema_version` | Phiên bản schema. |
| `difficulty_tags` | Thuộc tính độ khó; không dùng như shortcut để chọn agent. |

### 9.2 Task và context

| Trường | Ý nghĩa khi gán nhãn |
|---|---|
| `task` | Mục tiêu cuối người dùng yêu cầu. |
| `task_context.available_resources` | Tài nguyên công khai mà workflow có thể sử dụng. |
| `task_context.task_constraints` | Ràng buộc công khai của task. |
| `use_only_observable_state` | Không được suy diễn từ hidden transition/gold. |
| `hidden_source_location_provided` | Cho biết vị trí source ẩn có được cung cấp cho router hay không; không tiết lộ source thật. |

### 9.3 `observed_state`

| Trường | Ý nghĩa khi gán nhãn |
|---|---|
| `facts` | Các sự kiện đã trở thành quan sát công khai. |
| `available_artifacts` | Artifact hiện có ở dạng công khai/legacy. |
| `artifact_inventory` | Danh sách artifact hiện có đã chuẩn hóa. |
| `last_event.action` | Action gần nhất đã được thử. |
| `last_event.status` | Kết quả thực thi/rejection của event gần nhất. |
| `last_event.effect` | Thay đổi công khai do event gần nhất tạo ra. |
| `failure_count` | Tổng failure đã quan sát theo state contract. |
| `routing_rejection_count` | Số action bị từ chối trước execution. |
| `actor_failure_count` | Số failure sau khi executor đã chạy. |
| `fingerprint` | Dấu vân tay semantic của state; state giống fingerprint hỗ trợ phát hiện loop nhưng không thay thế việc đọc nội dung. |
| `step_index` | Vị trí của state trong trajectory đã mô phỏng/ghi nhận. |
| `open_requirements` | Những dependency hoặc công việc còn mở. Đây là tín hiệu chính để ghép role. |
| `goal_status` | Trạng thái goal (`incomplete` hoặc hoàn tất theo schema). Không tự mình đủ để kết luận `STOP` nếu final artifact thiếu. |

### 9.4 Execution memory và recent events

| Trường | Ý nghĩa khi gán nhãn |
|---|---|
| `execution_memory.attempts_by_agent` | Lịch sử lần gọi theo agent, gồm status/effect/step; dùng để nhận diện retry và loop. |
| `execution_memory.rejected_actions` | Action bị harness từ chối trước execution. |
| `execution_memory.completed_contributions` | Đóng góp đã hoàn tất và có thể dùng ở bước sau. |
| `recent_events` | Chuỗi event gần đây theo thứ tự thời gian, giúp hiểu state đã thay đổi thế nào. |

Các trường trên chỉ chứa public effect và trạng thái thực thi. Reviewer không
được giả định orchestrator nhìn thấy raw `contribution`, `candidate_answer` hoặc
`prior_live_contributions`. Raw output được lưu trong Track-B trace và có thể
được cấp cho executor sau, nhưng không nằm trong router observation của
V3-beta.

### 9.5 Budget

| Trường | Ý nghĩa khi gán nhãn |
|---|---|
| `budget.calls_remaining` | Số actor call còn lại. Action phải để workflow còn khả thi. |
| `budget.reserved_terminal_stop` | Có bước `STOP` zero-cost riêng sau execution budget. |

### 9.6 Role cards và candidates

| Trường | Ý nghĩa khi gán nhãn |
|---|---|
| `agent_pool` | Toàn bộ role card mà router thấy ở state. |
| `agent_id` | ID action ổn định. |
| `role_name` | Tên role để con người đọc. |
| `role_goal` | Mục tiêu chức năng của role. |
| `use_when` | Điều kiện tích cực để gọi role. |
| `avoid_when` | Điều kiện không nên gọi; nếu đang đúng thường làm action invalid. |
| `supported_inputs` | Loại input role có thể nhận. |
| `expected_outputs` | Artifact/effect mà role phải tạo. |
| `candidate_actions` | Tất cả action cần gán nhãn, gồm `STOP`. |

Các field `cost_class`, `action_cost`, `semantic_family` và `distractor_only` có
thể có trong source agent registry nhưng không nhất thiết xuất hiện ở blind view.
Không suy đoán các field bị ẩn.

## 10. Trường chỉ xem ở workflow/source audit

Các trường sau chứa gold-bearing information và không được dùng trong phase A:

| Trường | Ý nghĩa audit |
|---|---|
| `valid_actions` | Oracle action set hiện tại cần được con người kiểm tra, không mặc định đúng. |
| `action_rationales` | Precondition thỏa/vi phạm do formal generator tạo. |
| `terminal_valid` | Kết quả formal cho `STOP`. |
| `ground_truth_method` | Phương pháp sinh oracle. |
| `human_review_status` | Trạng thái thủ tục review, không phải nhãn chất lượng. |

### 10.1 Scenario/workflow fields

| Trường | Ý nghĩa audit |
|---|---|
| `initial_facts`, `initial_artifacts` | State khởi đầu. |
| `fact_descriptions`, `artifact_descriptions` | Diễn giải public cho fact/artifact symbol. |
| `goal_facts`, `goal_artifacts` | Điều kiện terminal bắt buộc. |
| `transitions` | Contract chuyển state của từng action. |
| `max_steps`, `max_cost` | Budget toàn workflow. |
| `distractor_ids` | Agent nhiễu được thêm có chủ đích; không được lộ ở blind input. |
| `public_context` | Context được phép công khai cho router. |
| `max_stagnant_repeats` | Ngưỡng loop không tiến triển. |

Mỗi transition có:

- `transition_id`: định danh transition;
- `action`: agent thực thi;
- `requires_all`: tất cả fact bắt buộc;
- `requires_any`: ít nhất một fact bắt buộc;
- `forbids`: fact không được hiện diện;
- `requires_artifacts`: artifact đầu vào bắt buộc;
- `requires_last_statuses` / `requires_last_actions`: ràng buộc event trước;
- `add_facts` / `remove_facts`: thay đổi fact;
- `add_artifacts` / `remove_artifacts`: thay đổi artifact;
- `status`: execution status sau transition;
- `public_effect`: effect mà router sẽ quan sát;
- `cost_override`: cost của transition;
- `minimum_steps_remaining`, `minimum_cost_remaining`: điều kiện khả thi còn lại.

### 10.2 Provenance

| Trường | Ý nghĩa audit |
|---|---|
| `source_type`, `source_dataset`, `source_version`, `source_split` | Nguồn và phiên bản gốc. |
| `source_item_id`, `source_path`, `source_sha256` | Khả năng truy vết đúng record/file. |
| `license` | Điều kiện sử dụng/phát hành. |
| `derivation_script`, `derivation_version` | Mã và phiên bản biến đổi dữ liệu. |
| `transformation_steps` | Các bước chuyển từ nguồn sang benchmark. |
| `validation_method` | Cách kiểm tra nguồn/gold. |

### 10.3 Actor input và evaluation

- `task_input`: dữ liệu private được cấp cho executor, không được lộ gold hoặc
  annotation-only metadata cho router.
- `evaluation.metric`: cách chấm end-task output.
- `evaluation.answer`: đáp án gold.
- `evaluation.aliases`: các đáp án tương đương được chấp nhận.
- `evaluation.supporting_ids`: evidence/support gold nếu source có cung cấp.
- `evaluation.source_item_id`: record nguồn dùng để đối chiếu.

## 11. Issue tags

| Tag | Khi sử dụng |
|---|---|
| `insufficient_public_state` | Public state thiếu thông tin cần để phân biệt validity. |
| `missing_valid_action` | Có action hợp lệ theo state/contract nhưng oracle không chứa. |
| `overbroad_valid_set` | Oracle chứa action không thể biện minh. |
| `ambiguous_role_card` | Role boundary, input hoặc output contract mơ hồ. |
| `hidden_information_required` | Nhãn chỉ biện minh được bằng thông tin router không thấy. |
| `budget_inconsistency` | Budget/cost làm đường đi không khả thi hoặc hiển thị mâu thuẫn. |
| `unjustified_stop` | `STOP` được coi hợp lệ khi terminal contract chưa đủ. |
| `semantic_loop_possible` | Action có thể lặp state không đổi hoặc retry không có thay đổi thực chất. |
| `transition_precondition_error` | Requires/forbids/status condition sai. |
| `transition_effect_error` | Fact/artifact/effect sau transition sai hoặc không nhất quán. |
| `gold_leakage` | Router input lộ answer, supporting label, preferred action hoặc transition gold. |
| `source_mismatch` | Task/corpus/item không khớp record nguồn. |
| `answer_dispute` | Evaluation answer hoặc alias có tranh chấp. |
| `license_or_provenance_issue` | Thiếu/sai license, checksum, source path hoặc derivation trail. |

Notes phải chỉ rõ fact, role contract, transition hoặc source record liên quan;
không chỉ ghi “không hợp lý”.

## 12. Workflow audit checklist

- `routing_labels_complete`: mọi state/candidate đã có nhãn.
- `public_state_sufficient`: public state đủ cho các quyết định routing.
- `invalid_actions_indefensible`: mọi action bị đánh invalid thực sự không thể
  biện minh từ contract.
- `role_overlap_genuine`: các role đồng hợp lệ thật sự có output contract tương
  đương ở state đó.
- `gold_leakage_absent`: input router/actor không lộ annotation-only gold.
- `transitions_coherent`: precondition, effect, status và memory nhất quán.
- `workflow_reachable`: tồn tại đường đi từ initial state tới goal trong budget.
- `terminal_contract_correct`: final artifact và `STOP` tuân thủ quy tắc terminal.
- `provenance_verified`: record truy vết được và license phù hợp.
- `evaluation_gold_verified`: answer/aliases/supporting IDs đúng source.

Workflow verdict:

- `pass`: không phát hiện lỗi;
- `pass_with_note`: đúng nhưng có giới hạn cần công bố;
- `needs_adjudication`: cần quyết định của reviewer thứ ba;
- `needs_fix`: giữ workflow nhưng phải sửa specification/generator/source;
- `exclude`: không thể sửa đáng tin cậy cho release hiện tại.

## 13. Quy tắc theo nhóm nguồn

### Controlled workflows

Ưu tiên kiểm tra semantics của transition, recovery, branch, conflict, loop và
budget. Không kết luận chỉ từ tên fact; phải xem fact thay đổi hành vi nào.

### MuSiQue-derived workflows

Kiểm tra question/corpus cùng source item; dependency giữa các hop có thể suy ra
từ supplied corpus; supporting IDs đúng paragraph; answer/aliases đúng; official
support labels không lọt vào router input.

### MMLU-Pro-derived workflows

Kiểm tra question, options và answer cùng source record; domain role hợp lý nhưng
category không được lộ như shortcut; split/group manifest nhất quán.

## 14. Ví dụ chuẩn

### Ví dụ A: chưa có plan

State nói dependencies chưa được sắp thứ tự. Workflow Planner nhận task
specification và tạo dependency plan.

- Workflow Planner: `valid`.
- Retriever cần evidence target từ plan: `invalid` nếu target chưa có.
- Composer: `invalid` vì evidence branches chưa đủ.
- `STOP`: `invalid` vì goal chưa hoàn tất.

### Ví dụ B: search thất bại, query đã đổi

Web Researcher đã fail với query cũ; state công khai có `revised_query_ready` và
query mới khác thực chất.

- gọi lại Web Researcher: có thể `valid`;
- gọi lại với query y hệt, không có evidence/state mới: `invalid` và cân nhắc
  `semantic_loop_possible`.

### Ví dụ C: candidate đã đủ và cần commit

State có candidate, evidence support và `finalization_pending`.

- Finalizer/Editor có compatible contract: `valid`;
- Evidence Retriever: `invalid` nếu không còn evidence dependency;
- `STOP`: vẫn `invalid` cho tới khi final artifact được commit.

### Ví dụ D: goal đã hoàn tất

State có `final_committed`, final artifact tồn tại và không còn open requirement.

- `STOP`: `valid`;
- gọi thêm verifier/retriever/finalizer: thường `invalid`, trừ khi task contract
  công khai còn bước bắt buộc chưa hoàn tất.

### Ví dụ E: hai role overlap thật

Corpus Evidence Analyst và Broad Corpus Evidence Resolver đều nhận supplied text
cùng evidence target, cùng tạo grounded evidence packet và không vi phạm
`avoid_when`.

- cả hai: `valid`;
- không ép chọn một role preferred.

### Ví dụ F: General Problem Solver ở state MuSiQue ban đầu

State chỉ có original question và supplied corpus; chưa có dependency plan,
grounded evidence packet hoặc completed evidence branches.

- Workflow Planner: có thể `valid` vì initial planning còn thiếu.
- Corpus Evidence Analyst/Broad Corpus Evidence Resolver: có thể `valid` vì
  corpus và unresolved evidence target đã quan sát được.
- General Problem Solver (`agent_14`): `invalid` vì các evidence-backed input
  được ghi trong role card chưa tồn tại.

Không dùng hidden transition để đưa ra kết luận này. Nếu một oracle chỉ có thể
được biện minh bằng precondition ẩn mà role card không thể hiện, gắn
`ambiguous_role_card` hoặc `hidden_information_required` và chuyển sang audit.

## 15. Adjudication

Đưa decision vào adjudication nếu:

- hai reviewer có valid-action set khác nhau;
- có ít nhất một nhãn `uncertain`;
- có issue ảnh hưởng state, role card, transition, source hoặc evaluation gold.

Adjudicator xem lập luận của hai reviewer trước, sau đó mới xem current oracle và
formal rationales. Current oracle là đối tượng cần kiểm tra, không phải đáp án mặc
định.

Resolution:

- `accept_current_label`;
- `modify_valid_set`;
- `modify_public_state`;
- `modify_role_card`;
- `modify_transition`;
- `correct_evaluation_gold`;
- `exclude_workflow`.

Mọi resolution phải có rationale chỉ rõ bằng chứng. Nếu sửa state, role card,
transition hoặc evaluation, phải sửa nguồn/generator rồi build lại; không chỉnh
tay một generated decision riêng lẻ.

## 16. Những thực hành bị cấm

- xem `valid_actions`, formal rationale hoặc transition trước khi hoàn tất blind
  review của workflow;
- dùng output của Jev/Luna/Sol/Gemini để quyết định nhãn;
- đánh invalid một role chỉ vì executor của một run từng reasoning sai;
- chọn đúng một action khi nhiều contract đều hợp lệ;
- coi tên role, workflow family hoặc source dataset là shortcut thay cho state;
- suy đoán source scope không được công khai;
- sửa generated JSONL riêng lẻ để làm oracle khớp annotation;
- dùng `uncertain` mà không ghi lý do cụ thể.

## 17. Điều kiện hoàn tất gán nhãn

Một decision hoàn tất khi mọi candidate action có nhãn, `state_sufficient` đã
được quyết định, mọi `uncertain` có lý do và issue tags/notes đủ cụ thể.

Một workflow hoàn tất khi blind review của mọi state xong, source/workflow audit
xong và mọi bất đồng đã được adjudicate. Trạng thái release nên giữ phân biệt:

- `reviewed_agreed`;
- `reviewed_adjudicated`;
- `excluded`.

Không gộp các trạng thái này thành một nhãn “approved”, vì cần giữ audit trail
cho release và các lần build sau.
