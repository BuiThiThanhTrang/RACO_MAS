# RoutingBench V3-beta Human Review Protocol

## 1. Mục tiêu

Human review xác nhận benchmark đo khả năng chọn next agent từ task state một
cách khách quan. Review không tối ưu label cho Jev, Luna, Sol, Gemini hoặc bất
kỳ router/executor cụ thể nào.

## 2. Thành phần và phạm vi

- 116 workflow.
- 1.156 decision state.
- 64 controlled workflow.
- 24 MuSiQue workflow.
- 28 MMLU-Pro workflow.
- Hai annotation độc lập cho mọi decision và workflow.
- Một adjudication cho mọi bất đồng hoặc uncertainty.

Không đổi `pending` thành `reviewed` chỉ dựa trên formal validator.

## 3. Vai trò

### Annotator

Đánh giá candidate actions chỉ bằng thông tin public mà router quan sát được.
Không xem gold, transition table hoặc baseline outputs trước khi submit.

### Source auditor

Sau khi blind review hoàn tất, kiểm tra transitions, provenance, source item,
actor input, official answer và leakage.

### Adjudicator

Giải quyết bất đồng, xem được cả annotation độc lập và gold-bearing sidecars.
Không mặc định current dataset oracle là đúng.

## 4. Calibration

Trước khi review toàn bộ, hai annotator independently review 12 workflow:

- một workflow từ mỗi controlled family;
- một MuSiQue 2-hop, 3-hop và 4-hop;
- một MMLU-Pro.

Sau đó thống nhất cách áp dụng rubric và review lại pilot bằng protocol cuối.
Luôn đưa `v3b-musique-3hop-06` vào priority audit.

## 5. Pha A: gold-blind decision review

Annotator được xem:

- task và task context công khai;
- observed state;
- execution memory và recent events;
- budget;
- qualitative role cards;
- candidate actions.

Annotator không được xem:

- `valid_actions` và `action_rationales`;
- scenario transitions và distractor IDs;
- evaluation answer, aliases và supporting IDs;
- kết quả hoặc lỗi của các router baseline.

### Rubric cho từng candidate action

Gán `valid` khi tất cả điều kiện sau đúng:

1. State có một nhu cầu quan sát được mà agent xử lý được.
2. Supported inputs của agent đã tồn tại hoặc được cấp trong lời gọi kế tiếp.
3. Không có `avoid_when` nào bị vi phạm.
4. Expected output làm tiến triển một open requirement hoặc hoàn tất workflow.
5. Action khả thi trong call budget hiện tại.
6. Action không lặp lại failure mà không có state/evidence thay đổi thực chất.
7. Không cần hidden information để biện minh.

Gán `invalid` khi có lý do rõ ràng vi phạm rubric. Gán `uncertain` khi role card
hoặc public state không đủ rõ và phải ghi lý do.

Không chọn một action “tốt nhất”. Specialist và generalist đều có thể hợp lệ nếu
cả hai contract đều thỏa mãn. Cost chỉ làm action invalid khi khiến workflow
không còn khả thi; không tạo preferred action ngầm.

### STOP

`STOP` chỉ valid khi final artifact đã commit, observable goal hoàn tất và không
còn dependency/verification/finalization bắt buộc. Candidate answer “có vẻ đúng”
không đủ để STOP.

## 6. Pha B: workflow/source audit

### Mọi workflow

Kiểm tra:

- public state đủ để biện minh mọi valid action;
- không có invalid action vẫn defensible;
- overlapping roles thực sự interchangeable theo output contract;
- transition preconditions và effects nhất quán;
- workflow có đường đi tới goal trong budget;
- failure origin phân biệt pre-execution rejection và post-execution failure;
- terminal STOP là quyết định zero-cost ngoài actor-call budget;
- router/actor input không chứa gold hoặc source-only annotation.

### Controlled workflows

Kiểm tra loop, recovery, conflict, branch và budget bằng semantics của transition;
không chỉ kiểm tra tên fact.

### MuSiQue

Kiểm tra câu hỏi/corpus đúng source item, bridge facts suy ra được từ supplied
corpus, hop dependency hợp lý, supporting IDs đúng paragraph, answer/aliases
đúng và official supporting labels không lọt vào router input.

### MMLU-Pro

Kiểm tra question/options/answer khớp source, role domain hợp lý nhưng không dùng
category làm shortcut, và domain split đúng manifest.

## 7. Issue tags

Sử dụng vocabulary có sẵn trong dashboard. Không tạo tag đồng nghĩa mới nếu tag
hiện có mô tả được lỗi. Notes phải chỉ rõ fact, role contract hoặc source record
liên quan.

## 8. Workflow verdict

- `pass`: không phát hiện lỗi.
- `pass_with_note`: đúng nhưng có giới hạn cần công bố.
- `needs_adjudication`: cần người thứ ba quyết định.
- `needs_fix`: giữ workflow nhưng phải sửa generator/source specification.
- `exclude`: workflow không thể sửa đáng tin cậy cho release này.

## 9. Adjudication

Đưa state vào adjudication nếu valid-action set khác nhau, có `uncertain`, hoặc
bất kỳ reviewer nào flag lỗi quan trọng. Adjudicator xem hai annotation trước,
sau đó mới đối chiếu current oracle và formal rationales.

Resolution hợp lệ:

- `accept_current_label`;
- `modify_valid_set`;
- `modify_public_state`;
- `modify_role_card`;
- `modify_transition`;
- `correct_evaluation_gold`;
- `exclude_workflow`.

Mọi resolution phải có rationale. Nếu sửa state, role card hoặc transition, sửa
generator/source specification và build lại; không hand-edit riêng generated
`decision_states.jsonl`. Review lại toàn bộ workflow và group liên quan.

## 10. Agreement report

Báo cáo tối thiểu:

- exact valid-set agreement;
- macro Jaccard;
- số state có uncertainty;
- disagreement/adjudication count;
- số action được thêm/bỏ sau adjudication;
- agreement theo family, split và source.

Không dùng một ngưỡng agreement tùy ý để tự động tuyên bố release hợp lệ. Điều
kiện quan trọng là mọi state được double-review và mọi bất đồng được giải quyết.

## 11. Điều kiện hoàn tất

Human review hoàn tất khi:

1. Hai reviewer submit đủ 1.156 decision.
2. Hai reviewer audit đủ 116 workflow.
3. Mọi uncertainty và disagreement đã adjudicate.
4. Mọi source/gold của 52 real-task workflow được kiểm tra.
5. Mọi sửa đổi được build lại và impacted workflows được re-review.
6. Formal validation pass, oracle đạt 100% và không có observability collision.
7. Review JSONL export và agreement report được lưu cùng release artifact.

Trạng thái cuối nên phân biệt:

- `reviewed_agreed`;
- `reviewed_adjudicated`;
- `excluded`.

Không nên gom tất cả thành một trạng thái `approved` vì sẽ làm mất audit trail.
