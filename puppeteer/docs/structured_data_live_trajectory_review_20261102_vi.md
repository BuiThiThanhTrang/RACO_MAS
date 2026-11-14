# Review live trajectory — structured data processing (2026-11-02)

## 1. Phạm vi và cohort hợp lệ

Backbone executor là `qwen/qwen3.5-9b`, router là `typesafe/jev-1.13`, ngân sách tối đa 8 executor calls, guard tắt và toàn bộ 10 role cộng `STOP` đều hiện diện trong mọi decision state.

Cohort đánh giá hợp lệ gồm:

- 9 episode từ `runs/structured_data_10_jev_qwen35_9b_live_20261102_b`, loại episode cũ của `sdp-spreadsheetbench-250-20`;
- episode thay thế của `sdp-spreadsheetbench-250-20` từ `runs/structured_data_250_20_jev_qwen35_9b_live_20261102_c`.

Episode `250-20` cũ không được dùng vì evaluator phát hiện `public/validation_contract.json` thiếu discriminator `kind`. Packet đã được sửa thành `kind = deduplicate_sum`, cập nhật checksum manifest, preflight lại và chạy live trajectory mới từ đầu. Preflight mới đạt reference public, reference private 2/2, independent evaluator, negative control và forbidden-import rejection. Run thay thế giữ nguyên source trong suốt execution và execution validator đạt.

Run `..._b` cũng giữ nguyên source trong thời gian chạy. Sau đó source chỉ thay đổi ở hai file của packet `250-20`; validator lịch sử xác nhận 9 episode còn lại có trace hợp lệ.

## 2. Kết quả task-level

| Chỉ số | Kết quả |
|---|---:|
| Packet trong cohort | 10 |
| Infrastructure-affected episode | 0 |
| Có candidate được chấp nhận | 5/10 |
| Có public execution thành công | 3/10 |
| Public validation pass và router chọn `STOP` | 2/10 |
| Trusted end-task hard pass | 1/10 |
| Trusted evaluator case pass | 3/29 |

Hai packet đạt terminal public là:

- `250-20`: trusted evaluator đạt 2/2 case, hard pass;
- `227-40`: public case đạt nhưng trusted evaluator chỉ đạt case 1, fail case 2 và 3; soft score `1/3`.

Ba packet có candidate nhưng không đạt terminal:

- `41-47` và `84-40`: candidate được ghi nhận nhưng public execution thất bại;
- `585-41`: public execution thành công, public validation fail, router đã chọn repair nhưng hết ngân sách trước khi có bản được kiểm thử và xác nhận lại.

Năm packet còn lại không tạo được candidate hợp lệ trước khi hết ngân sách: `120-24`, `398-14`, `47827`, `493-5`, `50631`.

## 3. Tín hiệu routing và execution

Cohort có 84 router decisions, trong đó 76 là primary decisions và 8 quyết định cuối bị suppress bởi budget. Tám quyết định bị suppress vẫn được giữ trong raw trace nhưng phải loại khỏi mẫu routing chính.

Phân bố action trong toàn bộ raw decisions:

| Action | Số lần |
|---|---:|
| Table Reshaping Specialist | 40 |
| Workbook Test Executor | 15 |
| Workbook Surveyor | 11 |
| Formula Computation Specialist | 7 |
| Workbook Repair Specialist | 6 |
| Workbook Integrity Validator | 3 |
| STOP | 2 |

Bốn role không được chọn trong cohort này là `Data Quality Profiler`, `Formula Dependency Analyst`, `Cross-sheet Reconciler` và `Formatting & Presentation Specialist`. Đây không phải agent-pool masking: cả 84 state đều có cùng 11 candidate actions và guard đều tắt.

Trong 68 role executions được dispatch:

- 25 success;
- 39 `invalid_output`;
- 4 failed sau khi tool đã chạy.

Ba mươi chín `invalid_output` gồm 21 lượt không tạo được tool call đúng protocol và 18 lượt `write_solution` bị candidate policy từ chối. Vì vậy bottleneck chính của cohort hiện tại là executor tạo candidate, không phải thiếu thông tin evaluator hay lỗi hạ tầng.

Các trajectory vẫn tạo ra bốn dạng phối hợp khác nhau:

1. survey → author lặp lại nhưng không có candidate;
2. survey → author → test, public execution fail → repair;
3. survey → author → test → validation fail → repair;
4. survey → author → test → validation pass → `STOP`.

Run `250-20` là positive trajectory hoàn chỉnh. Run `227-40` cho thấy public success không đồng nghĩa với khả năng tái sử dụng trên case chưa thấy, nên private evaluator thực sự thêm tín hiệu generalization.

## 4. Ý nghĩa đánh giá của domain

### Phần đã có giá trị

- Environment có public/private separation, source freeze, immutable input và independent final evaluator hoạt động thực tế.
- Outcome không bị ceiling: có cả success, public-only success, execution failure, validation failure và failure trước candidate.
- Trace đủ để đánh giá router về thời điểm survey, author, test, repair, validate và stop; hậu quả của action được giữ lại thay vì rollback che mất.
- Có thể thay decision model trong khi giữ nguyên packet, executor, budget và evaluator để so sánh outcome và routing efficiency.

### Phần chưa đủ để kết luận performance của decision model

- Hard pass `1/10` và candidate acceptance `5/10` tạo floor effect mạnh. Task success hiện bị executor contract adherence chi phối lớn.
- Chưa có human valid-action labels, nên `valid_action_accuracy` vẫn là `null`. Không thể gọi 76 primary decisions là đúng/sai chỉ từ task outcome.
- Mới có ba archetype: 4 aggregation, 4 cleaning và 2 formatting. Chưa có các packet `formula_dependency` và `reshape_join`, do đó các workflow cần formula analysis, reconciliation và cross-sheet coordination chưa được đại diện.
- Chỉ 6/10 executor roles được chọn. Cohort chưa đủ để đo toàn bộ khả năng phối hợp role của agent pool.

## 5. Kết luận readiness

Domain structured data hiện **đủ điều kiện để tiếp tục thu thập và human-review trajectory**, và environment đã cho tín hiệu task-level đáng tin cậy trên cohort hợp lệ.

Tuy nhiên, 10 packet hiện tại **chưa đủ để dùng task success làm kết luận độc lập về chất lượng router**. Kết quả phù hợp nhất để dùng ở giai đoạn này là:

- đánh giá qualitative/human labeling cho 76 primary decisions;
- đo progress milestones và routing efficiency với executor cố định;
- dùng hard pass và trusted-case coverage như outcome phụ có kiểm soát;
- bổ sung 10 packet còn lại, đặc biệt `formula_dependency` và `reshape_join`, trước khi so sánh decision models ở cấp domain.

Không nên sửa packet hoặc role card nhằm tối ưu Jev hay Qwen cho cohort này. Vấn đề cần tách riêng là độ tin cậy của executor protocol/candidate contract; nếu thay đổi phần đó, phải tạo một environment version mới và chạy lại toàn bộ router variants trên cùng version.

## 6. Cohort mười packet cuối và audit đủ 20 packet

### 6.1. Run và xử lý lỗi hạ tầng

Mười packet cuối được chạy trong
`runs/structured_data_final10_jev_qwen35_9b_live_20261102_b` với cùng Jev, Qwen3.5-9B, 16K output
cap, 8 executor calls, full pool + `STOP` và guard tắt.

Attempt `structured_data_final10_jev_qwen35_9b_live_20261102_a` bị dừng khi router transport chưa kết
nối được mạng và chưa tạo episode hoàn chỉnh; attempt này không nằm trong cohort hay mẫu đánh giá.

Hai episode `37456` và `45738` ban đầu bị `tool_environment/TypeError` trước khi executor model được gọi.
Nguyên nhân là workbook chứa `ArrayFormula`; workbook snapshot giữ object OpenPyXL thay vì chuyển nó
thành formula text trước khi serialize executor payload. Adapter đã được sửa để serialize cả cell value
và formula list thành text. State-contract audit sau sửa xác nhận executor context JSON-serializable cho
20/20 packet.

Hai packet lỗi hạ tầng được chạy lại từ đầu trong
`runs/structured_data_formula2_jev_qwen35_9b_live_20261102_c`. Cohort pilot hợp lệ cho mười packet cuối
gồm tám episode không bị ảnh hưởng từ run chính và hai episode rerun. Episode lỗi hạ tầng cũ bị loại,
không được tính như task failure. Cả hai run đều xác nhận source không đổi trong thời gian riêng của run;
run chính được giữ như historical freeze sau khi adapter được sửa.

### 6.2. Kết quả mười packet cuối

| Chỉ số | Kết quả |
|---|---:|
| Packet hợp lệ | 10 |
| Infrastructure-affected sau thay thế | 0 |
| Raw router decisions | 90 |
| Primary routing decisions | 80 |
| Budget-suppressed final decisions | 10 |
| Role executions | 72 |
| Accepted candidate | 6/10 |
| Có public output thành công | 3/10 |
| Có public validation artifact | 1/10 |
| Public validation pass | 0/10 |
| Trusted end-task hard pass | 0/10 |
| Trusted evaluator case pass | 0/30 |

Phân bố 72 role executions gồm 33 `success`, 29 `invalid_output` và 10 `failed`. Trong 29
`invalid_output`, 20 lượt không tạo được tool call đúng protocol và chín lượt tới tool nhưng candidate
source bị từ chối. Tám trong mười `failed` là candidate execution failure; hai lượt còn lại là semantic
validation failure.

Các trajectory tạo ra nhiều cấu trúc phối hợp quan sát được:

1. survey → author lặp lại nhưng không có accepted candidate;
2. survey → author → test precondition reject vì chưa có candidate;
3. survey → author → test failure → nhiều repair version → test lại;
4. survey → cross-sheet author → test success → validation fail → repair → test → validation fail;
5. survey → formula author → public execution success nhưng hết ngân sách trước validation;
6. survey → profile → quay lại formula authoring.

Jev dùng tám trong mười executor roles ở cohort mới. `Cross-sheet Reconciler` và `Data Quality Profiler`
đã xuất hiện, nhưng `Formula Dependency Analyst` và `Formatting & Presentation Specialist` vẫn không
được chọn. Đây không phải masking: mọi decision state có đủ mười role cộng `STOP`, không có valid-action
mask và guard tắt. Không episode nào chọn `STOP`; quyết định thứ chín của mỗi episode là non-STOP bị
suppress khi budget bằng 0.

### 6.3. Tổng hợp đủ 20 packet

Ghép cohort mười packet đầu đã review với cohort mười packet cuối cho kết quả:

| Chỉ số | Kết quả |
|---|---:|
| Packet | 20, đủ 5 selection archetypes |
| Accepted candidate | 11/20 |
| Public execution thành công | 6/20 |
| Public validation pass + router `STOP` | 2/20 |
| Trusted end-task hard pass | 1/20 |
| Trusted evaluator case pass | 3/59 |
| Raw router decisions | 174 |
| Primary routing decisions | 156 |
| Budget-suppressed decisions | 18 |

Domain hiện bao phủ aggregation, cleaning, formatting, formula/dependency và reshape/join. Trace có
inspection, authoring, cross-sheet work, testing, validation, repair, precondition rejection, premature
or missing stop và budget exhaustion. Do đó nó tạo đủ loại state transition để phân biệt router chọn
workflow, timing và recovery khác nhau.

Hard pass `1/20` cho thấy floor effect mạnh nếu chỉ dùng task success. Bottleneck chính vẫn là executor:
candidate acceptance/protocol và candidate runtime correctness. Vì mọi router variant sẽ dùng cùng
executor freeze, đây là một confound được kiểm soát trong so sánh tương đối, nhưng làm giảm statistical
power của hard-pass metric và có thể che khác biệt routing.

### 6.4. Quyết định admission

Structured data processing **đạt để đưa vào benchmark như domain thứ tư về routing/coordination**, vì:

- đủ 20 packet được chốt trước live, đủ năm archetype và có semantic oracle độc lập;
- public/private isolation, exact role-tool contract, state sufficiency, trace linkage và source freeze
  đã pass;
- trajectory có hậu quả action và recovery observable, không rollback che lỗi;
- full pool không masking tạo các route khác nhau có thể human-review và so giữa decision models.

Tuy nhiên không dùng hard pass làm metric duy nhất. Metric bundle cho domain này phải gồm:

1. trusted hard pass và trusted-case coverage;
2. milestone: accepted candidate, public execution, public validation và correct `STOP`;
3. human-labeled action appropriateness/recovery quality trên primary decisions;
4. routing efficiency: calls tới milestone, precondition rejection, author-loop không tiến triển và
   budget-suppressed final decision.

Trước run so sánh router chính thức cần freeze adapter đã sửa và chạy **tất cả router variants từ đầu**
trên cùng 20 packet, executor/config/evaluator. Không dùng cohort pilot ghép từ hai source hashes làm
leaderboard result. Human labels hiện vẫn pending; vì vậy admission này là environment/domain admission,
chưa phải một comparative router score hoàn chỉnh.
