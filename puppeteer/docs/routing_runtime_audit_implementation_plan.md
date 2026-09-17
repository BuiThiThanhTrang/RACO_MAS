# Implementation plan: kiểm chứng routing, hội thoại và tổng hợp đáp án

Trạng thái cập nhật 16/09/2026: **đã triển khai M0–M4 và kiểm chứng offline**; M5 mới hoàn tất chuẩn bị, chưa chạy dev bằng model thật vì thiếu cấu hình truy cập API.

- 83/83 unittest đạt; 7 trace smoke có 0 lỗi invariant.
- Đã khóa 20 ID smoke và 140 ID dev.
- Xem [hướng dẫn sử dụng](routing_runtime_audit_guide.md) và [báo cáo nghiệm thu](routing_runtime_audit_validation.md).
- Các tiêu chí yêu cầu dữ liệu live ở Definition of Done bên dưới vẫn chưa nghiệm thu. Chưa kết luận về giảm phạt cost.

Phần dưới giữ thiết kế và tiêu chí ban đầu để đối chiếu với báo cáo triển khai.

## 1. Mục tiêu và phạm vi

Trước khi train dài hoặc điều chỉnh reward, hệ thống phải trả lời được:

1. Path dừng vì chọn STOP, hết depth, lỗi thực thi hay không còn agent hợp lệ?
2. Hai path dùng cùng teammate có giữ hội thoại độc lập không?
3. Agent được đề xuất, được cấp chỗ, thực sự thực thi và được cập nhật gradient có khớp nhau không?
4. Xác suất dùng trong loss có đúng với cơ chế chọn một hoặc nhiều agent không?
5. Đáp án đúng bị mất ở bước suy luận tiếp theo, tổng hợp trong path hay bỏ phiếu giữa các path?

Phạm vi đánh giá đầu tiên là MMLU-Pro. Không kiểm tra dataset hoặc bộ chấm GSM-Hard. Với CW/SRDD, chỉ kiểm tra hồi quy lifecycle và tách trạng thái; không áp dụng majority vote trắc nghiệm cho tác vụ sinh nội dung.

Giữ pool, backbone, profile ban đầu, quyền tool, encoder, Skywork và trọng số reward cố định trong từng phép so sánh. Thí nghiệm giảm cost là giai đoạn tiếp theo, sau khi các kiểm tra bên dưới đạt yêu cầu. Không coi việc đạt các tiêu chí kỹ thuật là bằng chứng accuracy chắc chắn tăng.

## 2. Căn cứ từ code và log hiện tại

- `inference/policy/role_aware_reinforce.py`: `_choose()` lấy mẫu một action, thêm agent theo threshold rồi sắp xếp/cắt width; `forward()` ghi trajectory trước khi runtime hoàn tất cấp chỗ cho nhánh.
- `inference/reasoning/path.py`: STOP và giới hạn depth là điều kiện riêng; `split()` có thể cắt bớt agent theo số chỗ còn lại.
- `agent/register/register.py` và `agent/reasoning_agent.py`: registry dùng lại object teammate có hội thoại và cờ kích hoạt mutable giữa các path.
- `inference/reasoning/reasoning.py`: tổng hợp từng path, chấm reward từng path và update policy trước khi quyết định đáp án cuối; hòa phiếu hiện chọn phần tử cuối trong nhóm hòa.
- Nhóm log `logs/MMLU-Pro10-09-train` có 200 task/600 path; 546 path chạy 3 bước. Có 116 task có ít nhất một path đúng, nhưng 83 task có đầu ra cuối đúng. Đây là thống kê train, không phải kết quả dev/test và chưa xác định được run cũ chỉ gọi Domain Reasoner.

Các quan sát trên là lý do thiết kế kiểm tra. Chưa quy toàn bộ performance thấp cho một lỗi cụ thể.

## 3. Thứ tự triển khai và mốc nghiệm thu

| Mốc | Công việc | Điều kiện hoàn tất |
|---|---|---|
| M0 | Khóa cấu hình và tạo fixture | Có manifest, dữ liệu giả và mốc đối chiếu tái lập được |
| M1 | Ghi trace và lý do STOP | Mọi path kết thúc có đúng một lý do; bật log không đổi hành vi |
| M2 | Tách trạng thái theo path | Test xen kẽ cùng teammate và test fork không lẫn hội thoại |
| M3a | Đồng bộ chọn/cấp chỗ/thực thi/trajectory | Không có action ma hoặc reward gán nhầm nhánh |
| M3b | Làm rõ xác suất và loss | Cơ chế mới có phân phối xác định, test xác suất và gradient đạt |
| M4 | Phân tích và replay tổng hợp | Có bảng hòa phiếu, mất đáp án đúng và so sánh trên cùng candidate |
| M5 | Kiểm chứng trên dev | Báo cáo paired theo task, đóng băng phương án, cho phép pilot reward |

M1 chỉ thêm khả năng quan sát, giữ thuật toán cũ để có mốc đối chiếu. M2/M3/M4 là các thay đổi hành vi riêng, mỗi phần có diff và test độc lập. Không gộp tất cả thành một lần chạy rồi quy cải thiện cho một thành phần.

## 4. M0 — Manifest và dữ liệu kiểm tra

### Triển khai

- Đặt trace mới dưới `runs/<run_id>/audit/`; mỗi task có thư mục theo ID ổn định. Kết quả task chứa `run_id`, `task_id`, `attempt_id`, `trace_path`, split và offset để tìm ngược log.
- Manifest lưu: resolved config, seed router, seed model nếu provider hỗ trợ, checkpoint hash, pool/profile/split hashes, phiên bản routing/runtime/aggregation và mã nguồn liên quan.
- Git commit không đủ khi có sửa chưa commit: lưu hash nội dung các file liên quan và trạng thái dirty. Không thu thập khóa API hoặc giá trị biến môi trường bí mật.
- Threshold đang hard-code cần được ghi rõ trong manifest. Khi đưa thành config, giá trị mặc định phải tái tạo đúng code hiện tại; thay hệ số là một thay đổi thí nghiệm riêng.
- Chọn cố định 20 câu dev theo danh sách ID lưu trước khi xem kết quả để smoke-test. Đánh giá chính dùng toàn bộ dev 140 câu trong split manifest. Seed train và seed chọn tập là hai trường riêng.
- Tạo model/policy giả trả output định trước; các test M1–M4 không gọi API.
- Đọc log cũ bằng adapter riêng. Trường thiếu phải là `unknown`, không suy diễn xác suất STOP hoặc checkpoint từ tên thư mục. Chỉ so sánh trực tiếp các run có đủ metadata tương thích.

### Nghiệm thu

- Từ một dòng kết quả tìm được config, checkpoint và toàn bộ trace task.
- Chạy lại fixture cùng seed cho cùng quyết định, output và reward.
- Ghi rõ provider có hoặc không bảo đảm tái lập; cùng seed không được quảng bá là bảo đảm output API giống hệt.

## 5. M1 — Trace quyết định và kết thúc path

### Tệp và điểm tích hợp

- Thêm `role_aware/audit_trace.py`: schema và writer JSONL theo event.
- Policy ghi phân phối tại `_distribution()`/`_choose()`.
- Runtime ghi cấp chỗ, thực thi và kết thúc tại `reasoning.py`/`path.py`.
- Model wrapper của agent ghi lời gọi và token, kể cả retry, sửa định dạng và aggregation.
- Runner truyền `run_id`, task context và trace sink; không tự dựng thư mục log bằng timestamp không liên kết run.

### Schema tối thiểu

Mỗi event có `schema_version`, `run_id`, `task_id`, `attempt_id`, `event_seq`, `event_type`, `path_uid`, `parent_path_uid`, `decision_id`, `action_id` khi áp dụng. `path_uid` không đổi nếu danh sách path đổi vị trí; số index chỉ phục vụ hiển thị.

| Event | Trường cần ghi |
|---|---|
| `routing_decision` | Số bước đã chạy; depth còn lại; width còn lại; mask; toàn bộ xác suất trước/sau mask; `allow_stop`; `p_stop`; entropy; threshold; action lấy mẫu; agent vượt threshold; danh sách cuối; chế độ chọn; xác suất chọn thực tế |
| `allocation` | Action đề xuất, action được cấp chỗ, nhánh nhận action, action bị loại và lý do |
| `action_started` / `action_finished` | Teammate ID, role, backbone dùng nội bộ; trạng thái thành công/lỗi; session ID; prompt digest; output; số lượt API và token |
| `path_finished` | Lý do dừng; số bước thực thi; decision gây dừng nếu có; candidate trước/sau tổng hợp |
| `reward_assigned` / `policy_update` | Reward từng thành phần, leaf/path liên quan, decision/action được credit, giá trị return, estimator/version và optimizer có thực sự cập nhật không |
| `aggregation` | Candidate đầu vào, chuẩn hóa, số phiếu, hòa phiếu, cách phá hòa, đáp án cuối, call ID và cost nếu gọi model |

Quy ước dừng: `policy_stop`, `depth_limit`, `agent_terminated`, `no_valid_agent`, `execution_error`, `cancelled`. Retry cạn phải có error event; không biến thành STOP hợp lệ. Hết width chỉ chặn mở nhánh mới, không tự động kết thúc path hiện tại.

Nếu đạt depth trước khi gọi router, `p_stop = null` và `stop_probability_observed = false`; không gọi thêm encoder/router chỉ để có số đo. Nếu STOP bị mask, ghi xác suất thực thi bằng 0 và lưu phân phối trước mask riêng.

Trace dùng tensor đã detach để serialize, nhưng tensor `log_prob` dùng train phải còn computation graph. Writer không lấy mẫu RNG, không gọi model, không sửa thứ tự thực thi. ID dùng bộ đếm/nguồn độc lập với RNG router.

Thông tin đáp án chuẩn và correctness chỉ ghi vào artifact đánh giá sau khi đã chốt prediction. Router, prompt và aggregator không đọc artifact này.

### Test bắt buộc

1. Depth 2, policy chọn agent rồi STOP: đúng 1 activation, lý do `policy_stop`.
2. Depth 2, policy tiếp tục: đúng 2 activation, lý do `depth_limit`, không hỏi STOP thêm.
3. Không agent vượt threshold: vẫn ghi action fallback của legacy; không nhầm thành STOP.
4. STOP bị mask ở đầu task; mask rỗng, lỗi API, cancel có kết quả phân loại riêng.
5. Bật/tắt audit trên cùng fixture: quyết định, output, reward, gradient và số API call không đổi.

## 6. M2 — Hội thoại và tài nguyên thuộc từng path

### Thiết kế

Giữ teammate spec, RoleCard, backbone và client ở registry. Di chuyển trạng thái hội thoại vào `AgentSession`, thuộc `PathRuntimeContext`:

`(run_id, task_id, attempt_id, path_uid, teammate_id) -> AgentSession`

Session giữ `dialog_history`, `initial_dialog_history`, `system_prompt`, `last_prompt`, cờ kích hoạt và đường dẫn làm việc. Hàm activate/query/deactivate nhận session hiện hành, không đọc/ghi lịch sử mutable chung trên object registry. Query client có thể dùng chung nếu không giữ message history.

- Cùng teammate chạy lại trong một path: giữ lịch sử của teammate trong path đó.
- Đổi sang teammate khác: prompt nhận workflow được phép của cùng path.
- Khi fork: snapshot session và workflow tới thời điểm fork; copy dữ liệu mutable; hai con nhận phần lịch sử chung rồi phát triển độc lập.
- Root độc lập: session rỗng riêng. Task mới: không mang session task trước sang.
- Chỉ activate sau khi allocation được chấp nhận và ngay trước execution; không pre-activate toàn bộ danh sách ứng viên.
- Tool environment, file đầu ra và code/text artifact phải thuộc path. Với tool state không thể clone, báo lỗi/không hỗ trợ fork rõ ràng thay vì dùng chung âm thầm.
- Không deep-copy API client, logger hoặc tensor autograd. Prefix action có ID bất biến; state kế tiếp có session mới.

### Test chứng minh độc lập

Tạo hai nhánh dùng cùng teammate: A nhận marker `ONLY_A`, B nhận `ONLY_B`. Model giả thu toàn bộ messages thực sự gửi vào query.

- Chạy xen kẽ A1 → B1 → A2 → B2: request của A không chứa `ONLY_B`, và ngược lại.
- Chạy B trước A: output, prompt và số call của từng nhánh vẫn giống khi dùng seed/model giả riêng theo nhánh.
- Fork sau marker `SHARED`: hai con đều thấy SHARED; chỉ nhánh nhận marker mới được thấy marker đó.
- Sửa sâu một message/list/file ở A không thay đổi B.
- Reset task, lỗi giữa chừng, retry và aggregation không đưa session từ nhánh khác vào request.

Tiêu chí: 0 trường hợp lẫn marker, không chỉ kiểm tra hai object có địa chỉ khác nhau. Việc kiểm tra thứ tự thực thi dùng lịch cấp chỗ cố định; không đòi output live API tuyệt đối giống nhau.

## 7. M3a — Một nguồn sự thật cho action và trajectory

### Luồng đề xuất

`DecisionProposal -> CapacityReservation -> ExecutionReceipt -> Reward/Credit`

1. Runtime xác định số chỗ khả dụng tại đúng thời điểm quyết định.
2. Policy đề xuất hành động kèm distribution và provenance; chưa tự tạo path index trong danh sách trajectory.
3. Runtime cấp path UID và reservation, trả allocation thực tế.
4. Chỉ action được chấp nhận mới có transition thực thi. Action bị từ chối vẫn có audit event, không bị tính step/token hoặc nhận log-prob như thể đã chạy.
5. Kết quả thực thi nối bằng `decision_id`/`action_id`, không bằng vị trí trong list. STOP là decision hợp lệ không gọi agent, không cộng một agent step.
6. Action đã được chấp nhận nhưng thất bại vẫn có receipt và outcome theo protocol; không loại bỏ khỏi tập train chỉ vì thất bại.
7. Prefix khi fork dùng reference tới action đã chạy; không nhân đôi API call hoặc chi phí thực thi thật. Reward theo leaf có thể tham chiếu prefix nhưng phải phân biệt rõ với cost task đếm action/call duy nhất.

### Invariant và test

- Mỗi action được chấp nhận có đúng một receipt thành công/lỗi, hoặc một trạng thái interrupted chưa hoàn tất.
- Mỗi activation thực tế có một action ID được cấp chỗ; không xuất hiện action ma trong update.
- Width toàn hệ thống không vượt giới hạn; path depth tính số activation đã thực thi, không tính STOP.
- Test khi width đã đầy, chỉ còn một chỗ, nhiều path cùng muốn fork và index được sắp lại.
- Decision chung của prefix không bị ghi gradient lặp chỉ vì có nhiều leaf. Credit phải theo estimator được khai báo ở M3b.
- Task interrupted không update bằng trace dở dang. Resume tái chạy cả task với attempt ID mới; deduplicate kết quả theo task/attempt được commit, giữ log attempt cũ để truy vết.

## 8. M3b — Phân phối chọn nhiều agent và loss

### Hai mode tách biệt

**`legacy_threshold`:** giữ nguyên cơ chế hiện tại cho replay/đối chiếu; log agent bị cắt sau sampling và đánh dấu estimator là surrogate legacy. Không gọi tích các xác suất đơn lẻ là xác suất đúng của tập top-k/threshold.

**`categorical_set_v2`:** cơ chế đề xuất để train có xác suất rõ ràng:

1. Runtime cấp trước ngân sách action cho decision. Số action mục tiêu K là config cố định, cắt theo số teammate khả dụng và capacity trước khi sampling; chưa học số nhánh.
2. Ở root, STOP bị mask. Ở decision tiếp theo, lần lấy mẫu đầu có thể là STOP; nếu STOP thì kết thúc riêng path đó và không lấy thêm agent.
3. Nếu tiếp tục, lấy đủ K agent theo thứ tự, không hoàn lại. Sau mỗi lần chọn, mask agent đó, mask STOP trong các lượt bổ sung và chuẩn hóa lại.
4. Không sắp xếp/cắt bỏ action sau sampling. Các agent được cấp path theo thứ tự đã lấy mẫu.
5. K = 1 là đối chứng để kiểm tra REINFORCE chọn một action trước khi bật K > 1.

Với danh sách có thứ tự A = (a1, ..., aK):

`log q(A | s, capacity) = sum_j log p(aj | s, mask_j)`

Trong đó lần đầu có thể chứa STOP theo luật trên. Dùng log conditional probability thật tại từng lượt, không cộng log của phân phối ban đầu chưa mask. Mỗi joint decision giữ tensor log-prob một lần; bản ghi JSON chỉ chứa giá trị detached.

### Credit cho decision mở nhiều nhánh
Không gán tùy ý joint log-prob cho riêng reward của một nhánh. Để có bản triển khai đầu tiên kiểm chứng được, dùng score-function estimator theo toàn task:

- Giữ reward task, Skywork, step/token và gamma của từng leaf theo config đã khóa.
- Định nghĩa rõ `G_task = sum_leaf discounted_return(leaf)`. Đây là objective tổng chất lượng có phạt của các path; chưa phải accuracy của đáp án cuối hoặc chi phí thực thi task đếm duy nhất.
- Dùng `loss_policy = -sum_decision log_q(decision) * stop_gradient(G_task)`; mỗi decision thực thi xuất hiện một lần, kể cả STOP.
- Log entropy regularization riêng; chưa thêm baseline học được, clipping hoặc chuẩn hóa return trong bản đầu để tránh gộp nhiều thay đổi. Ghi gradient norm và variance của return để đánh giá ổn định.
- Đây là estimator mới với objective được công bố rõ; kết quả phải ghi version riêng, không mô tả là chỉ thay threshold. Full-task return có thể có variance cao: kiểm tra bằng fixture trước, pilot train sau. Việc đổi sang mean leaf reward, final-answer reward hoặc credit theo subtree là thí nghiệm tiếp theo.

### Test xác suất và gradient

- Pool giả 3 agent + STOP: liệt kê mọi ordered tuple hợp lệ, tổng xác suất bằng 1 trong sai số số học.
- Conditional log-prob khớp tích xác suất tính tay; unavailable agent và agent lặp có xác suất 0.
- Không có agent đã sample bị rơi khỏi execution vì capacity; scheduler cấp chỗ tuần tự trước sampling.
- Với reward giả xác định trước, so sánh gradient kỳ vọng bằng cách liệt kê toàn bộ outcomes với gradient của kỳ vọng reward tính trực tiếp. Thử K = 1, K = 2, STOP và cây có prefix chung.
- Thử reward của nhánh anh em thay đổi: joint decision nhận đúng credit toàn task; prefix không bị nhân gradient do copy trajectory.
- Checkpoint ghi `routing_mode`, `routing_version`, `objective_version`, `estimator_version`. Không resume optimizer âm thầm giữa legacy và v2; nạp trọng số cũ để chẩn đoán phải là chế độ explicit. So sánh train mới dùng khởi tạo chung đã lưu.

## 9. M4 — Tổng hợp đáp án và xác định nơi mất đáp án đúng

### Tách ba mốc output

Lưu riêng đáp án thô sau mỗi step, candidate cuối trước tổng hợp trong path, candidate sau tổng hợp trong path và đáp án cuối giữa các path. Mỗi bước model-assisted aggregation có call/token riêng; không bỏ token trả về như nhánh hiện tại.

Aggregator chỉ nhận câu hỏi, lựa chọn và candidate/provenance được phép. Không nhận đáp án chuẩn, correctness, terminal reward, hoặc profile vừa cập nhật từ nhãn của task hiện tại. Prediction và aggregation hoàn tất trước khi chấm/ghi evidence; train cũng dùng thứ tự này.

### Chỉ số và mẫu số

- `final_accuracy`: số task đúng / tổng task hoàn tất theo protocol.
- `any_path_correct`: số task có ít nhất một candidate path đúng / tổng task; là mức bao phủ candidate, không phải accuracy triển khai được.
- `lost_correct_rate`: số task có candidate đúng nhưng final sai / số task có candidate đúng. Báo cả count và tỷ lệ trên toàn tập.
- `tie_rate`: số task có nhiều đáp án hợp lệ cùng số phiếu cao nhất / số task có ít nhất một phiếu hợp lệ.
- `invalid_answer_rate`: số candidate không parse được / tổng candidate; task toàn invalid ghi riêng.
- Đúng→sai và sai→đúng theo các step liên tiếp, trước/sau tổng hợp trong path và trước/sau tổng hợp toàn task. Thiếu đáp án là trạng thái riêng, không âm thầm bỏ khỏi thống kê.
- Phân tầng theo số path, depth, role/backbone, STOP và chi phí; đây là mô tả, không tự diễn giải thành quan hệ nhân quả.

### Replay trên cùng candidate dev

Thêm script đọc candidate đã lưu, không chạy lại reasoning, so sánh:

| Phương án | Quy tắc |
|---|---|
| Legacy | Tái tạo đúng parser/bỏ phiếu/phá hòa cũ để đối chiếu |
| Majority chuẩn hóa | Chuẩn hóa chữ cái theo tập lựa chọn hợp lệ, loại phiếu invalid; phá hòa bằng RNG riêng có seed từ run/task, lấy trên danh sách đáp án đã sắp xếp để không phụ thuộc thứ tự path |
| Majority + verifier khi hòa | Chỉ khi hòa, verifier nhận câu hỏi và candidate, phải chọn trong các đáp án đang hòa; model/prompt cố định, không thấy nhãn; tính toàn bộ token bổ sung |

Không mặc định ưu tiên agent có router probability cao nhất: xác suất routing chưa phải độ tin cậy đáp án đã hiệu chuẩn. Baseline chọn candidate đầu có thể ghi thêm để chẩn đoán, không gọi là phương án tối ưu.

Verifier thấy chính xác cùng candidate snapshot giữa các phương án. Nếu muốn sinh đáp án mới ngoài candidate, đó là phương án khác và phải báo cáo riêng. Nếu verifier output lỗi/hết retry, fallback theo majority chuẩn hóa đã khai báo.

### Test

- F đúng, I sai, J sai: phát hiện tie và lost-correct của legacy; không ép aggregator luôn chọn F vì aggregator không được thấy nhãn.
- Upper/lowercase, output có giải thích, nhãn ngoài A–J, rỗng và toàn invalid.
- Đảo thứ tự candidate: majority chuẩn hóa cùng seed không đổi kết quả; verifier đo độ nhạy thứ tự riêng, không giả định bất biến.
- Thay nhãn gold trong evaluator: request aggregator và prediction vẫn giống, chỉ correctness thay đổi.
- Hai replay dùng cùng hash candidate; không vô tình trộn candidate khác checkpoint hoặc run.

## 10. M5 — Kiểm chứng dev và điều kiện cho phép train dài

### Trình tự chạy

1. Chạy test offline và smoke bằng model giả cho width/depth 1/1, 1/2, 3/2; gồm fork sau prefix, STOP, error, resume.
2. Chạy 20 câu dev cố định với audit; kiểm tra đủ trace, session, accounting và prediction trước scoring.
3. Đánh giá trên dev 140 câu, cùng pool/checkpoint/profile/tool/prompt. Router và profile đóng băng, kiểm tra hash trước/sau. Replay aggregation trên cùng output đã lưu.
4. Với routing/runtime sửa đổi, dùng cùng trọng số cũ chỉ để đo tác động inference trực tiếp; ghi rõ đây không phải phép so sánh các policy đã train tối ưu. Pilot train lại từ khởi tạo chung cho K = 1 và v2 K > 1 trước khi kết luận về learner.
5. Dùng seed 42 cho smoke; các ứng viên cuối chạy thêm seed 43 và 44 theo cùng protocol. Tách seed routing/eval với split seed; không tạo lại split khi đổi seed train.
6. Báo chênh lệch theo cặp trên cùng task, count đúng→sai/sai→đúng, độ biến thiên theo seed và khoảng tin cậy paired bootstrap theo task. Với nhiều seed, bootstrap theo cụm task; không coi các lần lặp cùng task là mẫu độc lập.
7. Chọn aggregation trên dev bằng accuracy, kèm cost; nếu kết quả chưa phân biệt rõ, giữ phương án đơn giản/ít chi phí hơn theo quy tắc chọn lưu trước. Dev dùng để chọn, không gọi điểm dev đã tối ưu là ước lượng cuối không thiên lệch.
8. Đóng băng protocol rồi mới final. Nếu final hiện tại đã được dùng để chọn setting, ghi rõ giới hạn đó và dành tập holdout chưa dùng cho quyết định tiếp theo; không đổi tên tập đã xem thành holdout mới.

### Kiểm tra riêng giả thuyết dừng quá sớm

Sau M2/M3, lấy các state sau bước đầu trên dev. So sánh hai continuation có kiểm soát từ cùng snapshot: cho STOP theo policy và ép thêm đúng một agent theo quy tắc cố định không dùng nhãn. Giữ output bước đầu cố định, cộng chi phí thực phát sinh. Ghi thay đổi đúng/sai và nhóm p(STOP).

Đây là can thiệp chẩn đoán: nếu ép tiếp tục không cải thiện đáp án, chỉ giảm phạt sẽ khó giải quyết vấn đề. Không dùng nhãn dev để quyết định STOP từng câu trong lần đánh giá chính.

### Definition of Done

- [ ] 100% path hoàn tất có đúng một lý do kết thúc; xác suất không quan sát được ghi null.
- [ ] 0 action ma, không gán reward nhầm UID; lỗi/interruption có accounting rõ.
- [ ] Test hội thoại xen kẽ/fork/reset không có marker từ nhánh khác.
- [ ] Xác suất multi-selection và gradient khớp fixture liệt kê outcomes.
- [ ] STOP không tính như một activation; mọi API call, aggregation và retry đều được tính một lần vào tổng task.
- [ ] Có report hòa phiếu, candidate coverage và mất đáp án đúng, phân biệt các tầng aggregation.
- [ ] Prediction hoàn tất trước scoring; dev không cập nhật policy/profile bằng nhãn.
- [ ] Có manifest và báo cáo so sánh dev tái lập được; các giới hạn thiếu log/API không deterministic được ghi rõ.

Chỉ sau mốc này mới chạy pilot các mức phạt 0,1/0,01/0 với routing, aggregation, profile và khởi tạo cố định. Không đặt tiêu chí nghiệm thu kiểu “accuracy phải tăng X%” trước khi có dữ liệu.

## 11. Danh sách tệp dự kiến

Tên module mới bên dưới là đề xuất, chưa tồn tại chỉ vì được liệt kê ở đây.

| Tệp | Thay đổi dự kiến |
|---|---|
| `role_aware/audit_trace.py` (mới) | Schema event, writer, manifest và kiểm tra liên kết ID |
| `agent/agent_session.py` (mới) | Session của teammate theo path, snapshot/fork |
| `agent/agent.py`, `agent/reasoning_agent.py` | Truyền session vào activate/query/deactivate; trace lời gọi |
| `agent/register/register.py` | Registry giữ spec/client; không sở hữu hội thoại dùng chung |
| `agent/agent_info/global_info.py`, `workflow.py` | Path UID, action/call reference, artifact riêng và accounting |
| `inference/reasoning/path.py`, `reasoning.py` | Reservation, execution receipt, stop reason, liên kết cây và thứ tự predict/evaluate |
| `inference/policy/role_aware_reinforce.py` | Trace phân phối, legacy mode, sampler v2, decision-level estimator |
| `role_aware/aggregation.py` (mới) | Candidate schema, majority, phá hòa và verifier interface |
| `tasks/runner.py`, `tasks/mmlu_pro.py` | Truyền context, liên kết result→trace, frozen eval |
| `config/runtime.py`, `role_aware/checkpointing.py` | Config audit/routing/aggregation, kiểm tra version resume |
| `scripts/analyze_routing_audit.py` (mới) | Validate trace, thống kê STOP, allocation, cost, chất lượng theo bước |
| `scripts/replay_aggregation.py` (mới) | Replay candidate dev, report paired và case inspection |
| `tests/test_routing_audit.py` (mới) | Event, STOP, accounting và bật/tắt instrumentation |
| `tests/test_path_sessions.py` (mới) | Cùng teammate, xen kẽ nhánh, fork, reset và artifact |
| `tests/test_routing_execution_alignment.py` (mới) | Capacity, UID, receipt và prefix credit |
| `tests/test_multiselect_policy.py` (mới) | Xác suất conditional, STOP, gradient liệt kê outcomes |
| `tests/test_aggregation_audit.py` (mới) | Tie/invalid/parser/replay, không đưa gold vào lựa chọn |

Chạy thêm test lifecycle, role-aware pipeline và trajectory reward hiện có bị tác động. Không gọi live API trong unit test. Các lệnh CLI cho audit/replay chỉ được đưa vào hướng dẫn chạy sau khi đã implement và xác minh help; tài liệu này chưa tuyên bố có sẵn CLI mới.

## 12. Sản phẩm bàn giao của giai đoạn triển khai

1. Trace mẫu cho dừng sớm, chạy đủ depth, fork và lỗi API.
2. Test report với các invariant nêu trên.
3. Báo cáo dev gồm bảng phương án, chi phí, STOP, chuyển đúng/sai và danh sách case mất đáp án đúng.
4. Cấu hình được đóng băng, hash checkpoint/profile/pool và estimator version.
5. Kết luận một trong ba hướng: sửa runtime tiếp; cải thiện routing/aggregation; hoặc đã đủ điều kiện thử giảm phạt cost.

Ưu tiên thực hiện đầu tiên: M0 + M1. Đây là mốc tạo bằng chứng để các thay đổi tiếp theo có thể được kiểm tra và quy nguyên nhân.
