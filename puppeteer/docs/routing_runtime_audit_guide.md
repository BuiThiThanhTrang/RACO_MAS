# Hướng dẫn audit routing trước khi train dài

Cập nhật: 16/09/2026. Phạm vi: **MMLU-Pro**.

## Trạng thái

Đã triển khai trace, session riêng theo path, đồng bộ cấp chỗ/execution/trajectory, sampler v2, replay tổng hợp và phép thử STOP từ snapshot. Kiểm thử và smoke offline dùng model giả.

**20 câu smoke / 140 câu dev bằng model thật chưa chạy**: môi trường thiếu cấu hình truy cập encoder và các model trong pool. Offline kiểm chứng phần mềm; chưa chứng minh accuracy tăng hoặc cần giảm phạt cost.

Xem [báo cáo nghiệm thu](routing_runtime_audit_validation.md) và [kế hoạch](routing_runtime_audit_implementation_plan.md).

## 1. Chạy kiểm tra offline

Các lệnh chạy từ D:\SideProject\ChatDev\puppeteer:

~~~powershell
$py = "../puppeteer_env/Scripts/python.exe"
& $py -m unittest discover -s tests -v
& $py scripts/smoke_audit_offline.py --output runs/audit_smoke_new
& $py scripts/analyze_routing_audit.py runs/audit_smoke_new --output runs/audit_smoke_new/analysis.json
& $py scripts/replay_aggregation.py runs/audit_smoke_new --output runs/audit_smoke_new/replay.json
~~~

Smoke yêu cầu thư mục mới. Điểm model giả không phải điểm benchmark. Kết quả mẫu đã lưu ở runs/audit_validation/.

## 2. Các chế độ routing

~~~yaml
policy:
  routing:
    mode: legacy_threshold
    threshold_multiplier: 1.5
    selection_count: 1
~~~

Legacy giữ selector cũ: lấy mẫu một action, thêm agent vượt threshold, sắp xếp và cắt theo width. Hệ số mặc định vẫn là 1.5 như workspace trước lần triển khai này. Loss mang nhãn legacy_surrogate_v1; log-prob từng agent không phải xác suất thật của toàn tập được chọn.

**Legacy chạy trên runtime đã sửa session/cấp chỗ**, nên không tái hiện đầy đủ runtime cũ còn lỗi. Log lịch sử thiếu metadata không phải đối chứng nhân quả.

~~~yaml
policy:
  routing:
    mode: categorical_set_v2
    selection_count: 2
~~~

V2 lấy mẫu có thứ tự, không hoàn lại. Lần đầu cho phép STOP nếu path đã chạy; STOP kết thúc lựa chọn. Sau khi chọn agent, lần lấy tiếp loại STOP và agent đã chọn rồi chuẩn hóa lại. K bị giới hạn bởi capacity và số agent khả dụng.

Xác suất quyết định là tích các xác suất có điều kiện. Objective sum_discounted_leaf_v1 cộng return chiết khấu của các leaf; estimator joint_episode_v2 dùng log-prob **mỗi quyết định một lần** và return toàn task. Prefix reward/cost vẫn xuất hiện trong mỗi leaf theo định nghĩa objective; API call vật lý của prefix chỉ tính một lần. Không cập nhật từ episode chưa hoàn thành.

Entropy là regularizer hiện có. Kiểm thử gradient chính xác kiểm tra thành phần REINFORCE với entropy coefficient bằng 0. Không tự thay đổi hệ số phạt step/token hiện có của bạn.

## 3. Đọc trace và chi phí

~~~text
runs/<run_id>/audit/
  manifest.json
  manifests/<manifest_hash>.json
  task-<hash_id>/attempt-0001/
    events.jsonl
    candidates.json
    evaluation.json
    path-0001/step_1.json
~~~

Manifest lưu config đã che khóa bí mật, hash mã nguồn/checkpoint/pool/profile/split. Bản theo hash giữ metadata của attempt cũ khi resume; manifest.json là bản mới nhất. Mỗi task/result liên kết run/task/attempt và trace; sự kiện liên kết manifest hash.

| Cần kiểm tra | Event/trường |
|---|---|
| STOP hay hết depth | path_finished: stop_reason, p_stop, steps_completed |
| Phân phối trước/sau mask | routing_decision: raw_probabilities, probabilities, mask |
| Threshold chọn/loại ai | threshold, sampled_first, threshold_candidates, selected |
| Agent nào chạy | allocation.accepted → action_started → action_finished; đối chiếu action_id/path_uid |
| Trajectory nhận reward | reward_assigned: action_ids, decision_ids; policy_update |
| Model gọi ở đâu | model_call_started/finished: call_id, purpose, tokens |

Mỗi path có session cho từng teammate và workspace riêng. Fork sao chép dữ liệu hội thoại/artifact; không sao chép client hoặc tensor autograd.

Lý do dừng gồm policy_stop, depth_limit, agent_terminated, no_valid_agent, execution_error, cancelled. p_stop=null nghĩa là chưa quan sát. Hết depth không gọi thêm router để đo STOP. STOP có receipt nhưng không có agent activation.

Candidate thô và hash nằm trong candidates.json; nhãn chỉ vào evaluation.json sau prediction. Snapshot bước đầu cũng loại nhãn.

### Accounting

- cost: tổng request tới thời điểm chốt prediction, gồm encoder, agent, retry, sửa format, path aggregation và verifier.
- cost_including_scoring: thêm HTTP reward scorer sau prediction.
- unknown_usage_calls: provider không trả usage hoặc request lỗi; không coi là 0 token đã xác nhận.
- model_cost: proxy 2 × model_size × tokens, **không phải tiền API**. Encoder/reward không có model size trong scope không đóng góp vào proxy.
- Reward giữ công thức workflow cost/normalizer cũ; không tự chuyển sang tổng cost audit.

OpenAI SDK đặt max_retries=0; wrapper chat/reward sở hữu retry để trace mỗi request. Encoder hiện không có vòng retry riêng. Giữ cơ chế này cố định giữa các run so sánh. Reward cache hit và reward chạy local không tính thành API request mới.

## 4. Chuẩn bị và chạy dev

~~~powershell
& $py scripts/prepare_audit_dev.py --config config/experiments/role_aware_mmlu_pro.yaml --output runs/audit_validation/dev_protocol
~~~

Không gọi model. Lệnh khóa 140 ID dev và 20 ID smoke đầu tiên theo split seed, tạo preset legacy/K1/K2 và kiểm tra sự hiện diện cấu hình truy cập. Preset mặc định initialized, frozen.

Sau khi kết nối model/encoder đã cấu hình, smoke với policy khởi tạo:

~~~powershell
& $py main.py MMLU-Pro dev --config runs/audit_validation/dev_protocol/mmlu_audit_legacy.yaml --policy_mode initialized --data_limit 20 --seed 42
~~~

Đánh giá checkpoint bằng --policy_mode evolved --checkpoint <checkpoint>. Giữ cùng checkpoint/profile/pool/tool giữa các phương án; evolved mặc định lấy profile từ checkpoint. Dùng profile ngoài thì truyền rõ --profile_source probe --profile_path <profile>.

Để chuyển trọng số checkpoint cũ sang runtime/routing mới nhằm chẩn đoán inference, thêm vào bản sao cấu hình:

~~~yaml
policy:
  routing:
    mode: categorical_set_v2
    selection_count: 2
    allow_policy_transfer: true
~~~

Cờ này cho phép chuyển trọng số có chủ đích; không cho resume optimizer khác phiên bản runtime/routing/objective/estimator. Kiểm tra chiều encoder và schema mạng vẫn bắt buộc.

Sau smoke hợp lệ, chạy --data_limit 140. Seed routing 42/43/44 độc lập với dataset.split_seed=42. Runner kiểm tra hash policy/profile trước và sau đánh giá. Không so policy mới khởi tạo với checkpoint đã train rồi quy chênh lệch cho routing.

## 5. Phân tích và replay aggregation

~~~powershell
& $py scripts/analyze_routing_audit.py runs/<run_id>/audit --output runs/<run_id>/audit_report.json
& $py scripts/replay_aggregation.py runs/<run_id>/audit --modes legacy majority --output runs/<run_id>/aggregation_replay.json
~~~

Report có STOP, tie, invalid, any-path-correct, lost-correct; tách chuyển đúng/sai ở bước suy luận và tổng hợp path, phân nhóm số path/depth. leaf_transition_counts có thể lặp prefix theo leaf; physical_agent_executions đếm action ID duy nhất.

Replay dùng cùng candidate hash và chốt mọi prediction trước khi đọc nhãn. Majority chuẩn hóa phá hòa bằng RNG riêng theo seed + task ID trên đáp án đã sắp xếp; không dùng xác suất router như độ tin cậy đáp án.

~~~powershell
& $py scripts/replay_aggregation.py runs/<run_id>/audit --modes legacy majority majority_verifier --verifier-model <model-name> --output runs/<run_id>/aggregation_verifier.json
~~~

Verifier gọi API khi hòa, phải chọn trong các đáp án hòa, có trace cost; output lỗi thì fallback deterministic. Replay mặc định chặn dữ liệu ngoài dev. --allow-non-dev-diagnostic chỉ để chẩn đoán, không biến test thành tập chọn phương án.

Paired bootstrap gom cụm theo task khi có nhiều seed. Chọn aggregation bằng dev, lưu quyết định trước khi đánh giá final.

## 6. STOP so với thêm một bước từ cùng prefix

~~~powershell
& $py scripts/check_stop_continuation.py runs/<run_id>/audit/task-<id>/attempt-0001/path-0001/step_1.json --teammate-id <teammate-co-dinh> --output runs/stop_probe_new
~~~

Lệnh gọi model thật: khôi phục cùng output/hội thoại bước đầu rồi thử dừng hoặc thêm đúng một teammate. Cả hai dùng cùng aggregation; nhãn chỉ nối sau khi có hai prediction. Report lưu p(STOP) của run nguồn và chi phí phát sinh từ prefix trở đi.

Chỉ định quy tắc teammate trước khi xem nhãn, không chọn nhánh tốt nhất từng câu bằng gold. Một snapshot chưa đủ kết luận cần giảm phạt; cần gom nhiều state dev/seed.

## 7. Bước tiếp theo

Chạy smoke live → kiểm tra audit → frozen dev/replay → continuation diagnostic. Chỉ sau các kiểm tra này mới pilot hệ số phạt cost với routing, aggregation, profile, khởi tạo và budget cố định. Không chạy GSM-Hard trong protocol này.

