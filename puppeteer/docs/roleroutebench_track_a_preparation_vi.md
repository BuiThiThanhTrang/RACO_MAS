# RoleRouteBench — cập nhật dataset và đầu vào thiết kế Track A

Ngày rà soát: 2026-11-05

## 1. Nguồn sự thật hiện hành

Tên công khai chuẩn của dataset là **RoleRouteBench**. `RoutingBench V4-beta`
chỉ còn là tên phát triển lịch sử và vẫn xuất hiện trong một số namespace, tên
thư mục, schema và runtime đã đóng băng.

- Hugging Face: `https://huggingface.co/datasets/TrangBui/RoleRouteBench`
- DOI: `10.57967/hf/11430`
- release hiện hành: `4.0.0-beta.2`
- latest publication revision và là nguồn làm việc cho Track A tiếp theo:
  `b70803c339a19c767fe9f0718c5218f51d4ed130`
- frozen Track B data revision ban đầu:
  `cdc67bad1458f6b939f759e3f764f2e93612f424`
- DOI snapshot revision:
  `bd936a4b61500e374837702ca08ac631950b2f96`
- GitHub tag: `routingbench-v4-beta.2`
- execution/code commit:
  `ee21943c3594c97f51f2082d0671dde44b7d2eb2`

`bd936a4` là snapshot gắn với DOI; `b70803c` là commit cập nhật sau DOI và là
authority hiện tại cho công việc tiếp theo. Hai revision có vai trò khác nhau,
không phải lỗi provenance. Khi báo cáo thí nghiệm phải ghi riêng publication
revision, frozen data revision và code/runtime revision. Không dùng tên repo cũ
`TrangBui/RoutingBench-V4` cho lệnh download hoặc publish mới.

### Publication hygiene cần chốt trước release kế tiếp

Local Hugging Face worktree đang ở commit mới nhất được chỉ định cho công việc
này: `b70803c`, với parent là DOI snapshot `bd936a4`. Mọi thiết kế và export
Track A tiếp theo phải lấy metadata từ `b70803c`; không quay lại metadata trước
DOI hoặc repo ID cũ.

Dataset card hiện quảng bá lệnh một bước
`python scripts/reproduce_roleroutebench.py`, nhưng repository được rà soát chưa
có file này. Wrapper dự kiến chỉ điều phối download, private-isolation sentinel
và release certificate; nó không sinh state, không gán nhãn và không chạy
evaluator Track A. Vì vậy file này **không phải điều kiện để tạo Track A**.
Trước release kế tiếp vẫn nên bổ sung wrapper đã test hoặc bỏ tuyên bố
one-command khỏi card để publication claim khớp artifact thực tế.

## 2. Trạng thái release bốn domain

Track B beta.2 là release live đầu tiên của bốn domain, gồm 82 packet và 82
canonical live trajectory trên split công khai `full`.

| Domain | Packet/trajectory | Role | Independent trusted pass | Primary decisions |
| --- | ---: | ---: | ---: | ---: |
| `database_migration_integrity` | 12 | 10 | 7 | 84 |
| `code_debugging` | 20 | 6 | 12 | 143 |
| `formal_proof` | 30 | 7 | 18 | 185 |
| `structured_data_processing` | 20 | 10 | 1 | 156 |
| **Tổng** | **82** | **33** | **38** | **568** |

Các số trên là kết quả tham chiếu của JEV 1.13, không phải oracle và không được
dùng làm nhãn đúng cho Track A. Mọi so sánh router phải giữ nguyên packet, role
card, adapter, executor, budget và evaluator của domain tương ứng.

Human review packet đã hoàn tất cho database, code debugging và formal proof.
Structured data processing vẫn là research preview với packet review pending.
Nhãn human review ở cấp decision state vẫn pending cho cả bốn domain.

## 3. Track A đang được publish là dữ liệu lịch sử

Track A trong beta.2 được giữ nguyên byte từ beta.1. Audit trực tiếp các parquet
đang publish cho kết quả:

| Thành phần | Hiện trạng |
| --- | --- |
| Decision rows | 2.604: development 504, validation 840, test 1.260 |
| Base workflows | 124 |
| Interactive scenarios nguồn | 620 |
| Task domains | 10 domain tổng hợp |
| Agent catalog | 42 role chung, không chia pool theo bốn domain live |
| Public views | `router_state`, `router_structured`, và development-only `router_raw_development` |
| Label store | `track_a_labels`, tách khỏi router input |
| Human review | pending |

Mười domain lịch sử là:

1. `transactional_service`;
2. `personal_device_tool`;
3. `multi_hop_knowledge`;
4. `structured_data_processing`;
5. `cross_app_workflow`;
6. `scientific_procedure`;
7. `code_debugging`;
8. `domain_qa_verification`;
9. `document_media_processing` (test-only holdout);
10. `resource_constrained_workflow` (test-only holdout).

Track A lịch sử chỉ trùng hai tên domain với release live hiện tại:
`code_debugging` và `structured_data_processing`. Nó không có
`database_migration_integrity` hoặc `formal_proof`, và role card/action space
không phải agent pool đã đóng băng của Track B. Vì vậy không được diễn giải
điểm Track A beta.2 là khả năng định tuyến trên bốn môi trường live hiện hành.

## 4. Phát hiện cần xử lý trước khi tạo Track A mới

### 4.1 Đơn vị dữ liệu và trùng lặp

2.604 rows hiện tại chỉ tương ứng 2.033 `decision_id` công khai duy nhất. Có
199 nhóm ID lặp, chứa 770 rows; kích thước nhóm lớn nhất là 5. Các bản lặp có
cùng router input và cùng tập `valid_actions`, nên đây không phải xung đột nhãn,
nhưng nếu chấm theo row chúng tạo trọng số ngầm cho một số public state.

Track A bốn-domain mới định nghĩa unit là **một primary decision step đã thực sự
chạy trong canonical Track B**, và phải copy exact public observation của step
đó. Audit canonical live runs hiện tại cho thấy 568/568 primary observations là
duy nhất, nên ánh xạ một-một không tạo duplicate weighting. Validator vẫn phải
kiểm tra fingerprint; nếu release tương lai xuất hiện state trùng, dataset phải
version một dedup/sample-weighting policy rõ ràng thay vì silently merge hoặc
nhân trọng số.

### 4.2 Không dùng hành động của JEV làm ground truth

Action đã chọn trong live trajectory chỉ là hành vi của router tham chiếu. Nó
không chứng minh action đó là action hợp lệ duy nhất, tốt nhất hoặc an toàn.
Nhãn Track A phải là tập action hợp lệ theo contract tại state đó, cộng `STOP`
khi terminal contract đã thỏa. Không tạo `preferred_action`, action utility,
capability score hoặc workflow bắt buộc.

### 4.3 Đồng bộ đúng môi trường live

Mỗi state mới phải dùng đúng full role pool của domain và đúng role card đã
đóng băng. Candidate action là toàn bộ pool cộng `STOP`; không masking agent
pool theo state. Router input không được chứa model/provider của executor,
private packet artifacts, independent evaluator result, valid-action labels,
oracle rationale hoặc hidden recovery annotation.

### 4.4 Tách dữ liệu router và evaluator

Phần router-visible tối thiểu gồm task contract công khai, current observable
state, artifact inventory công khai, observable action consequences, budget và
full role pool. Nhãn set-valued, unsafe-action annotation, terminal truth,
domain/phase analysis fields và rationale thuộc evaluator/analysis tables.

Quy tắc beta.2 vẫn giữ nguyên:

- mọi hậu quả tự động của action phải còn quan sát được;
- recovery ở cấp orchestrator là một routed action mới;
- raw decision tại zero remaining budget có thể giữ để audit nhưng phải loại
  khỏi primary routing decisions;
- evaluator-only packet material có thể tải công khai để minh bạch nhưng tuyệt
  đối không được đưa vào router/executor context.

### 4.5 Split policy

Track B hiện dùng một split `full` và không có holdout. Để thống nhất với chủ
trương dataset không chia train/test/holdout, Track A mới nên publish một split
`full`. Nếu cần phân tích theo nguồn hoặc domain, dùng metadata/provenance thay
vì partition có ý nghĩa leaderboard. Các split development/validation/test của
Track A beta.2 phải được ghi rõ là historical và không tái sử dụng ngầm.

### 4.6 Tên benchmark và metric grouping

Payload Track A beta.2 vẫn chứa `benchmark: "RoutingBench"`,
`benchmark_version: "4.0-beta"` và public `workflow_family` được làm mờ thành
`dynamic_workflow`. Đây là dữ liệu lịch sử hợp lệ và không được sửa tại chỗ.
Track A mới phải dùng canonical name `RoleRouteBench` cùng schema/release ID mới.

Evaluator cũ nhóm `by_domain` từ public `workflow_family`, nên với beta rows kết
quả có thể bị gom vào một nhóm `dynamic_workflow` thay vì task domain thật.
Evaluator mới phải join analysis label bằng `decision_id` và nhóm macro metric
theo trường `domain` evaluator-only. Không được đưa domain analysis label vào
router input chỉ để sửa metric.

## 5. Contract đề xuất cho Track A mới

Một row primary nên có:

- `decision_id`: ID ổn định từ canonical public fingerprint;
- `domain` và `packet_id` trong analysis metadata;
- `trajectory_id` và `source_decision_index` trong provenance;
- `router_input_json`: payload chính xác được phép đưa cho router;
- `candidate_actions`: full domain pool cộng `STOP`;
- `budget_remaining` và cờ xác định decision có thuộc primary set;
- `valid_actions`: evaluator-only, set-valued;
- `unsafe_actions`: evaluator-only khi contract cho phép xác định;
- `terminal_valid`: evaluator-only;
- `label_source`, `review_status`, reviewer/adjudication provenance;
- `environment_release_id`, packet checksum, role-pool checksum, runtime
  checksum và dataset/code revision.

Không đưa action as-run, pass/fail cuối, executor identity hoặc JEV rationale
vào `router_input_json`. Action as-run chỉ thuộc trajectory provenance để phân
tích off-policy.

## 6. Metric tối thiểu

Primary metrics nên được báo cáo tách biệt, không ghép composite score:

- set-valued valid-action accuracy;
- macro valid-action accuracy theo domain;
- premature `STOP` rate;
- terminal `STOP` precision và recall;
- unsafe-action selection rate nếu nhãn unsafe đã được review;
- accuracy theo state phase và budget band;
- coverage/rejection rate cho output router không parse được.

Kết quả Track A không thay thế Track B. Track A đo chất lượng một quyết định cô
lập; Track B mới đo hậu quả tích lũy và khả năng phối hợp qua nhiều bước.

## 7. Release gates trước khi sinh Track A

1. Chốt source set: chỉ canonical live states hay bổ sung state được replay có
   kiểm soát; không trộn state tổng hợp beta.1 mà không gắn namespace lịch sử.
2. Chốt canonical public fingerprint và chính sách dedup.
3. Export chính xác router payload từ frozen environment, không dựng lại bằng
   template độc lập.
4. Kiểm tra full-pool/no-masking cho mọi row.
5. Tạo set-valued labels từ contract và human review; không suy ra từ JEV.
6. Chạy private-isolation sentinel trên toàn bộ payload.
7. Kiểm tra zero-budget exclusion và `STOP` semantics.
8. Báo cáo label coverage, disagreement và adjudication; row chưa review không
   được tính vào primary benchmark score.
9. Test evaluator grouping để bốn domain xuất hiện riêng, không bị gom vào
   `dynamic_workflow`.
10. Freeze manifest/checksum và tạo release ID mới; không ghi đè beta.2.

## 8. Việc đã cập nhật trong repository

- downloader mặc định dùng `TrangBui/RoleRouteBench`;
- publisher mặc định dùng `TrangBui/RoleRouteBench`;
- tài liệu reproducibility dùng tên RoleRouteBench và URL repo mới;
- runtime, packet, adapter, role card, evaluator và trajectory beta.2 không bị
  thay đổi.

Tài liệu này là snapshot chuẩn bị cho thiết kế Track A kế tiếp. Các tài liệu có
tên `RoutingBench V4-*` trước đây vẫn được giữ làm lịch sử thiết kế và không tự
động trở thành contract của Track A mới.
