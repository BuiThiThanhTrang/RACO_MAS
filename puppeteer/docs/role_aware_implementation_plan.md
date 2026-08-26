# Kế hoạch triển khai hệ thống đa tác tử nhận biết vai trò

## Các quyết định thí nghiệm đã khóa

- Seed duy nhất: 42.
- Giới hạn topology: chiều rộng tối đa 4, chiều sâu tối đa 2.
- Agent pool: đúng 11 vai trò; mỗi vai trò có hai teammate trong S0, S1, S2 và S3, trừ khi số lượng agent chính là biến can thiệp.
- STOP là action của orchestrator, không phải teammate hay vai trò thứ 12.
- Probe của GSM-Hard gồm 10 item lấy từ chính GSM-Hard; final test còn 909 item.
- Reference profiling: mỗi teammate đang khả dụng chạy cùng một tập con cố định gồm 50 item lấy từ reference split của task.
- Capability feedback là evidence nhị phân ở cuối từng reasoning path; teammate không được chọn sẽ không thay đổi.
- Profile update: mọi teammate được kích hoạt đều cập nhật global reliability; capability chỉ cập nhật trên phần giao giữa capability scope của role và capability scope của task.
- EMA dùng alpha cố định bằng 0.2. Evidence từ các path song song được gộp bằng công thức alpha_batch = 1 - (1 - alpha)^n.
- Quyền dùng tool: Python chỉ được bật cho GSM-Hard và SRDD; MMLU-Pro và CW chạy trong closed-book setting.
- CommonGen judge dùng Gemini 3.5 Flash với đúng rubric ba tiêu chí, thang điểm 1-4 của baseline.
- Task Analyzer dùng BAAI/bge-large-en-v1.5, vector 1024 chiều, và chỉ được gọi qua embedding server từ xa; Puppeteer không load BGE local.
- Auxiliary scalar reward dùng Skywork/Skywork-Reward-V2-Llama-3.1-8B.
- Skywork chỉ chấm một lần sau khi từng reasoning path hoàn tất, với đầu vào gồm task, trajectory và candidate output; Skywork không tham gia Task Analyzer hoặc routing.
- Điểm Skywork được center bằng 2q - 1, nhân trọng số 0.1, chỉ dùng khi train policy; lỗi scorer quay về task reward.
- Capability profile tiếp tục dùng binary evidence từ task evaluator và không đọc điểm Skywork.

## Giai đoạn 1: Nền tảng và vòng đời runtime

- Mỗi run sử dụng một experiment config riêng và lưu resolved-config snapshot vào thư mục run.
- Registry, policy, profile store, agent graph và action graph thuộc riêng từng run.
- Luồng thực thi role-aware không được đọc registry hoặc network singleton.
- Khi chuyển sang data item tiếp theo, hệ thống chỉ reset episode state, không đăng ký lại agent.
- Số lượng và định danh agent phải giữ nguyên trong suốt một run.
- Config nguồn và config/policy.json không được ghi đè trong lúc chạy.

## Giai đoạn 2: Biểu diễn role và teammate

- RoleCard mô tả role goal, core functions, action được phép, action bị cấm, input/output schema, tool và capability prior.
- TeammateSpec chứa RoleCard cùng model backbone, provider profile và decoding config nội bộ.
- CapabilityProfile lưu capability mean, uncertainty, observation count, role adherence và global reliability.
- Model identity, provider, teammate ID và index chỉ được dùng nội bộ; router không được nhìn thấy những trường này.
- Role adherence được kiểm tra bằng rule:
  - Action phải nằm trong allowed_actions.
  - Action không được nằm trong forbidden_actions.
  - Tool phải được RoleCard cho phép.
  - Tool phải được run config bật cho task hiện tại.
  - Teammate phải đang ở trạng thái available.

## Giai đoạn 3: Router có khả năng tổng quát hóa

Kiến trúc router gồm bốn thành phần:

1. Frozen Task Analyzer tạo vector biểu diễn task và reasoning state.
2. Agent Encoder nhận RoleCard cùng CapabilityProfile đã loại bỏ model identity.
3. Set Transformer contextualize một agent pool có kích thước thay đổi.
4. Shared Scorer chấm điểm mọi teammate bằng cùng một bộ tham số.

STOP dùng một learned embedding riêng nhưng được chấm bởi cùng shared scorer.

Checkpoint chỉ phụ thuộc vào:

- Số chiều của task representation.
- Số chiều của public agent feature.
- Kiến trúc Agent Encoder, Set Transformer và Shared Scorer.

Checkpoint không phụ thuộc vào:

- Số agent trong pool.
- Teammate ID.
- Backbone identity.
- Thứ tự teammate trong pool.

Điều này cho phép cùng một policy chạy với pool 11, 22 hoặc 33 agent và đánh giá các scenario S1, S2, S3 mà không thay output layer.

## Giai đoạn 4: Capability evidence và credit assignment

Với mỗi reasoning path:

- Chỉ ghi nhận teammate thực sự được kích hoạt.
- Một teammate xuất hiện nhiều lần trong cùng path chỉ được tính một lần.
- Successful path tạo binary evidence r = 1.
- Failed path tạo binary evidence r = 0.
- Unselected teammate không được cập nhật.
- Profile được cập nhật từ kết quả của từng path, không dùng kết quả majority vote cuối cùng.

Khi một teammate xuất hiện trong nhiều path song song:

- Gộp reward bằng giá trị trung bình r_bar.
- Đặt n bằng số path đã quan sát teammate đó.
- Tính alpha_batch = 1 - (1 - alpha)^n.
- Chỉ thực hiện một batch update để kết quả không phụ thuộc thứ tự hoàn tất của path.
- Uncertainty chỉ giảm trên những capability thực sự được quan sát.

Các capability được cập nhật là:

role capability scope giao với task capability scope.

Global reliability được cập nhật cho mọi teammate đã chọn. Role adherence được cập nhật riêng, không trộn vào terminal capability reward.

## Giai đoạn 5: Probe profile và reference profiling

- Fresh train bắt buộc load probe_profiles.json được tạo từ đúng 10 item probe.
- Mỗi teammate khả dụng chạy toàn bộ 10 probe item bằng FixedTeammatePolicy; shared scorer không được train trong bước này.
- Reference profile không được load tự động vào fresh train; nó chỉ được dùng explicit trong các run so sánh sau này.
- Mỗi teammate khả dụng chạy 50 reference item cho GSM-Hard, MMLU-Pro và SRDD; CW dùng toàn bộ 40 item của reference split.
- Python Tool Agent bị loại khỏi reference assignment của MMLU-Pro và CW vì Python không được bật trong hai task này.
- Profile artifact và manifest được ghi atomic sau mỗi teammate.
- Reference profile là cơ sở để xác định oracle assignment.
- Không dùng output trung gian hoặc semantic judge riêng để cập nhật capability.

Tạo probe profiles trước fresh train bằng --build_probe_profiles. Reference comparison dùng --build_reference_profiles và chỉ load bằng --profile_source reference --profile_path <path>.

    python main.py gsm-hard probe --config config/experiments/role_aware_gsm.yaml --build_probe_profiles
    python main.py MMLU-Pro probe --config config/experiments/role_aware_mmlu_pro.yaml --build_probe_profiles
    python main.py SRDD probe --config config/experiments/role_aware_srdd.yaml --build_probe_profiles
    python main.py CW probe --config config/experiments/role_aware_cw.yaml --build_probe_profiles

## Giai đoạn 6: Dataset split và evaluator

Mọi split được tạo một lần với seed 42, lưu dưới dạng index manifest trong data/splits và không được tạo lại giữa các run.

| Dataset | Train | Dev | Reference | Probe | Final test |
|---|---:|---:|---:|---:|---:|
| GSM-Hard | 200 | 100 | 100 | 10 | 909 |
| MMLU-Pro | 200 | 140 | 280 | 10 | 2000 |
| SRDD | 200 | 100 | 100 | 10 | 790 |
| CW | 80 | 20 | 40 | 10 | 50 |

Quy tắc bổ sung:

- GSM probe lấy từ GSM-Hard.
- Các split không giao nhau.
- MMLU-Pro được stratify theo category từ official test split.
- Official MMLU-Pro validation split được giữ riêng, không trộn vào các split nghiên cứu.
- Phần MMLU-Pro test không được chọn sẽ không được dùng.
- Alias validation ánh xạ sang dev; alias test ánh xạ sang final.

### GSM-Hard

- Success khi numerical answer khớp ground truth trong tolerance của evaluator.
- Capability evidence là 1 nếu đúng, ngược lại là 0.

### MMLU-Pro

- Success khi predicted choice khớp đáp án.
- Capability evidence là 1 nếu đúng, ngược lại là 0.

### SRDD

Task bắt buộc có req = "code".

Binary success chỉ bằng 1 khi đồng thời:

- Executability = 1.
- Completeness = 1.
- Consistency >= 0.70.

Continuous evaluator reward vẫn được giữ cho policy training; profile chỉ nhận binary evidence.

### CW/CommonGen-Hard

Task bắt buộc có req = "text".

Gemini 3.5 Flash chấm ba tiêu chí:

- Grammar and Fluency.
- Context Relevance.
- Logic Consistency.

Mỗi tiêu chí có raw score từ 1 đến 4 và được chuẩn hóa về [0,1].

Binary success chỉ bằng 1 khi đồng thời:

- Coverage = 1.
- Grammar >= 0.75.
- Relevance >= 0.75.
- Consistency >= 0.75.

Continuous reward bằng coverage nhân trung bình ba quality score; profile chỉ nhận binary evidence.

## Giai đoạn 7: Thiết kế scenario

### S0

- Pool đã thấy trong training.
- Dùng Qwen và Llama.
- Hai teammate cho mỗi role.

### S1

- Giữ hai model family Qwen và Llama.
- Dùng các tổ hợp role-backbone chưa xuất hiện trong S0.
- Kiểm tra khả năng tổng quát hóa sang teammate identity/composition mới trong cùng model family.

### S2-backbone

- Thay backbone trong cùng model family.
- Model mới được phép khác kích thước.
- Role structure được giữ cố định.

### S2-role

- Giữ backbone slot nhưng gán lại RoleCard.
- Kiểm tra role mismatch và capability shift.

### S2-tools

- Một Python Tool Agent bị loại quyền run_python.
- Một Software Engineer được gán run_python.
- Quyền tool của task vẫn do run config quyết định ở lớp ngoài.

### S2-cardinality

- Small pool: một teammate cho mỗi role, tổng cộng 11 agent.
- Large pool: ba teammate cho mỗi role, tổng cộng 33 agent.

### S3-primary

- Pool hoàn toàn chưa thấy.
- Chỉ dùng Mistral family.
- Hai teammate cho mỗi role.

### S3-mixed

- Pool phụ để phân tích.
- Trộn Qwen, Llama và Mistral trong cùng pool.

Danh sách đường dẫn persona cho từng scenario nằm trong config/experiments/scenario_matrix.yaml.

## Giai đoạn 8: Trình tự chạy thí nghiệm

Chạy các lệnh từ thư mục puppeteer.

### Bước 1: Tạo lại deterministic artifacts khi cần

Chỉ chạy lại hai lệnh này nếu source dataset hoặc định nghĩa pool thay đổi:

    python -m scripts.build_dataset_splits
    python -m scripts.build_role_aware_pools

Không tạo lại split chỉ vì bắt đầu một run mới.

### Bước 2: Tạo probe profiles

Chạy bốn lệnh probe profiling ở phần trên. Fresh train fail-fast nếu thiếu artifact hoàn tất.

### Bước 3: Train S0

- Train trên đúng train split.
- Giữ seed 42.
- Dùng W3D4.
- Dùng online REINFORCE.
- Task-level reward giữ dạng continuous.
- Capability evidence giữ dạng binary.
- Fresh train lưu checkpoint_initial.pt bất biến sau khi nạp probe profile và trước item đầu tiên.
- latest.pt và snapshot được lưu mỗi 20 item; chỉ giữ ba snapshot gần nhất.
- Final checkpoint luôn được ghi khi run kết thúc bình thường.
- Resume phải truyền --checkpoint explicit và khôi phục policy, optimizer, profile, global_step, RNG cùng dataset progress.
- Nếu artifact có dòng sau checkpoint, các dòng đó được backup rồi replay để không duplicate.
- Run checkpoint là nguồn duy nhất của trạng thái resume.
- Reference profile chỉ được load explicit trong comparison run.

### Bước 4: Chọn cấu hình trên dev

- Chỉ dùng dev để kiểm tra topology sensitivity và checkpoint; routing threshold được tính động, không tune bằng config.
- Dev/probe/final không được update optimizer hoặc capability profile.
- policy_mode=evolved bắt buộc checkpoint explicit.
- Khóa topology W3D4 cho các run so sánh chính.
- W3D4 là cấu hình chính mặc định.
- Routing chọn các candidate có xác suất ít nhất `1 / số agent đang khả dụng`, sau đó giới hạn bởi `max_width`.

### Bước 5: Chạy probe

- Tạo probe profile riêng cho từng pool bằng đúng 10 item probe đã khóa.
- Probe không được dùng để cập nhật policy cho final.
- Artifact profile phải có manifest khớp fingerprint và teammate set của pool.
- Lưu profile và routing trace để phân tích generalization.

### Bước 6: Final evaluation

- Không tune trên final split.
- Chạy lần lượt S0, S1, từng biến thể S2, S3-primary và S3-mixed.
- Dùng --personas để thay pool theo scenario_matrix.yaml.
- Dùng cùng policy checkpoint, seed, evaluator và topology đã khóa.
- Với pool mới, chỉ transfer policy weights/global_step rồi load explicit probe
  profile của pool đó; không restore optimizer, RNG, progress hoặc profile S0.

## Cấu hình máy thuê đề xuất

Cho BGE-large Task Analyzer và Skywork Reward 8B được serve chung trên máy thuê:

- GPU: 1 RTX 5090, 32 GB VRAM.
- CPU: ít nhất 16 vCPU.
- RAM hệ thống: 64 GB.
- Lưu trữ: ít nhất 150 GB NVMe.
- Hệ điều hành: Ubuntu 22.04 hoặc 24.04.
- Runtime: Docker cùng NVIDIA Container Toolkit.
- Serving: OpenAI-compatible embedding server cho BGE và HTTP reward server cho Skywork.
- Dtype ưu tiên: BF16.
- Max input của BGE: 512 token.
- Embedding output: cố định 1024 chiều.
- Batch ban đầu: 1 cho cả hai service.

Máy local cần đặt:

    TASK_ANALYZER_BASE_URL=<openai-compatible-base-url>
    TASK_ANALYZER_API_KEY=<api-key>
    SKYWORK_REWARD_BASE_URL=<reward-server-base-url>
    SKYWORK_REWARD_API_KEY=<api-key>

BGE-large và Skywork đều được load trên máy serve; Puppeteer local chỉ gọi API.

## Tiêu chí hoàn tất

Implementation được coi là đạt khi:

- Cùng seed và cùng input tạo cùng routing/result trong deterministic evaluation mode.
- Chạy liên tiếp nhiều data item không làm tăng agent registry.
- Router không nhận teammate ID, model backbone hoặc provider.
- Policy checkpoint chạy được với pool có kích thước khác.
- STOP không xuất hiện như một teammate.
- Closed-book task không thể chọn agent cần tool bị tắt.
- Profile chỉ cập nhật teammate đã kích hoạt.
- Profile update không phụ thuộc thứ tự path song song.
- Mỗi task tạo JSONL artifact hợp lệ.
- SRDD và CW tạo đúng loại artifact.
- Config nguồn không bị chỉnh sửa trong lúc chạy.
- Không còn handoff_when trong source.
- Split manifest có đúng kích thước và không giao nhau.

## Kiểm thử

Kiểm tra cú pháp:

    python -m compileall -q agent config inference role_aware tasks scripts

Chạy toàn bộ test:

    python -m unittest discover -s tests -p test_*.py -v

Bộ test hiện tại bao phủ:

- Vòng đời registry qua hai data item.
- Không dùng singleton network.
- Router view không làm lộ model identity.
- Pool S0 có đúng 11 role và 22 teammate.
- S3-primary chỉ dùng Mistral.
- Tool masking trong closed-book setting.
- Policy xử lý pool 11, 22 và 33 agent.
- Set Transformer permutation-equivariant.
- Profile update bất biến theo thứ tự path.
- Global reliability và task-relevant capability update.
- Reference budget 50 item cho mỗi teammate.
- Split manifest đúng kích thước và không giao nhau.
- SRDD/CW có đúng req schema.
- Smoke test bốn task tạo JSONL artifact hợp lệ.
- Exact resume từ chối pool fingerprint khác.
- Policy-transfer chạy checkpoint S0 với external profile S1.
- Generalization evaluation không restore optimizer/progress hoặc update profile.
