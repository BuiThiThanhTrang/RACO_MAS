# Kế hoạch thay đổi agent pool cho MMLU-Pro

## Mục tiêu chung

Đánh giá liệu RoleAware orchestrator có chọn agent phù hợp với dạng câu hỏi
MMLU-Pro hay không. Hai phiên bản được tách riêng để không nhầm lẫn tác động
của **role card** với tác động của **capability profile**.

## Phiên bản 1 — Đổi role, giữ schema profile cũ

### Câu hỏi thí nghiệm

Khi chỉ thay cấu trúc role của pool, orchestrator có ưu tiên đúng specialist
cho từng dạng câu hỏi không?

### Thiết kế

- Thay các role tổng quát bằng tám role MMLU-Pro:
  - Quantitative & Formal Reasoner
  - Natural & Life Science Specialist
  - Computing & Engineering Specialist
  - Social, Legal & Business Specialist
  - Humanities & Behavioral Specialist
  - Generalist Independent Solver
  - Adversarial Verifier
  - Stop Controller
- Giữ nguyên 10 chiều capability cũ và cơ chế profile/evidence cũ.
- Giữ `baseline_threshold_v1`, baseline-compatible reward, `max_width = 3`,
  `max_depth = 2` để so sánh với baseline.
- Stop Controller là persona có action `terminate`, bắt buộc cho
  `baseline_threshold_v1`.

### Artifact

- Pool: `personas/role_aware/mmlu_pro_domain_specialist_split_backbone_pool.jsonl`
- Config: `config/diagnostics/mmlu_train_domain_specialist_baseline_reward_s42.yaml`
- Run đang tiếp tục: `mmlu_train_domain_specialist_baseline_reward_s42`

### Kết quả cần theo dõi

- Accuracy MMLU-Pro.
- Tần suất chọn từng role theo category.
- Tỷ lệ `agent_terminated` và số action/token trung bình mỗi câu.
- Entropy, phân phối xác suất và số agent vượt threshold.

## Phiên bản 2 — Đổi role và capability profile theo MMLU-Pro

### Câu hỏi thí nghiệm

Sau khi biểu diễn năng lực theo domain benchmark thay vì theo tool, code repair,
hay commonsense tổng quát, orchestrator có phân biệt specialist tốt hơn không?

### Thiết kế

Giữ tám role của phiên bản 1, nhưng thay capability schema bằng mười chiều:

```text
task_planning
formal_quantitative
natural_science
computing_engineering
social_legal_business
humanities_behavioral
general_reasoning
evidence_verification
answer_integration
stop_decision
```

- `tool_use` và `repair` bị loại khỏi profile vì MMLU-Pro không cần chúng.
- `software_engineering` được mở rộng thành `computing_engineering`.
- `domain_reasoning` được tách thành natural science, computing, social/legal/
  business, và humanities/behavioral.
- Stop Controller có `stop_decision` riêng.
- Prior được tính từ model-card evidence của backbone kết hợp role template;
  capability không liên quan bị relevance gate giữ ở mức thấp.

### Ghép backbone

| Role | Backbone |
| --- | --- |
| Quantitative & Formal | Qwen3.5-9B |
| Natural & Life Science | Qwen3.5-9B |
| Computing & Engineering | Llama-3.1-8B |
| Social, Legal & Business | Gemma-3-12B-it |
| Humanities & Behavioral | Mistral-Nemo-12B |
| Generalist / Verifier | Qwen3.5-9B |
| Stop Controller | Llama-3.1-8B; terminate là action local |

### Artifact

- Pool: `personas/role_aware/mmlu_pro_domain_specialist_domain_capability_pool.jsonl`
- Config: `config/diagnostics/mmlu_train_domain_capability_baseline_reward_s42.yaml`
- Chi tiết triển khai: `docs/mmlu_pro_domain_capability_schema_implementation.md`

### Điều kiện thí nghiệm

- Train mới từ đầu, không load checkpoint hay profile artifact của phiên bản 1.
- Schema profile thay đổi ý nghĩa feature, nên checkpoint/profile artifact cũ
  không tương thích với phiên bản 2.
- Chế độ mặc định là `mmlu_domain_v1`; chỉ đặt
  `PUPPETEER_CAPABILITY_SCHEMA=legacy_v1` khi resume phiên bản 1.

## So sánh công bằng

Giữ cố định seed, split, reward mode, routing mode, width/depth, backbone
assignment khi có thể. Báo cáo accuracy cùng chi phí, role frequency, stop rate
và entropy; không kết luận chỉ từ accuracy vì Stop Controller thay đổi số bước
và token tiêu thụ.
