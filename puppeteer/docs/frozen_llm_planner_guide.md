# Frozen LLM planner: MMLU-Pro và SRDD

Đường chạy này không train policy. Planner chọn actor từ câu hỏi, trạng thái path,
output các bước trước và static routing profile. Tất cả actors trong một pool dùng
cùng backbone `qwen-3.5-9b`.

Planner mặc định là `openai/gpt-6-luna` qua OpenRouter, với structured outputs,
`reasoning_effort=medium` và provider routing yêu cầu hỗ trợ đầy đủ các tham số.
Đặt `OPENROUTER_API_KEY` cho planner và `HF_TOKEN` cho actor trước khi chạy.
Adapter chuyển effort sang `reasoning: {effort: medium}` và chỉ gửi những tham số
được Luna công bố hỗ trợ để `require_parameters=true` không loại hết endpoint.

## Hai setting

- `naive`: role/routing profiles đóng băng; không lưu outcome history.
- `evolving`: profiles vẫn đóng băng; chỉ lưu success/failure của toàn bộ route sau
  khi result của item đã được ghi thành công.

Route experience không cập nhật capability hoặc skill của từng actor và không có
uncertainty field.

## Cấu hình

| Task | Naive | Evolving |
|---|---|---|
| MMLU-Pro | `config/experiments/frozen_mmlu_pro_naive.yaml` | `config/experiments/frozen_mmlu_pro_evolving.yaml` |
| SRDD | `config/experiments/frozen_srdd_naive.yaml` | `config/experiments/frozen_srdd_evolving.yaml` |

Cả bốn cấu hình dùng W3D2: tối đa ba reasoning paths và tối đa hai actor
activations trên mỗi path. Aggregation không tính vào depth.

`max_width=3` là trần, không phải quota. Planner có thể tạo 1, 2 hoặc 3 path
dựa trên expected value của các đóng góp khác biệt; nếu output planner không hợp
lệ, fallback chỉ mở 1 path để tránh âm thầm tăng chi phí. Actor Qwen3.5 chạy ở
non-thinking mode để bảo đảm phần trả lời hiển thị và marker `FINAL ANSWER` được
sinh trong output budget.

## Chạy smoke test

Chạy từ thư mục `puppeteer/`:

```powershell
python main.py MMLU-Pro test --config config/experiments/frozen_mmlu_pro_naive.yaml --data_limit 5
python main.py MMLU-Pro test --config config/experiments/frozen_mmlu_pro_evolving.yaml --data_limit 5
python main.py SRDD test --config config/experiments/frozen_srdd_naive.yaml --data_limit 2
python main.py SRDD test --config config/experiments/frozen_srdd_evolving.yaml --data_limit 2
```

## Chạy toàn bộ training split

Các lệnh sau vẫn dùng frozen planner, không train policy. `train` ở đây chỉ là
dataset split. Không truyền `--data_limit` nghĩa là chạy toàn bộ 400 mẫu MMLU-Pro
hoặc 200 mẫu SRDD trong manifest seed 42.

```powershell
$env:OPENROUTER_API_KEY = "<your-openrouter-api-key>"
$env:HF_TOKEN = "<your-huggingface-token>"
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"

python main.py MMLU-Pro train --config config/experiments/frozen_mmlu_pro_naive.yaml --policy_mode frozen --run_id "mmlu_pro_train_naive_luna_s42_$runStamp"
python main.py MMLU-Pro train --config config/experiments/frozen_mmlu_pro_evolving.yaml --policy_mode frozen --run_id "mmlu_pro_train_evolving_luna_s42_$runStamp"
python main.py SRDD train --config config/experiments/frozen_srdd_naive.yaml --policy_mode frozen --run_id "srdd_train_naive_luna_s42_$runStamp"
python main.py SRDD train --config config/experiments/frozen_srdd_evolving.yaml --policy_mode frozen --run_id "srdd_train_evolving_luna_s42_$runStamp"
```

Không truyền `--build_probe_profiles`, `--profile_path`, hoặc checkpoint policy cho
các run này. Planner dùng OpenRouter; actor backbone dùng provider profile
`huggingface_router`.

## Artifact

Mỗi run lưu resolved config, audit trace và result JSONL như pipeline hiện tại.
Setting evolving lưu thêm `route_experience.json` trong run directory. Store này
được cập nhật trong `complete_item()` sau khi dòng result đã được flush.

MMLU-Pro dùng majority aggregation cố định. SRDD dùng artifact selector không cần
label: file tồn tại, chạy thành công, và không chứa placeholder được ưu tiên theo
thứ tự deterministic. Timeout được tính là execution failure.

## Invariants cần giữ khi so sánh

- Cùng dataset split, item order và split seed.
- Cùng planner model/prompt, actor pool, W3D2 và aggregation.
- Không đưa backbone, provider, teammate ID hoặc gold answer vào planner prompt.
- Khác biệt duy nhất giữa naive và evolving là `experience.mode`.
