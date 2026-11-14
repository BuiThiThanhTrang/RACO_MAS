# RoutingBench V4-beta — thiết kế và triển khai

## 1. Mục tiêu

V4-beta đánh giá khả năng chọn **agent tiếp theo từ task state quan sát được** trong workflow nhiều bước. Benchmark không chấm backbone, không cung cấp capability score, không ép router phải chọn một action ưu tiên duy nhất và không dùng hidden fault label làm input.

V4-alpha.2 được giữ bất biến. V4-beta nằm ở namespace và thư mục dữ liệu riêng.

## 2. Quy mô

- 8 core task domains, mỗi domain có 13 base workflows: 3 development, 5 validation, 5 test.
- 2 unseen-domain holdouts, mỗi domain có 10 base workflows và chỉ xuất hiện ở test.
- Tổng cộng 124 base workflows.
- Mỗi base workflow có 5 counterfactual variants: clean, explicit invalid output, silent corruption, retryable failure và conflicting outputs.
- Tổng cộng 620 interactive scenarios.
- Mỗi base workflow sinh 4 matched Track C fault/control pairs: 496 pairs, 992 episodes.

Task domain và workflow pattern được lưu riêng. Domain dùng để phân tích loại task; pattern mô tả cấu trúc điều phối như dependency, branch, conflict, recovery, join, terminal control, tool routing hoặc budget-sensitive routing.

## 3. Migration từ V3

V3 chỉ đóng vai trò nguồn của:

- task seed;
- workflow pattern tham khảo;
- provenance và license.

Không sao chép role card, transition graph, valid-action label, preferred action, action utility hay capability score từ V3. Audit phân loại từng nguồn thành `eligible`, `needs_rewrite` hoặc `reject`. MuSiQue và MMLU-Pro task seeds đủ provenance được chọn deterministic; workflow tổng hợp của V3 chỉ được dùng làm pattern tham khảo.

## 4. Contract quan sát được

Router nhận task, current state, budget, agent role cards và memory view đã chọn. Router không nhận:

- valid actions hoặc action rationales;
- fault class/origin;
- oracle remaining calls;
- capability score hoặc backbone identity;
- preferred action hoặc utility;
- post-fault pair identity.

Raw executor output chỉ khả dụng ở development. Hai holdout domains không xuất hiện ở development/validation.

## 5. Execution và fault semantics

`ACTION_REJECTED` chỉ đại diện cho pre-execution rejection do precondition sai; nó không phải transition fixture. Post-execution outcomes dùng các status rõ ràng: success, inefficient, invalid output, no progress và retryable failure. Hard failure chỉ xuất hiện khi một contaminated artifact bị commit và kết thúc episode.

Clean/fault counterfactuals giữ nguyên task, pool, initial observation, budget và terminal contract. Fault được tạo sau khi executor đã chạy. Một alternate executor hợp lệ không bị biến thành invalid chỉ để ép fault.

## 6. Tracks

- Track A: next-agent selection tại state cô lập, ground truth là tập safe actions.
- Track B: router tự điều phối toàn workflow với deterministic simulator hoặc live executor.
- Track C: recovery từ matched post-fault checkpoints và matched clean controls.

Primary study gồm ba cấu hình: naive + state, structured + state, structured + history. Raw history và structured-history với rejection guard tắt chỉ là diagnostic trên development.

## 7. Metrics

Không dùng composite score. Báo cáo riêng:

- valid-action accuracy;
- workflow success/end-task accuracy;
- unsafe action rate và premature stop;
- recovery success, clean-control success, recovery gap;
- calls to detection/repair, excess calls, loop rate;
- router/executor failure attribution;
- micro và macro theo task domain;
- breakdown theo workflow pattern và fault variant;
- Wilson 95% confidence interval cho rate metrics.

## 8. Release gates

Automated validation kiểm tra quota, schema, source hashes, split groups, duplicate task hashes, holdout isolation, role-card contract, public/private leakage, counterfactual invariants, initial action equivalence, safe path và Track C prefix/budget matching.

Automated gates không thay thế human review. Mỗi base workflow và Track C pair phải được hai người review độc lập; bất đồng phải được adjudicate. Cho tới khi hoàn tất, metadata giữ `human_review_status: pending` và `release_ready: false`.

## 9. Lệnh chính

```powershell
python scripts/build_routingbench_v4_beta.py

python scripts/run_routing_benchmark_v4_beta.py `
  --config config/routingbench_v4_beta/jev_structured_history.yaml `
  --split development `
  --track all `
  --run_id "routingbench_v4beta_jev_structured_history_dev"

python scripts/analyze_routingbench_v4_beta_runs.py `
  runs/routingbench_v4beta_jev_structured_history_dev `
  --output runs/routingbench_v4beta_analysis.json
```

## 10. Trạng thái

Implementation và automated validation có thể hoàn tất tự động. Hai-reviewer human review là một thủ tục nghiên cứu bên ngoài và không được đánh dấu hoàn tất bởi generator.
