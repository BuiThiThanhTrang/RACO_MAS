# Tổng hợp thay đổi của run `mmlu_train_baseline_reward_ablation_s42_4`

## 1. Phạm vi so sánh

Tài liệu này mô tả run train 400 mẫu MMLU-Pro với seed 42.

Nó so sánh run này với các run diagnostic trước đó dùng S0 pool.

Các đối chiếu chính là entropy ablation, `k=2`, legacy threshold và surplus.

Đây không phải là bản sao tuyệt đối của baseline Puppeteer gốc.

Đây là một ablation trong kiến trúc `RoleAwareREINFORCE` hiện tại.

Mục tiêu là cô lập ảnh hưởng của reward và sampling gần với baseline.

## 2. Tóm tắt cấu hình của run

`dataset.name` là `MMLU-Pro`.

`dataset.mode` là `train`.

Run dùng 400 mẫu từ offset 0, với `split_seed=42`.

Các run diagnostic trước thường dùng 20, 50 hoặc 120 mẫu.

Do đó đây là run dài hơn để quan sát học thay vì chỉ smoke test.

`policy.type` vẫn là `role_aware_reinforce`.

Policy được train mới, không load policy checkpoint trước đó.

Thiết bị policy là CPU.

Audit được bật cho mọi task, routing decision, reward và policy update.

Checkpoint được lưu mỗi 5 item và giữ 5 snapshot gần nhất.

## 3. Thay đổi về reward

Run này dùng `policy.reward.mode=baseline_compatible_v1`.

Các run S0 trước chủ yếu dùng `role_aware_v1` mặc định.

`baseline_compatible_v1` là một nhánh reward riêng trong `RoleAwareREINFORCE`.

Nó không dùng Skywork trajectory reward.

`trajectory_reward.enabled=false` là yêu cầu bắt buộc của mode này.

Nó cũng không dùng `cost_reward.step_penalty` của role-aware reward cũ.

Nó cũng không dùng `cost_reward.token_cost_weight` của role-aware reward cũ.

Trong audit, `token_cost_penalty` vì thế luôn bằng 0.

Điều này không có nghĩa action cost bằng 0.

Mỗi action vẫn nhận một penalty dựa trên `runtime_action.cost`.

Penalty action được chuẩn hóa bởi `action_cost_normalization=100000`.

Action thông thường có factor `-1.0`.

Các action web có factor `-1.5`.

`step_scale=0.1` kiểm soát độ lớn cost theo bước.

`growth_rate=1.0` làm scale không tăng lũy tiến theo độ sâu.

Task đúng nhận reward cơ sở `+1.0`.

Task sai nhận reward cơ sở `-1.0`.

Terminal term dùng `terminal_factor=0.5`.

Terminal reward được cộng khi task đúng và trừ khi task sai.

Vì vậy path reward không đồng nhất hoàn toàn với accuracy.

Đây là lý do cần báo cáo cả accuracy lẫn path reward.

## 4. Thay đổi về training objective

Learning rate của run này là `5e-5`.

Nó cao hơn các run S0 gần đây dùng `1e-5` hoặc `5e-6`.

Discount factor là `gamma=0.9`.

Một số run S0 trước dùng `gamma=0.8`.

Entropy coefficient là `-0.1`.

Độ lớn này mạnh hơn đáng kể so với `-0.005`, `-0.001`, `0` hoặc `0.001` đã thử.

Theo quy ước loss hiện tại, hệ số âm khuyến khích entropy cao hơn.

Mục đích là giữ exploration trong lúc policy còn chưa phân biệt agent rõ.

`lambda_kl_loss=0.0`, nên không có KL regularization với prior policy.

`threshold_margin_coef=0.0` tắt hoàn toàn surplus bonus.

`threshold_margin_cap` không có tác dụng khi coefficient bằng 0.

Điều này tách ablation baseline reward khỏi bonus khoảng cách vượt threshold.

## 5. Thay đổi về sampling agent

Run này dùng `routing.mode=baseline_threshold_v1`.

Các run S0 trước dùng chủ yếu `legacy_threshold` hoặc `categorical_set_v2`.

Baseline mode không thêm action STOP nội bộ vào softmax của orchestrator.

Stop Controller là một persona bình thường trong pool và có thể được chọn.

Điều này khác legacy threshold, nơi STOP là action riêng khi path đã bắt đầu.

Ngưỡng của run này là `1.5 / N_agent`.

Pool có 14 agent, nên threshold xấp xỉ `0.1071`.

Agent có xác suất lớn hơn ngưỡng được chọn và sắp theo xác suất giảm dần.

Tập agent được chọn bị giới hạn bởi capacity và `max_width=3`.

Nếu không agent nào vượt ngưỡng, policy sample không hoàn lại tối đa 3 agent.

`max_depth=2`, nên mỗi path có tối đa hai bước agent.

Sampling này giống cấu trúc fallback threshold của baseline.

Tuy vậy baseline gốc dùng ngưỡng cứng `2 / N_agent`.

Run này cố ý dùng numerator `1.5`, không phải `2.0`.

Vì thế cần gọi nó là baseline-compatible, không phải baseline-identical.

Ngưỡng thấp hơn làm nhiều agent vượt ngưỡng hơn và có thể tạo nhiều path hơn.

## 6. Thay đổi về agent pool

Run này không dùng `personas/role_aware/s0_pool.jsonl`.

Nó dùng `personas/role_aware/puppeteer_model_card_role_aware_pool.jsonl`.

Pool có 14 agent, bảo toàn các slot agent của Puppeteer baseline.

Mỗi agent cũ được thêm role card và capability prior.

Nguồn metadata ghi rõ `derived_from=puppeteer_pool.jsonl`.

Prior profile có version `puppeteer-model-card-role-blend-v1`.

Prior được xây từ model card chính thức và benchmark công bố.

S0 pool trước đó có các cặp role-backbone chéo và profile version v2.

S0 cũng dùng các backbone mới như Qwen 3.5 và Gemma 3.

Pool hiện tại quay về backbone tương ứng baseline pool.

Các backbone gồm Qwen 2.5 7B/14B, Llama 3.1 8B, Llama 3.2 3B,

Mistral Nemo 12B và Ministral 3B.

Hai provider xuất hiện trong pool là Hugging Face Router và OpenRouter.

Vai trò gồm file/web/academic retrieval, Python, planning và reasoning.

Pool cũng có critic, reflector, decomposer, summarizer, integrator và repair.

Stop Controller có action `terminate` và là agent thứ 14.

Vai trò và tool/action contract được mô tả rõ trong từng role card.

## 7. Thay đổi về profile và cách profile cập nhật

Profile được khởi tạo từ `source=priors`, không phải artifact probe.

Lý do là prior khớp trực tiếp fingerprint của pool hiện tại.

`required_for_train=false` nên không cần tạo probe profile trước khi train.

`alpha=0.0001` rất nhỏ so với nhiều run S0 dùng `0.0002` hoặc `0.2`.

Do đó profile có cập nhật, nhưng thay đổi rất chậm trong 400 mẫu.

`reset_scope=run` đảm bảo profile được reset theo run mới.

Evidence chỉ được rút ở terminal của mỗi path.

`deduplicate_within_path=true` ngăn một agent được gọi lặp trong cùng path tạo evidence lặp.

`aggregate_parallel_evidence=true` gộp evidence từ các path song song.

`update_strategy=role_task_intersection` chỉ cập nhật capability liên quan role và task.

Global reliability vẫn được cập nhật.

Role adherence cũng được cập nhật tách biệt với capability.

Vì mode là `train`, profile update được bật trong runner.

Do đó đây không phải fixed-profile experiment, dù alpha rất nhỏ.

## 8. Thành phần giữ nguyên hoặc không phải biến số của run

Task analyzer primary vẫn là Qwen3-Embedding-8B, frozen, 4096 chiều.

Nó khác với hidden-state reward encoder 70B của baseline gốc.

Aggregation vẫn dùng mode `legacy` với seed 42.

Graph vẫn giới hạn width 3 và depth 2.

Tool được cho phép ở run là `run_python`.

Các thay đổi gần đây về chia sẻ history theo path không nên quy cho run này

nếu chúng được áp dụng sau khi run đã khởi động.

Khi tái lập, luôn dùng `resolved_config.yaml` của chính run này.

## 9. Ý nghĩa thực nghiệm

Run này thay đồng thời reward, sampling mode, pool và prior source.

Vì vậy nó không cô lập một thay đổi duy nhất so với các run S0 cũ.

Nó phù hợp để so sánh một cấu hình baseline-compatible tổng thể.

Nó không đủ để kết luận riêng reward hay riêng pool là nguyên nhân cải thiện.

Muốn cô lập reward, giữ nguyên pool S0 và chỉ đổi reward plus sampling.

Muốn cô lập pool, giữ nguyên reward và routing của S0.

Muốn tái lập baseline gần hơn, đổi numerator từ 1.5 sang 2.0.

Khi báo cáo, nêu rõ final evaluation mới là phép đo generalization.

Training accuracy và reward chỉ chứng minh optimizer học trên train split.
