# Chạy MBPP+ protocol v6 (end-to-end)

Protocol v6 hiện chỉ triển khai track end-to-end. Không có frozen candidate bank và
không có controlled recovery. Các config v5 vẫn giữ nguyên.

## Smoke test

Chạy một task cho từng workflow trước:

```powershell
routerlab run --config configs\mbpp_e1_modal_v6.yaml --limit 1 --run-id smoke-single-v6 --fast-check --fail-fast
routerlab run --config configs\mbpp_e2_modal_v6.yaml --limit 1 --run-id smoke-fixed-v6 --fast-check --fail-fast
routerlab run --config configs\mbpp_e3_jev_modal_v6.yaml --limit 1 --run-id smoke-jev-v6 --fast-check --fail-fast
routerlab run --config configs\mbpp_e3_luna_openrouter_modal_v6.yaml --limit 1 --run-id smoke-luna-v6 --fast-check --fail-fast
routerlab run --config configs\mbpp_e3_laya_modal_v6.yaml --limit 1 --run-id smoke-laya-v6 --fast-check --fail-fast
```

## Full run trên manifest chung

Không dùng `--fast-check` cho kết quả báo cáo:

```powershell
routerlab run --config configs\mbpp_e1_modal_v6.yaml --manifest manifests\mbpp_plus_full.json --run-id full-single-v6-s42
routerlab run --config configs\mbpp_e2_modal_v6.yaml --manifest manifests\mbpp_plus_full.json --run-id full-fixed-v6-s42
routerlab run --config configs\mbpp_e3_jev_modal_v6.yaml --manifest manifests\mbpp_plus_full.json --run-id full-jev-v6-s42
routerlab run --config configs\mbpp_e3_luna_openrouter_modal_v6.yaml --manifest manifests\mbpp_plus_full.json --run-id full-luna-v6-s42
routerlab run --config configs\mbpp_e3_laya_modal_v6.yaml --manifest manifests\mbpp_plus_full.json --run-id full-laya-v6-s42
```

JEV và Luna cần `OPENROUTER_API_KEY`. Tất cả config trên cần
`MODAL_PROXY_TOKEN`. Laya dùng checkpoint/revision local đã pin trong config.

Mỗi run vẫn có các file tổng hợp tương thích v5. V6 bổ sung thư mục
`episodes/<task_id>/` chứa state snapshot, quyết định, trajectory, request audit,
artifact và final evaluation riêng tư. Plus Test chỉ xuất hiện trong file private.

## Hai cohort dùng cho rerun v6

Full manifest 378 task được chia thành hai cohort rời nhau, giữ nguyên thứ tự từ
`mbpp_plus_full.json`:

| Cohort | Manifest | Số task | Nguồn chọn |
|---|---|---:|---|
| A: common-failure | `mbpp_plus_jev_luna_both_failed_v5_s42.json` | 124 | JEV và Luna đều fail final Plus trong run v5 |
| B: remaining | `mbpp_plus_remaining254_after_jev_luna_common_failures.json` | 254 | Phần bù của A trong full 378 |

Hai cohort không giao nhau và hợp của chúng đúng bằng full manifest 378 task. Cohort A
là một phân tích hậu nghiệm dựa trên outcome v5, không phải benchmark split độc lập.
Phải báo cáo A và B riêng; chỉ ghép lại thành kết quả full khi cùng method đã chạy cùng
protocol v6 trên cả hai cohort.

Các run v6 trên cohort A đã hoàn tất cho JEV, Luna và Fixed:

```text
subset124-jev-v6-s42
subset124-luna-v6-s42
subset124-fixed-v6-s42
```

Snapshot từ `metrics.json` sau khi cả ba run hoàn tất:

| Method | Final Base | Final Plus | Base recovery | Regression | Mean executor steps |
|---|---:|---:|---:|---:|---:|
| JEV | 38.71% | 12.90% | 0/74 | 1 | 4.63 |
| Luna | 39.52% | 14.52% | 0/74 | 0 | 5.21 |
| Fixed | 38.71% | 10.48% | 1/76 | 0 | 4.68 |

Các số này chỉ mô tả cohort A đã được chọn theo lỗi v5. Không dùng bảng này một mình
để kết luận ranking trên MBPP+; cần kết quả cohort B hoặc full 378 tương ứng.

Chạy cohort B cho bốn phương pháp:

```powershell
python -m routerlab.cli run --config configs\mbpp_e3_jev_modal_v6.yaml --manifest manifests\mbpp_plus_remaining254_after_jev_luna_common_failures.json --run-id remaining254-jev-v6-s42
python -m routerlab.cli run --config configs\mbpp_e3_luna_openrouter_modal_v6.yaml --manifest manifests\mbpp_plus_remaining254_after_jev_luna_common_failures.json --run-id remaining254-luna-v6-s42
python -m routerlab.cli run --config configs\mbpp_e1_modal_v6.yaml --manifest manifests\mbpp_plus_remaining254_after_jev_luna_common_failures.json --run-id remaining254-single-v6-s42
python -m routerlab.cli run --config configs\mbpp_e2_modal_v6.yaml --manifest manifests\mbpp_plus_remaining254_after_jev_luna_common_failures.json --run-id remaining254-fixed-v6-s42
```

Single Agent hiện chỉ được yêu cầu chạy cohort B. Không được ghép thành kết quả full
v6 của Single Agent nếu chưa chạy cùng config trên cohort A.

## Tiếp tục run bị gián đoạn

CLI hỗ trợ `--resume`. Chế độ này xác nhận `config.json` và `manifest.json` trùng hoàn
toàn, nạp lại kết quả đã hoàn tất, bỏ qua task cũ và append task còn thiếu vào cùng run:

```powershell
python -m routerlab.cli run --config <config> --manifest <manifest> --run-id <run-id> --resume
```

Không đổi cách viết đường dẫn manifest giữa lần chạy đầu và lần resume vì resolved
config được kiểm tra chính xác để tránh vô tình nối hai thí nghiệm khác nhau.
