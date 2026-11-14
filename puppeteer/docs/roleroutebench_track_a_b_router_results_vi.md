# RoleRouteBench: kết quả router trên Track A và Track B

Ngày tổng hợp: 2026-11-09.

## Định nghĩa metric Track B

Mỗi router ban đầu chạy 82 packet: database 12, code debugging 20, formal proof 30 và structured data processing 20. Ba episode formal proof của JEV có `infrastructure_failure = true` được loại khỏi cả tử số lẫn mẫu số, nên JEV có 79 episode hợp lệ; các router còn lại giữ 82.

- **Pass + STOP (P):** `result.passed = true` và router đã chọn `STOP`.
- **End-task pass, không STOP (E):** final task audit độc lập trả `passed = true`, nhưng state cuối không chọn `STOP`.
- Hai nhóm trên loại trừ nhau. `Track B success = P + E` và `Track B accuracy = (P + E) / N hợp lệ`.
- Ba episode formal proof lỗi hạ tầng của JEV là exclusion đã biết. Một episode còn lại chỉ được tính thành công nếu thuộc đúng P hoặc E như trên.

Formal proof trong bảng dùng các run có cùng executor Goedel-Prover-V2-32B. Structured data của JEV lấy từ canonical Track B package đã đóng băng.

## Bảng kết quả chính

| Router | Track A đúng / 568 | Track A accuracy | P: pass + STOP | E: end-task pass, không STOP | Track B success / N hợp lệ | Track B accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Sol | 534 | 94.01% | 52 | 0 | 52/82 | **63.41%** |
| JEV | 486 | 85.56% | 44 | 1 | 45/79 | **56.96%** |
| Drex 9B | 405 | 71.30% | 33 | 13 | 46/82 | **56.10%** |
| PPLX Decider Flash | 502 | 88.38% | 43 | 2 | 45/82 | **54.88%** |

## Chi tiết Track B theo domain

Mỗi ô có dạng `P + E = success / N (accuracy)`.

| Router | Database (12) | Code debugging (20) | Formal proof (30) | Structured data (20) |
|---|---:|---:|---:|---:|
| Sol | 8 + 0 = 8/12 (66.67%) | 15 + 0 = 15/20 (75.00%) | 26 + 0 = 26/30 (86.67%) | 3 + 0 = 3/20 (15.00%) |
| JEV | 6 + 1 = 7/12 (58.33%) | 12 + 0 = 12/20 (60.00%) | 24 + 0 = 24/27 (88.89%) | 2 + 0 = 2/20 (10.00%) |
| Drex 9B | 5 + 1 = 6/12 (50.00%) | 1 + 10 = 11/20 (55.00%) | 25 + 1 = 26/30 (86.67%) | 2 + 1 = 3/20 (15.00%) |
| PPLX Decider Flash | 4 + 1 = 5/12 (41.67%) | 11 + 0 = 11/20 (55.00%) | 25 + 1 = 26/30 (86.67%) | 3 + 0 = 3/20 (15.00%) |

## Khác biệt giữa các router

### Sol

Sol đứng đầu ở cả Track A và Track B. Toàn bộ 52 success của Sol đều là pass có STOP; không có success phải được cứu bởi end-task audit. Sol đạt kết quả tốt nhất ở database và code debugging, đồng hạng cao nhất ở formal proof. Điểm yếu chung vẫn là structured data processing, chỉ 3/20.

Terminal distribution của Sol là 52 `STOP`, 30 `CALL_LIMIT`, không có `PREMATURE_STOP`. Điều này cho thấy Sol có hành vi kết thúc chính xác hơn ba router còn lại trên tập run này.

### JEV

Sau khi loại ba episode formal lỗi hạ tầng, JEV có 44 pass có STOP và chỉ một task đạt đích mà không STOP. Kết quả này phản ánh termination tương đối tốt, nhưng số episode hợp lệ chạm `CALL_LIMIT` vẫn cao: 35/79. JEV tốt hơn PPLX ở database và code, cao nhất ở formal proof sau infrastructure exclusion, nhưng thấp nhất ở structured data.

### Drex 9B

Drex có Track A thấp nhất nhưng Track B vẫn gần JEV: Drex có nhiều hơn một success thô (46 so với 45), nhưng accuracy thấp hơn vì JEV đã loại ba episode lỗi hạ tầng khỏi mẫu số. Lý do chính là metric E: 13/46 success của Drex đã đạt task contract nhưng router không STOP. Riêng code debugging, Drex chỉ có 1 pass có STOP nhưng có thêm 10 end-task pass không STOP.

Terminal distribution của Drex là 33 `STOP` và 49 `CALL_LIMIT`. Vì vậy Drex không hẳn yếu ở việc đưa executor tới lời giải; điểm yếu rõ nhất là nhận biết thời điểm hoàn tất và kết thúc trajectory.

### PPLX Decider Flash

PPLX có Track A cao thứ hai nhưng Track B thấp nhất, dù chênh lệch với JEV/Drex chỉ một packet. Router này chọn STOP nhiều nhất: 67/82 episode, nhưng chỉ 43 episode trở thành pass có STOP; 24 episode kết thúc bằng `PREMATURE_STOP`.

PPLX vì thế thể hiện xu hướng ngược Drex: Drex thường không STOP khi task đã đạt, còn PPLX thường STOP trước khi terminal contract sẵn sàng. Đây là nguyên nhân chính khiến accuracy Track A khá cao nhưng không chuyển thành kết quả Track B tương ứng.

## Tương quan Track A và Track B

Ở mức bốn router, tương quan chỉ mang tính thăm dò vì `n = 4`:

- Pearson giữa Track A accuracy và Track B accuracy: **0.575**.
- Spearman: **0.400**.
- Nếu xem 16 điểm router × domain, Pearson chỉ còn **0.292**, Spearman **0.281**.

Không nên xem Track A là proxy trực tiếp cho Track B:

1. Track A chấm một quyết định độc lập trên state đã ghi lại, theo membership trong tập `valid_actions` do Terra gán nhãn. Nó đo chất lượng chọn next action trong điều kiện state cố định.
2. Track B là closed-loop. Quyết định hiện tại làm thay đổi state tiếp theo, đồng thời kết quả còn phụ thuộc executor, tool outcome, recovery, ngân sách và quyết định STOP.
3. State Track A được lấy từ trajectory của JEV, nên không bao phủ hoàn toàn state distribution do các router khác tự tạo ra trong Track B.
4. Structured data cho thấy khoảng cách rõ nhất: Track A đạt 51.28–98.72%, nhưng Track B chỉ đạt 10–15%. Chọn role hợp lệ tại từng state chưa đủ để bảo đảm executor tạo đúng artifact cuối.
5. Formal proof gần ceiling ở Track B (86.67–88.89% sau khi loại lỗi hạ tầng của JEV) vì cùng dùng executor 32B mạnh. Domain này hiện phân biệt router kém hơn code và database.

Kết luận: Track A vẫn hữu ích để đo chất lượng decision model ở cấp quyết định, còn Track B đo hiệu quả điều phối end-to-end. Hai track bổ sung cho nhau; không nên dùng accuracy Track A để dự đoán trực tiếp task success Track B.

## Nguồn kết quả

- Track A: `report/roleroutebench_track_a_{router}_vs_terra.json`.
- Track B: `evaluation_report.json` trong các run JEV, Drex, PPLX và Sol tương ứng.
- JEV structured data: `hf_release/RoutingBench-V4/data/track_b_results/full.parquet`.
