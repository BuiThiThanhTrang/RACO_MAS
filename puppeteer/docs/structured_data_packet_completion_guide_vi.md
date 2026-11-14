# Quy trình hoàn thiện packet cho domain structured data

Ngày cập nhật: **2026-11-02**  
Phạm vi hiện tại: toàn bộ 20 task SpreadsheetBench đã được chuyển thành reusable Python transformation.

## 1. Mục tiêu

Tài liệu này chuẩn hóa cách biến một record SpreadsheetBench thành packet có thể dùng để đánh giá
decision model điều phối nhiều executor. Môi trường phải cung cấp đủ thông tin và công cụ để agent giải
task, đồng thời giữ evaluator độc lập. Mục tiêu không phải tối ưu workflow cho router hoặc làm task dễ
hơn bằng hint định tuyến.

Quy trình tổng quát để các domain khác áp dụng nằm ở
`benchmark_domain_packet_completion_playbook_vi.md`; tài liệu hiện tại giữ phần implementation và bằng
chứng riêng của SpreadsheetBench.

Một packet chỉ được xem là hoàn thiện khi phân biệt được ba lớp kết quả:

1. **Candidate acceptance:** source code đúng representation contract và được worker chấp nhận.
2. **Public execution/validation:** cùng candidate chạy được trên public case và thỏa public contract.
3. **Independent task evaluation:** trusted evaluator chạy đúng candidate hash trên toàn bộ trusted
   cases và quyết định task pass/fail.

## 2. Output contract chuẩn

Artifact model phải nộp là Python source có entrypoint:

```python
def solve(input_path: str, output_path: str) -> None:
    ...
```

Candidate phải đọc workbook tại `input_path`, ghi workbook kết quả tại `output_path`, và không phụ
thuộc tên file hay dữ liệu cụ thể của case public. Không yêu cầu model trả trực tiếp một workbook duy
nhất vì thiết kế nguồn đánh giá một chương trình tái sử dụng trên nhiều input.

Runtime dùng exact adapter `spreadsheetbench.python_oj.v1`. Worker chỉ cấp surface cần thiết của
OpenPyXL và một số helper chuẩn đã khai báo. Candidate không được đọc private assets, arbitrary path,
network hoặc process state. Restricted worker là containment profile cho benchmark này, không được mô
tả như một OS sandbox tổng quát.

## 3. Quy trình hoàn thiện một packet

### Bước 1 — Khóa nguồn và audit metadata

- Ghi dataset/repository, revision, license, archive hash và member hash.
- Lấy case 1 từ Verified-400; lấy source-native cases từ full archive.
- Nếu byte của hai workbook case 1 khác nhau, so sheet name, cell value và formula. Chỉ chấp nhận
  khác biệt serialization/style không làm đổi ngữ nghĩa.
- Kiểm tra `answer_position`, instruction, sheet/range thực tế. Mọi metadata repair phải ghi rõ trong
  provenance; không sửa âm thầm.

### Bước 2 — Xác minh từng source-native case trước live

Viết reference transformation độc lập từ instruction, không dùng answer khi sinh output. Chạy nó trên
từng case và so vùng upstream OJ. Case chỉ được đưa vào evaluator khi reference witness pass.

Nếu answer upstream sai hoặc mâu thuẫn với instruction/input:

- loại case trước mọi live run;
- lưu case ID, hash và bằng chứng loại;
- không xem case đó là lỗi của model;
- không thay answer chỉ để packet pass, trừ khi có correction chính thức và correction được ghi
  provenance.

Reference transformation chỉ là công cụ xác minh môi trường, không được đưa vào public packet hoặc
model context.

### Bước 3 — Chia public/private mà không tạo holdout toàn dataset

- Case 1 là public development case để agent inspect, viết và test candidate.
- Case 2 và 3 là private evaluator cases nếu đã qua bước 2.
- Private inputs, answers, hashes và lỗi chi tiết không được materialize vào episode workspace hoặc
  trả về router/executor.
- Public contract có thể nói candidate sẽ được đánh giá trên input khác, nhưng không lộ dữ liệu riêng.

Việc này không tạo dataset split toàn cục. Nó chỉ giữ đúng source-native evaluator cases bên trong một
packet để kiểm tra khả năng tái sử dụng của artifact.

### Bước 4 — Chuyển instruction thành semantic validation contract

Không chỉ sao chép `answer_position`. Contract cần chỉ rõ:

- vùng và sheet được phép thay đổi;
- giá trị, công thức, aggregation, ordering và duplicate policy;
- các sheet/cell phải được bảo toàn;
- style có ý nghĩa task như number format, alignment, font hoặc fill;
- quy tắc chấp nhận value tương đương và formula tương đương khi workbook chưa có cached result.

Official OJ chủ yếu kiểm tra cell values trong `answer_position`; semantic checker của packet phải bổ
sung những yêu cầu có ý nghĩa nhưng OJ không quan sát, đặc biệt preservation và formatting.

### Bước 5 — Đảm bảo role implementation khớp role contract

Mỗi role card phải map tới implementation có đúng typed tools đã công bố. Ví dụ role viết solution phải
thực sự gọi được `write_solution`; role test chỉ chạy accepted candidate; role validation chỉ đọc public
output và public contract. Không suy implementation từ tên role hoặc cấp một generic tool surface khác
với contract.

Role cards mô tả capability, không chứa route mẫu, thứ tự role bắt buộc hay hint về selection archetype.
Router được nhìn toàn bộ pool và `STOP`; guard định tuyến giữ tắt.

### Bước 6 — Artifact và state semantics

- Candidate source được version hóa bất biến và có hash.
- Attempt bị parse/AST/runtime rejection được ghi như rejected artifact; nó không thay current accepted
  candidate.
- Execution artifact ghi candidate hash, input case, stdout/stderr, status và output hash.
- Validation chỉ hợp lệ khi dependency hash khớp execution/output hiện tại; artifact cũ phải bị đánh
  dấu stale.
- Mọi mutation tự động phải hiện trong trace. Không rollback/xóa dấu vết để che hậu quả action.

### Bước 7 — Independent final evaluation

Sau khi episode dừng, evaluator tin cậy:

1. lấy current accepted candidate và cố định hash;
2. chạy chính hash đó trên clean copy của tất cả trusted cases;
3. áp semantic contract cho từng case;
4. trả per-case status, soft score và hard pass;
5. không phản hồi private outcomes vào episode.

Infrastructure failure phải là `UNAVAILABLE`, không được tính như model task failure. Candidate syntax,
representation hoặc runtime failure là executor/task evidence, không phải infrastructure failure.

### Bước 8 — Preflight bắt buộc

Mỗi packet phải qua:

- manifest/checksum/path validation;
- public-only materialization test;
- reference candidate qua actual restricted worker;
- public validation và terminal-state test;
- independent evaluator trên mọi trusted case với cùng candidate hash;
- ít nhất một negative control có khả năng bắt shortcut;
- forbidden import/path test;
- rejected attempt không thay accepted pointer;
- stale/dependency test nếu flow tạo nhiều phiên bản artifact.

Preflight pass chứng minh môi trường có discriminative power; nó không phải performance của router hay
executor và không thay human review.

### Bước 9 — Live trajectory và benchmark admission

Live pilot phải pin router, executor backbone/revision, decoding, 16K output cap, max calls, pool,
adapter, worker và evaluator. Lưu raw routing decision, dispatch, executor output, tool result, artifact
lineage, budget và terminal reason.

Quyết định non-STOP cuối được sinh khi budget đã hết nhưng không dispatch vẫn giữ trong raw trace, song
loại khỏi primary routing denominator. Không tạo một execution giả hoặc tính nó như bước routing bình
thường.

Human reviewer kiểm tra task sufficiency, route/recovery, artifact consequence và failure attribution.
Chỉ trajectory có source freeze đúng, trace đầy đủ và evaluator coverage hợp lệ mới được admit làm
benchmark evidence.

## 4. Bài học từ packet `250-20`

- Sheet minh họa `Desired outcome` làm shortcut pass case 1 nhưng fail case 3. Private reusable case là
  điều kiện cần để packet có khả năng phân biệt solution tổng quát.
- Case 2 chứa answer không nhất quán nên phải bị loại trước live, không sửa oracle theo output mong muốn.
- Live pilot Jev + Qwen3.5-9B cho thấy router đã chọn inspection/authoring hợp lý ban đầu, nhưng executor
  sinh Python lỗi syntax. Đây là failure ở candidate representation/executor; môi trường vẫn ghi đúng
  rejected attempts và final FAIL.
- Một packet live chỉ là integration evidence. Không đủ để ước lượng performance của decision model ở
  mức domain.

## 5. Các batch đã áp dụng

### 5.1. Batch bốn packet đầu tiên sau pilot

| Packet | Public/private cases | Semantic oracle bổ sung | Repair hoặc lưu ý |
|---|---|---|---|
| `sdp-spreadsheetbench-398-14` | public 1; private 2, 3 | aggregate `SALE/RET`, balance, ordering, preservation; number format, font và alignment | Verified/full case 1 khác bytes nhưng values/formulas tương đương |
| `sdp-spreadsheetbench-41-47` | public 1; private 2, 3 | bucket theo prefix, aggregate duplicate, sort numeric suffix, clear output, preserve source | sửa missing leading quote trong `answer_position`, có ghi provenance |
| `sdp-spreadsheetbench-84-40` | public 1; private 2, 3 | tổng hợp các sheet trước `LIST`, balance và total row, preserve source | chấp nhận stored values hoặc formula chính xác khi upstream formula chưa có cached value |
| `sdp-spreadsheetbench-120-24` | public 1; private 2, 3 | conditional replacement, blue fill cho qualifying rows, preserve all other values | style fill là task requirement nên được semantic checker kiểm tra |

Cả 12 source-native cases của bốn packet đã pass reference verification. Cả bốn packet đã pass actual
worker/evaluator preflight, identity negative control và forbidden-import control. Trạng thái này là
**deterministic-ready cho human review và live pilot**, chưa phải live performance evidence.

### 5.2. Batch năm packet tiếp theo

| Packet | Public/private cases | Semantic oracle | Correction hoặc lưu ý |
|---|---|---|---|
| `sdp-spreadsheetbench-227-40` | public 1; private 2, 3 | xóa duplicate text phía sau trong từng hàng nhưng giữ nguyên vị trí cột; ghi kết quả vào A14:E17; bảo toàn source block | case riêng tư thay dữ liệu nguồn trong khi example block cũ vẫn còn, nên bắt được shortcut copy example |
| `sdp-spreadsheetbench-493-5` | public 1; private 2, 3 | xóa cả hai hàng có cùng reference+narrative và debit/credit đảo nhau; giữ thứ tự; rebase công thức Net Balance | không coi hai amount khác nhau hoặc reference khác nhau là cancelling pair |
| `sdp-spreadsheetbench-585-41` | public 1; private 2, 3 | bỏ ký tự không phải số, giữ decimal point, xử lý comma như grouping separator và trả numeric value | I1 bất thường trong upstream case 3 nằm ngoài answer range nên không được đưa vào contract |
| `sdp-spreadsheetbench-47827` | public 1; private 2, 3 | sort ba chữ số tăng dần, giữ leading zero; bold, fill `#EBF1DE`, number format `000`; bảo toàn phần còn lại | dùng correction của Verified-400/gold; instruction cũ trong full archive nói nhầm thứ tự giảm dần |
| `sdp-spreadsheetbench-50631` | public 1; private 2, 3 | thay H7:H38 bằng chuỗi công thức WORKDAY dùng B3, F3 và J5:J7; bảo toàn mọi formula/value/style khác và gridlines | chuẩn hóa answer range thành `Sheet1!H7:H38`; không biến các yêu cầu “recreate” thành mutation ngoài cột H |

Cả 15 source-native cases của batch này đã pass reference verification và independent evaluator. Cả
năm identity outputs bị reject. Với `227-40`, `493-5`, `585-41` và `47827`, một candidate hard-code
theo public case được xác nhận pass public validation nhưng fail private evaluation; đây là bằng chứng
trực tiếp rằng private cases bắt được shortcut. `50631` không dùng hard-code negative này vì công thức
theo layout cố định chính là artifact tổng quát hợp lệ cho task; identity vẫn bị reject.
Các packet vẫn giữ `human_review_status=pending` và `execution_readiness=pending_runtime_verification`
cho đến khi human review và live pilot hoàn tất.

### 5.3. Batch mười packet cuối

| Packet | Public/private cases | Semantic oracle | Correction hoặc lưu ý |
|---|---|---|---|
| `sdp-spreadsheetbench-183-8` | public 1; private 2, 3 | công thức weighted average J:L theo FY và generation weight; bảo toàn I và summary row 7 | Verified-400 thêm label `FY` ở H2; normalize edit range thành J3:L6 và bỏ cached summary row 7 khỏi evaluator |
| `sdp-spreadsheetbench-33722` | public 1; private 2, 3 | progressive median formula, loại các hàng phía trên khi copy xuống; border/alignment/number format | chuẩn hóa sheet cho E2:E6; checker kiểm tra cả formula và presentation contract |
| `sdp-spreadsheetbench-37456` | public 1; private 2, 3 | formula sales theo cửa sổ 06:00–05:59:59, header động, cumulative subtraction, currency | hỗ trợ cả upstream stored values và array/formula representation qua semantic witness độc lập |
| `sdp-spreadsheetbench-45738` | public 1; private 2, 3 | formula cashflow theo frequency/maturity, principal tại maturity, center alignment, giữ number format | row 5 nằm ngoài edit range; dấu `-` được thể hiện bằng zero-display number format sẵn có |
| `sdp-spreadsheetbench-13-1` | public 1; private 2, 3 | aggregate theo DATE+REF, sort date tăng dần rồi REF A–Z, total row, clear old sections | Verified-400 sửa semantics sort; full-archive answers cũ không sort, nên cases 2–3 chỉ cung cấp input và checker tự suy ra oracle mới |
| `sdp-spreadsheetbench-267-21` | public 1; private 2, 3 | lookup ID từ SH1/RP1 vào merging, ghi `-` khi thiếu, giữ phần còn lại | sửa `answer_sheet` và quote bị hỏng trong `answer_position` |
| `sdp-spreadsheetbench-531-18` | public 1; private 2, 3 | unpivot ITEM × DATE thành 968 hàng ITEM/DATE/QTY, kể cả zero/blank, order xác định | checker suy số hàng động từ workbook thay vì hard-code output public |
| `sdp-spreadsheetbench-341-14` | public 1; private 2, 3 | split multiline values, lặp key, giữ blank record, numeric coercion và order | normalized range A1:B30 bao phủ case lớn nhất; số hàng thực tế vẫn được suy động |
| `sdp-spreadsheetbench-58484` | public 1; private 2, 3 | công thức chỉ ghi count tại cuối mỗi run transfer 5551234; bảo toàn recreated formatting và separator column G | chuẩn hóa sheet cho H5:H26 |
| `sdp-spreadsheetbench-58032` | public 1; private 2, 3 | two-key lookup Title+Department, formula A2:A35, fill `#FCD5B4`, hide gridlines, bảo toàn format khác | chuẩn hóa sheet cho A2:A35; checker quan sát cả formula, fill và sheet view |

Kết quả deterministic readiness cho batch này:

- 30/30 trusted case executions PASS bằng reference candidates qua restricted worker;
- 10/10 independent final evaluations đạt hard pass và soft score `1.0` với cùng candidate hash;
- 10/10 identity negative controls bị semantic checker từ chối;
- 10/10 forbidden-import attempts bị từ chối và không thay accepted candidate pointer;
- 10/10 public executions cũ trở thành stale sau khi một candidate version mới được accept;
- public-only materialization, manifest integrity và private isolation PASS cho cả 10 packet.

Audit state contract trên toàn domain cũng PASS 20/20 packet: router observation có task, budget/state do
shared runner cung cấp, artifact summary, freshness, rejected-attempt count và full pool + `STOP`; executor
context có public task, execution/validation contract, workbook snapshot và current runtime artifacts.
`inspect_workbook` và `profile_workbook_data` đã được sửa để thực thi đúng `sheet`/`cell_range`, nhờ đó
agent có thể đọc các range ngoài preview 200 hàng mà không cần lộ raw private workbook.

Trạng thái sau batch 5.3 là **deterministic-ready cho human review và live trajectory** cho đủ 20
structured-data packets. Đây chưa phải performance evidence của router: `human_review_status` vẫn
`pending`, chưa có live outcome cho batch cuối và chưa freeze một cohort/config dùng để so decision models.

## 6. Checklist quyết định

Packet chỉ được chuyển sang live cohort khi tất cả câu trả lời dưới đây là “Có”:

- [ ] Source/revision/license/archive/member hashes đã khóa?
- [ ] Mọi trusted case có independent reference witness pass?
- [ ] Case lỗi đã bị loại trước live và có evidence?
- [ ] Public/private materialization isolation pass?
- [ ] Candidate contract là reusable transformation và cùng hash chạy mọi case?
- [ ] Semantic checker bao phủ value/formula/style/preservation cần thiết?
- [ ] Role implementation và typed tool contract khớp nhau?
- [ ] Accepted/rejected/stale artifact semantics được test?
- [ ] Positive witness, negative control và forbidden operation đều pass?
- [ ] Independent evaluator phân biệt FAIL với UNAVAILABLE?
- [ ] Human review hoàn tất và freeze ID được ghi?
- [ ] Live config pin đầy đủ, guard tắt, budget accounting nhất quán?

Nếu thiếu một mục liên quan source correctness, visibility, oracle hoặc evaluator independence, packet
không được chạy để tạo benchmark evidence. Nếu chỉ thiếu live/human gate, có thể giữ packet ở trạng thái
deterministic-ready nhưng không gắn nhãn benchmark-ready.
