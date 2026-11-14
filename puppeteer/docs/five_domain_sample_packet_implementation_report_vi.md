# Báo cáo chọn sample và dựng packet cho năm domain mới

Ngày chốt: 2026-11-01  
Trạng thái: packet construction hoàn tất; human review và runtime verification đang chờ.

## 1. Kết quả

- Đã cố định 20 sample cho mỗi domain, tổng cộng 100 sample.
- Không tạo train/dev/test split hoặc holdout.
- Sample được chọn trước live trajectory; không thay sample theo pass/fail của router hoặc executor.
- Đã materialize 100 artifact packet theo schema `4.2-artifact-rc1`.
- Đã export 5 machine-readable agent pool, mỗi pool có 10 role cards và `STOP`.
- Guard tắt; packet không chứa workflow graph, valid-action mask hoặc preferred route.
- Tất cả 100 packet qua checksum, manifest-shape và artifact-openability validation.
- Chưa chạy live trajectory và chưa đánh dấu packet nào `runtime_verified`.

Artifact chính:

- dataset packet: `puppeteer/data/RoutingBench/v4_five_domains_100_20261101/`;
- selection manifest: `selection_manifest.jsonl`;
- source revisions và checksums: `source_lock.json`;
- machine-readable role cards: `agent_pools/*.json`;
- dynamic-routing contract: `routing_contract.json`;
- validation result: `validation_report.json`;
- reproducible builder: `puppeteer/scripts/build_five_domain_packets.py`.

## 2. Nguồn đã khóa

| Domain | Canonical source | Revision/artifact lock | License policy |
|---|---|---|---|
| Structured data processing | SpreadsheetBench Verified 400 | Git `49b73a9`; HF artifact `ab0b742`; archive SHA-256 trong `source_lock.json` | CC BY-SA 4.0 |
| Scientific procedure | ScienceAgentBench verified | Git `c26e151`; HF artifact `9c6e96c`; parquet SHA-256 trong source lock | CC BY 4.0; loại IDs `3, 32, 46, 53, 54, 84` |
| Cross-app workflow | AppWorld | Git `42b5bcf`; package `0.1.3.post1`; protected data bundle `0.1.0` với ETag và SHA-256 | Apache 2.0 cộng yêu cầu protected material chỉ được public redistribution ở dạng encrypted |
| Multi-hop knowledge | MuSiQue Full dev | Git `922ac98`; acquisition mirror `a7d9f9a`; JSONL SHA-256 trong source lock | CC BY 4.0 |
| Document/media processing | Omni Parse Bench | Git `735b646`; HF artifact `74f8d39`; manifest/results hashes trong source lock | Per-page upstream license; tests theo license của benchmark/source |

## 3. Selection cố định

### 3.1 Structured data processing

Nguồn: SpreadsheetBench Verified 400. Mỗi packet chứa public instruction và input workbook; golden workbook ở `private/`.

| Archetype | Source IDs |
|---|---|
| Formula/dependency | `183-8`, `33722`, `37456`, `45738` |
| Reshape/join | `13-1`, `267-21`, `531-18`, `341-14` |
| Cleaning/normalization | `120-24`, `227-40`, `585-41`, `493-5` |
| Aggregation/calculation | `41-47`, `84-40`, `250-20`, `398-14` |
| Formatting/presentation | `50631`, `58484`, `58032`, `47827` |

Các sample này kết hợp formula, cross-sheet dependency, reshape, deduplication, aggregation và presentation constraints. Builder không chuyển archetype thành router-visible field.

### 3.2 Scientific procedure

Nguồn: ScienceAgentBench verified. Selection cân bằng bốn scientific domain: Computational Chemistry, Bioinformatics, Geographical Information Science, Psychology and Cognitive Science; mỗi scientific domain xuất hiện năm lần.

| Archetype | Source IDs |
|---|---|
| Direct analysis | `16`, `21`, `29`, `85` |
| Method comparison | `1`, `18`, `71`, `76` |
| Anomaly diagnosis | `9`, `69`, `77`, `59` |
| Result reproduction | `28`, `27`, `50`, `42` |
| Conclusion under uncertainty | `52`, `70`, `64`, `36` |

Sáu source IDs có upstream-license exception (`3, 32, 46, 53, 54, 84`) bị loại bằng hard assertion trong builder.

### 3.3 Cross-app workflow

Nguồn: AppWorld `test_normal`. Chọn một variant cho mỗi base scenario để tránh ba task gần như trùng nhau theo supervisor state. Tất cả sample được kiểm tra có `num_apps >= 2` trong protected metadata.

Selected task IDs:

```text
6f4b9a5_1  3d9a636_1  634f342_1  270f1ff_1  042a9fc_1
d194965_1  b9c5c9a_1  0d01c76_1  652485c_1  9dabbc9_1
ffe6d5e_1  83a7951_1  f323bae_1  9016950_1  90adc3f_1
b6d1104_1  3aa1a22_1  32616b5_1  f861c32_1  6b6ca61_1
```

Protected review đã cân bằng lookup/update, synchronized records, batch operation, transaction/notification và reconciliation/recovery. Mapping semantic từ task ID sang archetype không được xuất ra release artifact. Packet chỉ lưu task ID, source split, bundle locator và checksum; task instruction, state, evaluator và solution không xuất hiện ở dạng plaintext.

### 3.4 Multi-hop knowledge

Nguồn: MuSiQue Full dev. MuSiQue Full dùng cùng `id` cho một cặp answerable/unanswerable; packet identity vì vậy là `(source_id, answerability)`, không chỉ là `source_id`.

| Archetype | Source IDs | Variant |
|---|---|---|
| Short chain | `2hop__231890_595414`; `2hop__525242_52835`; `2hop__169443_83769`; `2hop__82910_75184` | answerable |
| Long chain | `4hop1__436202_765799_282674_759393`; `4hop1__17192_17130_70784_61381`; `4hop1__56635_49925_13759_736921`; `4hop1__33012_53706_795904_580996` | answerable |
| Branching evidence | `3hop2__19639_90098_10557`; `3hop2__326964_36852_7713`; `3hop2__31398_2130_2232`; `4hop3__323121_88460_30152_20999` | answerable |
| Ambiguity/conflict | `3hop1__857_846_7702`; `3hop1__104237_214799_259594`; `3hop1__103890_10659_23296`; `3hop1__30390_36635_87083` | answerable |
| Insufficient evidence | `2hop__153573_109006`; `3hop1__539312_744503_34754`; `3hop2__655849_223623_162182`; `4hop1__129721_32392_823060_610794` | unanswerable |

Public corpus đã xóa `is_supporting` và không chứa question decomposition. Private reference chứa answer aliases, support IDs và answerability label. Tổng cộng đúng 16 answerable và 4 unanswerable.

### 3.5 Document/media processing

Nguồn: Omni Parse Bench. Mỗi page có ít nhất hai test family trong tập `present`, `order`, `layout_kind`, `table_cell`, `repeat`.

| Archetype | Page IDs |
|---|---|
| OCR/reading order | `docparsing_images-images_99663c5b_images_legal_contract_00007-p1`; `realdoc_pages-images_990322d4_images_760b6f57-61b5-4b3e-9507-4c81ea277213-p1`; `multiling_finepdfs-images_aaz_Latn_38769501-p1`; `olmo_old_scans-31-p1` |
| Table | `tables_derived-PMC10771002.1_doc-p6`; `tables_derived-PMC6635801.1_doc-p5`; `tables_derived-PMC11327485.1_doc-p4`; `tables_derived-PMC10914731.1_doc-p9` |
| Formula | `olmo_arxiv_math-2503.07965_pg7-p1`; `olmo_arxiv_math-2503.08614_pg8-p1`; `olmo_arxiv_math-2503.08341_pg5-p1`; `olmo_arxiv_math-2503.07708_pg3-p1` |
| Form/key-value | `forms_documentcloud-493872-obamacare-individual-mandate-tax-compliance-form-p1`; `forms_federal-scan_osha_300-p1`; `forms_filled-filled_irs_1040-p2`; `forms_federal-scan_ntsb_6120_1-p1` |
| Degraded/mixed-layout | `olmo_long_tiny_text-17_pg2_pg1-p1`; `handwriting_samples-catt_mss154040168_p8`; `docparsing_images-images_00e18000_images_education_english_00083-p1`; `olmo_old_scans_math-5_pg174-p1` |

Mỗi packet chứa page public, private deterministic tests và source-specific license evidence. PMC pages còn ghi article-level credit từ `pmc_articles.csv`.

## 4. Packet contract

Mỗi packet có:

```text
packs/<domain>/<packet_id>/
  public/
  private/
  fixtures/
  manifest.json
```

- `public/` là input mà executor được phép dùng.
- `private/` chứa oracle/evaluator, không đưa vào router observation.
- `fixtures/` chứa provenance, license hoặc protected-source locator, không phải routing hint.
- `manifest.json` khóa SHA-256, visibility, adapter ID, source ID và review/readiness status.

Toàn bộ manifests hiện có:

```text
human_review_status = pending
packet_review_status = pending
execution_readiness = pending_runtime_verification
```

Các status này cố ý không được nâng lên chỉ vì packet build pass. Runtime verification cần adapter thật, deterministic checker thật và isolated execution.

## 5. Kiểm tra đã chạy

- 100 packet directories; đúng 20/domain.
- 100/100 `validate_artifact_pack` pass.
- 40/40 SpreadsheetBench input/golden workbooks mở được bằng `openpyxl`.
- 20/20 MuSiQue support-ID sets là tập con của public paragraph IDs.
- MuSiQue public corpus không có `is_supporting` hoặc question decomposition.
- 0 ScienceAgentBench selected IDs thuộc license-exception set.
- 20/20 Omni pages có ít nhất hai test families.
- 20 AppWorld tasks thuộc 20 base scenarios khác nhau và đều là multi-app theo protected metadata.
- Không có AppWorld selected instruction nào xuất hiện nguyên văn trong release packet tree.
- 5 agent pools x 10 role cards; mỗi pool có 11 router actions khi tính thêm `STOP`.
- Không có `weakness` field và không có `selection_archetype` trong bất kỳ `public/` artifact nào.
- Builder và artifact validator compile thành công.

`pytest` không có trong Python environment hiện tại, vì vậy existing pytest suite chưa được chạy trong bước này. Đây không được tính là packet failure; integration validation ở trên đã chạy trực tiếp bằng production validator.

## 6. Việc còn lại trước live trajectory

1. Human review `selection_manifest.jsonl` và một số packet đại diện cho mỗi archetype.
2. Implement hoặc nối năm adapter IDs đã ghi trong manifests.
3. Bootstrap official protected assets cho ScienceAgentBench và AppWorld ở runtime cache.
4. Chạy deterministic checker smoke test rồi mới đổi packet tương ứng thành `runtime_verified`.
5. Sau khi packet contract được chấp nhận, đóng băng selection và mới bắt đầu live trajectory.

Không cần thay agent pool hoặc tối ưu workflow để hoàn thành các bước trên.
