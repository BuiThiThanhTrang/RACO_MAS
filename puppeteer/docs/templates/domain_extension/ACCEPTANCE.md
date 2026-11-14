# Biên bản nghiệm thu domain mới

Sao chép biểu mẫu này cho từng domain/release. Mọi mục PASS phải có đường dẫn evidence và source/config hash; không đổi pending thành human_verified từ kết quả chạy tool.

- Domain / adapter ID / release ID:
- Packet IDs, nguồn, revision, split, license và selection policy:
- Người duyệt nội dung, ngày, phạm vi (nếu có):
- Runtime/toolchain/dependency revisions và provisioning:
- Source/config/bank hashes:

| Gate | PASS / FAIL / PENDING | Evidence |
|---|---|---|
| Goal/invariants/edit scope rõ, không đổi semantics nguồn | | |
| Manifest/hash/path/openability cho toàn bộ packet | | |
| Context/baseline/tool runtime thật | | |
| Private witness/checker positive và negative | | |
| Role identity khớp tool implementation/schema | | |
| Artifact A → B đọc đúng version/hash | | |
| Candidate đổi làm validation stale | | |
| Rejected candidate không mất candidate đang áp dụng | | |
| Private/fixture/reader/prompt isolation thực tế | | |
| Normalization và automatic side effects được log | | |
| Full pool, precondition, STOP, budget boundary | | |
| Live allowlist/actor codec/runtime/checker dispatch | | |
| End-to-end smoke thật, tách khỏi performance cohort | | |
| Freeze manifest/archive verify | | |

Sau live:

| Gate | PASS / FAIL / PENDING | Evidence |
|---|---|---|
| State/decision/execution join và raw metadata đầy đủ | | |
| Source không đổi trong run | | |
| Independent final task checker | | |
| Infrastructure validity và evaluation coverage | | |
| Full action pool/probabilities, guard policy | | |
| Human review queue và denominator exclusions | | |
| Human labels đúng run, coverage/action accuracy tách riêng | | |

Kết luận phải phân biệt runtime-ready, đủ điều kiện đánh giá task outcome, và đủ nhãn human để chấm routing. Mô tả hạn chế/khác biệt phiên bản thực tế; không gộp run hoặc bỏ sample fail sau khi xem kết quả.
