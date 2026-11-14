# RoutingBench V3-beta: Visibility Contract

## 1. Phạm vi revision

`role-card-r1` là revision dữ liệu của schema `3.0-beta`. Revision này:

- làm rõ phạm vi evidence-backed của `agent_14`;
- không thay đổi transition;
- không thay đổi `valid_actions` hoặc action rationales;
- không thay đổi task input, evaluation gold, provenance hoặc split;
- không thêm conversation memory vào prompt router.

## 2. Router được quan sát

Router nhận:

- task và public task context;
- facts, artifact inventory và open requirements;
- goal status;
- last event gồm action, public status và public effect;
- tối đa các recent event mà environment công khai;
- compact execution memory: attempts, rejected actions và public completed
  contributions;
- remaining budget và terminal-STOP policy;
- qualitative role cards;
- candidate actions.

## 3. Router không được quan sát

Router không nhận:

- scenario transition table hoặc formal preconditions;
- oracle `valid_actions`, `terminal_valid` hoặc action rationales;
- hidden simulator state;
- evaluation answer, aliases hoặc supporting IDs;
- provenance-only annotations;
- raw actor `contribution`;
- actor `candidate_answer`;
- `prior_live_contributions`;
- backbone identity, capability score, preferred action hoặc action utility.

## 4. Track-B data flow

```text
Live actor result
    ├── raw contribution/candidate answer → immutable run event
    ├── raw contribution                 → prior_live_contributions của actor sau
    └── status/public effect/artifact    → public environment state → router
```

Vì vậy, actor sau có thể sử dụng nội dung do actor trước tạo ra, nhưng router chỉ
định tuyến bằng public workflow state. Đây là chủ đích của V3-beta.

## 5. Track A

Track A dùng decision states cố định và không thực thi actor. Không có raw actor
output. Router chỉ nhận các public field trong decision record; gold-bearing
sidecar không được đưa vào prompt.

## 6. Regression guarantees

Test suite phải bảo đảm:

1. Raw contribution không xuất hiện trong router observation.
2. Candidate answer và `prior_live_contributions` không xuất hiện trong router
   observation.
3. Raw contribution vẫn xuất hiện trong run event.
4. Actor tiếp theo vẫn nhận được `prior_live_contributions`.
5. Router vẫn nhận public event effect và completed-contribution signal.
6. Sample `v3b-musique-4hop-03:state-000` giữ valid set
   `{agent_01, agent_02, agent_25}` sau role-card revision.

## 7. Nội dung hoãn sang phiên bản sau

Raw-history selection, artifact ledger, context compiler, rolling summary,
Jev-Memory, game environments và cumulative causal-error metrics không thuộc
V3-beta. Chúng cần một benchmark track riêng để không trộn state-aware routing
với memory-management quality.
