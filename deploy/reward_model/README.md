# Nemotron Reward Q5 trên Modal

Đây là backend thử nghiệm cho `nvidia/Llama-3.1-Nemotron-70B-Reward-HF`,
trả cả reward và hidden state `[8192]` mà Puppeteer dùng làm state của policy.
Không dùng endpoint chat/embedding thông thường để thay thế hai đầu ra này.

## Trạng thái kiểm chứng

- Đã có client, adapter, server, bộ probe và kiểm thử CPU với mock.
- Ngày 2026-09-14: CUDA image đã build thành công trên Modal; `build_check` xác nhận
  wheel 0.3.16 chứa `libggml-cuda.so`. Xem `VALIDATION.md`.
- Ngày 2026-09-14: probe Q5 đã chạy thành công 10 mẫu trên A100 80GB, sai số chạy
  lại bằng 0. BF16 chưa tải và **chưa được chạy đối chiếu với Q5**.
- Endpoint từ chối khởi động nếu thiếu báo cáo verification đạt yêu cầu cho đúng
  mã nguồn và model revision hiện tại. Chạy đủ các bước bên dưới trước khi deploy.
- Bộ probe đi kèm chỉ là kiểm tra tương thích ban đầu. Cần thêm mẫu hội thoại từ
  workload thực tế và đánh giá policy trước training dài.

## Cấu trúc

- `puppeteer/reward_api/client.py`: HTTP transport, retry, validate response.
- `puppeteer/reward_api/representation.py`: vector JSON → tensor trên device policy.
- `puppeteer/model/embedding.py`: chọn backend local/remote.
- `modal_app.py`: Modal Image/Volume, các function probe, endpoint có xác thực.
- `backend.py`: backend llama-cpp-python được pin, trích xuất raw logits/hidden state.
- `model.lock.json`: repository, commit, tên file, số byte và SHA-256 của GGUF.
- `probe.py`, `probe_cases.json`: baseline BF16, Q5 và phép đối chiếu.
- `compare_policy.py`: đo thay đổi phân phối xác suất trên checkpoint policy thật.

Client nằm ngoài package `model` vì package đó khởi tạo OpenAI client và đọc
cấu hình ngay khi import. Smoke test reward API không cần OpenAI key.

## 1. Chuẩn bị SDK

Chạy từ **thư mục gốc repository**, không phải `puppeteer`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r deploy/reward_model/requirements-local.txt
python -m modal setup
```

SDK chạy trên máy local. Các dependency CUDA/Transformers được cài trong Image
Linux trên Modal; máy local không cần GPU để gọi API.
Nếu cần thực hiện probe trên repo bị giới hạn truy cập, thêm HF token bằng Modal
Secret và gắn vào các function tải model. Các artifact mặc định là public.

### Kiểm tra build image trước

```powershell
python -X utf8 -m modal run -m deploy.reward_model.modal_app::build_check
```

Lệnh này build image và kiểm tra wheel chứa `libggml-cuda.so`, chạy trên CPU,
không load model và không cấp GPU. Nó chưa xác nhận inference trên CUDA.
Cảnh báo thiếu NVIDIA Driver trong lệnh kiểm tra CPU này là bình thường.

Nếu log báo `Could not find compiler set in environment variable CC: clang`,
image đang kế thừa `CC=clang` trong khi chỉ có GCC. Cấu hình hiện tại chỉ định
`CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, `CUDAHOSTCXX=/usr/bin/g++` và
`CUDACXX=/usr/local/cuda/bin/nvcc` trước khi build `llama-cpp-python`.

Bản 0.3.16 còn bật build công cụ multimodal `llama-mtmd-cli` theo mặc định.
Trong container build CPU, bước liên kết công cụ này có thể lỗi
`libcuda.so.1 not found` / `undefined reference to cuMemCreate`.
Image đặt `-DLLAVA_BUILD=OFF` vì backend chỉ xử lý text. `-DGGML_CUDA=ON`
vẫn giữ nguyên để build thư viện inference CUDA. Không đưa CUDA driver stub
vào đường dẫn thư viện runtime để che lỗi này.

Trên Windows, dùng UTF-8 khi đọc log để tránh lỗi `charmap`:

```powershell
python -X utf8 -m modal image logs <image-id>
```

## 2. Tải weights trước khi cấp GPU
```powershell
modal run -m deploy.reward_model.modal_app::download --include-reference
```
Lệnh này tải Q5 (49.949.818.016 byte), tokenizer và weights BF16 tham chiếu vào
Volume `puppeteer-nemotron-q5`. Cần khoảng 200 GB cho Q5 và reference cùng tồn tại.
Checksum Q5 phải khớp trước khi ghi manifest hoàn tất. Mọi revision được pin.
Tải và chạy probe có sử dụng tài nguyên trả phí của tài khoản Modal.

Nếu chỉ tải Q5, bỏ `--include-reference`; tuy nhiên probe BF16 vẫn cần reference
trong Volume trước khi chạy. Model không được tải qua mỗi request.

### Nếu probe báo thiếu `/models/download.json`

Đây là manifest chỉ được ghi sau khi tải Q5 + tokenizer và kiểm tra checksum
thành công. `build_check` chỉ cài thư viện, không tải model. `/models` là đường
mount bên trong container; đổi thành đường dẫn local không đưa weights vào Volume.

Để chuẩn bị riêng cho Q5:

```powershell
python -X utf8 -m modal run --detach -m deploy.reward_model.modal_app::download
python -X utf8 -m modal volume ls puppeteer-nemotron-q5 /
```

Chờ lệnh tải báo `Pinned model artifacts downloaded and checksum verified.` và
Volume có `download.json`, `gguf`, `tokenizer` trước khi gọi `probe_q5`.
Không tự tạo file manifest hoặc bỏ qua checksum. Nếu tải bị ngắt, chạy lại lệnh
`download`; cache Hugging Face có thể tái sử dụng phần đã tải.

## 3. Chạy probe GPU

Chạy tuần tự:

```powershell
modal run -m deploy.reward_model.modal_app::probe_q5
modal run -m deploy.reward_model.modal_app::probe_reference
```

- Q5: 1 × A100 80GB, 64 GiB RAM CPU.
- BF16: 2 × A100 80GB, 192 GiB RAM CPU, chỉ dùng cho đối chiếu.
- Backend Q5 load một bản weights và chạy hai forward pass: logits trước,
  hidden state sau. KV cache bị xóa giữa các pass/request. Đây là cách triển khai
  ưu tiên kiểm chứng đầu ra; latency sẽ cao hơn một backend tối ưu một pass.
- So sánh token IDs giữa tokenizer Hugging Face và tokenizer trong GGUF.
- Hidden state lấy ở token cuối, pooling NONE, causal attention, không L2 normalize.
- Reward lấy raw logit ở vocabulary index 0; probe đối chiếu với `generate()` của
  NVIDIA để phát hiện khác biệt do chuyển đổi hoặc xử lý scores.

`probe_cases.json` gồm cặp câu trả lời, nhiều lượt, Unicode, đầu vào dài và system
message ban đầu của Puppeteer. Sửa/thêm mẫu vào file này sẽ làm verification cũ
hết hiệu lực. Chạy lại cả hai probe sau khi đổi mẫu hoặc backend.

Kết quả lưu ở `/q5_probe.json` và `/reference_probe.json` trong Volume. Probe Q5
còn chạy lại mẫu đầu sau các mẫu khác để kiểm tra rò rỉ trạng thái KV.

Nếu `probe_reference` báo `Incorrect path_or_model_id: /models/reference/...`,
kiểm tra đã hoàn thành `download --include-reference` hay chưa. Lệnh `download`
không có cờ này chỉ chuẩn bị Q5 + tokenizer. Transformers có thể báo lỗi repo ID
khi một đường dẫn thư mục local không tồn tại; không cần đổi tên model trên Hub.

## 4. Đối chiếu và mở endpoint

Ví dụ ngưỡng khởi đầu, **không phải ngưỡng chất lượng đã được chứng minh**:

```powershell
modal run -m deploy.reward_model.modal_app::verify --min-cosine 0.99 --max-reward-error 1.0 --max-norm-relative-error 0.10
```

Mọi mẫu phải khớp token IDs, đạt cosine/norm/reward thresholds; mọi cặp phải giữ
thứ tự reward của BF16; sai số chạy lại mẫu đầu không quá `1e-4`.
Function in metrics và ghi `/verification.json`. Kết quả fail sẽ chặn server.
Không nới ngưỡng chỉ để làm lệnh thành công; xem từng sai lệch và tác động policy.

Mỗi lần bắt đầu probe/verify mới sẽ vô hiệu hóa báo cáo cũ. Nếu đã có deployment
đang chạy, dừng nó trước khi thay model/probe, rồi redeploy sau verification;
container đang warm không tự nạp lại file verification.

## 5. Deploy và kiểm tra HTTP

```powershell
modal deploy -m deploy.reward_model.modal_app
```

Dùng URL của `RewardServer.score` mà Modal in ra, không thêm `/v1` hoặc `/score`.
Tạo **Proxy Token** trong Modal workspace, rồi cấu hình trong shell:

```powershell
$env:REWARD_MODEL_URL = "https://<URL-do-Modal-tra-ve>"
$env:MODAL_KEY = "<proxy-token-id>"
$env:MODAL_SECRET = "<proxy-token-secret>"
python -m deploy.reward_model.smoke_test
```

Proxy Token khác token SDK dùng để deploy. Không commit token vào repository.
Endpoint sử dụng `requires_proxy_auth=True`. HTTP client không theo redirect để
tránh chuyển headers xác thực sang một URL khác.

Server giới hạn 1 container, xử lý tuần tự; tự scale về 0 sau khoảng 300 giây idle.
Request đầu có cold start. Timeout đọc mặc định là 180 giây, có thể tăng khi đo
cold start thực tế. Read timeout/retry có thể tạo lại công việc inference trên
server; không tự retry vô hạn.

## 6. Chạy Puppeteer

Pool hiện tại định tuyến Qwen/Llama qua Featherless AI trên Hugging Face
Inference Providers và Mistral qua OpenRouter. Đặt `HF_TOKEN` và
`OPENROUTER_API_KEY`, rồi kiểm tra
cấu hình từ thư mục `puppeteer`:

```powershell
python -X utf8 provider_smoke_test.py
python -X utf8 provider_smoke_test.py --live
```

Lệnh đầu không gọi inference. Lệnh `--live` gửi một request ngắn đến mỗi model
trong pool và có thể phát sinh chi phí provider.

Trong `puppeteer/config/global.yaml`:

```yaml
reward_model:
  backend: remote
  endpoint_url: "" # dùng REWARD_MODEL_URL ở trên
  connect_timeout_seconds: 10
  read_timeout_seconds: 180
  max_attempts: 3
```

Sau đó chạy project như cũ từ thư mục `puppeteer`, với môi trường dependencies
của project. Giữ cấu hình OpenAI/LLM providers như trước.

`REINFORCE_continuous.py` truyền `self.device` cho adapter. Máy không có CUDA có
thể chọn `device.type: cpu` trong `puppeteer/config/policy.json` cho policy; đây
là lựa chọn riêng, không tự động đổi device của toàn bộ ứng dụng.

Cấu hình không có mục `reward_model` vẫn chọn local để tương thích config cũ.
Config mẫu trong repository chọn remote. Rollback bằng `backend: local` và
`model_weight_path` hợp lệ; không có fallback local tự động khi API lỗi.

### Token budget và lỗi

Client remote tạo một bản sao của messages và giới hạn theo
reward_model.max_input_chars (mặc định 8.000 ký tự). Nếu cần rút gọn, client cắt
phần giữa của nội dung dài nhất, giữ đầu/cuối, thứ tự role và không sửa lịch sử
agent trong bộ nhớ. Server vẫn áp dụng chat template với
add_generation_prompt=False và từ chối quá 4.096 token. Giới hạn request bổ sung:
256 messages và 65.536 ký tự nội dung.
- HTTP 401/403: kiểm tra Proxy Tokens.
- HTTP 422: client in detail từ server cùng số ký tự đã gửi; giảm max_input_chars nếu request vẫn vượt 4.096 token.
- HTTP 429/408/500/502/503/504, lỗi kết nối, timeout: client retry tối đa đã cấu hình.
- Response sai shape/không hữu hạn/khác schema: dừng, không dùng vector/reward giả.
- `verification` thiếu/stale: chạy đúng download → probes → verify trước deploy.
- Tokenizer không khớp hoặc thiếu hidden-state/logits: backend fail; kiểm tra
  binding/GGUF, không thay bằng embedding pooling khác.

## 7. Đánh giá policy và kiểm thử

Tải các báo cáo về máy:

```powershell
modal volume get puppeteer-nemotron-q5 /reference_probe.json .cache/reference_probe.json
modal volume get puppeteer-nemotron-q5 /q5_probe.json .cache/q5_probe.json
```

Trong môi trường có PyTorch, chạy với checkpoint của project:

```powershell
python -m deploy.reward_model.compare_policy --checkpoint <checkpoint.pt> --reference .cache/reference_probe.json --candidate .cache/q5_probe.json
```

Script in top-1 agreement, thay đổi xác suất tối đa và mean total variation.
Đây là phép đo deterministic của phân phối policy, không phải benchmark chất
lượng task hoặc kiểm tra toàn bộ thuật toán lấy mẫu agent. Đánh giá một tập task
nhỏ với API thật trước khi training dài; cần cân nhắc train lại policy nếu state
Q5 thay đổi hành vi đáng kể.

Kiểm thử CPU (cần requests, numpy, torch; dùng môi trường project):

```powershell
python -m unittest discover -s tests -v
```

Tests kiểm tra HTTP/retry, contract, adapter CPU và forward/backward MLP, chunking,
cleanup sau lỗi, tokenizer mismatch và verification gate. Các mock không xác
nhận CUDA runtime, chất lượng GGUF hay endpoint thật.

## Tài liệu tham chiếu

- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/guide/lifecycle-functions
- https://modal.com/docs/guide/webhook-proxy-auth
- https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward-HF
- https://huggingface.co/mradermacher/Llama-3.1-Nemotron-70B-Reward-HF-GGUF
- https://github.com/abetlen/llama-cpp-python/tree/v0.3.16
