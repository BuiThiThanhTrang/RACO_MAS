# Hướng dẫn thiết lập và chạy Puppeteer

Tài liệu áp dụng cho repository:

~~~text
D:SideProjectchatdev_originalChatDev
~~~

Các lệnh bên dưới dùng PowerShell trên Windows. Lệnh Modal chạy từ thư mục gốc repository; lệnh benchmark chạy từ thư mục puppeteer.

# Phần 1 — Thiết lập toàn bộ hệ thống

## 1. Chuẩn bị

Cần có Git, Miniconda/Anaconda, Python 3.11 bản 64-bit, tài khoản Hugging Face có quyền dùng Inference Providers, tài khoản OpenRouter và tài khoản Modal có credit hoặc phương thức thanh toán.

Nếu prompt hiện đồng thời (.venv) và (puppeteer_env), thoát .venv trước:

~~~powershell
deactivate
conda activate puppeteer_env
~~~

Kiểm tra Python:

~~~powershell
python --version
python -c "import struct; print(struct.calcsize('P') * 8)"
~~~

Kết quả cần là Python 3.11 và 64. Python 32-bit hoặc phiên bản không có wheel tương thích có thể khiến pip cố build NumPy bằng MinGW và báo NumPy requires GCC >= 8.4.

## 2. Tạo môi trường và cài dependency

~~~powershell
Set-Location "D:SideProjectchatdev_originalChatDev"
conda create -n puppeteer_env python=3.11 -y
conda activate puppeteer_env
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r deploy/reward_model/requirements-local.txt
~~~

Kiểm tra interpreter:

~~~powershell
python -c "import sys; print(sys.executable)"
~~~

Đường dẫn phải trỏ tới môi trường puppeteer_env. Chạy test CPU:

~~~powershell
python -X utf8 -m unittest discover -s tests -v
~~~

Test CPU kiểm tra HTTP client, retry, schema, adapter reward và tool agent. Chúng không kiểm tra CUDA hoặc chất lượng Q5.

## 3. Thiết lập provider cho agent

Pool mặc định là puppeteer/personas/personas.jsonl.

| Nhóm model | Provider | Biến môi trường |
|---|---|---|
| Qwen và Llama | Hugging Face Inference Providers | HF_TOKEN |
| Mistral | OpenRouter | OPENROUTER_API_KEY |

Đặt credential trong chính PowerShell sẽ chạy Puppeteer:

~~~powershell
$env:HF_TOKEN = "<hugging-face-token>"
$env:OPENROUTER_API_KEY = "<openrouter-api-key>"
~~~

Không ghi token vào YAML, source code hoặc Git. Model Hugging Face hiện đã có provider suffix như :featherless-ai; token vẫn phải có quyền Inference Providers và provider tương ứng phải được bật trên tài khoản.

Kiểm tra routing mà chưa gọi inference:

~~~powershell
Set-Location "D:SideProjectchatdev_originalChatDevpuppeteer"
python -X utf8 provider_smoke_test.py
~~~

Gọi thử cả sáu model:

~~~powershell
python -X utf8 provider_smoke_test.py --live
~~~

Kiểm tra riêng từng model:

~~~powershell
python -X utf8 provider_smoke_test.py --live --model qwen-2.5-7b
python -X utf8 provider_smoke_test.py --live --model mistral-nemo-12b
~~~

Lệnh --live gửi request thật và có thể phát sinh chi phí. Không chạy 200 mẫu nếu một model trong pool còn lỗi.

## 4. Thiết lập Modal SDK

Chuyển về thư mục gốc repository:

~~~powershell
Set-Location "D:SideProjectchatdev_originalChatDev"
python -m modal setup
~~~

Credential Modal SDK dùng để build/deploy khác Modal Proxy Token dùng để gọi endpoint.

Kiểm tra image CUDA:

~~~powershell
python -X utf8 -m modal run -m deploy.reward_model.modal_app::build_check
~~~

Kết quả mong đợi:

~~~text
{"llama_cpp_python": "0.3.16", "cuda_library_present": true}
~~~

Lệnh này không tải model, không cấp GPU inference và không xác nhận chất lượng Q5.

## 5. Tải Q5 và BF16 reference

Quy trình xác minh đầy đủ cần cả Q5 và BF16:

~~~powershell
python -X utf8 -m modal run --detach -m deploy.reward_model.modal_app::download --include-reference
~~~

Chờ log kết thúc bằng:

~~~text
Pinned model artifacts downloaded and checksum verified.
~~~

Kiểm tra Volume:

~~~powershell
python -X utf8 -m modal volume ls puppeteer-nemotron-q5 /
~~~

Volume cần có Q5 GGUF, tokenizer, BF16 reference và download.json. Cần khoảng 200 GB khi Q5 và reference cùng tồn tại.

Nếu chỉ tải Q5:

~~~powershell
python -X utf8 -m modal run --detach -m deploy.reward_model.modal_app::download
~~~

Cách này chưa đủ để chạy probe_reference.

## 6. Probe và verification

Chạy tuần tự:

~~~powershell
python -X utf8 -m modal run -m deploy.reward_model.modal_app::probe_q5
python -X utf8 -m modal run -m deploy.reward_model.modal_app::probe_reference
~~~

probe_q5 dùng một A100 80 GB. probe_reference dùng hai A100 80 GB và chỉ phục vụ đối chiếu với Q5.

Sau khi cả hai probe hoàn thành:

~~~powershell
python -X utf8 -m modal run -m deploy.reward_model.modal_app::verify --min-cosine 0.99 --max-reward-error 1.0 --max-norm-relative-error 0.10
~~~

Chỉ deploy khi verification trả passed: true. Artifact nằm trong Modal Volume:

~~~text
/q5_probe.json
/reference_probe.json
/verification.json
~~~

## 7. Deploy reward model và tạo credential gọi API

~~~powershell
Set-Location "D:SideProjectchatdev_originalChatDev"
python -X utf8 -m modal deploy -m deploy.reward_model.modal_app
~~~

Modal sẽ in URL của RewardServer.score. Dùng nguyên URL này, không thêm /v1 hoặc /score.

Tạo Modal Proxy Token trong workspace rồi đặt:

~~~powershell
$env:REWARD_MODEL_URL = "https://<reward-server-score-url>.modal.run"
$env:MODAL_KEY = "<proxy-token-id>"
$env:MODAL_SECRET = "<proxy-token-secret>"
~~~

Kiểm tra endpoint:

~~~powershell
python -X utf8 -m deploy.reward_model.smoke_test
~~~

Response hợp lệ có reward hữu hạn, hidden state đúng 8.192 chiều, input_tokens từ 1 đến 4.096 và provenance của model. Request đầu sau khi scale về 0 có thể chậm vì cold start; read timeout hiện là 600 giây.

## 8. Cấu hình Puppeteer

Trong puppeteer/config/global.yaml, giữ reward remote:

~~~yaml
model_weight_path: null

reward_model:
  backend: remote
  endpoint_url: ""
  connect_timeout_seconds: 10
  read_timeout_seconds: 600
  max_attempts: 3
~~~

Khi endpoint_url trống, client đọc REWARD_MODEL_URL.

max_input_chars giới hạn bản sao hội thoại gửi tới reward model. Khi context dài hơn 8.000 ký tự, client giữ nguyên lịch sử agent trong bộ nhớ nhưng cắt phần giữa của nội dung dài nhất trong bản request. Cách này tránh vượt trần 4.096 token do một agent sinh output quá dài.

Để giữ tool-use agents:

~~~yaml
external_tools_enabled: true

graph:
  max_parallel_paths: 4
  max_step_num: 2
~~~

Trong puppeteer/config/policy.json, để policy cập nhật trọng số:

~~~json
"training": {
  "loading": false,
  "training": true
}
~~~

device.type chỉ điều khiển policy network local. Có thể dùng cpu; reward model vẫn chạy trên GPU Modal.

## 9. Dữ liệu và split

| Dataset | File nguồn |
|---|---|
| MMLU-Pro | data/MMLU-Pro/test.parquet và validation.parquet |
| GSM-Hard | data/GSM-Hard/test.parquet |
| SRDD | data/SRDD/SRDD.csv |
| CW | data/CW/creative_writing.jsonl |

Manifest seed 42 nằm trong puppeteer/data/splits. Cả bốn dataset đọc nguồn và chỉ số từ manifest: train → train, validation → dev, test → final. Các loader giữ nguyên thứ tự split, kiểm tra seed/số lượng/index và áp dụng data_limit sau khi chọn split. Không shuffle lại hoặc fallback sang toàn bộ nguồn khi thiếu split.

Kiểm tra credential trước run:

~~~powershell
$required = @("HF_TOKEN", "OPENROUTER_API_KEY", "REWARD_MODEL_URL", "MODAL_KEY", "MODAL_SECRET")
$required | ForEach-Object {
  [PSCustomObject]@{
    Name = $_
    Present = [bool](Get-Item -Path ("Env:" + $_) -ErrorAction SilentlyContinue)
  }
}
~~~

Tất cả phải hiện Present = True.

# Phần 2 — Chạy các chế độ và dataset

## 1. Cú pháp chung

Mọi benchmark chạy từ thư mục puppeteer:

~~~powershell
Set-Location "D:SideProjectchatdev_originalChatDevpuppeteer"
~~~

~~~text
python -X utf8 main.py <TASK> <MODE> --data_limit <N> --seed 42 --personas "personas/personas.jsonl"
~~~

Task hợp lệ: MMLU-Pro, gsm-hard, SRDD và CW.

Mode CLI hợp lệ: train, validation và test.

Mode chọn split cho cả bốn dataset và đặt tên thư mục result/checkpoint. Nó không tự bật hoặc tắt cập nhật policy; hành vi đó do config/policy.json -> training.training quyết định.

## 2. Train MMLU-Pro với đúng 200 mẫu

Đặt training.training thành true và training.loading thành false, sau đó chạy:

~~~powershell
python -X utf8 main.py MMLU-Pro train --data_limit 200 --seed 42 --personas "personas/personas.jsonl"
~~~

Loader sẽ:

1. Đọc data/splits/mmlu_pro_seed42.json.
2. Đọc data/MMLU-Pro/test.parquet.
3. Lấy đúng 200 row index trong splits.train và giữ nguyên thứ tự.
4. Chỉ gửi 200 câu hỏi này vào agent graph.
5. Ghi question_id vào result.

Smoke run một mẫu:

~~~powershell
python -X utf8 main.py MMLU-Pro train --data_limit 1 --seed 42 --personas "personas/personas.jsonl"
~~~

--data_limit nhỏ hơn 200 lấy phần đầu split. Giá trị lớn hơn 200 vẫn chỉ có tối đa 200 mẫu.

## 3. MMLU-Pro validation và test

Validation đọc các chỉ số splits.dev trong mmlu_pro_seed42.json (140 mẫu), từ nguồn khai báo trong manifest:

~~~powershell
python -X utf8 main.py MMLU-Pro validation --data_limit 20 --personas "personas/personas.jsonl"
~~~

Test đọc các chỉ số splits.final trong mmlu_pro_seed42.json (2.000 mẫu):

~~~powershell
python -X utf8 main.py MMLU-Pro test --data_limit 100 --personas "personas/personas.jsonl"
~~~

validation → dev và test → final; official validation.parquet được giữ trên đĩa nhưng không được hai mode này chọn. Truyền --seed để chọn manifest tương ứng; mặc định 42.

## 4. GSM-Hard

~~~powershell
python -X utf8 main.py gsm-hard validation --data_limit 20 --personas "personas/personas.jsonl"
python -X utf8 main.py gsm-hard test --data_limit 100 --personas "personas/personas.jsonl"
~~~

GSM-Hard đọc gsm_hard_seed42.json: train 200 mẫu, validation/dev 100 mẫu, test/final 909 mẫu. Nguồn và thứ tự mẫu lấy từ manifest; không shuffle.

## 5. SRDD

~~~powershell
python -X utf8 main.py SRDD validation --data_limit 20 --personas "personas/personas.jsonl"
python -X utf8 main.py SRDD test --data_limit 100 --personas "personas/personas.jsonl"
~~~

SRDD đọc srdd_seed42.json: train 200 mẫu, validation/dev 100 mẫu, test/final 790 mẫu. Task có req=code để agent lưu mã nguồn vào code_path cho bộ chấm điểm. Giữ external_tools_enabled: true để Python Tool Agent hoạt động.

## 6. CW

~~~powershell
python -X utf8 main.py CW validation --data_limit 20 --personas "personas/personas.jsonl"
python -X utf8 main.py CW test --data_limit 50 --personas "personas/personas.jsonl"
~~~

CW đọc cw_seed42.json: train 80 mẫu, validation/dev 20 mẫu, test/final 50 mẫu. Task có req=text để agent lưu câu trả lời vào tệp văn bản cho bộ chấm điểm.

## 7. Mode train cho dataset khác

CLI cho phép:

~~~powershell
python -X utf8 main.py gsm-hard train --data_limit 200 --seed 42
python -X utf8 main.py SRDD train --data_limit 200 --seed 42
python -X utf8 main.py CW train --data_limit 80 --seed 42
~~~

Cả ba lệnh đều đọc splits.train và nhận --seed. data_limit lớn hơn kích thước split chỉ trả về các mẫu có trong split; không lấy thêm từ dev/reference/probe/final.

## 8. Output, log và checkpoint

Kết quả:

~~~text
results<TASK>_<MODE>~~~

Ví dụ:

~~~text
resultsMMLU-Pro_trainMMLU-Pro_train.jsonl
resultsMMLU-Pro_validationMMLU-Pro_validation.jsonl
resultsgsm-hard_testgsm-hard.jsonl
resultsSRDD_validationsrdd.jsonl
resultsCW_testcw.jsonl
~~~

Log:

~~~text
logs<TASK><timestamp>~~~

Log có thể gồm meta.log, model_query.log, train.log, path logs, graph HTML và workflow PNG.

Checkpoint:

~~~text
checkpoint<TASK>_<MODE>policy_net_<timestamp>.pt
~~~

Khi chạy lại cùng task/mode, result hiện được mở ở chế độ ghi mới và có thể ghi đè file cũ. Sao lưu thư mục result trước khi chạy lại. Checkpoint có timestamp nên không ghi đè checkpoint cũ.

MMLU-Pro hỗ trợ resume bằng --checkpoint và --data_start. Chương trình chỉ append
khi file result có đúng số dòng data_start và toàn bộ ID là prefix chính xác của
split/seed. Checkpoint phải chứa policy và optimizer tương ứng với các dòng đã hoàn
thành. Các dataset khác chưa hỗ trợ data_start.

Ví dụ resume MMLU-Pro từ 72/200:

~~~powershell
python -X utf8 main.py MMLU-Pro train --data_start 72 --data_limit 128 --seed 42 --checkpoint "checkpoint/MMLU-Pro_train/policy_net_20260915_042325.pt" --personas "personas/personas.jsonl"
~~~

## 9. Ghi log console của run dài

~~~powershell
python -X utf8 main.py MMLU-Pro train --data_limit 200 --seed 42 --personas "personas/personas.jsonl" 2>&1 | Tee-Object -FilePath "mmlu_pro_train_seed42.log"
~~~

Trong Modal dashboard, container có thể scale về 0 sau khoảng 300 giây không có request. Request sau đó tạo cold start và load model lại từ Volume.

## 10. Checklist trước run 200 mẫu

- Python 3.11 64-bit và đúng puppeteer_env.
- 33 test CPU hiện có chạy thành công.
- Provider smoke test --live thành công cho cả sáu model.
- verification.json có passed: true.
- Reward smoke test trả hidden state 8.192 chiều.
- HF_TOKEN, OPENROUTER_API_KEY, REWARD_MODEL_URL, MODAL_KEY và MODAL_SECRET có trong cùng PowerShell.
- global.yaml dùng reward_model.backend: remote.
- policy.json có training.training: true.
- Pool là personas/personas.jsonl.
- external_tools_enabled: true.
- data/splits/mmlu_pro_seed42.json tồn tại.
- Tài khoản Hugging Face, OpenRouter và Modal còn đủ quota/credit.
