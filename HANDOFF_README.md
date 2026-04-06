# 项目交接说明（给 Project 3 前端同学）

## 项目简介

这是本项目的后端服务，名称可以理解为 **SLM Expert Committee Backend**。  
它的职责是接收前端提交的 AI Agent 信息，然后调用 3 个“评审模块（Judge）”完成风险评估，最后返回给前端展示结果。

当前系统结构如下：

- **Judge 1**：使用本地 **Llama 3.2 3B LoRA 权重**
- **Judge 2**：使用本地 **Llama 3.2 3B LoRA 权重**
- **Judge 3**：**不使用本地模型权重**，而是通过 **Gemini API** 进行评估

前端同学可以把它理解成一个“多专家评审后端”，后端会输出：

- 每个 Judge 的单独结果
- 一个 `critique_round`，用于展示“评审讨论/分歧”
- 一个最终综合判断结果，供 Dashboard 展示

## 环境配置

建议你在项目根目录下按下面步骤准备环境。

### 1. 创建虚拟环境

```powershell
python -m venv .venv
```

### 2. 激活虚拟环境

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```powershell
pip install -r requirements.txt
```

### 4. 复制环境变量模板

```powershell
Copy-Item .env.example .env
```

### 5. 填写 Gemini API Key

打开 `.env` 文件，把下面这一项改成你自己的 key：

```env
GEMINI_API_KEY=your_google_api_key_here
```

说明：

- Judge 1 / Judge 2 依赖本地模型配置
- Judge 3 依赖 `GEMINI_API_KEY`
- 如果没有填写 `GEMINI_API_KEY`，Judge 3 会直接报错，无法正常评估

## 启动服务

在项目根目录运行下面这条命令：

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

启动成功后，后端地址是：

`http://127.0.0.1:8000`

## 接口对接

### Swagger 调试地址

启动后请直接打开：

`http://127.0.0.1:8000/docs`

这里可以看到所有接口，并且可以直接在线测试请求和响应，非常适合前端联调。

### 前端重点关注的返回内容

前端 Dashboard 对接时，请重点查看响应 JSON 中的以下内容：

- `critique_round`
- 最终判定结果

特别提醒：

- 项目讨论里有时会把最终判定叫做 **`final_verdict`**
- **但当前后端实际返回中，没有单独名为 `final_verdict` 的字段**
- 目前最终结果实际位于：`results.synthesis_output`

也就是说，前端如果要拿“最终结论”，建议重点解析这些字段：

- `results.synthesis_output.final_recommendation`
- `results.synthesis_output.final_risk_tier`
- `results.synthesis_output.rationale`
- `results.synthesis_output.next_actions`

而如果你要做“专家讨论 / 辩论 / 分歧展示”界面，请重点解析：

- `results.critique_round.agreement_points`
- `results.critique_round.disagreement_points`
- `results.critique_round.arbitration_notes`
- `results.critique_round.reconciled_risk_score`
- `results.critique_round.reconciled_risk_tier`

## 你最常用的联调流程

建议你按下面顺序操作：

1. 先启动后端服务
2. 打开 `http://127.0.0.1:8000/docs`
3. 在 Swagger UI 里测试接口
4. 确认返回 JSON 结构
5. 再把这些字段接到 React Dashboard

## 额外提醒

- 请不要把真实 `.env` 文件提交或打包给别人
- 如果后续后端部署到服务器，前端请把 API Base URL 放到环境变量里，不要写死 `localhost`
- 如果你联调时发现 Judge 3 报 Gemini key 错误，优先检查 `.env` 是否正确填写

如果你只是做前端联调，最重要的两件事是：

- 服务能跑起来
- 你能从 `/docs` 看懂并测试返回 JSON

做到这两点，基本就可以顺利开始做 Dashboard 了。
