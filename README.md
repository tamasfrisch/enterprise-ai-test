# The Economics of Enterprise AI Tasks

## 1. Research objective

This study compares the cost, performance, and task quality of hosted large language models when used in three enterprise AI application architectures: simple chat, retrieval-augmented generation (RAG), and agentic workflows.

## 2. Research questions

RQ1. How does the cost of completing enterprise AI tasks differ across simple chat, RAG, and agentic workflows?

RQ2. How do hosted models differ in cost and task quality across these architectures?

RQ3. To what extent are cost differences associated with token consumption and the number of model calls?

## 3. Experimental design

- Deployment: hosted model APIs only.
- Providers under consideration: OpenAI, Anthropic, and Google.
- Architectures: simple chat, RAG, and bounded agentic workflow.
- Initial benchmark: nine tasks.
- Repetitions: three per model-task combination.
- Planned main experiment: 81 executions, subject to API availability and budget.
- Corpus: synthetic corporate documents for Nordstar Industrial Systems GmbH.

## 4. Measurements

For each execution, collect where available:

- Model and architecture identifiers.
- Question ID and repetition number.
- Input and output tokens.
- Number of model calls.
- End-to-end latency.
- API cost.
- Model response.
- Task success score.

## 5. Quality evaluation

Answers will be evaluated against predefined expected answers and task-specific criteria.

A successful answer must satisfy the required factual criteria and must not contain material unsupported claims.

The evaluation procedure will be documented before the main experiment.

## 6. Cost calculation

API cost will be calculated from the actual input and output token usage and the applicable price for the selected model.

For multi-call workflows, the costs of all model calls will be included.

Cost per successful task will be calculated as total API cost divided by the number of successful executions. If no executions succeed, this metric will be reported as undefined.

## 7. Scope and limitations

This is a small, controlled exploratory study. Its results should not be generalized to all enterprise AI applications without further testing.

The study measures API usage costs rather than the full total cost of ownership of an enterprise AI system.

Self-hosted models are excluded because the available hardware does not support a direct self-hosting experiment within the current timeframe.

## 8. Reproducibility

The study will retain the benchmark questions, expected answers, model identifiers, pricing assumptions, retrieval configuration, prompts, raw outputs, and analysis scripts..
