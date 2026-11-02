# Nordstar Synthetic Corpus — Enterprise AI Economics Benchmark

## Purpose
A synthetic corporate document corpus for a controlled research experiment comparing task quality and economics across simple chat, retrieval-augmented generation (RAG), and bounded agentic workflows.

## Contents
- `corpus/`: 21 fictional policy, operational, and project documents.
- `manifest.csv`: filenames and intended roles.
- `benchmark_questions.csv`: three benchmark tasks and expected facts.
- `ground_truth.csv`: weighted criteria for answer scoring.

## Research and provenance note
Nordstar Industrial Systems GmbH and all policies, people, projects, and events in this corpus are fictional. The corpus was authored for this benchmark and should be described as researcher-created synthetic data in the methodology.

## Benchmark tasks
- TASK-01: For a EUR 75,000 purchase, the required approvals are department manager, Procurement, and Finance.
- TASK-02: Emergency purchase documentation must record the reason, amount, supplier, and circumstances requiring urgency, within five business days.
- TASK-03: Project Orion's pilot target is 30 September 2026; status is AMBER; analytics is three weeks behind because of data-quality issues. Use both the brief and the later status report.

## Scoring
Award each criterion's weight when satisfied; otherwise award zero. A task is successful only if its score is 1.0 and there is no material factual error. Record partial scores as a secondary quality measure. Fix the rubric before full runs and, where practical, evaluate outputs without knowing the model or architecture.

## Version and date handling
The Project Orion status report is dated 15 September 2026 and reports AMBER status. The postmortem dated 20 September explains a contributing cause but does not replace the current status report.

## Experimental cautions
- This corpus is small and synthetic; results may not generalize to real enterprise repositories.
- Three tasks are a pilot benchmark, not a broad evaluation. Treat results as exploratory.
- Keep the same questions across architectures and models for paired comparisons.
- Log all model calls, token usage, latency, failures, and provider-reported cost when available.
- Verify current provider pricing and exact model IDs before paid API calls.
- For multi-call agentic workflows, sum usage and cost across every model call.
- Report task success and cost per successful task. If no tasks succeed in a condition, cost per successful task is undefined.
- Check any generated citations against the source documents.
