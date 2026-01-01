# AI Usage

## Tools used
- ChatGPT/Codex for coding assistance across the project.
- Claude AI for double-checking some code and explanations.

## How AI assisted
- Project structure and orchestration (entry point `main.py`, facade files `src/data_loader.py`, `src/models.py`, `src/evaluation.py`).
- Guidance on time-based splits, use of `migration_lag1`, avoiding leakage, and aligning with the course rulebook requirements.
- Drafting/refactoring model scripts (OLS, Ridge, Random Forest, Gradient Boosting, Decision Tree), utilities, and plotting helpers.
- Assistance on notebooks and exploratory analysis scripts, including feature engineering ideas and diagnostics.
- LaTeX report adjustments and formatting fixes (consistency checks, tables/figures, text rewrites).

## My contributions
- Co-wrote the code with AI assistance; a majority of the code was AI-generated, but I continuously adapted it to match the course structure and requirements.
- Designed the project end-to-end: research question, modeling strategy, and model selection (OLS, Ridge, Random Forest, Gradient Boosting, Decision Tree).
- Built and cleaned the dataset from Swiss Federal Statistical Office (OFS/BFS) and Swiss National Bank sources.
- Defined the feature engineering logic, including lag structure and shock/shock-exposure construction.
- Ran all experiments locally, validated outputs, and generated the final results/figures.
- Wrote and maintained tests, and manually corrected bugs when they appeared.
- Wrote the LaTeX report and final document structure, and manually adjusted scripts to make them testable.
- Reviewed, adapted, and debugged all AI-suggested code before integrating it; all results were produced and checked locally by me.
