# AI Tools Usage

## AI Tools Used

- **ChatGPT** (OpenAI): Primary assistant for code structure, debugging, and LaTeX formatting
- **Claude** (Anthropic): Secondary assistant for code review and explanation verification

## How AI Assisted

### 1. Project Structure & Boilerplate
- Generated initial project structure (`main.py`, `src/` modules)
- Suggested modular architecture following course rulebook requirements
- Provided template code for data loading, model training, and evaluation pipelines

### 2. Debugging & Error Resolution
- Helped diagnose pandas merge errors and data type mismatches
- Suggested fixes for scikit-learn API usage (e.g., `random_state`, `TimeSeriesSplit`)
- Identified issues with feature scaling in Ridge regression

### 3. Learning New Libraries
- Explained `scikit-learn` best practices for time-based splits
- Guided implementation of `GridSearchCV` with temporal cross-validation
- Clarified `matplotlib` plotting for true vs predicted scatter plots

### 4. Code Review & Refactoring
- Suggested moving repeated preprocessing logic into reusable functions
- Recommended breaking long functions into testable units
- Improved variable naming for clarity (e.g., `X_train_scaled` vs `X`)

### 5. Documentation & LaTeX
- Drafted initial LaTeX table structures for model comparison
- Suggested phrasing improvements for methodology section
- Helped format bibliography entries in APA style
- Fixed LaTeX compilation errors (missing packages, figure placement)

### 6. Guidance on Best Practices
- Advised on avoiding data leakage (temporal split, no future info in training)
- Suggested using `migration_lag1` to capture persistence
- Recommended hyperparameter tuning strategies (trial-and-error vs grid search)

## My Contributions

### Research & Design
- **Defined the research question**: "Can housing and labour market conditions predict inter-cantonal migration?"
- **Selected the modeling strategy**: OLS → Ridge → RF → GB progression to show regularization benefits
- **Chose evaluation metrics**: R², RMSE, time-based train/test split

### Data Construction
- **Built the entire dataset from scratch**: 
  - Downloaded data from Swiss Federal Statistical Office (BFS/OFS)
  - Downloaded mortgage data from Swiss National Bank (SNB)
  - Matched variables across sources and years in Excel
  - Cleaned and converted to CSV
- **Designed feature engineering**: 
  - Constructed `shock_exposure = exposure_index × change_mrtgrate`
  - Defined lag structure for `migration_lag1`
  - Created log transformations and interactions

### Implementation & Validation
- **Wrote all final code**: AI provided templates, but I adapted, debugged, and validated every line
- **Ran all experiments locally**: Generated all results, figures, and tables on my machine
- **Hyperparameter tuning**: Iteratively tested max_depth, max_features, learning_rate via trial-and-error
- **Debugging**: Manually fixed bugs in data preprocessing, model pipelines, and plotting
- **Testing**: Wrote and maintained unit tests for data loading and preprocessing functions

### Writing & Reporting
- **Wrote the entire LaTeX report**: AI suggested formatting, but all content is mine
- **Interpreted all results**: Analysis of why Ridge > OLS, why GB > RF, feature importance discussion
- **Reviewed all AI suggestions**: Never copy-pasted without understanding; adapted to my project needs

## Declaration

**I understand all code in this project.** AI tools were used as **assistants** for:
- Learning syntax (e.g., how to use `TimeSeriesSplit`)
- Debugging errors (e.g., "why does my Ridge model return NaN?")
- Refactoring for clarity (e.g., breaking long functions)
- Formatting help (e.g., LaTeX table alignment)

**AI did NOT**:
- Write my research question or methodology
- Construct my dataset
- Run my experiments
- Interpret my results
- Make modeling decisions

All final code was written, reviewed, tested, and validated by me. I take full responsibility for the project's design, implementation, and results.

---

**Author**: Maxime Ducotterd  
**Date**: January 2025