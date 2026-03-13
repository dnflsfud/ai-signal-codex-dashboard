\# AGENTS.md



This project builds an AI-signal-driven equity portfolio pipeline.



\## Project root

Working project directory:

C:\\Users\\westl\\PycharmProjects\\pythonProject\\venv\_vf\\machine\\re\_study\\c2\\ai\_signal\_codex



\## External source file

Use this Excel file as the primary input dataset:

C:\\Users\\westl\\PycharmProjects\\pythonProject\\venv\_vf\\machine\\re\_study\\RL\_Universe\_Data.xlsx



Do not assume the Excel file is inside the project root.

Read it from the absolute path above unless explicitly changed.



\## Core objective

Build a long-only portfolio process where:

1\. ML forecasts 20-business-day specific returns, not full returns.

2\. Common components are removed before modeling.

3\. Model outputs are converted to cross-sectional alpha z-scores.

4\. Final portfolio construction uses a constrained mean-variance optimizer.

5\. Rebalancing is weekly and retraining is quarterly.



\## Working rules

\- Use BusinessDays as the canonical trading calendar.

\- Do not introduce look-ahead bias.

\- Keep code modular and production-style.

\- Add logging, assertions, docstrings, and type hints.

\- Put assumptions in config.yaml.

\- Save intermediate and final outputs under results/.

\- Prefer robust fallbacks over fragile perfect-looking implementations.

\- Because the universe is small, avoid overfitting and document limitations clearly.



\## File handling rules

\- Read input data from the external Excel path above.

\- Write all generated code and outputs only inside this project root.

\- Do not modify the original Excel file.

\- If access to the external Excel path requires approval, ask for it explicitly in the app flow.



\## Expected files

\- load\_data.py

\- feature\_engineering.py

\- target\_builder.py

\- model\_train.py

\- optimizer.py

\- backtest.py

\- run\_pipeline.py

\- config.yaml

\- README.md

