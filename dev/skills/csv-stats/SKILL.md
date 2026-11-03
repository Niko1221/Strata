---
name: csv-stats
description: Summarises a CSV file - row count, columns, and for every number column min, max, mean and empty cells - with the bundled script instead of reading the whole file into the context. Use it when someone asks about the contents, size, columns, ranges or quality of a .csv file, or wants a quick overview of tabular data.
compatibility: Python 3.8 or newer (standard library only)
---

# CSV statistics

Large CSV files do not fit the context, and counting by eye is wrong. Let the script do the arithmetic.

## Steps

1. Run the script on the file (the path is relative to this skill's base directory):

   ```
   python scripts/csv_stats.py <path-to-file.csv>
   ```

   It prints JSON: `rows`, `columns`, and per column `type` (number or text), `empty`, and for numbers `min`, `max`, `mean`.
   The command runs through the agent's shell, so Manual mode asks first and Plan mode does not run it.
2. Explain the result in plain words: what the file holds, which columns have gaps, which values look odd (a max far above the mean).
3. Read rows of the file itself only when the person asks about specific rows.

The same script is offered as the tool `csv-stats` (tools.json): its input is the path of the file.
