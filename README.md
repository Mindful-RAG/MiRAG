# MiRAG

## Running the App

## Installation

The project uses [uv](https://github.com/astral-sh/uv)

1. **Normal Run**: Processes all data from scratch
   ```
   uv run mirag
   ```

2. **Continue From Previous Run**: Loads the existing output file and only processes the error entries
   ```
   uv run mirag --continue-from-file
   ```

3. **Process Errors Only**: Same as continue-from-file, just an alternative flag name
   ```
   uv run mirag --process-errors-only
   ```

The continuation process works as follows:

1. Loads the previous results from the specified output file
2. Identifies entries marked as "error" in that file
3. For each error entry, looks up the corresponding item in the dataset by its ID
4. Attempts to process only those failed items
5. Updates the original output file with the newly processed results
6. Updates the summary file with current completion statistics

This approach allows you to run the script multiple times, progressively filling in error entries until all items are successfully processed. This is especially useful for long-running processes where some items may fail due to temporary issues like API rate limits, network problems, or timeouts.

The completion percentage in the summary file helps you track how much of the dataset has been successfully processed, making it easier to monitor progress across multiple runs.
