# tds_project2_finalversion# Autonomous Agent Server

## Overview

This project implements an autonomous agent powered by LangGraph and LangChain, capable of performing web navigation, data analysis, media transcription, and code execution to solve data-oriented tasks. The agent operates within a stateful graph workflow and is exposed via a FastAPI server for task submission and status monitoring. It is optimized for speed and efficiency, with built-in tools for handling web content, PDFs, audio/video files, and Python-based computations.

The agent is designed to process tasks such as solving puzzles or extracting information from URLs, using a combination of browser automation (Playwright), API interactions, and local computation. It includes safeguards for stateless execution in sensitive operations and supports dynamic package installation for extended functionality.

## Features

- **Web Navigation and Interaction**: Tools for navigating URLs, clicking elements, typing text, scrolling, and capturing screenshots.
- **File Handling**: Reading PDFs, analyzing audio/video files via transcription (using OpenAI Whisper), and parsing data formats like CSV/JSON/XML.
- **Code Execution**: A safe, stateful Python REPL for data processing, visualization (Matplotlib/Seaborn), machine learning (Torch/Statsmodels), geospatial analysis (GeoPandas), and network analysis (NetworkX).
- **Dynamic Package Installation**: Ability to install additional Python packages on-the-fly if needed.
- **Task Management**: FastAPI endpoints for submitting tasks (with authentication via email and secret) and checking task status.
- **Optimization**: Rate limiting, non-interactive backends for visualizations, and resource blocking (e.g., images/CSS) for faster web loading.
- **Error Handling**: Smart hints for common execution errors and robust exception management.

## Requirements

- Python 3.10 or higher.
- Access to an OpenAI API key for models like GPT and Whisper.
- Environment variables for configuration (see below).
- Internet access for package installation and API calls.

## Installation

1. Clone the repository or download the script.

2. Install base dependencies (the script handles this automatically on first run, but you can run it manually):
   ```
   pip install fastapi uvicorn langgraph langchain langchain-openai langchain-community langchain-core playwright pandas numpy requests beautifulsoup4 pypdf openai nest-asyncio pillow matplotlib seaborn networkx torch statsmodels geopandas
   ```

3. Install Playwright browsers:
   ```
   playwright install chromium
   playwright install-deps
   ```

4. Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   MY_SECRET=your_secret_key
   MY_EMAIL=your_email_address
   ```

## Usage

### Running the Server

Execute the script to start the FastAPI server:
```
python agent_server.py
```
The server will run on `http://0.0.0.0:8000`.

### Endpoints

- **Health Check**: `GET /healthz`
  - Returns: `{"status": "ok"}`

- **Submit Task**: `POST /submit_task`
  - Request Body (JSON):
    ```
    {
      "email": "your_email_address",
      "secret": "your_secret_key",
      "url": "https://example.com/task-url"
    }
    ```
  - Response: `{"message": "OK", "task_id": "uuid-string"}`
  - Authentication: Must match `MY_EMAIL` and `MY_SECRET` from `.env`.
  - Behavior: Starts a background task to process the URL with the agent.

- **Check Status**: `GET /status/{task_id}`
  - Returns: Task status, logs, and result (e.g., `{"status": "finished", "logs": [...], "result": "..."}`).

### Agent Behavior

- The agent starts at the provided URL and uses tools to interact, analyze, and submit data.
- It persists state across tool calls for the same task.
- Tasks are designed to complete within a short time frame (e.g., 3 minutes), with iteration limits to prevent infinite loops.
- Upon completion, the agent checks for success conditions (e.g., `{"correct": true}` in responses) and logs progress.

## Configuration

- **Environment Variables**:
  - `OPENAI_API_KEY`: Required for OpenAI integrations.
  - `MY_SECRET`: Secret key for task submission authentication.
  - `MY_EMAIL`: Email for task submission authentication.

- **Customization**:
  - Modify the `REQUIRED_PACKAGES` list to include additional libraries.
  - Adjust the system prompt in `agent_reasoning_node` for task-specific instructions.
  - Extend tools by adding to `agent_tools` and binding them to the model.

## Troubleshooting

- **Package Installation Errors**: Ensure pip is up-to-date and you have write permissions in the Python environment.
- **Browser Issues**: If Playwright fails, verify Chromium is installed and check for firewall restrictions.
- **API Rate Limits**: The agent uses an in-memory rate limiter; adjust parameters in `InMemoryRateLimiter` if needed.
- **Task Failures**: Check logs in the status endpoint for details. Common issues include invalid URLs or authentication failures in submissions.

## Limitations

- The agent operates headlessly and may not handle complex JavaScript-heavy sites perfectly.
- Media transcription relies on OpenAI Whisper, which may incur costs.
- No persistent storage for tasks beyond in-memory `TASK_STORE`.
- Security: Tools like `python_interpreter` are sandboxed but should not be exposed to untrusted inputs.

## Contributing

Contributions are welcome. Please submit pull requests for bug fixes, new tools, or optimizations.

## License

This project is licensed under the MIT License. See the LICENSE file for details (if not present, assume standard open-source terms).
