# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install uv for package management
RUN pip install uv

# Copy the requirements file first to leverage Docker cache
COPY pyproject.toml .
# Install dependencies using uv
# Using --system to install globally in the container environment
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the application code
COPY nari_tts ./nari_tts
COPY app ./app

# Expose the port the app runs on
EXPOSE 7860

# Define the entrypoint for the container
ENTRYPOINT [\"python\", \"app/app.py\"]

# Default command (can be overridden, e.g., to specify repo-id)
# Example: CMD [\"--repo-id\", \"buttercrab/nari-tts\", \"--server-name\", \"0.0.0.0\"]
CMD [\"--help\"] # Show help by default if no args given 