# The v4 emulation target as a container. Build context is the repo root so the
# server can import the shared env/ package:
#     docker build -f Target/Dockerfile -t v4-target .
#     docker run -p 5000:5000 v4-target
FROM python:3.12-slim

WORKDIR /app

# only the pieces the target needs at runtime
COPY Target/ ./Target/
COPY env/ ./env/
COPY telemetry/ ./telemetry/
COPY Target/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
# production WSGI server, not the Flask dev server
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--call", "Target.mock_server:create_app"]
