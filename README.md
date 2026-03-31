# Face Recognition Service

Python microservice for face enrollment and verification using FastAPI and DeepFace library.

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

## Endpoints

- `POST /enroll` - Store face embedding for employee
- `POST /verify` - Match face against all enrolled employees
- `POST /verify-specific` - Verify face against specific employee
- `GET /health` - Health check

## Environment Variables

- `FACE_MATCH_THRESHOLD` - Matching threshold (default: 0.6)
- `PORT` - Server port (default: 5000)
