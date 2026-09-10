# Axis Biometric Service

High-performance face biometrics, multi-pose liveness composition, and 1:N face identification microservice built with **FastAPI**, **InsightFace (`buffalo_l`)**, and **Oracle REST Data Services (ORDS)**.

---

## Tech Stack

* **Python**: `>=3.12,<3.13` (Managed via Poetry)
* **Framework**: FastAPI (`0.115.0`) + Uvicorn
* **Computer Vision**: InsightFace (`0.7.3`), ONNX Runtime, OpenCV, NumPy

---

## Project Structure

```text
axis-biometric-service/
├── app/
│   ├── __init__.py
│   ├── config.py         # Pydantic Settings & Environment Variables
│   ├── engine.py         # InsightFace Model Wrapper & Vector Math
│   ├── main.py           # FastAPI Endpoints & Lifespan Management
│   └── schemas.py        # Pydantic Request/Response DTO Models
├── .env                  # Environment Configuration (Git-ignored)
├── pyproject.toml        # Poetry Dependency Manifest
└── README.md

```

---

## Setup & CLI Deployment Guide (Node.js/Deployment Worker)

Since your terminal environment only has Node.js installed by default, follow these exact CLI commands to install Python 3.12, set up Poetry, configure the environment, and run the service.

### Step 1: Install Python 3.12 & Poetry (CLI)

```bash
# Update system and install Python 3.12 (Ubuntu / Debian example)
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip

# Install Poetry package manager
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# Verify installations
python3 --version
poetry --version

```

### Step 2: Initialize & Run Service

```bash
# 1. Clone and enter repository
cd axis-biometric-service

# 2. Configure Poetry environment to use Python 3.12
poetry env use python3.12

# 3. Install dependencies
poetry install

# 4. Create .env configuration file
cat << EOF > .env
APP_NAME="Axis Pharma Biometric Engine"
APP_VERSION="1.0.0"
DEBUG=false
INSIGHTFACE_MODEL_NAME="buffalo_l"
DETECTION_SIZE=640
SIMILARITY_THRESHOLD=0.68
MAX_IMAGE_SIZE_MB=10
EXECUTION_PROVIDER="CPUExecutionProvider"
ORDS_FACE_LOGIN_URL="http://92.204.189.99:8080/ords/api/App/face-login"
EOF

# 5. Run development server
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```

---

## API Endpoints Reference

Base URL: `http://localhost:8000`

### 1. Health Check

* **GET** `/health`
* **Response**: Returns engine status, model load state, and execution provider.

### 2. Register Face (3-Pose Composition)

* **POST** `/api/v1/face/register`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
* `cnic`: String identifier
* `center_frame`: Image file
* `left_frame`: Image file
* `right_frame`: Image file

* **Response**: Returns averaged 512-D unit vector embedding.

### 3. Verify Face (1:1 Match)

* **POST** `/api/v1/face/verify`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
* `stored_embedding_json`: JSON array string (512 floats)
* `live_frame`: Image file snapshot

* **Response**: Returns cosine similarity score, pass boolean against threshold (`0.68`), and confidence tier (`HIGH`, `MEDIUM`, `LOW`, `REJECTED`).

### 4. Face Login (1:N Identification & Session Initializer)

* **POST** `/api/v1/face/login`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
* `live_frame`: Image file snapshot
* `role`: Target role (`CUSTOMER`, `ADMIN`, `GUEST`)
* `device_id`: Client hardware/device token
* `auth_token`: App session token
* `fcm_token`: Firebase push notification token

* **Behavior**: Fetches registered records from ORDS, computes cosine similarity across candidates, matches against a strict **High Tier Confidence threshold ($\ge 0.78$)**, and executes the ORDS POST session initialization callback.

---

## Node.js & Gateway Integration Example

```javascript
const FormData = require('form-data');
const axios = require('axios');
const fs = require('fs');

async function verifyLiveFace(storedEmbeddingArray, imageFilePath) {
  const form = new FormData();
  form.append('stored_embedding_json', JSON.stringify(storedEmbeddingArray));
  form.append('live_frame', fs.createReadStream(imageFilePath));

  const response = await axios.post('http://localhost:8000/api/v1/face/verify', form, {
    headers: { ...form.getHeaders() },
    maxContentLength: Infinity,
    maxBodyLength: Infinity
  });
  
  return response.data;
}

```

---

## Production Deployment

* **Model Weights**: InsightFace automatically downloads `buffalo_l` weights to `~/.insightface/models/` on initial startup. Ensure write permissions or container image bundling.
* **Workers**: Run with Uvicorn workers configured for available CPU cores:

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

```
