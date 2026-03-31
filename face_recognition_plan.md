# Face Recognition for Time-In System

## Overview
Implementation plan for face recognition feature to enable automatic time-in when an employee's face is recognized.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Mobile App     │────▶│  PHP Backend     │────▶│  Python Face    │
│  (Expo/RN)      │◄────│  (existing)      │◄────│  Microservice   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
       │                         │
       └─────────────────────────┘
                   │
            ┌─────────────┐
            │  MySQL DB   │
            └─────────────┘
```

---

## Folder Structure

```
jajr_mobileapp/                    (existing)
├── app/
├── components/
│   ├── FaceRecognitionModal.tsx   (camera + face capture)
│   └── FaceEnrollmentModal.tsx    (one-time face setup)
├── services/
│   └── faceRecognitionService.ts  (API calls)
└── ...

face_recog_py/                         (Python microservice - ADJUSTED: same repo)
├── app.py                              (FastAPI entry point)
├── face_recognition_service.py         (face matching logic)
├── requirements.txt
├── Dockerfile                          (optional)
├── models/                             (cached face encodings)
│   └── .gitkeep
└── README.md
```

---

## Database Changes

```sql
-- Add to employees table
ALTER TABLE employees 
ADD COLUMN face_embedding TEXT NULL,        -- JSON array of 128-dimension encoding
ADD COLUMN face_image_url VARCHAR(255) NULL, -- Reference photo URL
ADD COLUMN face_enrolled_at TIMESTAMP NULL;    -- When face was enrolled
```

---

## Python Face Service Endpoints

### 1. `POST /enroll`
Store face embedding for an employee during first-time setup.

**Request:**
```json
{
  "employee_id": 123,
  "image": "base64_encoded_image"
}
```

**Response:**
```json
{
  "success": true,
  "face_embedding": [0.123, -0.456, ...],  // 128 dimensions
  "message": "Face enrolled successfully"
}
```

### 2. `POST /verify`
Match captured face against all enrolled employees.

**Request:**
```json
{
  "image": "base64_encoded_image",
  "threshold": 0.6  // optional, default 0.6
}
```

**Response:**
```json
{
  "success": true,
  "matched": true,
  "employee_id": 123,
  "confidence": 0.85,  // 0.0 to 1.0
  "message": "Face matched"
}
```

### 3. `POST /verify-specific`
Verify face matches a specific employee (for re-verification).

**Request:**
```json
{
  "employee_id": 123,
  "image": "base64_encoded_image"
}
```

---

## PHP Backend API Changes

### New Endpoints

#### `POST /enroll_face_api.php`
```php
// Flow:
// 1. Receive image from mobile app
// 2. Send to Python service /enroll
// 3. Store face_embedding in database
// 4. Return success to app
```

#### `POST /verify_face_api.php`
```php
// Flow:
// 1. Receive image from mobile app
// 2. Send to Python service /verify
// 3. If matched, proceed with time-in
// 4. Return result to app
```

#### `GET /check_face_enrollment.php`
Check if employee has enrolled their face.

---

## Mobile App Components

### 1. FaceRecognitionModal

```typescript
interface FaceRecognitionModalProps {
  visible: boolean;
  onClose: () => void;
  onFaceRecognized: (employeeId: number) => void;
  mode: 'timeIn' | 'verification';
}
```

**Features:**
- Full-screen camera preview
- Face detection overlay (green box when face detected)
- Auto-capture when face is centered and stable
- Liveness detection (blink or smile check)
- Loading state during verification
- Error handling for no match / poor lighting

### 2. FaceEnrollmentModal

```typescript
interface FaceEnrollmentModalProps {
  visible: boolean;
  employeeId: number;
  onClose: () => void;
  onEnrolled: () => void;
}
```

**Features:**
- Guide user through face enrollment
- Multiple angle captures (optional)
- Preview captured image
- Confirm/retake flow

### 3. Integration with Existing Time-In

Modify `handleSelfTimeIn()` in `home.tsx`:

```typescript
const handleSelfTimeIn = useCallback(async () => {
  // 1. Check if face enrolled
  const faceStatus = await ApiService.checkFaceEnrollment(currentUser.userId);
  
  if (!faceStatus.enrolled) {
    // Show enrollment modal for first-time users
    setFaceEnrollmentVisible(true);
    return;
  }
  
  // 2. Show face recognition modal
  setFaceRecognitionVisible(true);
}, [currentUser]);

// 3. On face recognized, proceed with actual time-in
const onFaceRecognized = async (employeeId: number) => {
  setFaceRecognitionVisible(false);
  await performSelfTimeIn(branch);  // existing time-in logic
};
```

---

## Dependencies

### Mobile App
```bash
# Already have expo-camera, need face detection:
npx expo install expo-face-detector
```

### Python Service
```txt
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
face-recognition==1.3.0
Pillow==10.1.0
numpy==1.26.2
python-multipart==0.0.6
```

---

## Security Considerations

1. **Liveness Detection**: Ensure person is real, not a photo
2. **Threshold Tuning**: Set appropriate confidence threshold (0.6 recommended)
3. **Image Storage**: Don't store raw face images, only embeddings
4. **HTTPS**: All API calls must be encrypted
5. **Rate Limiting**: Prevent brute force attacks on face verification

---

## Implementation Checklist

### Phase 1: Python Service
- [x] Create `face_recog_py/` folder structure
- [x] Create `requirements.txt` (OpenCV - no heavy ML dependencies)
- [x] Setup virtual environment
- [x] Install dependencies (OpenCV + scikit-learn)
- [x] Implement `app.py` with FastAPI endpoints
- [x] Implement `face_recognition_service.py` (OpenCV-based)
- [x] Test with /health endpoint
- [ ] Test with sample images
- [ ] Create Dockerfile (optional)

### Phase 2: Backend APIs
- [x] Add face columns to database
- [x] Create `enroll_face_api.php`
- [x] Create `verify_face_api.php`
- [x] Create `check_face_enrollment.php`
- [ ] Test integration with Python service

### Phase 3: Mobile App
- [x] Create `FaceRecognitionModal` component
- [x] Create `FaceEnrollmentModal` component
- [x] Add face service API methods (`faceRecognitionService.ts`)
- [ ] Integrate with existing time-in flow (home.tsx)
- [ ] Add UI elements (Face Time In button)

### Phase 4: Testing
- [ ] Test face enrollment
- [ ] Test face recognition in good lighting
- [ ] Test with poor lighting
- [ ] Test with glasses/mask
- [ ] Test liveness detection
- [ ] Test edge cases (no face, multiple faces)

---

## Quick Start Commands

```bash
# 1. Setup Python service
cd jajr_face_service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run Python service
uvicorn app:app --host 0.0.0.0 --port 5000 --reload

# 3. Test endpoints
curl -X POST http://localhost:5000/verify \
  -H "Content-Type: application/json" \
  -d '{"image": "base64...", "threshold": 0.6}'
```

---

## Notes for Tomorrow

1. Python service uses **face_recognition** library which requires dlib
2. On macOS: `brew install cmake` before pip install
3. On Ubuntu/Debian: `apt-get install cmake libopenblas-dev liblapack-dev libjpeg-dev`
4. First run will download face recognition models (~100MB)
5. Consider running Python service on same server as PHP backend for low latency

---

## Alternative: Cloud Option (Future)

If self-hosting becomes problematic, consider:
- **AWS Rekognition**: $0.001 per face verification
- **Azure Face API**: Free tier 30,000 transactions/month
- **Google Vision API**: Face detection + custom model

Cloud APIs eliminate server maintenance but add ongoing cost.
