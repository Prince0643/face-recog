#!/bin/bash
# test_face_api.sh - Quick test script for face recognition API

echo "=== Face Recognition API Test ==="
echo ""

echo "1. Testing /health endpoint..."
curl -s http://localhost:5000/health | jq .
echo ""

echo "2. Getting enrolled employees..."
curl -s http://localhost:5000/enrolled-employees | jq .
echo ""

echo "3. Test enrollment (requires base64 image):"
echo "   curl -X POST http://localhost:5000/enroll \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"employee_id\": 1, \"image\": \"data:image/jpeg;base64,...\"}'"
echo ""

echo "4. Test verification (requires base64 image):"
echo "   curl -X POST http://localhost:5000/verify \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"image\": \"data:image/jpeg;base64,...\", \"threshold\": 0.6}'"
echo ""

echo "=== PHP API Test ==="
echo ""

echo "5. Check enrollment via PHP:"
curl -s "https://jajr.xandree.com/check_face_enrollment.php?employee_id=1" | jq .
