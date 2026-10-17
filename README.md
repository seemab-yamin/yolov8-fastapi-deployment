# yolov8-model-deployment

A production-style REST API for real-time object detection using YOLOv8s. Built with FastAPI, containerized with Docker, and deployed via GitHub Actions to Docker Hub. Designed to reflect a real ML deployment pipeline: model serving, API design, and CI/CD, not just a demo.

## Curl Requests

Set your API key once before running any request:

```bash
export RUNPOD_API_KEY=your_api_key_here
export RUNPOD_ENDPOINT=https://n8w7mvnfkkbbgr.api.runpod.ai
```

**Health Check**
```bash
curl --request GET "$RUNPOD_ENDPOINT/ping" \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --header "accept: application/json"
```

**Object Detection**
```bash
curl --request POST "$RUNPOD_ENDPOINT/predict" \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --form "file=@/path/to/your/image.jpg"
```