# Korean Audio Dataset API

## Setup
```bash
pip install -r requirements.txt
```

## Run locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Test locally
In a second terminal:
```bash
python test_client.py
```

## Expose to the internet (required for submission)
```bash
ngrok http 8000
# or
cloudflared tunnel --url http://localhost:8000
```

Submit the printed public URL + `/analyze`, e.g.:
`https://your-tunnel-url.ngrok-free.app/analyze`

Keep the server + tunnel running until grading is done.
