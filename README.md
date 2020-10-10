# Using YOLOv4 to detect objects in live stream youtube

## Prerequisites

- Conda (Python 3.8.5)
- CUDA 10+

## Setup

```sh
sh setup.sh
```

## Run the Demo

```sh
conda activate vizyal
python src/main.py --url <"YOUTUBE_LIVE_STREAM_URL">
```

If you want to choose the quality of video, add **--quality**. Example:
```sh
python src/main.py --url <"YOUTUBE_LIVE_STREAM_URL"> --quality 1080
```