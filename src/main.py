import argparse
import cv2
import youtube_dl
import darknet
import ultis
import caches

parser = argparse.ArgumentParser()
parser.add_argument("--url", help="Live youtube url")
parser.add_argument("--quality", help="quality of video", default=720)
args = parser.parse_args()

darknet_network, class_names, _ = darknet.load_network(
    "./cfg/vizyal.cfg",
    "./cfg/vizyal.data",
    "./models/vizyal.weights",
    batch_size=1
)
caches.darknet_width = darknet.network_width(darknet_network)
caches.darknet_height = darknet.network_height(darknet_network)

if __name__ == '__main__':
    ydl_opts = {}
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    info_dict = ydl.extract_info(args.url, download=False)
    formats = info_dict.get('formats', None)

    frame_id = 0

    for f in formats:
        if (f.get('height', None) == int(args.quality)):
            src_video = f.get('url', None)
            cap = cv2.VideoCapture(src_video)
            caches.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            caches.frame_heights = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if (frame_id % 60 == 0):
                    darknet_image = darknet.make_image(
                        caches.darknet_width, caches.darknet_height, 3)
                    resized_frame = cv2.resize(
                        frame, (caches.darknet_width, caches.darknet_height), interpolation=cv2.INTER_LINEAR)
                    darknet.copy_image_from_bytes(
                        darknet_image, resized_frame.tobytes())

                    instances = darknet.detect_image(
                        darknet_network, class_names, darknet_image)
                    ultis.write_data(frame, instances)
                frame_id += 1
