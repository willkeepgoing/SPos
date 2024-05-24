# yolo
# python detect.py --source /data/yukunping/classification-of-sitting-posture/two_9.jpg --save-txt --class 0

# 图片处理
# python frame.py --func get_image --video_path ./three.jpg

# 仅截帧
# python frame.py --func frame --frame_gap 5 --video_path /two.mp4

# 视频处理
python frame.py --func video --frame_gap 5 --video_path /test2.mp4

# 服务开启
# python3 httpserver.py