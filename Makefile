help:
	echo "help"
	echo "make mot17_s"
export YOLOX_DATADIR=/media/peter/share2507/AI/ultralytics_poetry/ByteTrack/datasets
mot17_s:
	python3 tools/track.py \
      -f exps/example/mot/yolox_s_mix_det.py \
      -c pretrained/bytetrack_s_mot17.pth.tar \
      -b 1 \
      -d 1 \
      --fp16 \
      --fuse

mot17_x:
	python3 tools/track.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse
