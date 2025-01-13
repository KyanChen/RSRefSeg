import cv2
import datasets
import random
import sys
sys.path.append(sys.path[0] + '/..')
import numpy as np
from pycocotools import mask

data_root = '/mnt/dataset/cky_data/RRSIS-D/images/rrsisd/JPEGImages'
dataset = datasets.load_dataset('json', data_files='datainfo/rrsisd_train.jsonl')['train']

for idx in range(len(dataset)):
	# idx = random.randint(0, len(dataset))
	item = dataset[idx]
	seg_mask = item['segmentation'][0]
	img_path = data_root + '/' + item['file_name']
	img = cv2.imread(img_path)
	ori_img = img.copy()
	seg_mask = mask.decode(seg_mask)
	seg_mask = cv2.resize(seg_mask, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
	img[seg_mask == 1] = img[seg_mask == 1] * 0.4 + 0.6 * np.array([0, 0, 255])
	sent = item['sent']
	print(sent)
	cv2.imshow('img', img)
	print(item['file_name'])
	key_pressed = cv2.waitKey(0)
	if key_pressed == ord('q'):
		break
	elif key_pressed == ord('s'):
		cv2.imwrite(f'{item["file_name"]}.png', img)
		cv2.imwrite(f'{item["file_name"]}_ori.png', ori_img)
		cv2.imwrite(f'{item["file_name"]}_seg.png', seg_mask * 255)
cv2.destroyAllWindows()
