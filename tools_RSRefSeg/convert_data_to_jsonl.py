import json
import os
import pickle
from tqdm import tqdm

data_folder = '/mnt/dataset/cky_data/RRSIS-D'
# image data: images/rrsisd/JPEGImages
# annotation data: images/rrsisd/ann_split
# instance segmentation data: rrsisd/instances.json
# refs(unc).p data: rrsisd/refs(unc).p

ins_data = json.load(open(os.path.join(data_folder, 'rrsisd/instances.json')))
refs_data = pickle.load(open(os.path.join(data_folder, 'rrsisd/refs(unc).p'), 'rb'))

'''
refs_data[0] data example:
{'image_id': 2934, 'split': 'train', 
'sentences': [{'tokens': ['The', 'baseball', 'field', 'on', 'the', 'top'], 
'raw': 'The baseball field on the top', 'sent_id': 2934, 'sent': 'The baseball field on the top'}], 
'file_name': '02934.jpg', 'category_id': 4, 'ann_id': 2934, 'sent_ids': [2934], 'ref_id': 2934}
'''

'''
ins_data['categories'] data example:
[{'name': 'Expressway-Service-area', 'id': 0}, {'name': 'Expressway-toll-station', 'id': 1}, {'name': 'airplane', 'id': 2}, {'name': 'airport', 'id': 3}, {'name': 'baseballfield', 'id': 4}, {'name': 'basketballcourt', 'id': 5}, {'name': 'bridge', 'id': 6}, {'name': 'chimney', 'id': 7}, {'name': 'dam', 'id': 8}, {'name': 'golffield', 'id': 9}, {'name': 'groundtrackfield', 'id': 10}, {'name': 'harbor', 'id': 11}, {'name': 'overpass', 'id': 12}, {'name': 'ship', 'id': 13}, {'name': 'stadium', 'id': 14}, {'name': 'storagetank', 'id': 15}, {'name': 'tenniscourt', 'id': 16}, {'name': 'trainstation', 'id': 17}, {'name': 'vehicle', 'id': 18}, {'name': 'windmill', 'id': 19}]
'''

'''
ins_data['images'] data example:
{'file_name': '06000.jpg', 'height': 800, 'width': 800, 'id': 6000}
'''

'''
ins_data['annotations'] data example:
{'bbox': [588, 592, 633, 623], 'categories_id': 18, 'id': 6000, 'image_id': 6000, 'segmentation': [{'size': [800, 800], 'counts': 'XXXX'}], 'area': 279213}
'''

ann_data = []
category_id_to_name = {category['id']: category['name'] for category in ins_data['categories']}
ann_id_to_ann = {ann['id']: ann for ann in ins_data['annotations']}

for refs_item in tqdm(refs_data):
	split = refs_item['split']  # train, val, test
	image_id = refs_item['image_id']
	assert len(refs_item['sentences']) == 1, f'len(refs_item["sentences"])={len(refs_item["sentences"])}'
	sent = refs_item['sentences'][0]['sent']
	file_name = refs_item['file_name']
	category_id = refs_item['category_id']
	ann_id = refs_item['ann_id']
	ann_item = ann_id_to_ann[ann_id]
	assert category_id == ann_item['categories_id'], f'category_id={category_id}, ann_item["categories_id"]={ann_item["categories_id"]}'
	assert image_id == ann_item['image_id'], f'image_id={image_id}, ann_item["image_id"]={ann_item["image_id"]}'
	bbox = ann_item['bbox']
	segmentation = ann_item['segmentation']
	area = ann_item['area']
	ann_data.append({
		'split': split,
		'image_id': image_id,
		'sent': sent,
		'file_name': file_name,
		'category_id': category_id,
		'category_name': category_id_to_name[category_id],
		'ann_id': ann_id,
		'bbox': bbox,
		'segmentation': segmentation,
		'area': area
	})

split_set = set([item['split'] for item in ann_data])
print(f'split_set: {split_set}')
os.makedirs('datainfo', exist_ok=True)
# save as jsonl file
for split in split_set:
	with open(f'datainfo/rrsisd_{split}.jsonl', 'w') as f:
		for item in ann_data:
			if item['split'] == split:
				f.write(json.dumps(item) + '\n')
