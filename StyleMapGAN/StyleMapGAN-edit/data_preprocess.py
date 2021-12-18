import os


print('>>> Start segmentation...')
os.system(f'cd face_parsing; python test.py --input ../images/image_result.png')
print('>>> End segmentation...')
