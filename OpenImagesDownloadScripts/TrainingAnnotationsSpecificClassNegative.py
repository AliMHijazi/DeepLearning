import pandas as pd

chunksize = 10 ** 6

with open('ImageListTrainingNegative.txt', 'r') as f:
    image_ids = [line.strip().split('/')[-1] for line in f]

annotations = pd.DataFrame()
for i, chunk in enumerate(pd.read_csv('oidv6-train-annotations-bbox.csv', chunksize=chunksize)):
    print(f'Processing chunk {i + 1}')
    # Filter the annotations to only include the selected image IDs
    chunk = chunk[chunk['ImageID'].isin(image_ids)]
    annotations = pd.concat([annotations, chunk], ignore_index=True)

annotations.to_csv('TrainingAnnotationsNegative.csv', index=False)
print('Finished writing to .csv file')
