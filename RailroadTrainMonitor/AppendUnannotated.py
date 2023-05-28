import os
import pandas as pd
input_file = 'TrainingAnnotationsNegative.csv'
output_file = 'TrainingAnnotationsNegative.csv'
negative_label = 'Negative'
image_dir = 'TrainingImagesNegativeUnannotated'
image_ids = [os.path.splitext(filename)[0] for filename in os.listdir(image_dir)]
annotations = pd.read_csv(input_file)
new_annotations = pd.DataFrame({'ImageID': image_ids, 'LabelName': negative_label})
annotations = pd.concat([annotations, new_annotations], ignore_index=True)
annotations.to_csv(output_file, index=False)
