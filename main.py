from PIL import Image
import os

# Define the path to your dataset
dataset_dir = 'datasetnum'  # Change this to your dataset folder path
output_dir = 'dataset'  # Change this to your desired output folder

# Make sure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Traverse through each class folder (0, 1, 2, ..., 9)
for class_folder in os.listdir(dataset_dir):
    class_folder_path = os.path.join(dataset_dir, class_folder)
    
    # Check if it's a directory (class folder)
    if os.path.isdir(class_folder_path):
        # Create an output folder for this class in the output directory
        output_class_folder = os.path.join(output_dir, class_folder)
        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        # Process each image in the class folder
        for filename in os.listdir(class_folder_path):
            # Only process image files
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Get the path of the image
                image_path = os.path.join(class_folder_path, filename)
                img = Image.open(image_path)

                # Flip the image horizontally (use Image.FLIP_TOP_BOTTOM for vertical flip)
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                # Save the flipped image to the output directory
                flipped_image_path = os.path.join(output_class_folder, filename)
                flipped_img.save(flipped_image_path)

                print(f'Flipped image {filename} in class {class_folder}')

print('Image flipping complete for all classes.')
