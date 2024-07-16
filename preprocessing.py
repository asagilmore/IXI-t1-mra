import os
import numpy as np
import logging
import ants
import ray
import argparse


def get_matched_ids(dirs, split_char="-"):
    '''
    returns a sorted set of all ids that exist in all given dirs
    '''
    files = [os.listdir(dir) for dir in dirs]
    file_ids = [[file.split(split_char)[0] for file in file_list] for
                file_list in files]
    sets = [set(file_id) for file_id in file_ids]
    matched = set.intersection(*sets)
    return sorted(matched)


def apply_zscore_and_scale(input_ants_image):
    '''
    Apply z-score normalization to input_data and scale the result to [0, 1].
    takes Numpy array as argument
    '''
    input_data = input_ants_image.numpy()
    mean = np.mean(input_data)
    std = np.std(input_data)
    zscore_data = (input_data - mean) / std
    scaled_data = (zscore_data - np.min(zscore_data)) / (np.max(
                   zscore_data) - np.min(zscore_data))

    normalized_image = ants.from_numpy(scaled_data,
                                       origin=input_ants_image.origin,
                                       spacing=input_ants_image.spacing,
                                       direction=input_ants_image.direction)

    return normalized_image


# This function preforms the following processing on an
# Resample the images to the same resolution
# Apply z-score standardization to both images
# Optionally apply n3 bias correction to either image
# if Resample_to_mask is True, the images are resampled to the mask,
# otherwise the mask is resampled to the resolution of the images
# The processed images are saved in out_folder
@ray.remote
def process_id(id, image_folder, mask_folder, out_folder,
               Resample_to_mask=True, N3_to_image=True, N3_to_mask=False):
    try:
        files_1 = [file for file in os.listdir(image_folder) if not
                   file.startswith('.')]
        files_2 = [file for file in os.listdir(mask_folder) if not
                   file.startswith('.')]

        file_1 = [file for file in files_1 if file.startswith(id)]
        file_2 = [file for file in files_2 if file.startswith(id)]
        if len(file_1) == 1 and len(file_2) == 1:
            #load images as float32
            img_1 = ants.image_read(os.path.join(image_folder, file_1[0]))
            img_2 = ants.image_read(os.path.join(mask_folder, file_2[0]))

            if Resample_to_mask:
                img_1 = ants.resample_image_to_target(img_1, img_2,
                                                      interp_type='linear')
            else:
                img_2 = ants.resample_image_to_target(img_2, img_1,
                                                      interp_type='linear')

            if N3_to_image:
                img_1 = ants.n4_bias_field_correction(img_1)
            if N3_to_mask:
                img_2 = ants.n4_bias_field_correction(img_2)

            # Apply z-score standardization
            img_1 = apply_zscore_and_scale(img_1)
            img_2 = apply_zscore_and_scale(img_2)

            # Save processed images using ANTsPy
            ants.image_write(img_1, os.path.join(out_folder, 'images',
                             f"{id}_image.nii.gz"))
            ants.image_write(img_2, os.path.join(out_folder, 'masks',
                             f"{id}_mask.nii.gz"))
            return True
        else:
            print(f"ID {id} has more than one file in one of the folders "
                    "skipping")
            return False
    except Exception as e:
        logging.error(f"Error processing ID {id}: {e}")


def process_folder(image_folder, mask_folder, out_folder, N3_to_image=False,
                   N3_to_mask=False, Resample_to_mask=True):

    id_list = get_matched_ids([image_folder, mask_folder])

    args_list = [(id, image_folder, mask_folder, out_folder, Resample_to_mask,
                  N3_to_image, N3_to_mask) for id in id_list]

    ray.init(num_cpus=32)
    results = ray.get([process_id.remote(*args) for args in args_list])
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing script')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--image_folder', type=str, help='Image folder')
    parser.add_argument('--mask_folder', type=str, help='Mask folder')

    args = parser.parse_args()

    process_folder(args.img_folder, args.mask_folder, args.out_dir,
                   N3_to_image=True,
                   N3_to_mask=False, Resample_to_mask=True)
