import matplotlib.pyplot as plt
import napari
import numpy as np
import os
import pandas as pd
import traceback
from aicsimageio import AICSImage
from aicspylibczi import CziFile
from datetime import datetime
from pathlib import Path
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.io import imread, imsave
from skimage.measure import label, regionprops_table, regionprops
from skimage.morphology import white_tophat, reconstruction, remove_small_objects, disk, binary_closing, binary_erosion, binary_dilation
from scipy import ndimage as ndi
from scipy.ndimage import median_filter, binary_fill_holes
from skimage.segmentation import watershed, clear_border, find_boundaries, relabel_sequential

# === CONFIGURABLE PARAMETERS ===

dapi_thresh = 40         # Threshold for blue/DAPI channel
foci_thresh = 50       # Threshold for foci channel
min_dapi_area = 50       # Minimum DAPI region size
min_foci_area = 3         # Minimum red/green particle size
z_slice = 15            # Z slice number desired for 3d images
dapi_channel = 2        # DAPI Channel Number (Check in ImageJ) Adjustable
foci_channel = 0        # Foci Channel Number (Check in ImageJ) Adjustable
green_channel = 1       # Green Channel Number (Check in ImageJ) Adjustable



def measure_foci(img_path, img, labeled_dapi, foci_channel, foci_thresh, green_channel, min_area, dapi_channel):

    filename = Path(img_path).name
    foci = img.get_image_data("ZYX", C=foci_channel)
    foci = np.max(foci, axis=0)

    green = img.get_image_data("ZYX", C=green_channel)
    green = np.max(green, axis=0)
    thresh_green = 20 # Adjustable
    green_mask = green > thresh_green
    green_mask = remove_small_objects(green_mask, min_size=25) # Adjustable

    foci_records = []

    h, w = labeled_dapi.shape

    labeled_foci_global = np.zeros_like(labeled_dapi, dtype=np.int32)
    large_mask_global = np.zeros_like(labeled_dapi, dtype=np.uint16)
    foci_offset = 0

    props = regionprops(labeled_dapi)
    to_remove = set()

    for prop in props:
        label_id = prop.label
        obj_mask = (labeled_dapi == label_id)
        obj_perimeter_mask = find_boundaries(obj_mask, mode='inner')
        perimeter_count = np.sum(obj_perimeter_mask)
        
        if perimeter_count == 0: 
            continue 

        coords = prop.coords  
        on_edge = np.sum((coords[:, 0] == 0) | (coords[:, 0] == h - 1) |
                        (coords[:, 1] == 0) | (coords[:, 1] == w - 1))
        
        percent_on_edge = (on_edge / prop.perimeter) * 100 if prop.perimeter > 0 else 0
        
        if percent_on_edge > 15: # Adjustable
            to_remove.add(prop.label)
            continue

        dilated_mask = binary_dilation(obj_mask)

        neighbor_ids = np.unique(labeled_dapi[dilated_mask])
        neighbor_ids = [n for n in neighbor_ids if n != 0 and n != label_id]

        for n_id in neighbor_ids:
            neighbor_mask = (labeled_dapi == n_id)
            shared_wall = np.logical_and(obj_perimeter_mask, binary_dilation(neighbor_mask))
            shared_count = np.sum(shared_wall)
            percent_shared = (shared_count / perimeter_count) * 100
            is_small = prop.area < 4000 # Adjustable

            if percent_shared > 15 and is_small: # Adjustable
                to_remove.add(label_id)
            elif percent_shared > 25: # Adjustable
                to_remove.add(label_id)


    kept_labels = [l for l in np.unique(labeled_dapi) if l != 0 and l not in to_remove]
    sequenced_labels = labeled_dapi.copy()
    to_remove_list = list(to_remove)
    mask_to_erase = np.isin(sequenced_labels, to_remove_list)
    sequenced_labels[mask_to_erase] = 0
    sequenced_labels, forward_map, inverse_map = relabel_sequential(sequenced_labels)
    final_props = regionprops(sequenced_labels)

    nucleus_summaries = []

    for prop in final_props:
        
        nucleus_id = prop.label
        minr, minc, maxr, maxc = prop.bbox

        green_crop = green_mask[minr:maxr, minc:maxc]
        overlap_count = np.sum(np.logical_and(prop.image, green_crop))
        percent_green = overlap_count / prop.area

        cyclin_status = "+" if percent_green > 0.1 else "-" # Adjustable

        if cyclin_status == "+":
            n_foci = "NA"
        else:

            foci_crop = foci[minr:maxr, minc:maxc]
            nucleus_mask_crop = prop.image

            footprint = disk(4)
            foci_enhanced = white_tophat(foci_crop, footprint)
            foci_vals_enhanced = foci_enhanced[nucleus_mask_crop]

            if foci_vals_enhanced.size > 0:
                t_foci = np.mean(foci_vals_enhanced) + (2 * np.std(foci_vals_enhanced)) # Adjustable
                t_foci = max(t_foci, 60) # Adjustable
            else:
                t_foci = 60 # Adjustable

            foci_mask = (foci_enhanced > t_foci) & nucleus_mask_crop
            
            #foci_mask = binary_dilation(foci_mask, disk(1))
            #foci_mask = binary_erosion(foci_mask, disk(1))
            foci_mask = binary_closing(foci_mask, disk(1))
            foci_mask = remove_small_objects(foci_mask, min_size=3) # Adjustable
            
            labeled_foci = label(foci_mask, connectivity=1)

            n_foci = labeled_foci.max()
            n_pixels = prop.area

            if labeled_foci.max() > 0:

                relabeled_foci = labeled_foci.copy()
                mask = relabeled_foci > 0
                relabeled_foci[mask] += foci_offset

                global_window = labeled_foci_global[minr:maxr, minc:maxc]
                global_window[mask] = relabeled_foci[mask]

                foci_offset += int(labeled_foci.max())

        nucleus_summaries.append({
            "Image_Name": filename,
            "Nucleus_ID": nucleus_id,
            "Cyclin2A +/-": cyclin_status,
            "Num_Foci": n_foci,
        })

    all_labels = np.unique(labeled_dapi)
    kept_labels = [l for l in all_labels if l != 0 and l not in to_remove]
    visual_labels = np.where(np.isin(labeled_dapi, kept_labels), labeled_dapi, 0)
    number_labels = np.zeros_like(visual_labels, dtype=np.int32)

    point_props = regionprops_table(sequenced_labels, properties=('label', 'centroid'))
    df = pd.DataFrame(point_props)
    points = df[['centroid-0', 'centroid-1']].values
    labels_text = [str(l) for l in df['label']]

    napari_dapi = img.get_image_data("ZYX", C=dapi_channel)
    napari_dapi = np.max(napari_dapi, axis=0)
    napari_foci = img.get_image_data("ZYX", C=foci_channel)
    napari_foci = np.max(napari_foci, axis=0)

    """viewer = napari.Viewer()
    viewer.add_image(napari_dapi, name="DAPI", colormap="blue", blending="additive")
    viewer.add_image(napari_foci, name="Foci", colormap="red", blending="additive")
    viewer.add_image(green_mask, name="Green", colormap="green", blending="additive")
    
    # label layers (global)
    viewer.add_labels(visual_labels, name="Nuclei Labels", opacity=0.35)
    viewer.add_labels(labeled_foci_global, name="Foci", opacity=0.6)

    points_layer = viewer.add_points(points, size=3)
    points_layer.face_color = 'transparent'
    points_layer.edge_color = 'white'
    points_layer.text = {
        'string': labels_text,
        'size': 8,
        'color': 'white',
        'anchor': 'center',
    }

    napari.run()"""
    
    group_name = czi_path.parent.name
    output_path = output_root / f"{group_name}_{czi_path.stem}_VALIDATION.jpg"
    save_validation_sheet(napari_dapi, napari_foci, sequenced_labels, labeled_foci_global, green_mask, df, output_path)
    
    return nucleus_summaries

def watershed_dapi(img, dapi_channel, dapi_thresh = 30, min_pixels=50, seed_distance=0,
                  smoothing_sigma=1.5, peak_threshold=5):
    
    dapi = img.get_image_data("ZYX", C=dapi_channel)
    dapi = np.max(dapi, axis=0)
    dapi = median_filter(dapi, size=(3, 3))
    dapi = gaussian(dapi, sigma=smoothing_sigma, preserve_range=True)

    binary = dapi > dapi_thresh

    #structuring_element = np.ones((3, 3))
    binary = binary_closing(binary, disk(1))
    #binary = binary_fill_holes(binary)

    pad_size = 20 
    padded = np.pad(binary, pad_size, mode='edge') 
    distance = ndi.distance_transform_edt(padded)
    distance = distance[pad_size:-pad_size, pad_size:-pad_size]

    filtered_dist = gaussian(distance, sigma=1.5, preserve_range=True)
    max_dist = np.max(filtered_dist)
    h = 0.05*max_dist # Adjustable
    h_maxima_dist = reconstruction(np.maximum(filtered_dist - h, 0), filtered_dist, method='dilation')
 
    footprint = np.ones((50, 50))# Adjustable (About 1.5x average radius)

    coordinates = peak_local_max(
        h_maxima_dist,
        footprint=footprint,
        labels=binary,
        min_distance=seed_distance,
        threshold_abs=peak_threshold
    )

    # --- Convert coordinates to markers array ---
    mask = np.zeros_like(distance, dtype=bool)
    if len(coordinates) > 0:
        mask[tuple(coordinates.T)] = True
    markers, _ = ndi.label(mask)

    # --- Run watershed ---
    labeled_mask = watershed(-h_maxima_dist, markers, mask=binary)
    
    print(f"Watershed produced {labeled_mask.max()} labels.")

    return labeled_mask


def save_validation_sheet(dapi, foci, labels, labeled_foci_global, green_mask, df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(dapi, cmap='gray')
    axes[0].imshow(foci, cmap='Reds', alpha=0.5)
    axes[0].imshow(green_mask, cmap='Greens', alpha=0.2)
    axes[0].set_title("Raw", fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(dapi, cmap='gray', alpha=0.5)
    axes[1].imshow(labels, cmap='nipy_spectral', alpha=0.3) # Color-coded nuclei

    foci_mask = np.ma.masked_where(labeled_foci_global == 0, labeled_foci_global)
    axes[1].imshow(foci_mask, cmap='prism', alpha=0.8) # High contrast for tiny foci
  
    for _, row in df.iterrows():
        axes[1].text(row['centroid-1'], row['centroid-0'], str(int(row['label'])), 
                     color='white', fontsize=8, ha='center', fontweight='bold')
    
    axes[1].set_title(f"Final Count: {len(df)} Nuclei", fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight') # Adjustable
    #plt.show() # Adjustable
    plt.close()

# === MAIN EXECUTION ===
if __name__ == "__main__":
    input_dir = input("Enter input directory path: ").strip('"')
    output_dir = input("Enter output directory path: ").strip('"')
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    summaries_by_folder = {}
    all_image_summaries = []


    for czi_path in input_root.rglob("*.czi"):
        folder_name = czi_path.parent.name

        img = AICSImage(czi_path)
        dapi = img.get_image_data("ZYX", C=dapi_channel)
        dapi = np.max(dapi, axis=0)
        dapi = white_tophat(dapi, disk(3))
        print(f"Loaded {czi_path}")
           
        try:  
            labeled_dapi = watershed_dapi(img, dapi_channel)

            df_summary = measure_foci(
                czi_path,
                img, 
                labeled_dapi,
                foci_channel=foci_channel,
                foci_thresh=foci_thresh,
                green_channel=green_channel,
                min_area=min_foci_area,
                dapi_channel=dapi_channel,
            )

            if folder_name not in summaries_by_folder:
                summaries_by_folder[folder_name] = []

            summaries_by_folder[folder_name].extend(df_summary)

        except Exception as e:
            print(f"❌ Error processing {czi_path}: {e}")
            traceback.print_exc()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    for folder_name, rows in summaries_by_folder.items():
        df_folder = pd.DataFrame(rows)
        summary_filename = f"{folder_name}_{timestamp}_summary.csv"
        summary_path = output_root / summary_filename
        
        df_folder.to_csv(summary_path, index=False)
        print(f"Saved summary for {folder_name} to {summary_path}")
