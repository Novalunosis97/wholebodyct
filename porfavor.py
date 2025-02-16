import os
os.environ["DISPLAY"] = ":99"

import streamlit as st
import os
import tempfile
import zipfile
import io
import torch
import numpy as np
from monai.bundle import ConfigParser, download
from monai.transforms import LoadImage, EnsureChannelFirst, Orientation, Compose
from skimage import measure
import vtk
import shutil
import time
import glob
from monai.data import ITKReader
import pydicom
import nibabel as nib

# Set page configuration
st.set_page_config(page_title="3D Whole Body CT Segmentation", layout="wide")

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .footer {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-header'>3D Whole Body CT Segmentation</h1>", unsafe_allow_html=True)
st.markdown("""
<p class='info-text'>
Upload a CT scan in NIfTI format (.nii or .nii.gz) or DICOM files (.dcm) to generate 3D models of segmented organs and structures.
The segmentation uses deep learning to identify over 100 different anatomical structures.
</p>
""", unsafe_allow_html=True)

# Import the visualization functions
def save_vtk_polydata(mesh, filename):
    # Ensure normals are properly calculated
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.ConsistencyOn()  # Ensure consistency of normals
    normals.SplittingOff()  # Disable splitting
    normals.AutoOrientNormalsOn()  # Automatically orient normals
    normals.Update()
    mesh_with_normals = normals.GetOutput()

    writer = vtk.vtkOBJWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh_with_normals)
    writer.Write()
    
def save_combined_obj(meshes, filename, color_map):
    """
    Save all meshes to a single OBJ file with material definitions for colors
    """
    # Create MTL file for materials
    mtl_filename = os.path.splitext(filename)[0] + '.mtl'
    with open(mtl_filename, 'w') as mtl_file:
        mtl_file.write("# Material definitions for CT segmentation\n\n")
        
        # Write material definitions
        for label, color in color_map.items():
            if label == 0:  # Skip background
                continue
            mtl_file.write(f"newmtl material_{label}\n")
            mtl_file.write(f"Ka {color[0]} {color[1]} {color[2]}\n")  # Ambient color
            mtl_file.write(f"Kd {color[0]} {color[1]} {color[2]}\n")  # Diffuse color
            mtl_file.write(f"Ks 0.1 0.1 0.1\n")  # Specular color
            mtl_file.write("d 1.0\n")  # Opacity
            mtl_file.write("illum 2\n\n")  # Illumination model
    
    # Create OBJ file with references to materials
    with open(filename, 'w') as obj_file:
        obj_file.write(f"mtllib {os.path.basename(mtl_filename)}\n\n")
        
        vertex_offset = 1  # OBJ indices start at 1
        
        # Process each mesh
        for label, mesh in meshes.items():
            if label == 0:  # Skip background
                continue
                
            # Get points and polygons from VTK mesh
            points = mesh.GetPoints()
            polys = mesh.GetPolys()
            
            # Write material for this segment
            obj_file.write(f"g segment_{label}\n")
            obj_file.write(f"usemtl material_{label}\n")
            
            # Write vertices
            for i in range(points.GetNumberOfPoints()):
                point = points.GetPoint(i)
                obj_file.write(f"v {point[0]} {point[1]} {point[2]}\n")
            
            # Write faces
            polys.InitTraversal()
            id_list = vtk.vtkIdList()
            while polys.GetNextCell(id_list):
                if id_list.GetNumberOfIds() == 3:  # Triangle
                    v1 = id_list.GetId(0) + vertex_offset
                    v2 = id_list.GetId(1) + vertex_offset
                    v3 = id_list.GetId(2) + vertex_offset
                    obj_file.write(f"f {v1} {v2} {v3}\n")
                    
            # Update vertex offset for next mesh
            vertex_offset += points.GetNumberOfPoints()
            
            # Add a newline between segments
            obj_file.write("\n")

def visualize_3d_multilabel(segmentation, output_directory, return_preview=False):
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if segmentation.ndim == 4 and segmentation.shape[0] == 1:
        segmentation = np.squeeze(segmentation, axis=0)

    if segmentation.ndim != 3:
        raise ValueError(f'Input volume should be a 3D numpy array. Got shape: {segmentation.shape}')

    segmentation = np.asarray(segmentation, dtype=np.uint8)

    # Define color map (abbreviated for space)
    color_map = {
        0: [0, 0, 0],              # Background - Black
        1: [0.8, 0.3, 0.3],        # Spleen - Light Red
        2: [0.6, 0.2, 0.2],        # Right Kidney - Medium Red
        3: [0.6, 0.2, 0.2],        # Left Kidney - Medium Red
        4: [0.7, 0.4, 0.3],        # Gallbladder - Light Brown
        5: [0.5, 0.1, 0.1],        # Liver - Dark Red
        6: [0.8, 0.6, 0.5],        # Stomach - Light Brown
        7: [0.9, 0.1, 0.1],        # Aorta - Bright Red
        8: [0.4, 0.6, 0.8],        # Inferior Vena Cava - Light Blue
        9: [0.4, 0.6, 0.8],        # Portal Vein and Splenic Vein - Light Blue
        10: [0.5, 0.3, 0.3],       # Pancreas - Medium Brown
        11: [0.7, 0.4, 0.4],       # Adrenal Gland Right - Medium Red
        12: [0.7, 0.4, 0.4],       # Adrenal Gland Left - Medium Red
        13: [0.4, 0.3, 0.3],       # Lung Upper Lobe Left - Dark Brown
        14: [0.4, 0.3, 0.3],       # Lung Lower Lobe Left - Dark Brown
        15: [0.4, 0.3, 0.3],       # Lung Upper Lobe Right - Dark Brown
        16: [0.4, 0.3, 0.3],       # Lung Middle Lobe Right - Dark Brown
        17: [0.4, 0.3, 0.3],       # Lung Lower Lobe Right - Dark Brown
        18: [0.3, 0.3, 0.3],       # Vertebrae L5 - Dark Grey
        19: [0.3, 0.3, 0.3],       # Vertebrae L4 - Dark Grey
        20: [0.3, 0.3, 0.3],       # Vertebrae L3 - Dark Grey
        21: [0.3, 0.3, 0.3],       # Vertebrae L2 - Dark Grey
        22: [0.3, 0.3, 0.3],       # Vertebrae L1 - Dark Grey
        23: [0.3, 0.3, 0.3],       # Vertebrae T12 - Dark Grey
        24: [0.3, 0.3, 0.3],       # Vertebrae T11 - Dark Grey
        25: [0.3, 0.3, 0.3],       # Vertebrae T10 - Dark Grey
        26: [0.3, 0.3, 0.3],       # Vertebrae T9 - Dark Grey
        27: [0.3, 0.3, 0.3],       # Vertebrae T8 - Dark Grey
        28: [0.3, 0.3, 0.3],       # Vertebrae T7 - Dark Grey
        29: [0.3, 0.3, 0.3],       # Vertebrae T6 - Dark Grey
        30: [0.3, 0.3, 0.3],       # Vertebrae T5 - Dark Grey
        31: [0.3, 0.3, 0.3],       # Vertebrae T4 - Dark Grey
        32: [0.3, 0.3, 0.3],       # Vertebrae T3 - Dark Grey
        33: [0.3, 0.3, 0.3],       # Vertebrae T2 - Dark Grey
        34: [0.3, 0.3, 0.3],       # Vertebrae T1 - Dark Grey
        35: [0.3, 0.3, 0.3],       # Vertebrae C7 - Dark Grey
        36: [0.3, 0.3, 0.3],       # Vertebrae C6 - Dark Grey
        37: [0.3, 0.3, 0.3],       # Vertebrae C5 - Dark Grey
        38: [0.3, 0.3, 0.3],       # Vertebrae C4 - Dark Grey
        39: [0.3, 0.3, 0.3],       # Vertebrae C3 - Dark Grey
        40: [0.3, 0.3, 0.3],       # Vertebrae C2 - Dark Grey
        41: [0.3, 0.3, 0.3],       # Vertebrae C1 - Dark Grey
        42: [0.4, 0.3, 0.3],       # Esophagus - Dark Brown
        43: [0.5, 0.4, 0.4],       # Trachea - Brown
        44: [1, 0, 0],             # Heart Myocardium - Red
        45: [1, 0.4, 0.4],         # Heart Atrium Left - Light Red
        46: [1, 0, 0],             # Heart Ventricle Left - Red
        47: [1, 0.4, 0.4],         # Heart Atrium Right - Light Red
        48: [1, 0, 0],             # Heart Ventricle Right - Red
        49: [0, 0, 1],             # Pulmonary Artery - Blue
        50: [0.5, 0.5, 1],         # Brain - Light Blue
        51: [0.6, 0.3, 0],         # Iliac Artery Left - Orange
        52: [0.6, 0.3, 0],         # Iliac Artery Right - Orange
        53: [0.3, 0.6, 0.6],       # Iliac Vein Left - Teal
        54: [0.3, 0.6, 0.6],       # Iliac Vein Right - Teal
        55: [0.5, 0.3, 0.3],       # Small Bowel - Light Brown
        56: [0.7, 0.2, 0.2],       # Duodenum - Light Red
        57: [0.2, 0.2, 0.7],       # Colon - Dark Blue
        58: [0.3, 0.3, 0.3],       # Rib Left 1 - Dark Grey
        59: [0.3, 0.3, 0.3],       # Rib Left 2 - Dark Grey
        60: [0.3, 0.3, 0.3],       # Rib Left 3 - Dark Grey
        61: [0.3, 0.3, 0.3],       # Rib Left 4 - Dark Grey
        62: [0.3, 0.3, 0.3],       # Rib Left 5 - Dark Grey
        63: [0.3, 0.3, 0.3],       # Rib Left 6 - Dark Grey
        64: [0.3, 0.3, 0.3],       # Rib Left 7 - Dark Grey
        65: [0.3, 0.3, 0.3],       # Rib Left 8 - Dark Grey
        66: [0.3, 0.3, 0.3],       # Rib Left 9 - Dark Grey
        67: [0.3, 0.3, 0.3],       # Rib Left 10 - Dark Grey
        68: [0.3, 0.3, 0.3],       # Rib Left 11 - Dark Grey
        69: [0.3, 0.3, 0.3],       # Rib Left 12 - Dark Grey
        70: [0.3, 0.3, 0.3],       # Rib Right 1 - Dark Grey
        71: [0.3, 0.3, 0.3],       # Rib Right 2 - Dark Grey
        72: [0.3, 0.3, 0.3],       # Rib Right 3 - Dark Grey
        73: [0.3, 0.3, 0.3],       # Rib Right 4 - Dark Grey
        74: [0.3, 0.3, 0.3],       # Rib Right 5 - Dark Grey
        75: [0.3, 0.3, 0.3],       # Rib Right 6 - Dark Grey
        76: [0.3, 0.3, 0.3],       # Rib Right 7 - Dark Grey
        77: [0.3, 0.3, 0.3],       # Rib Right 8 - Dark Grey
        78: [0.3, 0.3, 0.3],       # Rib Right 9 - Dark Grey
        79: [0.3, 0.3, 0.3],       # Rib Right 10 - Dark Grey
        80: [0.3, 0.3, 0.3],       # Rib Right 11 - Dark Grey
        81: [0.3, 0.3, 0.3],       # Rib Right 12 - Dark Grey
        82: [0.5, 0.2, 0.2],       # Humerus Left - Dark Red
        83: [0.6, 0.3, 0.3],       # Humerus Right - Medium Red
        84: [0.6, 0.3, 0.3],       # Scapula Left - Medium Red
        85: [0.7, 0.4, 0.4],       # Scapula Right - Medium Red
        86: [0.7, 0.4, 0.4],       # Clavicula Left - Medium Red
        87: [0.8, 0.5, 0.5],       # Clavicula Right - Light Red
        88: [0.3, 0.2, 0.2],       # Femur Left - Dark Brown
        89: [0.4, 0.3, 0.3],       # Femur Right - Dark Brown
        90: [0.3, 0.2, 0.2],       # Hip Left - Dark Brown
        91: [0.4, 0.3, 0.3],       # Hip Right - Dark Brown
        92: [0.3, 0.2, 0.2],       # Sacrum - Dark Brown
        93: [0.3, 0.2, 0.2],       # Face - Dark Brown
        94: [0.5, 0.3, 0.3],       # Gluteus Maximus Left - Medium Brown
        95: [0.5, 0.3, 0.3],       # Gluteus Maximus Right - Medium Brown
        96: [0.5, 0.3, 0.3],       # Gluteus Medius Left - Medium Brown
        97: [0.5, 0.3, 0.3],       # Gluteus Medius Right - Medium Brown
        98: [0.5, 0.3, 0.3],       # Gluteus Minimus Left - Medium Brown
        99: [0.5, 0.3, 0.3],       # Gluteus Minimus Right - Medium Brown
        100: [0.5, 0.3, 0.3],      # Autochthon Left - Medium Brown
        101: [0.5, 0.3, 0.3],      # Autochthon Right - Medium Brown
        102: [0.4, 0.2, 0.2],      # Iliopsoas Left - Dark Brown
        103: [0.4, 0.2, 0.2],      # Iliopsoas Right - Dark Brown
        104: [0.3, 0.3, 0.8],      # Urinary Bladder - Dark Blue
    }
    
    if return_preview:
        # For preview, we'll just return the first few labels
        preview_labels = [1, 2, 5]  # Example: spleen, kidney, liver
        meshes = []
        
        for label_idx in preview_labels:
            if label_idx not in np.unique(segmentation):
                continue
                
            verts, faces, _, _ = measure.marching_cubes(segmentation == label_idx, level=0.5)
            
            # Simplify mesh for preview
            color = color_map.get(label_idx, [0.5, 0.5, 0.5])
            meshes.append((verts, faces, color, label_idx))
            
        return meshes
    else:
        # Dictionary to store all meshes
        all_meshes = {}
        
        # Process all labels and create VTK meshes
        for label_idx in np.unique(segmentation):
            if label_idx == 0:  # Skip background
                continue

            verts, faces, _, _ = measure.marching_cubes(segmentation == label_idx, level=0.5)
            
            # Create VTK mesh
            mesh = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            triangles = vtk.vtkCellArray()

            for i, vert in enumerate(verts):
                points.InsertNextPoint(vert)
            for face in faces:
                triangle = vtk.vtkTriangle()
                for j in range(3):
                    triangle.GetPointIds().SetId(j, face[j])
                triangles.InsertNextCell(triangle)

            mesh.SetPoints(points)
            mesh.SetPolys(triangles)

            # Apply smoothing
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputData(mesh)
            smoother.SetNumberOfIterations(30)
            smoother.SetRelaxationFactor(0.1)
            smoother.FeatureEdgeSmoothingOff()
            smoother.BoundarySmoothingOn()
            smoother.Update()
            
            # Store the smoothed mesh
            all_meshes[label_idx] = smoother.GetOutput()
        
        # Save individual meshes (kept for backward compatibility)
        for label_idx, mesh in all_meshes.items():
            output_path = os.path.join(output_directory, f'label_{label_idx}.obj')
            save_vtk_polydata(mesh, output_path)
        
        # Save combined mesh with materials
        combined_output_path = os.path.join(output_directory, 'combined_segmentation.obj')
        save_combined_obj(all_meshes, combined_output_path, color_map)

def save_metadata(metadata, output_directory):
    metadata_path = os.path.join(output_directory, "metadata.json")
    with open(metadata_path, "w") as f:
        import json
        json.dump(metadata, f, indent=4)

def convert_dicom_to_nifti(dicom_files, output_nifti_path):
    """Convert a series of DICOM files to a single NIfTI file"""
    # Use MONAI's ITKReader for DICOM to NIfTI conversion
    reader = ITKReader()
    try:
        # Sort DICOM files by instance number to ensure correct ordering
        dicom_files = sorted(dicom_files, key=lambda x: pydicom.dcmread(x).InstanceNumber)
        
        # Read the series of DICOM files
        img_data, metadata = reader.read(dicom_files)
        
        # Convert to NIfTI using nibabel
        affine = np.eye(4)  # Default affine if not available
        if 'affine' in metadata:
            affine = metadata['affine']
        
        nifti_img = nib.Nifti1Image(img_data, affine)
        nib.save(nifti_img, output_nifti_path)
        
        return output_nifti_path
    except Exception as e:
        raise ValueError(f"Error converting DICOM to NIfTI: {str(e)}")

def process_ct_scan(input_file_path, output_directory, is_dicom=False, dicom_files=None):
    # Create temporary download directory
    download_dir = tempfile.mkdtemp()
    nifti_path = input_file_path
    
    try:
        # Convert DICOM to NIfTI if needed
        if is_dicom:
            # If input is a list of DICOM files
            if dicom_files:
                # Convert to NIfTI
                nifti_path = os.path.join(tempfile.mkdtemp(), 'converted.nii.gz')
                nifti_path = convert_dicom_to_nifti(dicom_files, nifti_path)
            # If input is a directory of DICOM files
            elif os.path.isdir(input_file_path):
                dicom_files = glob.glob(os.path.join(input_file_path, '*.dcm'))
                if not dicom_files:
                    raise ValueError("No DICOM files found in the directory")
                # Convert to NIfTI
                nifti_path = os.path.join(tempfile.mkdtemp(), 'converted.nii.gz')
                nifti_path = convert_dicom_to_nifti(dicom_files, nifti_path)
            # If input is a zip file of DICOM files
            else:
                temp_dicom_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(input_file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dicom_dir)
                dicom_files = glob.glob(os.path.join(temp_dicom_dir, '*.dcm'))
                if not dicom_files:
                    raise ValueError("No DICOM files found in the zip file")
                # Convert to NIfTI
                nifti_path = os.path.join(tempfile.mkdtemp(), 'converted.nii.gz')
                nifti_path = convert_dicom_to_nifti(dicom_files, nifti_path)
        
        # Download the model if needed
        model_name = "wholeBody_ct_segmentation"
        download(name=model_name, bundle_dir=download_dir)
        model_path = os.path.join(download_dir, 'wholeBody_ct_segmentation', 'models', 'model_lowres.pt')
        config_path = os.path.join(download_dir, 'wholeBody_ct_segmentation', 'configs', 'inference.json')
        
        # Load configuration
        config = ConfigParser()
        config.read_config(config_path)
        
        # Preprocess the data
        preprocessing = config.get_parsed_content("preprocessing")
        data = preprocessing({'image': nifti_path})
        
        # Set device and load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = config.get_parsed_content("network")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Run inference
        inferer = config.get_parsed_content("inferer")
        postprocessing = config.get_parsed_content("postprocessing")
        
        with torch.no_grad():
            data['pred'] = inferer(data['image'].unsqueeze(0).to(device), network=model)
        data['pred'] = data['pred'][0]
        data['image'] = data['image'][0]
        data = postprocessing(data)
        
        # Generate segmentation meshes and get preview
        preview_meshes = visualize_3d_multilabel(data['pred'], output_directory, return_preview=True)
        
        # Also save the actual meshes
        visualize_3d_multilabel(data['pred'], output_directory, return_preview=False)
        
        # Save metadata (abbreviated here for readability)
        metadata = {'color_map': [
            {"label": 0, "color": [0, 0, 0]},             # Background - Black
            {"label": 1, "color": [0.8, 0.3, 0.3]},       # Spleen - Light Red
            {"label": 2, "color": [0.6, 0.2, 0.2]},       # Right Kidney - Medium Red
            {"label": 3, "color": [0.6, 0.2, 0.2]},       # Left Kidney - Medium Red
            {"label": 4, "color": [0.7, 0.4, 0.3]},       # Gallbladder - Light Brown
            {"label": 5, "color": [0.5, 0.1, 0.1]},       # Liver - Dark Red
            {"label": 6, "color": [0.8, 0.6, 0.5]},       # Stomach - Light Brown
            {"label": 7, "color": [0.9, 0.1, 0.1]},       # Aorta - Bright Red
            {"label": 8, "color": [0.4, 0.6, 0.8]},       # Inferior Vena Cava - Light Blue
            {"label": 9, "color": [0.4, 0.6, 0.8]},       # Portal Vein and Splenic Vein - Light Blue
            {"label": 10, "color": [0.5, 0.3, 0.3]},      # Pancreas - Medium Brown
            {"label": 11, "color": [0.7, 0.4, 0.4]},      # Adrenal Gland Right - Medium Red
            {"label": 12, "color": [0.7, 0.4, 0.4]},      # Adrenal Gland Left - Medium Red
            {"label": 13, "color": [0.4, 0.3, 0.3]},      # Lung Upper Lobe Left - Dark Brown
            {"label": 14, "color": [0.4, 0.3, 0.3]},      # Lung Lower Lobe Left - Dark Brown
            {"label": 15, "color": [0.4, 0.3, 0.3]},      # Lung Upper Lobe Right - Dark Brown
            {"label": 16, "color": [0.4, 0.3, 0.3]},      # Lung Middle Lobe Right - Dark Brown
            {"label": 17, "color": [0.4, 0.3, 0.3]},      # Lung Lower Lobe Right - Dark Brown
            {"label": 18, "color": [0.3, 0.3, 0.3]},      # Vertebrae L5 - Dark Grey
            {"label": 19, "color": [0.3, 0.3, 0.3]},      # Vertebrae L4 - Dark Grey
            {"label": 20, "color": [0.3, 0.3, 0.3]},      # Vertebrae L3 - Dark Grey
            {"label": 21, "color": [0.3, 0.3, 0.3]},      # Vertebrae L2 - Dark Grey
            {"label": 22, "color": [0.3, 0.3, 0.3]},      # Vertebrae L1 - Dark Grey
            {"label": 23, "color": [0.3, 0.3, 0.3]},      # Vertebrae T12 - Dark Grey
            {"label": 24, "color": [0.3, 0.3, 0.3]},      # Vertebrae T11 - Dark Grey
            {"label": 25, "color": [0.3, 0.3, 0.3]},      # Vertebrae T10 - Dark Grey
            {"label": 26, "color": [0.3, 0.3, 0.3]},      # Vertebrae T9 - Dark Grey
            {"label": 27, "color": [0.3, 0.3, 0.3]},      # Vertebrae T8 - Dark Grey
            {"label": 28, "color": [0.3, 0.3, 0.3]},      # Vertebrae T7 - Dark Grey
            {"label": 29, "color": [0.3, 0.3, 0.3]},      # Vertebrae T6 - Dark Grey
            {"label": 30, "color": [0.3, 0.3, 0.3]},      # Vertebrae T5 - Dark Grey
            {"label": 31, "color": [0.3, 0.3, 0.3]},      # Vertebrae T4 - Dark Grey
            {"label": 32, "color": [0.3, 0.3, 0.3]},      # Vertebrae T3 - Dark Grey
            {"label": 33, "color": [0.3, 0.3, 0.3]},      # Vertebrae T2 - Dark Grey
            {"label": 34, "color": [0.3, 0.3, 0.3]},      # Vertebrae T1 - Dark Grey
            {"label": 35, "color": [0.3, 0.3, 0.3]},      # Vertebrae C7 - Dark Grey
            {"label": 36, "color": [0.3, 0.3, 0.3]},      # Vertebrae C6 - Dark Grey
            {"label": 37, "color": [0.3, 0.3, 0.3]},      # Vertebrae C5 - Dark Grey
            {"label": 38, "color": [0.3, 0.3, 0.3]},      # Vertebrae C4 - Dark Grey
            {"label": 39, "color": [0.3, 0.3, 0.3]},      # Vertebrae C3 - Dark Grey
            {"label": 40, "color": [0.3, 0.3, 0.3]},      # Vertebrae C2 - Dark Grey
            {"label": 41, "color": [0.3, 0.3, 0.3]},      # Vertebrae C1 - Dark Grey
            {"label": 42, "color": [0.4, 0.3, 0.3]},      # Esophagus - Dark Brown
            {"label": 43, "color": [0.5, 0.4, 0.4]},      # Trachea - Brown
            {"label": 44, "color": [1, 0, 0]},            # Heart Myocardium - Red
            {"label": 45, "color": [1, 0.4, 0.4]},        # Heart Atrium Left - Light Red
            {"label": 46, "color": [1, 0, 0]},            # Heart Ventricle Left - Red
            {"label": 47, "color": [1, 0.4, 0.4]},        # Heart Atrium Right - Light Red
            {"label": 48, "color": [1, 0, 0]},            # Heart Ventricle Right - Red
            {"label": 49, "color": [0, 0, 1]},            # Pulmonary Artery - Blue
            {"label": 50, "color": [0.5, 0.5, 1]},        # Brain - Light Blue
            {"label": 51, "color": [0.6, 0.3, 0]},        # Iliac Artery Left - Orange
            {"label": 52, "color": [0.6, 0.3, 0]},        # Iliac Artery Right - Orange
            {"label": 53, "color": [0.3, 0.6, 0.6]},      # Iliac Vein Left - Teal
            {"label": 54, "color": [0.3, 0.6, 0.6]},      # Iliac Vein Right - Teal
            {"label": 55, "color": [0.5, 0.3, 0.3]},      # Small Bowel - Light Brown
            {"label": 56, "color": [0.7, 0.2, 0.2]},      # Duodenum - Light Red
            {"label": 57, "color": [0.2, 0.2, 0.7]},      # Colon - Dark Blue
            {"label": 58, "color": [0.3, 0.3, 0.3]},      # Rib Left 1 - Dark Grey
            {"label": 59, "color": [0.3, 0.3, 0.3]},      # Rib Left 2 - Dark Grey
            {"label": 60, "color": [0.3, 0.3, 0.3]},      # Rib Left 3 - Dark Grey
            {"label": 61, "color": [0.3, 0.3, 0.3]},      # Rib Left 4 - Dark Grey
            {"label": 62, "color": [0.3, 0.3, 0.3]},      # Rib Left 5 - Dark Grey
            {"label": 63, "color": [0.3, 0.3, 0.3]},      # Rib Left 6 - Dark Grey
            {"label": 64, "color": [0.3, 0.3, 0.3]},      # Rib Left 7 - Dark Grey
            {"label": 65, "color": [0.3, 0.3, 0.3]},      # Rib Left 8 - Dark Grey
            {"label": 66, "color": [0.3, 0.3, 0.3]},      # Rib Left 9 - Dark Grey
            {"label": 67, "color": [0.3, 0.3, 0.3]},      # Rib Left 10 - Dark Grey
            {"label": 68, "color": [0.3, 0.3, 0.3]},      # Rib Left 11 - Dark Grey
            {"label": 69, "color": [0.3, 0.3, 0.3]},      # Rib Left 12 - Dark Grey
            {"label": 70, "color": [0.3, 0.3, 0.3]},      # Rib Right 1 - Dark Grey
            {"label": 71, "color": [0.3, 0.3, 0.3]},      # Rib Right 2 - Dark Grey
            {"label": 72, "color": [0.3, 0.3, 0.3]},      # Rib Right 3 - Dark Grey
            {"label": 73, "color": [0.3, 0.3, 0.3]},      # Rib Right 4 - Dark Grey
            {"label": 74, "color": [0.3, 0.3, 0.3]},      # Rib Right 5 - Dark Grey
            {"label": 75, "color": [0.3, 0.3, 0.3]},      # Rib Right 6 - Dark Grey
            {"label": 76, "color": [0.3, 0.3, 0.3]},      # Rib Right 7 - Dark Grey
            {"label": 77, "color": [0.3, 0.3, 0.3]},      # Rib Right 8 - Dark Grey
            {"label": 78, "color": [0.3, 0.3, 0.3]},      # Rib Right 9 - Dark Grey
            {"label": 79, "color": [0.3, 0.3, 0.3]},      # Rib Right 10 - Dark Grey
            {"label": 80, "color": [0.3, 0.3, 0.3]},      # Rib Right 11 - Dark Grey
            {"label": 81, "color": [0.3, 0.3, 0.3]},      # Rib Right 12 - Dark Grey
            {"label": 82, "color": [0.5, 0.2, 0.2]},      # Humerus Left - Dark Red
            {"label": 83, "color": [0.6, 0.3, 0.3]},      # Humerus Right - Medium Red
            {"label": 84, "color": [0.6, 0.3, 0.3]},      # Scapula Left - Medium Red
            {"label": 85, "color": [0.7, 0.4, 0.4]},      # Scapula Right - Medium Red
            {"label": 86, "color": [0.7, 0.4, 0.4]},      # Clavicula Left - Medium Red
            {"label": 87, "color": [0.8, 0.5, 0.5]},      # Clavicula Right - Light Red
            {"label": 88, "color": [0.3, 0.2, 0.2]},      # Femur Left - Dark Brown
            {"label": 89, "color": [0.4, 0.3, 0.3]},      # Femur Right - Dark Brown
            {"label": 90, "color": [0.3, 0.2, 0.2]},      # Hip Left - Dark Brown
            {"label": 91, "color": [0.4, 0.3, 0.3]},      # Hip Right - Dark Brown
            {"label": 92, "color": [0.3, 0.2, 0.2]},      # Sacrum - Dark Brown
            {"label": 93, "color": [0.3, 0.2, 0.2]},      # Face - Dark Brown
            {"label": 94, "color": [0.5, 0.3, 0.3]},      # Gluteus Maximus Left - Medium Brown
            {"label": 95, "color": [0.5, 0.3, 0.3]},      # Gluteus Maximus Right - Medium Brown
            {"label": 96, "color": [0.5, 0.3, 0.3]},      # Gluteus Medius Left - Medium Brown
            {"label": 97, "color": [0.5, 0.3, 0.3]},      # Gluteus Medius Right - Medium Brown
            {"label": 98, "color": [0.5, 0.3, 0.3]},      # Gluteus Minimus Left - Medium Brown
            {"label": 99, "color": [0.5, 0.3, 0.3]},      # Gluteus Minimus Right - Medium Brown
            {"label": 100, "color": [0.5, 0.3, 0.3]},     # Autochthon Left - Medium Brown
            {"label": 101, "color": [0.5, 0.3, 0.3]},     # Autochthon Right - Medium Brown
            {"label": 102, "color": [0.4, 0.2, 0.2]},     # Iliopsoas Left - Dark Brown
            {"label": 103, "color": [0.4, 0.2, 0.2]},     # Iliopsoas Right - Dark Brown
            {"label": 104, "color": [0.3, 0.3, 0.8]}      # Urinary Bladder - Dark Blue
        ], 
        'combined_file' : 'combined_segmentation.obj'
    }
        save_metadata(metadata, output_directory)
        
        return preview_meshes
        
    finally:
        # Clean up the download directory
        shutil.rmtree(download_dir, ignore_errors=True)
        # Clean up temp directories if created for DICOM conversion
        if is_dicom and nifti_path != input_file_path:
            if os.path.exists(os.path.dirname(nifti_path)):
                shutil.rmtree(os.path.dirname(nifti_path), ignore_errors=True)
        if is_dicom and 'temp_dicom_dir' in locals():
            shutil.rmtree(temp_dicom_dir, ignore_errors=True)

# Add option for file type selection
file_type = st.radio(
    "Select input file type:",
    ("NIfTI (.nii, .nii.gz)", "DICOM folder (.dcm files)", "DICOM zip (.zip with .dcm files)"),
    index=0
)

uploaded_file = None
uploaded_files = None
is_dicom = False

if file_type == "NIfTI (.nii, .nii.gz)":
    uploaded_file = st.file_uploader("Choose a NIfTI file", type=["nii", "nii.gz"])
    is_dicom = False
elif file_type == "DICOM folder (.dcm files)":
    st.markdown("""
    <p class='info-text'>
    You can upload multiple DICOM files from a folder. All .dcm files will be processed as a single series.
    </p>
    """, unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose DICOM files", type=["dcm"], accept_multiple_files=True)
    is_dicom = True
else:  # DICOM zip
    st.markdown("""
    <p class='info-text'>
    You can upload a zip file containing DICOM images. All .dcm files will be processed as a single series.
    </p>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a zip file containing DICOM images", type=["zip"])
    is_dicom = True

# Check if files are uploaded
if uploaded_file is not None or (uploaded_files is not None and len(uploaded_files) > 0):
    # Display file details
    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1e6:.2f} MB"
        }
        st.write(file_details)
    elif uploaded_files is not None:
        st.write(f"Selected {len(uploaded_files)} DICOM files")
        total_size = sum(file.size for file in uploaded_files)
        st.write(f"Total size: {total_size / 1e6:.2f} MB")
    
    # Create button to start processing
    if st.button("Process CT Scan"):
        # Create temporary directories
        temp_input_dir = tempfile.mkdtemp()
        temp_output_dir = tempfile.mkdtemp()
        
        try:
            # Handle different input types
            if uploaded_file is not None:
                # Save uploaded file
                input_file_path = os.path.join(temp_input_dir, uploaded_file.name)
                with open(input_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                dicom_files = None
            else:  # uploaded_files is not None
                # Save all uploaded DICOM files
                dicom_files = []
                for file in uploaded_files:
                    file_path = os.path.join(temp_input_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    dicom_files.append(file_path)
                input_file_path = temp_input_dir  # Use the directory as input path
            
            # Show progress
            with st.spinner("Processing CT scan... This may take a few minutes."):
                # Process the CT scan
                preview_meshes = process_ct_scan(input_file_path, temp_output_dir, is_dicom=is_dicom, dicom_files=dicom_files)
                
                # If we have preview meshes, we could render them here (would need a 3D renderer)
                
                # Create a ZIP file with all results
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for root, _, files in os.walk(temp_output_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_output_dir)
                            zip_file.write(file_path, arcname)
                
                # Reset buffer position
                zip_buffer.seek(0)
                
                # Show success message
                st.markdown("<div class='success-box'>Processing completed successfully!</div>", unsafe_allow_html=True)
                
                # Provide download button
                st.download_button(
                    label="Download 3D Model Files (ZIP)",
                    data=zip_buffer,
                    file_name="segmentation_results.zip",
                    mime="application/zip"
                )
                
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            
        finally:
            # Clean up temporary directories
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)

# About section
st.markdown("<h2 class='sub-header'>About this Application</h2>", unsafe_allow_html=True)
st.markdown("""
<p class='info-text'>
This application uses deep learning to segment different organs and structures from CT scans.
The model can identify over 100 different anatomical structures including organs, blood vessels,
bones, muscles, and more. The output is provided as a set of 3D model files (.obj) that can be
viewed in any 3D viewer software.
</p>

<h3>Supported File Types</h3>
<p class='info-text'>
- <strong>NIfTI (.nii, .nii.gz)</strong>: Medical imaging format commonly used in research<br>
- <strong>DICOM (.dcm)</strong>: Standard format for medical images in clinical settings, uploaded as individual files or as a zip file
</p>

<h3>Privacy Notice</h3>
<p class='info-text'>
Your CT scans are processed locally in this application and are not stored permanently.
All data is deleted after processing is complete.
</p>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
Created with ❤️ | <a href="https://github.com/yourusername/ct-segmentation-app" target="_blank">View on GitHub</a>
</div>
""", unsafe_allow_html=True)