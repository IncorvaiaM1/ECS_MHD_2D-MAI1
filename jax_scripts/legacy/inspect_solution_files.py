import numpy as np
from pathlib import Path

def inspect_solution_files(library_path):
    """
    Thoroughly inspect the structure of .npy solution files
    """
    library_path = Path(library_path)
    
    # Find first solution to inspect
    array_files = [f for f in library_path.glob("*.npy") 
                   if not f.stem.endswith('_meta')]
    
    if not array_files:
        print("No solution files found!")
        return

    print(f"Found {len(array_files)} solution files\n")
    print("="*70)
    
    # Inspect first 3 solutions in detail
    for i, array_file in enumerate(array_files[:3]):
        sol_name = array_file.stem
        meta_file = library_path / f"{sol_name}_meta.npy"
        
        print(f"\n{'='*70}")
        print(f"SOLUTION {i+1}: {sol_name}")
        print(f"{'='*70}")
        
        # Load and inspect array file
        print(f"\n--- Array File: {array_file.name} ---")
        ft_data = np.load(array_file)
        print(f"Shape: {ft_data.shape}")
        print(f"Dtype: {ft_data.dtype}")
        print(f"Is complex: {np.iscomplexobj(ft_data)}")
        print(f"Value range: [{ft_data.real.min():.6f}, {ft_data.real.max():.6f}]")
        
        # Try converting to real space
        print(f"\n--- After irfftn() ---")
        real_data = np.fft.irfftn(ft_data)
        print(f"Real space shape: {real_data.shape}")
        print(f"Real space dtype: {real_data.dtype}")
        print(f"Value range: [{real_data.min():.6f}, {real_data.max():.6f}]")
        
        # Try to interpret dimensions
        print(f"\n--- Dimension interpretation ---")
        if len(real_data.shape) == 3:
            print(f"3D array detected: possibly (field_components, nx, ny)")
            print(f"  Dimension 0 (fields): {real_data.shape[0]}")
            print(f"  Dimension 1 (x): {real_data.shape[1]}")
            print(f"  Dimension 2 (y): {real_data.shape[2]}")
            
            # Try to guess what fields are present
            if real_data.shape[0] == 3:
                print(f"  → Likely: [u, v, b] or [ux, uy, bz]")
            elif real_data.shape[0] == 4:
                print(f"  → Likely: [ux, uy, bx, by]")
            elif real_data.shape[0] == 5:
                print(f"  → Likely: [ux, uy, uz, bx, by] or similar")
        
        elif len(real_data.shape) == 2:
            print(f"2D array detected: possibly (nx, ny) - single field")
            print(f"  Grid size: {real_data.shape[0]} x {real_data.shape[1]}")
        
        # Load and inspect metadata
        if meta_file.exists():
            print(f"\n--- Metadata File: {meta_file.name} ---")
            meta = np.load(meta_file, allow_pickle=True)
            print(f"Metadata type: {type(meta)}")
            print(f"Metadata shape: {meta.shape if hasattr(meta, 'shape') else 'N/A'}")
            
            if isinstance(meta, np.ndarray):
                print(f"Number of entries: {len(meta)}")
                print(f"\nFirst 10 metadata entries:")
                for j, entry in enumerate(meta[:10]):
                    print(f"  [{j}]: {entry} (type: {type(entry).__name__})")
                    
                # Specifically extract period and shift
                if len(meta) >= 2:
                    print(f"\n  → Period (meta[0]): {meta[0]}")
                    print(f"  → Shift (meta[1]): {meta[1]}")
        else:
            print(f"\n--- No metadata file found ---")
        
        print(f"\n{'='*70}\n")
    
    # Summary of all files
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL SOLUTIONS")
    print(f"{'='*70}\n")
    
    shapes_dict = {}
    for array_file in array_files:
        ft_data = np.load(array_file)
        real_data = np.fft.irfftn(ft_data)
        shape_key = str(real_data.shape)
        
        if shape_key not in shapes_dict:
            shapes_dict[shape_key] = []
        shapes_dict[shape_key].append(array_file.stem)
    
    print("Unique shapes found:")
    for shape, solutions in shapes_dict.items():
        print(f"\n  Shape {shape}: {len(solutions)} solutions")
        if len(solutions) <= 5:
            for sol in solutions:
                print(f"    - {sol}")
        else:
            print(f"    - {solutions[0]}")
            print(f"    - {solutions[1]}")
            print(f"    ... ({len(solutions)-2} more)")

def compare_with_npz_format(library_path, example_npz_path=None):
    """
    Compare .npy format with your existing .npz format
    """
    print(f"\n{'='*70}")
    print("NPZ FORMAT COMPARISON")
    print(f"{'='*70}\n")
    
    if example_npz_path and Path(example_npz_path).exists():
        print(f"Loading example NPZ: {example_npz_path}")
        npz_data = np.load(example_npz_path)
        
        print(f"\nNPZ file contents:")
        for key in npz_data.files:
            data = npz_data[key]
            print(f"  '{key}': shape={data.shape}, dtype={data.dtype}")
        
        print(f"\n→ You'll need to map the .npy data to these field names")
    else:
        print("No example NPZ provided.")
        print("\nTypical NPZ format for MHD might contain:")
        print("  'u' or 'velocity': velocity field")
        print("  'b' or 'magnetic': magnetic field")
        print("  'T' or 'period': temporal period")
        print("  'shift': spatial shift")
        print("\nPlease provide an example .npz file to compare!")


if __name__ == "__main__":
    # Inspect the Re=40 solutions
    library_path = r"C:\Users\micha\Downloads\Re40_all_for_matt\Re40_all_for_matt"
    inspect_solution_files(library_path)
    
    # If you have an example NPZ file from your continuation code, add path here:
    example_npz = r"solutions\Re40\2.npz"
    compare_with_npz_format(library_path, example_npz)