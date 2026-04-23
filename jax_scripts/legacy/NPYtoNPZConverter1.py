import numpy as np
from pathlib import Path
import re

class NPYtoNPZConverter:
    """
    Convert Re=40 .npy solution files to the NPZ format used by your continuation code
    """
    
    def __init__(self, template_npz_path, magnetic_strength=0.0, imposed_field=None):
        """
        Initialize with a template NPZ to copy grid/operator structure
        
        Args:
            template_npz_path: Path to an existing .npz file from your continuation code
            magnetic_strength: float - strength of initial current field (0.0 = none)
            imposed_field: tuple (bx, by) - imposed magnetic field, e.g., (0.1, 0.0)
                          If None, uses template's b0
        """
        self.template = np.load(template_npz_path)
        self.magnetic_strength = magnetic_strength
        
        # Extract grid and operator info from template
        self.grid_info = {
            'x': self.template['x'],
            'y': self.template['y'],
            'kx': self.template['kx'],
            'ky': self.template['ky'],
            'mask': self.template['mask'],
            'to_u': self.template['to_u'],
            'to_v': self.template['to_v'],
            'inv_lap': self.template['inv_lap']
        }
        
        # Physical parameters - can override b0 (imposed magnetic field)
        self.params = {
            'nu': self.template['nu'],
            'eta': self.template['eta'],
            'b0': np.array(imposed_field) if imposed_field is not None else self.template['b0'],
            'forcing': self.template['forcing']
        }
        
        # Continuation settings (use template defaults, can be overridden)
        self.cont_settings = {
            'steps': self.template.get('steps', 1000),
            'shift_reflect_ny': self.template.get('shift_reflect_ny', 0),
            'rot': self.template.get('rot', False),
            'ministeps': self.template.get('ministeps', 10),
            'num_checkpoints': self.template.get('num_checkpoints', 10),
            'input_keys': self.template.get('input_keys', np.array(['T', 'sx', 'fields'])),
            'param_keys': self.template.get('param_keys', self.template['param_keys'])
        }
        
        print(f"Template loaded from: {template_npz_path}")
        print(f"Grid size: {self.grid_info['x'].shape}")
        print(f"nu={self.params['nu']}, eta={self.params['eta']}")
        print(f"b0 (imposed field)={self.params['b0']}")
        print(f"magnetic_strength (initial current)={self.magnetic_strength}")
    
    def load_npy_solution(self, array_file, meta_file):
        """
        Load and parse a single .npy solution pair
        
        The array file is in Fourier space (rfft format) - (128, 65) complex
        After irfftn, it becomes (128, 128) real space vorticity
        
        Returns:
            dict with fields, period, shift
        """
        # Load Fourier space data - this is ALREADY the vorticity in Fourier space!
        # Shape: (128, 65) complex from rfft2
        vorticity_fourier = np.load(array_file)
        
        # Also get real space version for inspection
        vorticity_real = np.fft.irfft2(vorticity_fourier)
        
        solution = {
            'vorticity_fourier': vorticity_fourier,
            'vorticity_real': vorticity_real
        }
        
        # Load metadata if exists
        if meta_file.exists():
            meta = np.load(meta_file, allow_pickle=True)
            
            # Parse metadata (shape should be (8,))
            # [0]: period, [1]: shift, [2-7]: other info
            if len(meta) >= 2:
                solution['T'] = float(meta[0])  # Period
                solution['sx'] = float(meta[1])  # Shift
            
            # Store rest for inspection
            solution['meta_full'] = meta
        else:
            # Use defaults if no metadata
            solution['T'] = 10.0
            solution['sx'] = 0.0
            print(f"Warning: No metadata for {array_file.stem}, using defaults")
        
        return solution
    
    def interpret_fields(self, vorticity_fourier, magnetic_strength=0.0):
        """
        Create fields array from vorticity (Navier-Stokes → MHD conversion)
        
        IMPORTANT: Your Newton solver expects REAL SPACE fields!
        The .npz template has fields in real space (2, 128, 128), not Fourier space.
        
        Input:
            vorticity_fourier: (128, 65) complex array - vorticity in Fourier space
            magnetic_strength: float - how strong to initialize the current field
                             0.0 = pure Navier-Stokes (no magnetic field)
                             0.1 = weak magnetic seed
                             1.0 = comparable to vorticity
        
        Output:
            fields: (2, 128, 128) REAL array
              fields[0] = vorticity ω in REAL space
              fields[1] = current j in REAL space
        """
        # Convert vorticity to real space
        vorticity_real = np.fft.irfft2(vorticity_fourier)  # Shape: (128, 128) real
        
        # Create current field based on magnetic_strength
        if magnetic_strength == 0.0:
            # No magnetic field (pure Navier-Stokes)
            current_real = np.zeros_like(vorticity_real)
        else:
            # Seed with a scaled version of vorticity (common initialization)
            # Or add random noise, or copy vorticity structure
            current_real = magnetic_strength * vorticity_real
            
            # Alternative: Add random perturbation
            # np.random.seed(42)
            # current_real = magnetic_strength * np.random.randn(*vorticity_real.shape)
        
        # Stack: fields[0] = vorticity, fields[1] = current
        fields = np.stack([vorticity_real, current_real], axis=0)  # Shape: (2, 128, 128) REAL
        
        return fields
    
    def convert_solution(self, npy_library_path, solution_number, output_path):
        """
        Convert a single .npy solution to .npz format
        
        Args:
            npy_library_path: Path to directory with .npy files
            solution_number: Which solution to convert (e.g., 0, 1, 2, ...)
            output_path: Where to save the .npz file
        """
        npy_library_path = Path(npy_library_path)
        
        # Construct filenames
        array_file = npy_library_path / f"soln_array_Re40_{solution_number}.npy"
        meta_file = npy_library_path / f"soln_meta_Re40_{solution_number}.npy"
        
        if not array_file.exists():
            raise FileNotFoundError(f"Solution file not found: {array_file}")
        
        # Load the solution
        solution = self.load_npy_solution(array_file, meta_file)
        
        # Create fields array (vorticity + current with magnetic_strength)
        fields = self.interpret_fields(solution['vorticity_fourier'], self.magnetic_strength)
        
        # Check shape compatibility
        template_shape = self.template['fields'].shape
        if fields.shape != template_shape:
            print(f"Warning: Shape mismatch!")
            print(f"  Converted: {fields.shape}")
            print(f"  Template:  {template_shape}")
            # You may need to resize/interpolate here
        
        # Package into NPZ format
        npz_data = {
            # Solution data
            'T': solution['T'],
            'sx': solution['sx'],
            'fields': fields,
            
            # Grid info (from template)
            **self.grid_info,
            
            # Physical parameters (from template)
            **self.params,
            
            # Continuation settings (from template)
            **self.cont_settings
        }
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **npz_data)
        
        print(f"\n✓ Converted solution {solution_number}")
        print(f"  Period: {solution['T']:.4f}")
        print(f"  Shift: {solution['sx']:.4f}")
        print(f"  Saved to: {output_path}")
        
        return npz_data
    
    def batch_convert(self, npy_library_path, output_dir, solution_numbers=None):
        """
        Convert multiple solutions
        
        Args:
            npy_library_path: Path to .npy library
            output_dir: Where to save converted .npz files
            solution_numbers: List of solution numbers to convert (None = all)
        """
        npy_library_path = Path(npy_library_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all solution numbers if not specified
        if solution_numbers is None:
            array_files = list(npy_library_path.glob("soln_array_Re40_*.npy"))
            solution_numbers = []
            for f in array_files:
                match = re.search(r'soln_array_Re40_(\d+)\.npy', f.name)
                if match:
                    solution_numbers.append(int(match.group(1)))
            solution_numbers = sorted(solution_numbers)
        
        print(f"\n{'='*70}")
        print(f"Converting {len(solution_numbers)} solutions...")
        print(f"{'='*70}\n")
        
        converted = []
        failed = []
        
        for sol_num in solution_numbers:
            try:
                output_path = output_dir / f"Re40_solution_{sol_num}.npz"
                self.convert_solution(npy_library_path, sol_num, output_path)
                converted.append(sol_num)
            except Exception as e:
                print(f"✗ Failed to convert solution {sol_num}: {e}")
                failed.append(sol_num)
        
        print(f"\n{'='*70}")
        print(f"Conversion complete!")
        print(f"  Success: {len(converted)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Output directory: {output_dir}")
        print(f"{'='*70}\n")
        
        return converted, failed


def inspect_metadata_structure(npy_library_path):
    """
    Helper to understand what's in the metadata files
    """
    npy_library_path = Path(npy_library_path)
    
    print("\n=== Inspecting Metadata Structure ===\n")
    
    # Load first few metadata files
    for i in range(min(5, 10)):
        meta_file = npy_library_path / f"soln_meta_Re40_{i}.npy"
        if meta_file.exists():
            meta = np.load(meta_file, allow_pickle=True)
            print(f"Solution {i} metadata (shape={meta.shape}):")
            for j, val in enumerate(meta):
                print(f"  [{j}]: {val} (type: {type(val).__name__})")
            print()


if __name__ == "__main__":
    # First, inspect metadata to understand structure
    npy_library = r"C:\Users\micha\Downloads\Re40_all_for_matt\Re40_all_for_matt"
    inspect_metadata_structure(npy_library)
    
    # ===== CONFIGURE MAGNETIC FIELD HERE =====
    # Weak magnetic seed - good starting point for MHD continuation
    magnetic_strength = 0.1  # 10% of vorticity strength
    imposed_field = (0.1, 0.0)  # Small imposed field in x-direction
    
    # Initialize converter with your template NPZ and magnetic settings
    template_npz = r"solutions\Re40\2.npz"
    converter = NPYtoNPZConverter(template_npz, 
                                 magnetic_strength=magnetic_strength,
                                 imposed_field=imposed_field)
    
    # Convert a single solution first (test)
    output_dir = Path(r"C:\Users\micha\Downloads\converted_Re40_solutions_weak_B")
    print("\n=== Test conversion (solution 0) ===")
    converter.convert_solution(npy_library, 0, output_dir / "test_solution_0.npz")
    
    # If that works, batch convert all (or subset)
    print("\n=== Batch convert first 10 solutions ===")
    converter.batch_convert(npy_library, output_dir, solution_numbers=list(range(10)))
    
    # To convert ALL solutions, uncomment:
    # converter.batch_convert(npy_library, output_dir)