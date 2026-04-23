import numpy as np
import os
from pathlib import Path
import json




#computed solution x1 = 2x0 - x_(-1) possible to increment so that its continuous - predictor corrector scheme
# if it doesnt converge in 30 - 45 min instead of 10 hours
# also make sure it is continuous using the previous solution and adding the mag field so that it isnt cold 

# can write an sbatch and ssh in pace then can have it so that when a job ends it starts a new job automatically until all solutions are done
# try to get for .3 and bascially try to build a bifurcation diagram 
# Raley bernard convention 


# C:\Users\micha\ECS_MHD_2D\jax_scripts\MHDSolutionExtractor.py
class MHDSolutionExtractor:
    """Extract and process MHD solutions from .npy files"""
    
    def __init__(self, library_path):
        self.library_path = Path(library_path)
        self.solutions = {}
        
    def load_solution(self, solution_name):
        """
        Load a single solution from array and meta files
        
        Args:
            solution_name: Name of solution (without .npy extension)
            
        Returns:
            dict with 'data', 'period', 'shift', and other metadata
        """
        # Load array file (initial condition in Fourier space)
        array_file = self.library_path / f"{solution_name}.npy"
        meta_file = self.library_path / f"{solution_name}_meta.npy"
        
        if not array_file.exists():
            raise FileNotFoundError(f"Array file not found: {array_file}")
        
        # Load the Fourier transform data
        ft_data = np.load(array_file)
        
        # Convert from Fourier space to real space
        real_data = np.fft.irfftn(ft_data)
        
        solution_dict = {
            'name': solution_name,
            'ft_data': ft_data,  # Keep FT for reference
            'real_data': real_data,
            'shape': real_data.shape
        }
        
        # Load metadata if it exists
        if meta_file.exists():
            meta = np.load(meta_file, allow_pickle=True)
            
            # First two entries are period and shift
            if len(meta) >= 2:
                solution_dict['period'] = float(meta[0])
                solution_dict['shift'] = float(meta[1])
            
            # Store rest of metadata
            if len(meta) > 2:
                solution_dict['additional_meta'] = meta[2:]
        else:
            print(f"Warning: No metadata file found for {solution_name}")
        
        return solution_dict
    
    def scan_library(self):
        """
        Scan library directory and identify all solutions
        
        Returns:
            list of solution names (without extensions)
        """
        # Find all .npy files that are not metadata files
        array_files = [f.stem for f in self.library_path.glob("*.npy") 
                      if not f.stem.endswith('_meta')]
        
        print(f"Found {len(array_files)} solutions in {self.library_path}")
        return sorted(array_files)
    
    def load_all_solutions(self):
        """Load all solutions in the library"""
        solution_names = self.scan_library()
        
        for name in solution_names:
            try:
                self.solutions[name] = self.load_solution(name)
                print(f"Loaded: {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        
        return self.solutions
    
    def filter_traveling_waves(self):
        """Extract only traveling wave solutions (TWs)"""
        # TWs typically have 'TW' in the name
        tw_solutions = {k: v for k, v in self.solutions.items() 
                       if 'TW' in k.upper() or 'tw' in k}
        
        print(f"Found {len(tw_solutions)} traveling wave solutions")
        return tw_solutions
    
    def export_for_jax(self, solution_name, output_path):
        """
        Export solution in format suitable for JAX MHD code
        
        Args:
            solution_name: Name of solution to export
            output_path: Path to save output
        """
        if solution_name not in self.solutions:
            raise ValueError(f"Solution {solution_name} not loaded")
        
        sol = self.solutions[solution_name]
        
        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save real space initial condition
        np.save(output_path / f"{solution_name}_ic.npy", sol['real_data'])
        
        # Save metadata as JSON for easy reading
        meta = {
            'name': solution_name,
            'period': sol.get('period', None),
            'shift': sol.get('shift', None),
            'shape': sol['shape'],
            'reynolds_number': 40
        }
        
        with open(output_path / f"{solution_name}_info.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Exported {solution_name} to {output_path}")
        
    def summary_statistics(self):
        """Print summary statistics of loaded solutions"""
        if not self.solutions:
            print("No solutions loaded")
            return
        
        print("\n=== Solution Library Summary ===")
        print(f"Total solutions: {len(self.solutions)}")
        
        # Count TWs vs UPOs
        tw_count = sum(1 for k in self.solutions.keys() if 'TW' in k.upper())
        print(f"Traveling Waves: {tw_count}")
        print(f"UPOs: {len(self.solutions) - tw_count}")
        
        # Period statistics
        periods = [s.get('period') for s in self.solutions.values() 
                  if s.get('period') is not None]
        if periods:
            print(f"\nPeriod range: {min(periods):.3f} - {max(periods):.3f}")
            print(f"Mean period: {np.mean(periods):.3f}")
        
        # Shape info
        shapes = set(s['shape'] for s in self.solutions.values())
        print(f"\nData shapes found: {shapes}")


# Example usage
if __name__ == "__main__":
    # Path to your library
    library_path = r"C:\Users\micha\Downloads\Re40_all_for_matt\Re40_all_for_matt"
    
    # Initialize extractor
    extractor = MHDSolutionExtractor(library_path)
    
    # Load all solutions
    print("Loading solutions...")
    extractor.load_all_solutions()
    
    # Print summary
    extractor.summary_statistics()
    
    # Filter TWs if you want only traveling waves
    tw_solutions = extractor.filter_traveling_waves()
    print(f"\nTraveling wave solutions: {list(tw_solutions.keys())}")
    
    # Export solutions for use in your JAX code
    output_dir = r"C:\Users\micha\Downloads\processed_mhd_solutions"
    print(f"\n=== Exporting all solutions to {output_dir} ===")
    for sol_name in extractor.solutions.keys():
        extractor.export_for_jax(sol_name, output_dir)
    
    # Display info about first few solutions
    print("\n=== First 5 Solutions Details ===")
    for i, (sol_name, sol_data) in enumerate(list(extractor.solutions.items())[:5]):
        print(f"\n{i+1}. {sol_name}")
        print(f"   Period: {sol_data.get('period')}")
        print(f"   Shift: {sol_data.get('shift')}")
        print(f"   Shape: {sol_data['shape']}")
        print(f"   Data range: [{sol_data['real_data'].min():.3f}, {sol_data['real_data'].max():.3f}]")
    
    print(f"\n=== Complete! ===")
    print(f"All {len(extractor.solutions)} solutions exported to: {output_dir}")
    print(f"Each solution has:")
    print(f"  - *_ic.npy (initial condition in real space)")
    print(f"  - *_info.json (metadata with period, shift, etc.)")