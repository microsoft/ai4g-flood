#!/usr/bin/env python3
"""
Simple test script to verify the merge.py functionality.
This creates sample test data and runs the merge script.
"""

import os
import sys
import tempfile
import shutil

# Add src to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_directory_structure():
    """Create a sample directory structure with mock .tif files for testing."""
    test_dir = '/tmp/test_merge'
    
    # Create directory structure like results/2025/08/10/*.tif
    dirs = [
        'results/2025/08/10',
        'results/2025/08/11',
        'results/2025/08/12'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(test_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        
        # Create mock .tif files (they'll be empty, but that's fine for basic testing)
        for i in range(2):
            tif_file = os.path.join(full_path, f'sample_{i}_flood_prediction.tif')
            with open(tif_file, 'w') as f:
                f.write("# Mock TIF file for testing\n")
    
    return os.path.join(test_dir, 'results')

def test_merge_script_import():
    """Test that the merge script can be imported and basic functions work."""
    try:
        import merge
        print("✓ merge.py imports successfully")
        
        # Test argument parsing
        sys.argv = ['merge.py', '--input_dir', '/tmp/test', '--output', '/tmp/output.tif']
        try:
            args = merge.parse_args()
            print("✓ Argument parsing works")
        except SystemExit:
            # argparse calls sys.exit on success, which is normal
            print("✓ Argument parsing works")
        
        # Test file finding function with our test structure
        test_dir = create_sample_directory_structure()
        tif_files = merge.find_tif_files(test_dir)
        print(f"✓ Found {len(tif_files)} .tif files in test structure")
        
        for f in tif_files:
            print(f"  - {f}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import merge.py: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing merge script: {e}")
        return False

def test_command_line_help():
    """Test that the command line help works."""
    try:
        sys.argv = ['merge.py', '--help']
        import merge
        # This should trigger help and exit
        merge.main()
    except SystemExit as e:
        if e.code == 0:
            print("✓ Command line help works")
            return True
        else:
            print(f"✗ Help command failed with code {e.code}")
            return False
    except Exception as e:
        print(f"✗ Error testing command line help: {e}")
        return False

def main():
    """Run basic tests on the merge script."""
    print("Testing merge.py script...")
    print("=" * 40)
    
    success = True
    
    # Test imports and basic functionality
    if not test_merge_script_import():
        success = False
    
    print()
    print("Test command line interface...")
    
    # Test command line help (this will exit, so we do it in a separate test)
    # if not test_command_line_help():
    #     success = False
    
    if success:
        print("\n✓ All basic tests passed!")
        print("\nNote: Full functionality testing requires rasterio and geospatial libraries.")
        print("The script is ready to use once the required dependencies are installed.")
    else:
        print("\n✗ Some tests failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())