"""
Test script to validate README instructions work correctly
"""

from cell_analysis import main
import os
import sys

def test_readme_example():
    """Test the exact Quick Start example from README"""
    
    print("=" * 60)
    print("Testing README Quick Start Example")
    print("=" * 60)
    
    # Use the sample image in sample_data folder
    image_path = "sample_data/sample_test_image_small.tif"
    output_dir = "example_outputs"
    
    # Check if sample image exists
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image not found at {image_path}")
        return False
    
    print(f"✓ Image found: {image_path}")
    print(f"  File size: {os.path.getsize(image_path) / (1024*1024):.1f} MB")
    
    # Run the exact code from README
    try:
        print("\nRunning analysis (this may take a few minutes)...")
        results = main(
            image_path=image_path,
            output_dir=output_dir,
            qki_channel=1,      # QKI protein
            iba1_channel=2,     # Microglia marker (IBA1)
            dapi_channel=0,     # Nuclei (DAPI)
            tcell_channel=3,    # T cell marker (CD3)
            max_workers=4       # Reduced for local testing
        )
        
        # Print results as shown in README
        print(f"\n✓ Analysis complete!")
        print(f"Found {results['stats']['iba1_count']} microglia")
        print(f"Found {results['stats']['tcell_count']} T cells")
        
    except Exception as e:
        print(f"\n❌ ERROR: Analysis failed!")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify expected outputs exist
    print("\nVerifying outputs...")
    expected_files = [
        "stats.csv",
        "all_cells_comprehensive.csv",
        "iba1_cells.csv",
        "tcells.csv",
        "composite.png",
        "morphology.png",
        "morphology_pie_chart.png"
    ]
    
    missing = []
    for file in expected_files:
        filepath = os.path.join(output_dir, file)
        if os.path.exists(filepath):
            print(f"  ✓ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n❌ {len(missing)} expected files are missing!")
        return False
    
    print(f"\n✅ ALL TESTS PASSED!")
    print(f"   README instructions work correctly")
    print(f"   Results saved in: {output_dir}/")
    return True

if __name__ == "__main__":
    success = test_readme_example()
    sys.exit(0 if success else 1)