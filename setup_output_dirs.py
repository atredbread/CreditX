"""
Utility script to set up output directories with proper permissions.
"""
import os
import sys
from pathlib import Path

def setup_directories():
    """Create all necessary output directories with proper permissions."""
    base_dir = Path(__file__).parent
    output_dirs = [
        "output/classifications",
        "output/region_reports",
        "output/email_summaries",
        "output/features",
        "output/logs"
    ]
    
    try:
        # Create each directory if it doesn't exist
        for dir_path in output_dirs:
            full_path = base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created/Verified directory: {full_path}")
            
            # Try to set write permissions (may require admin on Windows)
            try:
                os.chmod(full_path, 0o777)  # rwxrwxrwx
                print(f"  - Set permissions for: {full_path}")
            except Exception as e:
                print(f"  - Warning: Could not set permissions for {full_path}: {str(e)}")
        
        print("\n✅ Output directory structure is ready!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error setting up directories: {str(e)}")
        return False

if __name__ == "__main__":
    print("Setting up output directories...\n")
    success = setup_directories()
    
    if not success:
        print("\n⚠️  Some directories might not have been created. You may need to run this script as administrator.")
        print("   Try running the command prompt as administrator and run this script again.")
        sys.exit(1)
    
    # Test writing a file
    test_file = Path("output/classifications/test_permissions.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("Test file - can be deleted")
        test_file.unlink()  # Clean up
        print("\n✓ Successfully tested file creation in output directories.")
    except Exception as e:
        print(f"\n⚠️  Could not write test file: {str(e)}")
        print("   Please check directory permissions or run as administrator.")
        sys.exit(1)
