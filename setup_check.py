#!/usr/bin/env python3
"""
Setup script for ISS Docking Model Training

This script checks for required packages and provides installation instructions.
"""

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def main():
    print("🔍 ISS Docking Model - Dependency Check")
    print("=" * 50)
    
    # Required packages
    packages = [
        ("tensorflow", "tensorflow"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("opencv-python", "cv2"),
    ]
    
    # Optional packages
    optional_packages = [
        ("scikit-learn", "sklearn"),
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    print("📦 Checking required packages:")
    for package, import_name in packages:
        if check_package(package, import_name):
            print(f"   ✅ {package}")
        else:
            print(f"   ❌ {package}")
            missing_required.append(package)
    
    # Check optional packages
    print("\n📦 Checking optional packages:")
    for package, import_name in optional_packages:
        if check_package(package, import_name):
            print(f"   ✅ {package}")
        else:
            print(f"   ⚠️  {package} (optional)")
            missing_optional.append(package)
    
    # Installation instructions
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("💡 Install required packages:")
        print(f"   pip install {' '.join(missing_required)}")
        print("\n🚫 Cannot run training until required packages are installed.")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("💡 Install optional packages for better functionality:")
        print(f"   pip install {' '.join(missing_optional)}")
    
    print("\n✅ All required packages are installed!")
    print("🚀 You can now run: python train.py")
    
    # Quick install option
    if missing_required or missing_optional:
        print("\n💡 Quick install all packages:")
        print("   pip install -r requirements.txt")
    
    return True

if __name__ == "__main__":
    main()