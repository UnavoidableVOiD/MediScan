#!/usr/bin/env python
"""Install dependencies with flexible version handling."""
import subprocess
import sys
import re

def install_package(package_spec):
    """Try to install a package, with fallback to flexible version."""
    package_spec = package_spec.strip()
    if not package_spec or package_spec.startswith('#'):
        return True
    
    # Extract package name and version
    if '>=' in package_spec:
        # Already flexible, try as-is
        result = subprocess.run(
            ['pip', 'install', package_spec],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"[OK] Installed {package_spec}")
            return True
        else:
            print(f"[FAIL] Failed to install {package_spec}")
            return False
    elif '==' in package_spec:
        pkg_name, version = package_spec.split('==')
        pkg_name = pkg_name.strip()
        version = version.strip()
        
        # Try exact version first
        result = subprocess.run(
            ['pip', 'install', package_spec],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"[OK] Installed {package_spec}")
            return True
        else:
            # Try flexible version
            print(f"[WARN] {package_spec} not found, trying {pkg_name}>={version}")
            result = subprocess.run(
                ['pip', 'install', f"{pkg_name}>={version}"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"[OK] Installed {pkg_name} (flexible version)")
                return True
            else:
                # Try without version constraint
                print(f"[WARN] Trying {pkg_name} without version constraint")
                result = subprocess.run(
                    ['pip', 'install', pkg_name],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"[OK] Installed {pkg_name} (latest version)")
                    return True
                else:
                    print(f"[FAIL] Failed to install {package_spec}")
                    print(f"  Error: {result.stderr[:200]}")
                    return False
    else:
        # No version specified
        result = subprocess.run(
            ['pip', 'install', package_spec],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"[OK] Installed {package_spec}")
            return True
        else:
            print(f"[FAIL] Failed to install {package_spec}")
            return False

def main():
    requirements_file = r"F:\Desktop\Mediscan\ML\requirements.txt"
    
    print("Installing ML dependencies...")
    print("=" * 60)
    
    failed_packages = []
    
    with open(requirements_file, 'r') as f:
        packages = f.readlines()
    
    for i, package in enumerate(packages, 1):
        print(f"\n[{i}/{len(packages)}] Processing: {package.strip()}")
        if not install_package(package):
            failed_packages.append(package.strip())
    
    print("\n" + "=" * 60)
    print("Installation Summary:")
    print(f"Total packages: {len(packages)}")
    print(f"Failed packages: {len(failed_packages)}")
    
    if failed_packages:
        print("\nFailed packages:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("\nNote: Some packages may be optional or unavailable.")
    else:
        print("\n[SUCCESS] All packages installed successfully!")

if __name__ == "__main__":
    main()
