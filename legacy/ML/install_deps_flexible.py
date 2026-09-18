#!/usr/bin/env python
"""Install dependencies with flexible version handling and skip unavailable packages."""
import subprocess
import sys

# Packages that don't exist or have alternative names
PACKAGE_ALTERNATIVES = {
    'bce-python-sdk': None,  # Skip - not available
    'fpdf': 'fpdf2',  # Alternative package name
    'hf-xet': None,  # Skip - not available
    'imbalanced-learn': 'imblearn',  # Alternative package name
}

def install_package(package_spec):
    """Try to install a package with flexible version handling."""
    package_spec = package_spec.strip()
    if not package_spec or package_spec.startswith('#'):
        return True, None
    
    # Check if package has an alternative
    pkg_name = package_spec.split('==')[0].split('>=')[0].strip()
    if pkg_name in PACKAGE_ALTERNATIVES:
        alt = PACKAGE_ALTERNATIVES[pkg_name]
        if alt is None:
            print(f"[SKIP] {package_spec} - package not available")
            return True, "skipped"
        else:
            print(f"[ALT] {package_spec} -> using {alt}")
            package_spec = alt
    
    # Try installing with exact version first
    result = subprocess.run(
        ['pip', 'install', package_spec],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        return True, "installed"
    
    # If exact version fails and it's pinned, try flexible
    if '==' in package_spec:
        pkg_name, version = package_spec.split('==')
        pkg_name = pkg_name.strip()
        version = version.strip()
        
        # Try >= version
        result = subprocess.run(
            ['pip', 'install', f"{pkg_name}>={version}"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"[FLEX] {package_spec} -> installed {pkg_name}>={version}")
            return True, "flexible"
        
        # Try without version
        result = subprocess.run(
            ['pip', 'install', pkg_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"[FLEX] {package_spec} -> installed {pkg_name} (latest)")
            return True, "flexible"
    
    # Failed
    error_msg = result.stderr[:150] if result.stderr else "Unknown error"
    print(f"[FAIL] {package_spec}")
    print(f"       {error_msg}")
    return False, error_msg

def main():
    requirements_file = r"F:\Desktop\Mediscan\ML\requirements.txt"
    
    print("Installing ML dependencies (flexible mode)...")
    print("=" * 70)
    
    installed = []
    failed = []
    skipped = []
    
    with open(requirements_file, 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    for i, package in enumerate(packages, 1):
        print(f"\n[{i}/{len(packages)}] {package}")
        success, status = install_package(package)
        
        if status == "skipped":
            skipped.append(package)
        elif success:
            installed.append(package)
        else:
            failed.append((package, status))
    
    print("\n" + "=" * 70)
    print("Installation Summary:")
    print(f"  Installed: {len(installed)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Failed: {len(failed)}")
    
    if skipped:
        print(f"\nSkipped packages ({len(skipped)}):")
        for pkg in skipped:
            print(f"  - {pkg}")
    
    if failed:
        print(f"\nFailed packages ({len(failed)}):")
        for pkg, error in failed:
            print(f"  - {pkg}")
    
    if len(installed) + len(skipped) == len(packages):
        print("\n[SUCCESS] All available packages processed!")
    else:
        print(f"\n[PARTIAL] {len(installed)}/{len(packages)} packages installed successfully")

if __name__ == "__main__":
    main()
