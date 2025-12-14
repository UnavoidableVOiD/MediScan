import os

def list_files(startpath):
    # Folders to ignore
    exclude_dirs = {'venv', 'env', '.git', '__pycache__', 'node_modules', '.idea', '.vscode', 'lib', 'include', 'bin', 'scripts'}
    
    print(f"\n--- PROJECT STRUCTURE: {os.path.basename(os.path.abspath(startpath))} ---")
    
    for root, dirs, files in os.walk(startpath):
        # Modify dirs in-place to skip ignored folders
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        level = root.replace(startpath, '').count(os.sep)
        # Calculate indentation based on depth
        indent = ' |   ' * level
        print(f'{indent}{os.path.basename(root)}/')
        
        subindent = ' |   ' * (level + 1)
        for f in files:
            # Ignore hidden files and compiled python files
            if f.startswith('.') or f.endswith('.pyc'): continue
            print(f'{subindent}{f}')

if __name__ == "__main__":
    list_files('.')