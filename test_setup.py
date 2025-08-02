#!/usr/bin/env python3
"""
Test script to verify all components of the Adversarial Attack Visualizer work correctly.
Run this script to ensure the application is properly set up and functional.
"""

import sys
import os
import importlib.util
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'tkinter',
        'matplotlib',
        'numpy',
        'PIL',
        'requests'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                from PIL import Image, ImageTk
            elif package == 'tkinter':
                import tkinter as tk
                from tkinter import ttk, filedialog, messagebox
            else:
                importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("All imports successful!")
        return True

def test_attack_modules():
    """Test attack module imports."""
    print("\nTesting attack modules...")
    
    attack_modules = ['fgsm', 'pgd', 'deepfool', 'cw']
    failed_modules = []
    
    # Add attacks directory to path
    attacks_dir = Path(__file__).parent / 'attacks'
    if attacks_dir.exists():
        sys.path.insert(0, str(attacks_dir))
    
    for module_name in attack_modules:
        try:
            module = importlib.import_module(module_name)
            # Check if the attack function exists
            if hasattr(module, f'{module_name}_attack'):
                print(f"  ✓ {module_name}")
            else:
                print(f"  ! {module_name}: function {module_name}_attack not found")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            failed_modules.append(module_name)
    
    if failed_modules:
        print(f"\nFailed attack modules: {', '.join(failed_modules)}")
        return False
    else:
        print("All attack modules loaded successfully!")
        return True

def test_torch_setup():
    """Test PyTorch setup and CUDA availability."""
    print("\nTesting PyTorch setup...")
    
    try:
        import torch
        import torchvision.models as models
        
        print(f"  ✓ PyTorch version: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        else:
            print("  i Using CPU (CUDA not available)")
            device = torch.device("cpu")
        
        # Test model loading
        print("  - Testing model loading...")
        model = models.resnet18(weights='IMAGENET1K_V1').eval().to(device)
        print("  ✓ ResNet-18 loaded successfully!")
        
        # Test basic tensor operations
        test_tensor = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(test_tensor)
        print(f"  ✓ Model inference test passed! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ PyTorch test failed: {e}")
        return False

def test_gui_components():
    """Test GUI components without actually opening the window."""
    print("\nTesting GUI components...")
    
    try:
        import tkinter as tk
        from tkinter import ttk
        
        # Create a test root window (hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test basic widgets
        test_frame = ttk.Frame(root)
        test_button = ttk.Button(test_frame, text="Test")
        test_label = ttk.Label(test_frame, text="Test Label")
        test_scale = ttk.Scale(test_frame, from_=0, to=1)
        
        print("  ✓ Basic tkinter widgets work")
        
        # Test matplotlib with tkinter
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
        
        print("  ✓ Matplotlib integration works")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"  ✗ GUI test failed: {e}")
        return False

def test_file_structure():
    """Test project file structure."""
    print("\nTesting file structure...")
    
    required_files = [
        'scripts/run_attack.py',
        'attacks/fgsm.py',
        'attacks/pgd.py', 
        'attacks/deepfool.py',
        'attacks/cw.py',
        'requirements.txt',
        'README.md'
    ]
    
    project_root = Path(__file__).parent
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - Missing!")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        return False
    else:
        print("All required files present!")
        return True

def run_integration_test():
    """Run a simple integration test."""
    print("\nRunning integration test...")
    
    try:
        # Add scripts directory to path
        scripts_dir = Path(__file__).parent / 'scripts'
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
        
        # Import the main GUI class
        import importlib.util
        script_path = Path(__file__).parent / 'scripts' / 'run_attack.py'
        if script_path.exists():
            spec = importlib.util.spec_from_file_location("run_attack", script_path)
            run_attack_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_attack_module)
            AdversarialAttackGUI = run_attack_module.AdversarialAttackGUI
        else:
            raise ImportError("run_attack.py not found in scripts directory")
        
        print("  ✓ Main GUI class imported successfully")
        
        # Test creating an instance (without showing the GUI)
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # This will test model loading and initialization
        print("  - Testing GUI initialization...")
        gui = AdversarialAttackGUI(root)
        print("  ✓ GUI initialized successfully!")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests."""
    print("Adversarial Attack Visualizer - System Test\n")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Attack Modules Test", test_attack_modules), 
        ("PyTorch Setup Test", test_torch_setup),
        ("GUI Components Test", test_gui_components),
        ("File Structure Test", test_file_structure),
        ("Integration Test", run_integration_test)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"  ✗ {test_name} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("All tests passed! The application is ready to use.")
        print("\nTo start the application, run:")
        print("   python scripts/run_attack.py")
    else:
        print("Some tests failed. Please check the error messages above.")
        print("\nCommon fixes:")
        print("   - Run: pip install -r requirements.txt")
        print("   - Check file structure and paths")
        print("   - Ensure Python 3.8+ is being used")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
