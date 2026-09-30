#!/usr/bin/env python3
"""System test for the Adversarial Attack Visualizer.

Verifies dependencies, project structure, and — most importantly — that every
attack implementation actually fools a small classifier on synthetic data.
Runs offline (no model download or network access required).
"""

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """All required third-party packages import."""
    print("Testing imports...")
    required = ['torch', 'torchvision', 'matplotlib', 'numpy', 'PIL',
                'requests', 'dotenv']
    failed = []
    for package in required:
        try:
            __import__(package)
            print(f"  + {package}")
        except ImportError as e:
            print(f"  x {package}: {e}")
            failed.append(package)
    if failed:
        print("Run: pip install -r requirements.txt")
    return not failed


def test_tkinter():
    """tkinter is available (needs a display for the real GUI)."""
    print("\nTesting tkinter...")
    try:
        import tkinter  # noqa: F401
        print("  + tkinter importable")
        return True
    except ImportError as e:
        print(f"  x tkinter: {e} (install your OS's python3-tk package)")
        return False


def test_file_structure():
    """Expected project files exist."""
    print("\nTesting file structure...")
    required = [
        'scripts/run_attack.py',
        'attacks/__init__.py',
        'attacks/fgsm.py',
        'attacks/pgd.py',
        'attacks/deepfool.py',
        'attacks/cw.py',
        'utils/model_utils.py',
        'utils/image_utils.py',
        'requirements.txt',
        '.env.example',
        'README.md',
    ]
    missing = []
    for rel in required:
        if (PROJECT_ROOT / rel).exists():
            print(f"  + {rel}")
        else:
            print(f"  x {rel} - missing!")
            missing.append(rel)
    return not missing


def _tiny_model_and_input():
    """A small linear classifier whose decision boundary attacks can cross."""
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 10)).eval()
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        label = model(image).argmax(dim=1)
    return model, image, label


def test_attacks():
    """Each attack returns a valid [0, 1] image and (usually) flips the label."""
    print("\nTesting attack implementations...")
    import torch
    from attacks import ATTACKS

    model, image, label = _tiny_model_and_input()
    params = {
        'FGSM': {'epsilon': 0.1},
        'PGD': {'epsilon': 0.1, 'iters': 20},
        'DeepFool': {'num_classes': 10, 'max_iter': 30},
        'CW': {'c': 5.0, 'max_iter': 100},
    }

    all_valid = True
    for name, attack_fn in ATTACKS.items():
        try:
            calls = []
            adv = attack_fn(model, image.clone(), label,
                            callback=lambda i, l, c: calls.append(i),
                            **params[name])
            assert adv.shape == image.shape, "shape mismatch"
            assert adv.min() >= 0 and adv.max() <= 1, "output outside [0, 1]"
            assert not torch.isnan(adv).any(), "NaN in output"
            assert calls, "progress callback never invoked"
            with torch.no_grad():
                adv_pred = model(adv).argmax(dim=1)
            flipped = (adv_pred != label).item()
            assert flipped, "attack failed to change the prediction"
            print(f"  + {name}: valid output, label flipped")
        except Exception as e:
            print(f"  x {name}: {e}")
            traceback.print_exc()
            all_valid = False
    return all_valid


def test_normalized_model():
    """NormalizedModel matches manual normalization + underlying model."""
    print("\nTesting NormalizedModel wrapper...")
    import torch
    import torch.nn as nn
    from utils.model_utils import NormalizedModel

    torch.manual_seed(0)
    inner = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4)).eval()
    wrapped = NormalizedModel(inner).eval()

    x = torch.rand(1, 3, 8, 8)
    mean = wrapped.mean
    std = wrapped.std
    with torch.no_grad():
        expected = inner((x - mean) / std)
        actual = wrapped(x)
    ok = torch.allclose(expected, actual)
    print(f"  {'+' if ok else 'x'} normalization applied inside forward pass")
    return ok


def test_image_utils():
    """Image helper functions behave sensibly without network access."""
    print("\nTesting image utilities...")
    from utils.image_utils import get_placeholder_url, search_pexels

    ok = True
    url = get_placeholder_url("test term")
    if url.startswith('https://') and 'test' in url:
        print("  + placeholder URL generation")
    else:
        print(f"  x unexpected placeholder URL: {url}")
        ok = False

    if search_pexels("cat", api_key=None) is None:
        print("  + Pexels search skipped cleanly without API key")
    else:
        print("  x Pexels search should return None without an API key")
        ok = False
    return ok


def main():
    print("Adversarial Attack Visualizer - System Test")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("tkinter", test_tkinter),
        ("File structure", test_file_structure),
        ("NormalizedModel", test_normalized_model),
        ("Attack implementations", test_attacks),
        ("Image utilities", test_image_utils),
    ]

    passed = 0
    for name, fn in tests:
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"  x {name} crashed: {e}")
            traceback.print_exc()
        print()

    print("=" * 50)
    print(f"Test results: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("All tests passed. Start the app with:")
        print("   python scripts/run_attack.py")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
