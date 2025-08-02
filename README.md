<<<<<<< HEAD
# adversarial-visualizer
=======
# 🔍 Adversarial Attack Visualizer

A comprehensive GUI application for visualizing and analyzing adversarial attacks on deep neural networks. This tool provides an intuitive interface for generating, visualizing, and understanding adversarial examples using state-of-the-art attack methods.

## ✨ Features

### 🚀 Core Functionality
- **Multiple Attack Methods**: FGSM, PGD, DeepFool, and Carlini-Wagner (CW)
- **Real-time Visualization**: 5-panel display showing original image, perturbation overlay, adversarial image, confidence heatmap, and attack progression
- **Interactive Parameters**: Adjustable epsilon, iterations, and other attack parameters via GUI sliders
- **Image Search**: Built-in image search using Pexels API, Unsplash, and Bing
- **Progress Tracking**: Live visualization of attack convergence

### 📊 Advanced Visualizations
- **3D Attack Surface**: Visualize attack success rates across different parameters
- **Gradient Flow**: Show how gradients flow during attacks
- **Vulnerability Heatmap**: Identify which image regions are most susceptible to attacks
- **Attack Progression**: Real-time plotting of loss and confidence changes

### 🎬 Export Capabilities
- **Video Export**: Generate smooth transition videos between original and adversarial images
- **Results Export**: Detailed attack results with success metrics
- **Graph Spacing**: Optimized layout for better visualization

### 🛡️ Analysis Tools
- **Confidence Analysis**: Top-5 class confidence visualization
- **Attack Success Metrics**: Comprehensive success rate analysis
- **Parameter Sensitivity**: Interactive parameter tuning

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended but not required)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd adversarial_attack_visualizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (optional)**
   ```bash
   # Create .env file for API keys
   echo "PEXELS_API_KEY=your_pexels_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   python scripts/run_attack.py
   ```

## 📖 Usage Guide

### Basic Usage
1. **Select Attack Method**: Choose from FGSM, PGD, DeepFool, or CW
2. **Adjust Parameters**: Use sliders to set epsilon and iteration values
3. **Load Image**: Either search for images or provide a direct URL
4. **Run Attack**: Click "Search & Attack" or "Load & Attack"
5. **Analyze Results**: View the 5-panel visualization and results text

### Advanced Features
- **3D Attack Surface**: Click "📊 3D Attack Surface" to analyze attack effectiveness across parameter ranges
- **Gradient Flow**: Use "🌊 Gradient Flow" to visualize gradient information
- **Vulnerability Heatmap**: Click "🔥 Vulnerability Heatmap" to see vulnerable image regions
- **Video Export**: Generate transition videos with "🎬 Export Attack Video"

### Interactive Controls
- **Epsilon Slider**: Adjust perturbation magnitude (0.001 - 0.1)
- **Iterations Slider**: Set number of attack iterations (10 - 100)
- **Real-time Updates**: Enable for immediate parameter changes
- **Animation Controls**: Start/stop transition animations

## 🏗️ Architecture

### File Structure
```
adversarial_attack_visualizer/
├── scripts/
│   └── run_attack.py          # Main GUI application
├── attacks/
│   ├── fgsm.py               # FGSM attack implementation
│   ├── pgd.py                # PGD attack implementation
│   ├── deepfool.py           # DeepFool attack implementation
│   └── cw.py                 # Carlini-Wagner attack implementation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

### Key Components

#### Main GUI Class (`AdversarialAttackGUI`)
- **Initialization**: Sets up model, transforms, and UI components
- **Image Processing**: Handles image loading, preprocessing, and attack execution
- **Visualization**: Manages matplotlib plots and real-time updates
- **Export Functions**: Handles video generation and results export

#### Attack Methods
- **FGSM**: Fast Gradient Sign Method - single-step attack
- **PGD**: Projected Gradient Descent - iterative FGSM with momentum
- **DeepFool**: Finds minimal perturbations to cross decision boundaries
- **CW**: Carlini-Wagner - optimization-based attack with L2 constraints

## 🎯 Attack Methods Detail

### FGSM (Fast Gradient Sign Method)
- **Speed**: Very fast (single iteration)
- **Parameters**: Epsilon (perturbation magnitude)
- **Use Case**: Quick adversarial example generation

### PGD (Projected Gradient Descent)
- **Speed**: Medium (multiple iterations)
- **Parameters**: Epsilon, alpha (step size), iterations, momentum
- **Use Case**: Strong iterative attacks with momentum

### DeepFool
- **Speed**: Variable (until convergence)
- **Parameters**: Max iterations, overshoot, number of classes
- **Use Case**: Minimal perturbation attacks

### Carlini-Wagner (CW)
- **Speed**: Slow (optimization-based)
- **Parameters**: c (regularization), kappa (confidence), iterations
- **Use Case**: Strong L2-norm constrained attacks

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```
PEXELS_API_KEY=your_pexels_api_key_here
```

### Model Configuration
The application uses ResNet-18 pre-trained on ImageNet by default. To use different models, modify the initialization in `run_attack.py`:

```python
self.model = models.resnet50(weights='IMAGENET1K_V1').eval().to(self.device)
```

### Attack Parameters
Default parameters are optimized for ImageNet classification:
- **Epsilon**: 0.03 (3% of input range)
- **Iterations**: 40 for iterative methods
- **Learning Rate**: 0.01 for optimization-based methods

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or use CPU
   self.device = torch.device("cpu")
   ```

2. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install --upgrade -r requirements.txt
   ```

3. **Image Search Failures**
   - Check internet connection
   - Verify API keys in .env file
   - Try different search terms

4. **Video Export Issues**
   ```bash
   # Install OpenCV
   pip install opencv-python
   ```

### Performance Optimization

1. **GPU Usage**: Ensure CUDA is available for faster processing
2. **Memory Management**: Close unused windows and clear variables
3. **Parameter Tuning**: Start with smaller iteration counts for testing

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make changes and test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write unit tests for new features

### Adding New Attack Methods
1. Create new file in `attacks/` directory
2. Implement attack function with progress tracking
3. Add to GUI dropdown and processing logic
4. Update documentation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **Adversarial Robustness Toolbox**: For attack implementation references
- **OpenCV**: For video processing capabilities
- **Matplotlib**: For comprehensive visualization tools

## 📚 References

1. Goodfellow, I. J., et al. "Explaining and harnessing adversarial examples." ICLR 2015.
2. Madry, A., et al. "Towards deep learning models resistant to adversarial attacks." ICLR 2018.
3. Moosavi-Dezfooli, S. M., et al. "DeepFool: a simple and accurate method to fool deep neural networks." CVPR 2016.
4. Carlini, N., & Wagner, D. "Towards evaluating the robustness of neural networks." S&P 2017.

## 📧 Support

For questions, issues, or contributions, please:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include system information and error logs
4. Provide minimal reproduction steps

---

**Made with ❤️ for the adversarial ML community**
>>>>>>> d5c6def (Initial commit: Adversarial Attack Visualizer)
