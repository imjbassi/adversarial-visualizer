import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import time
import os
from dotenv import load_dotenv

class AdversarialAttackGUI:
    def __init__(self, root):
        # Load environment variables
        load_dotenv()
        self.pexels_api_key = os.getenv('PEXELS_API_KEY', '563492ad6f91700001000001d543bbd7e8dc42a7a4ec9c09c65188fc')  # Fallback to demo key
        
        # Rest of your initialization code
        self.root = root
        self.root.title("Adversarial Attack Visualizer")
        self.root.geometry("1400x800")
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights='IMAGENET1K_V1').eval().to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Store current images for visualization
        self.current_original = None
        self.current_perturbed = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=2)
        main_container.rowconfigure(0, weight=1)

        # Left panel for controls
        left_panel = ttk.Frame(main_container, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right panel for visualization
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_controls(left_panel)
        self.setup_visualization(right_panel)
        
    def setup_controls(self, parent):
        # Create a canvas and scrollbar for the entire left panel
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Now create all controls in the scrollable_frame instead of parent
        
        # Attack method selection (move to top)
        attack_frame = ttk.LabelFrame(scrollable_frame, text="Attack Method", padding="10")
        attack_frame.pack(fill=tk.X, pady=(0, 10))
        self.attack_method_var = tk.StringVar(value="FGSM")
        attack_dropdown = ttk.Combobox(attack_frame, textvariable=self.attack_method_var, state="readonly")
        attack_dropdown['values'] = ("FGSM", "PGD", "DeepFool", "CW")
        attack_dropdown.pack(fill=tk.X)
        attack_dropdown.bind("<<ComboboxSelected>>", self.on_attack_method_change)

        # Interactive parameter controls
        self.setup_interactive_controls(scrollable_frame)

        # Title
        title_label = ttk.Label(scrollable_frame, text="Adversarial Attack Visualizer", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))

        # Search input
        search_frame = ttk.LabelFrame(scrollable_frame, text="Search for Image", padding="10")
        search_frame.pack(fill=tk.X, pady=(0, 10))
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=28)
        self.search_entry.pack(fill=tk.X, pady=(0, 5))
        self.search_entry.bind('<Return>', lambda e: self.search_and_attack())
        self.search_btn = ttk.Button(search_frame, text="Search & Attack", command=self.search_and_attack)
        self.search_btn.pack(fill=tk.X)

        # URL input
        url_frame = ttk.LabelFrame(scrollable_frame, text="Direct Image URL", padding="10")
        url_frame.pack(fill=tk.X, pady=(0, 10))
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(url_frame, textvariable=self.url_var, width=28)
        self.url_entry.pack(fill=tk.X, pady=(0, 5))
        self.url_entry.bind('<Return>', lambda e: self.load_from_url())
        self.url_btn = ttk.Button(url_frame, text="Load & Attack", command=self.load_from_url)
        self.url_btn.pack(fill=tk.X)

        # Advanced visualizations
        self.setup_advanced_viz_controls(scrollable_frame)

        # Progress bar
        self.progress = ttk.Progressbar(scrollable_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 5))

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(scrollable_frame, textvariable=self.status_var, wraplength=280, anchor='center', justify='center')
        self.status_label.pack(pady=(0, 10))

        # Add Results text widget
        results_frame = ttk.LabelFrame(scrollable_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_text = tk.Text(results_frame, height=8, width=36, wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Close button
        close_btn = ttk.Button(scrollable_frame, text="Close", command=self.close_program)
        close_btn.pack(fill=tk.X, pady=(10, 5))
    
    def close_program(self):
        """Properly close the program and release resources"""
        try:
            # Close matplotlib figure to release resources
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            
            # Close the window and exit
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Error during close: {e}")
            sys.exit(0)  # Force exit if needed
    
    def setup_visualization(self, parent):
        # Visualization frame
        viz_frame = ttk.LabelFrame(parent, text="Visualization", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure with 2 rows: top row with 4 panels, bottom row with 1 panel
        self.fig = plt.figure(figsize=(20, 8))
        
        # Top row: 4 panels for images and predictions
        self.ax1 = plt.subplot2grid((2, 4), (0, 0))  # Original Image
        self.ax2 = plt.subplot2grid((2, 4), (0, 1))  # Perturbation
        self.ax3 = plt.subplot2grid((2, 4), (0, 2))  # Adversarial Image
        self.ax4 = plt.subplot2grid((2, 4), (0, 3))  # Top 5 Predictions
        
        # Bottom row: 1 panel spanning full width for attack progression
        self.ax5 = plt.subplot2grid((2, 4), (1, 0), colspan=4)  # Attack Progression
        
        self.fig.patch.set_facecolor('white')
        
        # Adjust spacing
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10, 
                                wspace=0.3, hspace=0.4)

        # Set up axes titles and properties
        self.ax1.set_title("Original Image")
        self.ax1.axis('off')
        self.ax2.set_title("Perturbation Overlay")
        self.ax2.axis('off')
        self.ax3.set_title("Adversarial Image")
        self.ax3.axis('off')
        self.ax4.set_title("Top 5 Predictions")
        self.ax4.set_xlabel('Rank')
        self.ax4.set_ylabel('Confidence')
        
        # Set up the ax5 for attack progression - now full width at bottom
        self.ax5.set_title("Attack Progression")
        self.ax5.set_xlabel('Iteration')
        self.ax5.set_ylabel('Value')
        
        self.attack_history = {'iterations': [], 'loss': [], 'confidence': []}

        # Add to tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize with placeholder
        self.show_placeholder()
        
    def show_placeholder(self):
        """Show placeholder images and heatmap."""
        placeholder = np.ones((224, 224, 3)) * 0.8
        for ax, title in zip([self.ax1, self.ax2, self.ax3], 
                         ["Original Image\n(No image loaded)", 
                          "Perturbation\n(Will show blend)", 
                          "Adversarial Image\n(After attack)"]):
            ax.clear()
            ax.imshow(placeholder)
            ax.set_title(title)
            ax.axis('off')
        
        # Top 5 Predictions placeholder
        self.ax4.clear()
        self.ax4.bar([], [])
        self.ax4.set_title("Top 5 Predictions")
        self.ax4.set_xlabel('Rank')
        self.ax4.set_ylabel('Confidence')
        self.ax4.text(2, 0.5, 'Predictions will\nshow here after\nattack', ha='center', va='center', 
                     fontsize=10, alpha=0.6, style='italic')
        
        # Attack progression placeholder - now spans full width
        self.ax5.clear()
        self.ax5.set_title("Attack Progression")
        self.ax5.set_xlabel('Iteration')
        self.ax5.set_ylabel('Value')
        self.ax5.text(25, 0.5, 'Attack progression will show here', ha='center', va='center', 
                     fontsize=12, alpha=0.6, style='italic')
        self.ax5.set_xlim(0, 50)
        self.ax5.set_ylim(0, 1)
        self.ax5.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def on_attack_method_change(self, event=None):
        self.attack_method = self.attack_method_var.get()

    def search_and_attack(self):
        search_term = self.search_var.get().strip()
        if not search_term:
            self.show_error("Please enter a search term")
            return
        self.disable_controls()
        self.progress.start()
        self.status_var.set(f"Searching for '{search_term}'...")
        # Use dropdown value for attack method
        attack_method = getattr(self, 'attack_method', self.attack_method_var.get())
        thread = threading.Thread(target=self._search_worker, args=(search_term, attack_method))
        thread.daemon = True
        thread.start()

    def load_from_url(self):
        url = self.url_var.get().strip()
        if not url:
            self.show_error("Please enter an image URL")
            return
        
        self.disable_controls()
        self.progress.start()
        self.status_var.set("Loading image from URL...")
        
        thread = threading.Thread(target=self._url_worker, args=(url,))
        thread.daemon = True
        thread.start()
    
    def disable_controls(self):
        self.search_btn.config(state='disabled')
        self.url_btn.config(state='disabled')
    
    def enable_controls(self):
        self.search_btn.config(state='normal')
        self.url_btn.config(state='normal')
    
    def _search_worker(self, search_term, attack_method):
        try:
            self.status_var.set(f"Searching for images of '{search_term}'...")
            image_url = self.find_image_url(search_term)
            
            if image_url:
                self.status_var.set(f"Found image for '{search_term}', processing...")
                self._process_image(image_url, f"Search: {search_term}", attack_method)
            else:
                error_msg = f"Could not find any images for '{search_term}'. Please try another search term."
                self.root.after(0, lambda msg=error_msg: self.show_error(msg))
        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            self.root.after(0, lambda msg=error_msg: self.show_error(msg))

    def _url_worker(self, url):
        try:
            attack_method = getattr(self, 'attack_method', 'FGSM')
            self._process_image(url, "Direct URL", attack_method)
        except Exception as e:
            error_msg = f"URL error: {str(e)}"
            self.root.after(0, lambda msg=error_msg: self.show_error(msg))
            
    def find_image_url(self, search_term):
        """Find an image URL for the search term using multiple sources."""
        try:
            # Try Pexels first (most reliable)
            image_url = self.search_pexels(search_term)
            if image_url:
                return image_url

            # Try Unsplash as fallback
            image_url = self.search_unsplash(search_term)
            if image_url:
                return image_url

            # Final fallback to dummy image
            return self.get_dummy_image_url(search_term)
        except Exception as e:
            print(f"Search error: {e}")
            return self.get_dummy_image_url(search_term)

    def search_pexels(self, search_term):
        """Search Pexels API for images."""
        try:
            encoded_term = urllib.parse.quote(search_term)
            pexels_url = f"https://api.pexels.com/v1/search?query={encoded_term}&per_page=1"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Authorization': self.pexels_api_key
            }
            
            response = requests.get(pexels_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'photos' in data and len(data['photos']) > 0:
                    return data['photos'][0]['src']['medium']
        except Exception as e:
            print(f"Pexels search failed: {e}")
        return None
    
    def search_unsplash(self, search_term):
        """Search Unsplash for images."""
        try:
            timestamp = int(time.time())
            encoded_term = urllib.parse.quote(search_term)
            url = f"https://source.unsplash.com/featured/400x400/?{encoded_term}&sig={timestamp}"
            
            response = requests.get(
                url, 
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Cache-Control': 'no-cache'
                },
                allow_redirects=True,
                timeout=10
            )
            
            if response.status_code == 200 and response.url != url:
                return response.url
        except Exception as e:
            print(f"Unsplash search error: {e}")
        return None

    def get_dummy_image_url(self, search_term):
        """Return a dummy image URL with the search term."""
        encoded_term = urllib.parse.quote(search_term)
        return f"https://dummyimage.com/400x400/3498db/ffffff&text={encoded_term}"
    
    def load_image_from_url(self, url):
        """Load image from URL with proper headers and better error handling"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Referer': 'https://www.google.com/'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=15, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"URL returned non-image content: {content_type}")
                
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise Exception(f"Failed to load image from {url}: {str(e)}")

    def tensor_to_image(self, tensor):
        """Convert tensor to displayable image"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()

    def _process_image(self, image_url, source, attack_method):
        try:
            image = self.load_image_from_url(image_url)
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                original_output = self.model(input_tensor)
                original_pred = original_output.argmax(dim=1)
                original_confidence = torch.softmax(original_output, dim=1).max().item()
            
            self.attack_history = {'iterations': [], 'loss': [], 'confidence': []}
            
            epsilon = getattr(self, 'epsilon_var', type('', (), {'get': lambda: 0.03})()).get()
            iterations = getattr(self, 'iterations_var', type('', (), {'get': lambda: 40})()).get()
            
            if attack_method == 'PGD':
                perturbed = self.pgd_attack_with_progress(input_tensor.clone(), original_pred, 
                                                        epsilon=epsilon, alpha=0.01, iters=iterations)
            elif attack_method == 'DeepFool':
                perturbed = self.deepfool_attack_with_progress(input_tensor.clone(), original_pred, 
                                                             num_classes=1000, max_iter=iterations)
            elif attack_method == 'CW':
                perturbed = self.cw_attack_with_progress(input_tensor.clone(), original_pred, 
                                                       c=1e-2, kappa=0, max_iter=iterations)
            else:  # FGSM
                perturbed = self.fgsm_attack_with_progress(input_tensor.clone(), original_pred, 
                                                         epsilon=epsilon)
            
            with torch.no_grad():
                adv_output = self.model(perturbed)
                adv_pred = adv_output.argmax(dim=1)
                adv_confidence = torch.softmax(adv_output, dim=1).max().item()
            
            self.current_original = input_tensor
            self.current_perturbed = perturbed
            
            self.root.after(0, lambda src=source, url=image_url, 
                            op=original_pred.item(), oc=original_confidence,
                            ap=adv_pred.item(), ac=adv_confidence, 
                            ot=input_tensor, pt=perturbed, adv_out=adv_output: 
                            self.show_results(src, url, op, oc, ap, ac, ot, pt, adv_out))
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.root.after(0, lambda msg=error_msg: self.show_error(msg))
    
    def show_results(self, source, url, original_pred, original_conf, adv_pred, adv_conf, original_tensor, perturbed_tensor, adv_output):
        """Display the attack results"""
        try:
            # Convert tensors to images
            original_img = self.tensor_to_image(original_tensor)
            adversarial_img = self.tensor_to_image(perturbed_tensor)
            perturbation = adversarial_img - original_img
            
            # Clear all axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
                ax.clear()
            
            # Show original image
            self.ax1.imshow(original_img)
            self.ax1.set_title(f"Original Image\nPred: {original_pred} ({original_conf:.3f})")
            self.ax1.axis('off')
            
            # Show perturbation with enhanced visibility
            perturbation_enhanced = np.abs(perturbation) * 10  # Amplify for visibility
            perturbation_enhanced = np.clip(perturbation_enhanced, 0, 1)
            self.ax2.imshow(perturbation_enhanced)
            self.ax2.set_title("Perturbation\n(Enhanced 10x)")
            self.ax2.axis('off')
            
            # Show adversarial image
            self.ax3.imshow(adversarial_img)
            self.ax3.set_title(f"Adversarial Image\nPred: {adv_pred} ({adv_conf:.3f})")
            self.ax3.axis('off')
            
            # Show confidence comparison with better formatting
            confidence_data = torch.softmax(adv_output, dim=1).squeeze().cpu().detach().numpy()
            top_5_indices = np.argsort(confidence_data)[-5:][::-1]
            top_5_confidences = confidence_data[top_5_indices]
            
            # Create color-coded bars
            colors = ['red' if i == 0 and adv_pred != original_pred else 
                     'green' if i == 0 and adv_pred == original_pred else 
                     'lightblue' for i in range(5)]
            
            bars = self.ax4.bar(range(5), top_5_confidences, color=colors)
            self.ax4.set_title("Top 5 Predictions", fontsize=12, fontweight='bold')
            self.ax4.set_xlabel("Rank", fontsize=10)
            self.ax4.set_ylabel("Confidence", fontsize=10)
            self.ax4.set_xticks(range(5))
            
            # Better labels with class names if available
            labels = []
            for i in range(5):
                conf_pct = top_5_confidences[i] * 100
                labels.append(f"#{i+1}\nClass {top_5_indices[i]}\n{conf_pct:.1f}%")
            
            self.ax4.set_xticklabels(labels, fontsize=8)
            self.ax4.set_ylim(0, 1)
            self.ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on top of bars
            for i, (bar, conf) in enumerate(zip(bars, top_5_confidences)):
                self.ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                             f'{conf:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Show attack progression with better visualization (now full width)
            if self.attack_history['iterations']:
                iterations = self.attack_history['iterations']
                losses = self.attack_history['loss']
                confidences = self.attack_history['confidence']
                
                # Normalize losses for better visualization
                if len(losses) > 1:
                    losses = np.array(losses)
                    losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
                
                self.ax5.plot(iterations, losses, 'r-', label='Loss (normalized)', 
                             linewidth=2, marker='o', markersize=4)
                self.ax5.plot(iterations, confidences, 'b-', label='Confidence', 
                             linewidth=2, marker='s', markersize=4)
                self.ax5.set_title("Attack Progression", fontsize=14, fontweight='bold')
                self.ax5.set_xlabel("Iteration", fontsize=12)
                self.ax5.set_ylabel("Value", fontsize=12)
                self.ax5.legend(fontsize=11)
                self.ax5.grid(True, alpha=0.3)
                
                # Set proper limits
                if iterations:
                    self.ax5.set_xlim(-0.5, max(iterations) + 0.5)
                self.ax5.set_ylim(-0.05, 1.05)
            else:
                # Show gradient magnitude instead
                if hasattr(self, 'current_original') and self.current_original is not None:
                    grad_img = self.visualize_gradients(original_tensor, original_pred)
                    self.ax5.imshow(grad_img, cmap='hot')
                    self.ax5.set_title("Gradient Magnitude")
                    self.ax5.axis('off')
            
            self.canvas.draw()
            
            # Update results text
            results_text = f"Source: {source}\n"
            results_text += f"Original: Class {original_pred} ({original_conf:.3f})\n"
            results_text += f"Adversarial: Class {adv_pred} ({adv_conf:.3f})\n"
            results_text += f"Attack Success: {'Yes' if adv_pred != original_pred else 'No'}\n"
            results_text += f"Confidence Drop: {original_conf - adv_conf:.3f}\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_text)
            
        except Exception as e:
            print(f"Error in show_results: {e}")
        finally:
            self.progress.stop()
            self.enable_controls()
            self.status_var.set("Attack completed")
    
    def show_error(self, message):
        """Display error message"""
        self.progress.stop()
        self.enable_controls()
        self.status_var.set("Error occurred")
        messagebox.showerror("Error", message)
        
        # Also add to results text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error: {message}")
    
    def visualize_gradients(self, input_tensor, target_class):
        """Visualize gradient magnitude for the input"""
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([target_class]).to(self.device))
        loss.backward()
        
        gradients = input_tensor.grad.data.abs()
        grad_img = self.tensor_to_image(gradients)
        grad_magnitude = np.mean(grad_img, axis=2)  # Average across color channels
        return grad_magnitude

    def setup_interactive_controls(self, parent):
        """Add interactive parameter controls"""
        params_frame = ttk.LabelFrame(parent, text="Attack Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Epsilon slider
        epsilon_frame = ttk.Frame(params_frame)
        epsilon_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(epsilon_frame, text="Epsilon:", width=12).pack(side=tk.LEFT)
        
        self.epsilon_var = tk.DoubleVar(value=0.03)
        epsilon_scale = ttk.Scale(epsilon_frame, from_=0.001, to=0.1, 
                            variable=self.epsilon_var, orient=tk.HORIZONTAL)
        epsilon_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.epsilon_label = ttk.Label(epsilon_frame, text="0.030", width=8)
        self.epsilon_label.pack(side=tk.RIGHT)
        epsilon_scale.bind("<Motion>", self.update_epsilon_label)
        
        # Iterations slider (for PGD/DeepFool)
        iter_frame = ttk.Frame(params_frame)
        iter_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(iter_frame, text="Iterations:", width=12).pack(side=tk.LEFT)
        
        self.iterations_var = tk.IntVar(value=40)
        iter_scale = ttk.Scale(iter_frame, from_=10, to=100, 
                         variable=self.iterations_var, orient=tk.HORIZONTAL)
        iter_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.iter_label = ttk.Label(iter_frame, text="40", width=8)
        self.iter_label.pack(side=tk.RIGHT)
        iter_scale.bind("<Motion>", self.update_iter_label)

    def update_epsilon_label(self, event):
        self.epsilon_label.config(text=f"{self.epsilon_var.get():.3f}")

    def update_iter_label(self, event):
        self.iter_label.config(text=f"{self.iterations_var.get()}")

    def setup_advanced_viz_controls(self, parent):
        """Add buttons for advanced visualizations"""
        viz_frame = ttk.LabelFrame(parent, text="Advanced Visualizations", padding="10")
        viz_frame.pack(fill=tk.X, pady=(0, 10))
        
        surface_btn = ttk.Button(viz_frame, text="📊 3D Attack Surface", 
                               command=self.plot_attack_surface)
        surface_btn.pack(fill=tk.X, pady=(0, 5))
        
        gradient_btn = ttk.Button(viz_frame, text="🌊 Gradient Flow", 
                                command=self.visualize_gradient_flow)
        gradient_btn.pack(fill=tk.X, pady=(0, 5))
        
        heatmap_btn = ttk.Button(viz_frame, text="🔥 Vulnerability Heatmap", 
                               command=self.create_vulnerability_heatmap)
        heatmap_btn.pack(fill=tk.X)

    def plot_attack_surface(self):
        """Complete the 3D attack surface visualization"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            if not hasattr(self, 'current_original') or self.current_original is None:
                messagebox.showwarning("No Image", "Please run an attack first!")
                return
            
            # Create new window for 3D visualization
            surface_window = tk.Toplevel(self.root)
            surface_window.title("3D Attack Surface")
            surface_window.geometry("800x600")
            
            fig = plt.Figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Test different epsilon values and attack methods with more variation
            epsilon_range = np.linspace(0.01, 0.1, 8)
            methods = ['FGSM', 'PGD', 'DeepFool', 'CW']
            
            success_rates = []
            confidence_changes = []
            epsilons = []
            method_nums = []
            iterations_tested = []
            
            # Progress tracking
            total_tests = len(methods) * len(epsilon_range)
            current_test = 0
            
            for i, method in enumerate(methods):
                for j, eps in enumerate(epsilon_range):
                    try:
                        # Create a clean copy for testing
                        test_image = self.current_original.clone()
                        
                        # Get original prediction
                        with torch.no_grad():
                            original_output = self.model(test_image)
                            original_pred = original_output.argmax(dim=1)
                            original_conf = torch.softmax(original_output, dim=1).max().item()
                        
                        # Run attack with varied parameters
                        iterations = 10 + j * 5  # Vary iterations based on epsilon
                        
                        if method == 'FGSM':
                            perturbed = self.fgsm_attack_with_progress(test_image, original_pred, epsilon=eps)
                        elif method == 'PGD':
                            perturbed = self.pgd_attack_with_progress(test_image, original_pred, 
                                                                   epsilon=eps, iters=iterations)
                        elif method == 'DeepFool':
                            perturbed = self.deepfool_attack_with_progress(test_image, original_pred, 
                                                                         max_iter=iterations)
                        else:  # CW
                            perturbed = self.cw_attack_with_progress(test_image, original_pred, 
                                                                   c=eps/10, max_iter=iterations)
                        
                        # Calculate success metrics
                        with torch.no_grad():
                            adv_output = self.model(perturbed)
                            adv_pred = adv_output.argmax(dim=1)
                            adv_conf = torch.softmax(adv_output, dim=1).max().item()
                            
                            success = 1.0 if original_pred != adv_pred else 0.2  # Partial credit for near misses
                            conf_change = abs(original_conf - adv_conf)
                        
                        success_rates.append(success)
                        confidence_changes.append(conf_change)
                        epsilons.append(eps)
                        method_nums.append(i + (j * 0.1))  # Add slight offset for better 3D distribution
                        iterations_tested.append(iterations)
                        
                    except Exception as e:
                        print(f"Error with {method} eps={eps}: {e}")
                        # Add default values to maintain structure
                        success_rates.append(0.0)
                        confidence_changes.append(0.0)
                        epsilons.append(eps)
                        method_nums.append(i + (j * 0.1))
                        iterations_tested.append(10)
                    
                    current_test += 1
                    surface_window.title(f"3D Attack Surface - Progress: {current_test}/{total_tests}")
                    surface_window.update()
            
            # Create 3D surface with better visualization
            if success_rates:
                # Create a mesh-like surface by interpolating data
                scatter = ax.scatter(epsilons, method_nums, success_rates, 
                                   c=confidence_changes, cmap='viridis', s=80, alpha=0.8)
                
                ax.set_xlabel('Epsilon')
                ax.set_ylabel('Attack Method')
                ax.set_zlabel('Success Rate')
                ax.set_title('Attack Success Surface')
                
                # Set method labels
                ax.set_yticks([0, 1, 2, 3])
                ax.set_yticklabels(['FGSM', 'PGD', 'DeepFool', 'CW'])
                
                # Add colorbar
                fig.colorbar(scatter, label='Confidence Change', shrink=0.8)
                
                # Add grid for better visualization
                ax.grid(True, alpha=0.3)
            
            canvas = FigureCanvasTkAgg(fig, surface_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            surface_window.title("3D Attack Surface - Complete")
            
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to create 3D surface: {str(e)}")

    def visualize_gradient_flow(self):
        """Show gradient flow during attack"""
        try:
            if not hasattr(self, 'current_original') or self.current_original is None:
                messagebox.showwarning("No Image", "Please run an attack first!")
                return
            
            grad_window = tk.Toplevel(self.root)
            grad_window.title("Gradient Flow Visualization")
            grad_window.geometry("800x400")
            
            fig = plt.Figure(figsize=(12, 4))
            ax1, ax2, ax3 = fig.subplots(1, 3)
            
            # Get gradients at different stages
            image = self.current_original.clone().requires_grad_(True)
            
            # Initial gradient
            output = self.model(image)
            loss = torch.nn.functional.cross_entropy(output, torch.tensor([0]).to(self.device))
            initial_grad = torch.autograd.grad(loss, image, retain_graph=True)[0]
            
            # Show gradient magnitudes
            grad_mag = initial_grad.squeeze().abs().mean(dim=0).cpu().numpy()
            
            ax1.imshow(self.tensor_to_image(image))
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            im2 = ax2.imshow(grad_mag, cmap='hot')
            ax2.set_title("Gradient Magnitude")
            ax2.axis('off')
            fig.colorbar(im2, ax=ax2, shrink=0.8)
            
            ax3.imshow(self.tensor_to_image(self.current_perturbed))
            ax3.set_title("Adversarial Result")
            ax3.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, grad_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to create gradient flow: {str(e)}")

    def create_vulnerability_heatmap(self):
        """Create heatmap showing which image regions are most vulnerable"""
        try:
            if not hasattr(self, 'current_original') or self.current_original is None:
                messagebox.showwarning("No Image", "Please run an attack first!")
                return
            
            heatmap_window = tk.Toplevel(self.root)
            heatmap_window.title("Vulnerability Heatmap")
            heatmap_window.geometry("600x400")
            
            fig = plt.Figure(figsize=(10, 4))
            ax1, ax2 = fig.subplots(1, 2)
            
            # Calculate perturbation magnitude per pixel
            perturbation = (self.current_perturbed - self.current_original).abs()
            vulnerability_map = perturbation.squeeze().mean(dim=0).cpu().detach().numpy()
            
            ax1.imshow(self.tensor_to_image(self.current_original))
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            im2 = ax2.imshow(vulnerability_map, cmap='Reds', alpha=0.7)
            ax2.imshow(self.tensor_to_image(self.current_original), alpha=0.3)
            ax2.set_title("Vulnerability Heatmap")
            ax2.axis('off')
            fig.colorbar(im2, ax=ax2, shrink=0.8, label='Perturbation Magnitude')
            
            canvas = FigureCanvasTkAgg(fig, heatmap_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to create vulnerability heatmap: {str(e)}")

    def update_attack_progress(self, iteration, loss_val, confidence):
        """Update the live attack visualization"""
        self.attack_history['iterations'].append(iteration)
        self.attack_history['loss'].append(loss_val)
        self.attack_history['confidence'].append(confidence)
        
        # Update in the main thread
        def update_plot():
            self.ax5.clear()
            if len(self.attack_history['iterations']) > 0:
                # Normalize loss values to 0-1 range for better visualization
                loss_vals = np.array(self.attack_history['loss'])
                conf_vals = np.array(self.attack_history['confidence'])
                iter_vals = np.array(self.attack_history['iterations'])
                
                # Normalize loss to 0-1 range
                if len(loss_vals) > 1 and np.max(loss_vals) > np.min(loss_vals):
                    loss_normalized = (loss_vals - np.min(loss_vals)) / (np.max(loss_vals) - np.min(loss_vals))
                else:
                    loss_normalized = loss_vals
                
                self.ax5.plot(iter_vals, loss_normalized, 'r-', label=f'Loss (norm)', linewidth=2, marker='o', markersize=3)
                self.ax5.plot(iter_vals, conf_vals, 'b-', label='Confidence', linewidth=2, marker='s', markersize=3)
                
                self.ax5.set_title('Attack Progression', fontsize=10)
                self.ax5.set_xlabel('Iteration', fontsize=9)
                self.ax5.set_ylabel('Value', fontsize=9)
                self.ax5.legend(fontsize=8)
                self.ax5.grid(True, alpha=0.3)
                
                # Better axis limits
                max_iter = max(iter_vals) if len(iter_vals) > 0 else 50
                self.ax5.set_xlim(-1, max_iter + 1)
                self.ax5.set_ylim(-0.1, 1.1)
                
                # Add value annotations for key points
                if len(iter_vals) > 0:
                    # Annotate first and last points
                    self.ax5.annotate(f'{conf_vals[0]:.3f}', 
                                    (iter_vals[0], conf_vals[0]), 
                                    textcoords="offset points", 
                                    xytext=(0,10), 
                                    ha='center', fontsize=7)
                    if len(iter_vals) > 1:
                        self.ax5.annotate(f'{conf_vals[-1]:.3f}', 
                                        (iter_vals[-1], conf_vals[-1]), 
                                        textcoords="offset points", 
                                        xytext=(0,10), 
                                        ha='center', fontsize=7)
            else:
                # Show placeholder when no data
                self.ax5.set_title('Attack Progression', fontsize=10)
                self.ax5.set_xlabel('Iteration', fontsize=9)
                self.ax5.set_ylabel('Value', fontsize=9)
                self.ax5.grid(True, alpha=0.3)
                self.ax5.set_xlim(0, 50)
                self.ax5.set_ylim(0, 1)
                self.ax5.text(25, 0.5, 'Attack Progress\nWill Show Here', ha='center', va='center', 
                             fontsize=10, alpha=0.6, style='italic')
            
            self.canvas.draw_idle()
        
        self.root.after(0, update_plot)
    
    def fgsm_attack_with_progress(self, image, label, epsilon=0.03):
        """FGSM attack with progress tracking"""
        # Get initial loss/confidence
        with torch.no_grad():
            output = self.model(image)
            initial_loss = torch.nn.functional.cross_entropy(output, label).item()
            initial_conf = torch.softmax(output, dim=1).max().item()
            self.update_attack_progress(0, initial_loss, initial_conf)
        
        # Perform FGSM attack
        image.requires_grad = True
        output = self.model(image)
        loss = torch.nn.functional.cross_entropy(output, label)
        self.model.zero_grad()
        loss.backward()
        result = image + epsilon * image.grad.sign()
        result = torch.clamp(result, 0, 1)
        
        # Get final loss/confidence 
        with torch.no_grad():
            final_output = self.model(result)
            final_loss = torch.nn.functional.cross_entropy(final_output, label).item()
            final_conf = torch.softmax(final_output, dim=1).max().item()
            self.update_attack_progress(1, final_loss, final_conf)
        
        return result
    
    def pgd_attack_with_progress(self, image, label, epsilon=0.03, alpha=0.01, iters=40, momentum=0.9):
        """PGD attack with progress tracking"""
        ori_image = image.clone().detach()
        grad_accum = torch.zeros_like(image)
        
        # Initial state
        with torch.no_grad():
            initial_output = self.model(image)
            initial_loss = torch.nn.functional.cross_entropy(initial_output, label).item()
            initial_conf = torch.softmax(initial_output, dim=1).max().item()
            self.update_attack_progress(0, initial_loss, initial_conf)
        
        for i in range(iters):
            image.requires_grad = True
            output = self.model(image)
            loss = torch.nn.functional.cross_entropy(output, label)
            confidence = torch.softmax(output, dim=1).max().item()
            
            # Update progress more frequently for better visualization
            if i % max(1, iters // 10) == 0 or i == iters - 1:
                self.update_attack_progress(i + 1, loss.item(), confidence)
            
            self.model.zero_grad()
            loss.backward()
            
            grad = image.grad.sign()
            grad_accum = momentum * grad_accum + grad
            adv_image = image + alpha * grad_accum.sign()
            
            eta = torch.clamp(adv_image - ori_image, -epsilon, epsilon)
            image = torch.clamp(ori_image + eta, 0, 1).detach()
        
        return image
    
    def deepfool_attack_with_progress(self, image, label, num_classes=10, overshoot=0.02, max_iter=50):
        """DeepFool attack with progress tracking"""
        image = image.clone().detach().requires_grad_(True)
        pert_image = image.clone().detach()
        output = self.model(image)
        _, orig_label = output.max(1)
        if label is not None:
            orig_label = label
        
        num_classes = min(num_classes, output.shape[1])
        
        loops = 0
        while loops < max_iter:
            output = self.model(pert_image)
            logits = output[0]
            orig_class = orig_label.item()
            pert_image.requires_grad = True
            
            # Track progress
            loss = torch.nn.functional.cross_entropy(output, orig_label)
            confidence = torch.softmax(output, dim=1).max().item()
            if loops % max(1, max_iter // 10) == 0:
                self.update_attack_progress(loops, loss.item(), confidence)
            
            try:
                grad_orig = torch.autograd.grad(logits[orig_class], pert_image, retain_graph=True, allow_unused=True)[0]
                if grad_orig is None:
                    grad_orig = torch.zeros_like(pert_image)
            except Exception:
                grad_orig = torch.zeros_like(pert_image)
            
            min_dist = float('inf')
            w = None
            valid_grad_found = False
            
            for k in range(num_classes):
                if k == orig_class:
                    continue
                    
                try:
                    grad_k = torch.autograd.grad(logits[k], pert_image, retain_graph=True, allow_unused=True)[0]
                    if grad_k is None:
                        continue
                        
                    w_k = grad_k - grad_orig
                    norm_w_k = torch.norm(w_k.flatten()) + 1e-8
                    
                    f_k = (logits[k] - logits[orig_class]).item()
                    if abs(f_k) < 1e-8:
                        continue
                        
                    dist = abs(f_k) / norm_w_k
                    
                    if dist < min_dist:
                        min_dist = dist
                        w = w_k
                        valid_grad_found = True
                except Exception:
                    continue
            
            if not valid_grad_found or w is None:
                noise = torch.randn_like(pert_image) * 0.01
                pert_image = torch.clamp(pert_image + noise, 0, 1).detach().requires_grad_(True)
                loops += 1
                continue
                
            r_i = min_dist * w / (torch.norm(w.flatten()) + 1e-8)
            pert_image = pert_image + (1 + overshoot) * r_i
            pert_image = torch.clamp(pert_image, 0, 1).detach().requires_grad_(True)
            
            with torch.no_grad():
                new_output = self.model(pert_image)
                new_label = new_output.max(1)[1].item()
                if new_label != orig_class:
                    break
                    
            loops += 1
        
        # Final progress update
        final_output = self.model(pert_image)
        final_loss = torch.nn.functional.cross_entropy(final_output, orig_label).item()
        final_conf = torch.softmax(final_output, dim=1).max().item()
        self.update_attack_progress(loops, final_loss, final_conf)
        
        return pert_image.detach()
    
    def cw_attack_with_progress(self, image, label, c=1e-2, kappa=0, max_iter=100, lr=0.01):
        """Carlini-Wagner attack with progress tracking"""
        batch_size = image.shape[0]
        w = torch.zeros_like(image, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=lr)
        
        def tanh_rescale(x):
            return (torch.tanh(x) + 1) / 2
        
        for i in range(max_iter):
            adv_image = tanh_rescale(w)
            output = self.model(adv_image)
            
            # L2 distance
            l2_dist = torch.norm((adv_image - image).view(batch_size, -1), dim=1)
            
            # Classification loss
            real = output.gather(1, label.unsqueeze(1)).squeeze(1)
            other = output.masked_fill(torch.zeros_like(output).scatter_(1, label.unsqueeze(1), 1).bool(), -1e9).max(1)[0]
            
            # Prevent NaN by clamping extreme values
            real = torch.clamp(real, -10, 10)
            other = torch.clamp(other, -10, 10)
            
            loss1 = torch.clamp(real - other + kappa, min=0)
            loss2 = l2_dist
            
            # Handle potential NaN/inf values
            loss1 = torch.where(torch.isnan(loss1) | torch.isinf(loss1), torch.zeros_like(loss1), loss1)
            loss2 = torch.where(torch.isnan(loss2) | torch.isinf(loss2), torch.ones_like(loss2), loss2)
            
            loss = loss1 + c * loss2
            loss = loss.mean()
            
            # Track progress
            confidence = torch.softmax(output, dim=1).max().item()
            if i % max(1, max_iter // 20) == 0:
                self.update_attack_progress(i, loss.item(), confidence)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
            
            optimizer.step()
        
        # Final result
        result = tanh_rescale(w)
        final_output = self.model(result)
        final_loss = torch.nn.functional.cross_entropy(final_output, label).item()
        final_conf = torch.softmax(final_output, dim=1).max().item()
        self.update_attack_progress(max_iter, final_loss, final_conf)
        
        return result.detach()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdversarialAttackGUI(root)
    root.mainloop()