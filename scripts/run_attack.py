"""Adversarial Attack Visualizer GUI.

Interactive tool for running FGSM / PGD / DeepFool / CW attacks against a
pretrained ResNet-18 and visualizing the results.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks import ATTACKS
from utils.model_utils import load_model, build_transform
from utils.image_utils import (find_image_url, load_image_from_url,
                               load_image_from_file)


class AdversarialAttackGUI:
    def __init__(self, root):
        load_dotenv()
        self.pexels_api_key = os.getenv('PEXELS_API_KEY', '')

        self.root = root
        self.root.title("Adversarial Attack Visualizer")
        self.root.geometry("1400x800")
        self.root.protocol("WM_DELETE_WINDOW", self.close_program)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.categories = load_model(self.device)
        self.transform = build_transform(224)

        self.current_original = None
        self.current_perturbed = None
        self.attack_history = {'iterations': [], 'loss': [], 'confidence': []}

        self.setup_ui()

    # ------------------------------------------------------------------ UI

    def setup_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_container, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_controls(left_panel)
        self.setup_visualization(right_panel)

    def setup_controls(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        title = ttk.Label(frame, text="Adversarial Attack Visualizer",
                          font=("Arial", 13, "bold"))
        title.pack(pady=(0, 10))

        # Attack method
        attack_frame = ttk.LabelFrame(frame, text="Attack Method", padding="10")
        attack_frame.pack(fill=tk.X, pady=(0, 10))
        self.attack_method_var = tk.StringVar(value="FGSM")
        dropdown = ttk.Combobox(attack_frame, textvariable=self.attack_method_var,
                                state="readonly", values=list(ATTACKS.keys()))
        dropdown.pack(fill=tk.X)

        # Parameters
        self.setup_parameter_controls(frame)

        # Image sources
        search_frame = ttk.LabelFrame(frame, text="Search for Image", padding="10")
        search_frame.pack(fill=tk.X, pady=(0, 10))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(fill=tk.X, pady=(0, 5))
        search_entry.bind('<Return>', lambda e: self.search_and_attack())
        self.search_btn = ttk.Button(search_frame, text="Search & Attack",
                                     command=self.search_and_attack)
        self.search_btn.pack(fill=tk.X)

        url_frame = ttk.LabelFrame(frame, text="Direct Image URL", padding="10")
        url_frame.pack(fill=tk.X, pady=(0, 10))
        self.url_var = tk.StringVar()
        url_entry = ttk.Entry(url_frame, textvariable=self.url_var)
        url_entry.pack(fill=tk.X, pady=(0, 5))
        url_entry.bind('<Return>', lambda e: self.load_from_url())
        self.url_btn = ttk.Button(url_frame, text="Load & Attack",
                                  command=self.load_from_url)
        self.url_btn.pack(fill=tk.X)

        file_frame = ttk.LabelFrame(frame, text="Local Image", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        self.file_btn = ttk.Button(file_frame, text="Open File & Attack",
                                   command=self.load_from_file)
        self.file_btn.pack(fill=tk.X)

        # Advanced visualizations
        viz_frame = ttk.LabelFrame(frame, text="Advanced Visualizations", padding="10")
        viz_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(viz_frame, text="3D Attack Surface",
                   command=self.plot_attack_surface).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(viz_frame, text="Gradient Flow",
                   command=self.visualize_gradient_flow).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(viz_frame, text="Vulnerability Heatmap",
                   command=self.create_vulnerability_heatmap).pack(fill=tk.X)

        # Export
        export_frame = ttk.LabelFrame(frame, text="Export", padding="10")
        export_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(export_frame, text="Save Figure...",
                   command=self.save_figure).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(export_frame, text="Save Adversarial Image...",
                   command=self.save_adversarial_image).pack(fill=tk.X)

        # Progress and status
        self.progress = ttk.Progressbar(frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 5))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.status_var, wraplength=280,
                  anchor='center', justify='center').pack(pady=(0, 10))

        # Results
        results_frame = ttk.LabelFrame(frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_text = tk.Text(results_frame, height=8, width=36, wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical",
                                          command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(frame, text="Close", command=self.close_program).pack(
            fill=tk.X, pady=(10, 5))

    def setup_parameter_controls(self, parent):
        params_frame = ttk.LabelFrame(parent, text="Attack Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))

        epsilon_frame = ttk.Frame(params_frame)
        epsilon_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(epsilon_frame, text="Epsilon:", width=12).pack(side=tk.LEFT)
        self.epsilon_var = tk.DoubleVar(value=0.03)
        self.epsilon_label = ttk.Label(epsilon_frame, text="0.030", width=8)
        self.epsilon_label.pack(side=tk.RIGHT)
        ttk.Scale(epsilon_frame, from_=0.001, to=0.1, variable=self.epsilon_var,
                  orient=tk.HORIZONTAL,
                  command=lambda v: self.epsilon_label.config(
                      text=f"{float(v):.3f}")).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        iter_frame = ttk.Frame(params_frame)
        iter_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(iter_frame, text="Iterations:", width=12).pack(side=tk.LEFT)
        self.iterations_var = tk.IntVar(value=40)
        self.iter_label = ttk.Label(iter_frame, text="40", width=8)
        self.iter_label.pack(side=tk.RIGHT)
        ttk.Scale(iter_frame, from_=10, to=100, variable=self.iterations_var,
                  orient=tk.HORIZONTAL,
                  command=lambda v: self.iter_label.config(
                      text=f"{int(float(v))}")).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def setup_visualization(self, parent):
        viz_frame = ttk.LabelFrame(parent, text="Visualization", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(20, 8))
        self.ax1 = plt.subplot2grid((2, 4), (0, 0))
        self.ax2 = plt.subplot2grid((2, 4), (0, 1))
        self.ax3 = plt.subplot2grid((2, 4), (0, 2))
        self.ax4 = plt.subplot2grid((2, 4), (0, 3))
        self.ax5 = plt.subplot2grid((2, 4), (1, 0), colspan=4)

        self.fig.patch.set_facecolor('white')
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10,
                                 wspace=0.3, hspace=0.4)

        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.show_placeholder()

    def show_placeholder(self):
        placeholder = np.ones((224, 224, 3)) * 0.8
        for ax, title in zip([self.ax1, self.ax2, self.ax3],
                             ["Original Image\n(No image loaded)",
                              "Perturbation\n(Will show here)",
                              "Adversarial Image\n(After attack)"]):
            ax.clear()
            ax.imshow(placeholder)
            ax.set_title(title)
            ax.axis('off')

        self.ax4.clear()
        self.ax4.set_title("Top 5 Predictions")
        self.ax4.set_xlabel('Rank')
        self.ax4.set_ylabel('Confidence')
        self.ax4.text(0.5, 0.5, 'Predictions will\nshow here after\nattack',
                      transform=self.ax4.transAxes, ha='center', va='center',
                      fontsize=10, alpha=0.6, style='italic')

        self.ax5.clear()
        self.ax5.set_title("Attack Progression")
        self.ax5.set_xlabel('Iteration')
        self.ax5.set_ylabel('Value')
        self.ax5.text(0.5, 0.5, 'Attack progression will show here',
                      transform=self.ax5.transAxes, ha='center', va='center',
                      fontsize=12, alpha=0.6, style='italic')
        self.ax5.grid(True, alpha=0.3)
        self.canvas.draw()

    # -------------------------------------------------------------- helpers

    def class_name(self, class_id):
        if 0 <= class_id < len(self.categories):
            return self.categories[class_id]
        return f"class {class_id}"

    def set_status(self, message):
        """Thread-safe status update."""
        self.root.after(0, lambda: self.status_var.set(message))

    def close_program(self):
        try:
            plt.close('all')
            self.root.quit()
            self.root.destroy()
        except Exception:
            sys.exit(0)

    def disable_controls(self):
        for btn in (self.search_btn, self.url_btn, self.file_btn):
            btn.config(state='disabled')

    def enable_controls(self):
        for btn in (self.search_btn, self.url_btn, self.file_btn):
            btn.config(state='normal')

    def tensor_to_image(self, tensor):
        """Convert a [0, 1] pixel-space tensor to a displayable numpy image."""
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    def start_worker(self, target, *args):
        self.disable_controls()
        self.progress.start()
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()

    # -------------------------------------------------------- input actions

    def search_and_attack(self):
        term = self.search_var.get().strip()
        if not term:
            self.show_error("Please enter a search term")
            return
        self.set_status(f"Searching for '{term}'...")
        self.start_worker(self._search_worker, term)

    def load_from_url(self):
        url = self.url_var.get().strip()
        if not url:
            self.show_error("Please enter an image URL")
            return
        self.set_status("Loading image from URL...")
        self.start_worker(self._url_worker, url)

    def load_from_file(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        self.set_status(f"Loading {os.path.basename(path)}...")
        self.start_worker(self._file_worker, path)

    def _search_worker(self, term):
        try:
            image_url = find_image_url(term, self.pexels_api_key)
            self.set_status(f"Found image for '{term}', running attack...")
            image = load_image_from_url(image_url)
            self._process_image(image, f"Search: {term}")
        except Exception as e:
            self.report_error(f"Search error: {e}")

    def _url_worker(self, url):
        try:
            image = load_image_from_url(url)
            self._process_image(image, "Direct URL")
        except Exception as e:
            self.report_error(f"URL error: {e}")

    def _file_worker(self, path):
        try:
            image = load_image_from_file(path)
            self._process_image(image, f"File: {os.path.basename(path)}")
        except Exception as e:
            self.report_error(f"File error: {e}")

    # --------------------------------------------------------------- attack

    def run_attack(self, attack_method, input_tensor, label, epsilon, iterations,
                   callback=None):
        """Dispatch to the selected attack with appropriate parameters."""
        attack_fn = ATTACKS[attack_method]
        kwargs = {'callback': callback}
        if attack_method == 'FGSM':
            kwargs['epsilon'] = epsilon
        elif attack_method == 'PGD':
            kwargs.update(epsilon=epsilon, iters=iterations)
        elif attack_method == 'DeepFool':
            kwargs.update(num_classes=10, max_iter=iterations)
        elif attack_method == 'CW':
            kwargs.update(c=1.0, max_iter=iterations)
        return attack_fn(self.model, input_tensor, label, **kwargs)

    def _process_image(self, image, source):
        try:
            attack_method = self.attack_method_var.get()
            epsilon = self.epsilon_var.get()
            iterations = self.iterations_var.get()

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                original_output = self.model(input_tensor)
                original_pred = original_output.argmax(dim=1)
                original_conf = torch.softmax(original_output, dim=1).max().item()

            self.attack_history = {'iterations': [], 'loss': [], 'confidence': []}
            self.set_status(f"Running {attack_method} attack...")

            perturbed = self.run_attack(attack_method, input_tensor,
                                        original_pred, epsilon, iterations,
                                        callback=self.update_attack_progress)

            with torch.no_grad():
                adv_output = self.model(perturbed)
                adv_pred = adv_output.argmax(dim=1)
                adv_conf = torch.softmax(adv_output, dim=1).max().item()

            self.current_original = input_tensor
            self.current_perturbed = perturbed

            self.root.after(0, lambda: self.show_results(
                source, attack_method, original_pred.item(), original_conf,
                adv_pred.item(), adv_conf, input_tensor, perturbed, adv_output))
        except Exception as e:
            self.report_error(f"Processing error: {e}")

    def show_results(self, source, attack_method, original_pred, original_conf,
                     adv_pred, adv_conf, original_tensor, perturbed_tensor,
                     adv_output):
        try:
            original_img = self.tensor_to_image(original_tensor)
            adversarial_img = self.tensor_to_image(perturbed_tensor)
            perturbation = adversarial_img - original_img

            for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5):
                ax.clear()

            self.ax1.imshow(original_img)
            self.ax1.set_title(f"Original\n{self.class_name(original_pred)} "
                               f"({original_conf:.3f})")
            self.ax1.axis('off')

            self.ax2.imshow(np.clip(np.abs(perturbation) * 10, 0, 1))
            self.ax2.set_title("Perturbation\n(Enhanced 10x)")
            self.ax2.axis('off')

            self.ax3.imshow(adversarial_img)
            self.ax3.set_title(f"Adversarial\n{self.class_name(adv_pred)} "
                               f"({adv_conf:.3f})")
            self.ax3.axis('off')

            # Top-5 predictions on the adversarial image
            confidences = torch.softmax(adv_output, dim=1).squeeze().cpu().numpy()
            top5 = np.argsort(confidences)[-5:][::-1]
            top5_conf = confidences[top5]
            colors = ['red' if adv_pred != original_pred else 'green'] + \
                     ['lightblue'] * 4
            bars = self.ax4.bar(range(5), top5_conf, color=colors)
            self.ax4.set_title("Top 5 Predictions (adversarial)",
                               fontsize=11, fontweight='bold')
            self.ax4.set_ylabel("Confidence")
            self.ax4.set_xticks(range(5))
            self.ax4.set_xticklabels(
                [self.class_name(int(c))[:14] for c in top5],
                fontsize=7, rotation=20, ha='right')
            self.ax4.set_ylim(0, 1)
            self.ax4.grid(True, alpha=0.3, axis='y')
            for bar, conf in zip(bars, top5_conf):
                self.ax4.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + 0.01, f'{conf:.3f}',
                              ha='center', va='bottom', fontsize=8)

            self.plot_attack_history(self.ax5)
            self.canvas.draw()

            l2 = float(np.sqrt(np.sum(perturbation ** 2)))
            linf = float(np.max(np.abs(perturbation)))
            success = adv_pred != original_pred
            results_text = (
                f"Source: {source}\n"
                f"Attack: {attack_method}\n"
                f"Original: {self.class_name(original_pred)} "
                f"({original_conf:.3f})\n"
                f"Adversarial: {self.class_name(adv_pred)} ({adv_conf:.3f})\n"
                f"Attack Success: {'Yes' if success else 'No'}\n"
                f"Perturbation L2: {l2:.4f}\n"
                f"Perturbation L-inf: {linf:.4f}\n"
            )
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_text)
        except Exception as e:
            print(f"Error in show_results: {e}")
        finally:
            self.progress.stop()
            self.enable_controls()
            self.status_var.set("Attack completed")

    def plot_attack_history(self, ax):
        ax.set_title("Attack Progression", fontsize=13, fontweight='bold')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        iters = self.attack_history['iterations']
        if not iters:
            ax.text(0.5, 0.5, 'No progression data', transform=ax.transAxes,
                    ha='center', va='center', alpha=0.6, style='italic')
            return

        losses = np.array(self.attack_history['loss'], dtype=float)
        confs = np.array(self.attack_history['confidence'], dtype=float)
        if len(losses) > 1 and losses.max() > losses.min():
            losses = (losses - losses.min()) / (losses.max() - losses.min())

        ax.plot(iters, losses, 'r-', label='Loss (normalized)', linewidth=2,
                marker='o', markersize=3)
        ax.plot(iters, confs, 'b-', label='Confidence', linewidth=2,
                marker='s', markersize=3)
        ax.legend(fontsize=10)
        ax.set_xlim(-0.5, max(iters) + 0.5)
        ax.set_ylim(-0.05, 1.05)

    def update_attack_progress(self, iteration, loss_val, confidence):
        """Attack progress callback; safe to call from worker threads."""
        self.attack_history['iterations'].append(iteration)
        self.attack_history['loss'].append(loss_val)
        self.attack_history['confidence'].append(confidence)

        def redraw():
            self.ax5.clear()
            self.plot_attack_history(self.ax5)
            self.canvas.draw_idle()

        self.root.after(0, redraw)

    # ------------------------------------------------------- error handling

    def report_error(self, message):
        """Thread-safe error reporting."""
        self.root.after(0, lambda: self.show_error(message))

    def show_error(self, message):
        self.progress.stop()
        self.enable_controls()
        self.status_var.set("Error occurred")
        messagebox.showerror("Error", message)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error: {message}")

    # ---------------------------------------------------------------- export

    def save_figure(self):
        if self.current_original is None:
            messagebox.showwarning("No Results", "Please run an attack first!")
            return
        path = filedialog.asksaveasfilename(
            title="Save figure", defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches='tight')
            self.status_var.set(f"Figure saved to {os.path.basename(path)}")

    def save_adversarial_image(self):
        if self.current_perturbed is None:
            messagebox.showwarning("No Results", "Please run an attack first!")
            return
        path = filedialog.asksaveasfilename(
            title="Save adversarial image", defaultextension=".png",
            filetypes=[("PNG image", "*.png")])
        if path:
            img = (self.tensor_to_image(self.current_perturbed) * 255)
            from PIL import Image
            Image.fromarray(img.astype(np.uint8)).save(path)
            self.status_var.set(f"Image saved to {os.path.basename(path)}")

    # -------------------------------------------- advanced visualizations

    def plot_attack_surface(self):
        """Sweep epsilon across attack methods and plot a 3D success surface."""
        if self.current_original is None:
            messagebox.showwarning("No Image", "Please run an attack first!")
            return

        surface_window = tk.Toplevel(self.root)
        surface_window.title("3D Attack Surface - Computing...")
        surface_window.geometry("800x600")
        progress_label = ttk.Label(surface_window,
                                   text="Computing attack surface...")
        progress_label.pack(pady=20)

        def worker():
            epsilon_range = np.linspace(0.01, 0.1, 6)
            methods = list(ATTACKS.keys())
            points = []
            total = len(methods) * len(epsilon_range)
            done = 0

            with torch.no_grad():
                original_output = self.model(self.current_original)
                original_pred = original_output.argmax(dim=1)
                original_conf = torch.softmax(original_output, dim=1).max().item()

            for mi, method in enumerate(methods):
                for eps in epsilon_range:
                    try:
                        perturbed = self.run_attack(
                            method, self.current_original.clone(),
                            original_pred, eps, 15)
                        with torch.no_grad():
                            adv_output = self.model(perturbed)
                            adv_pred = adv_output.argmax(dim=1)
                            adv_conf = torch.softmax(adv_output,
                                                     dim=1).max().item()
                        success = 1.0 if (original_pred != adv_pred).item() else 0.0
                        points.append((eps, mi, success,
                                       abs(original_conf - adv_conf)))
                    except Exception as e:
                        print(f"Attack surface error ({method}, eps={eps}): {e}")
                        points.append((eps, mi, 0.0, 0.0))
                    done += 1
                    self.root.after(0, lambda d=done: progress_label.config(
                        text=f"Computing attack surface... {d}/{total}"))

            def show():
                if not surface_window.winfo_exists():
                    return
                progress_label.destroy()
                fig = plt.Figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                eps_v, method_v, succ_v, conf_v = zip(*points)
                scatter = ax.scatter(eps_v, method_v, succ_v, c=conf_v,
                                     cmap='viridis', s=80, alpha=0.8)
                ax.set_xlabel('Epsilon')
                ax.set_ylabel('Attack Method')
                ax.set_zlabel('Success')
                ax.set_title('Attack Success Surface')
                ax.set_yticks(range(len(methods)))
                ax.set_yticklabels(methods)
                fig.colorbar(scatter, label='Confidence Change', shrink=0.8)
                canvas = FigureCanvasTkAgg(fig, surface_window)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                surface_window.title("3D Attack Surface")

            self.root.after(0, show)

        threading.Thread(target=worker, daemon=True).start()

    def visualize_gradient_flow(self):
        """Show input-gradient magnitude alongside original and adversarial."""
        if self.current_original is None:
            messagebox.showwarning("No Image", "Please run an attack first!")
            return
        try:
            grad_window = tk.Toplevel(self.root)
            grad_window.title("Gradient Flow Visualization")
            grad_window.geometry("900x400")

            fig = plt.Figure(figsize=(12, 4))
            ax1, ax2, ax3 = fig.subplots(1, 3)

            image = self.current_original.clone().requires_grad_(True)
            output = self.model(image)
            pred = output.argmax(dim=1)
            loss = torch.nn.functional.cross_entropy(output, pred)
            grad = torch.autograd.grad(loss, image)[0]
            grad_mag = grad.squeeze(0).abs().mean(dim=0).cpu().numpy()

            ax1.imshow(self.tensor_to_image(self.current_original))
            ax1.set_title("Original Image")
            ax1.axis('off')

            im2 = ax2.imshow(grad_mag, cmap='hot')
            ax2.set_title("Input Gradient Magnitude")
            ax2.axis('off')
            fig.colorbar(im2, ax=ax2, shrink=0.8)

            ax3.imshow(self.tensor_to_image(self.current_perturbed))
            ax3.set_title("Adversarial Result")
            ax3.axis('off')

            canvas = FigureCanvasTkAgg(fig, grad_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Visualization Error",
                                 f"Failed to create gradient flow: {e}")

    def create_vulnerability_heatmap(self):
        """Show which image regions received the largest perturbation."""
        if self.current_original is None or self.current_perturbed is None:
            messagebox.showwarning("No Image", "Please run an attack first!")
            return
        try:
            heatmap_window = tk.Toplevel(self.root)
            heatmap_window.title("Vulnerability Heatmap")
            heatmap_window.geometry("700x400")

            fig = plt.Figure(figsize=(10, 4))
            ax1, ax2 = fig.subplots(1, 2)

            perturbation = (self.current_perturbed - self.current_original).abs()
            vulnerability = perturbation.squeeze(0).mean(dim=0).cpu().numpy()

            ax1.imshow(self.tensor_to_image(self.current_original))
            ax1.set_title("Original Image")
            ax1.axis('off')

            ax2.imshow(self.tensor_to_image(self.current_original), alpha=0.4)
            im2 = ax2.imshow(vulnerability, cmap='Reds', alpha=0.6)
            ax2.set_title("Perturbation Magnitude")
            ax2.axis('off')
            fig.colorbar(im2, ax=ax2, shrink=0.8, label='Perturbation Magnitude')

            canvas = FigureCanvasTkAgg(fig, heatmap_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Visualization Error",
                                 f"Failed to create heatmap: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AdversarialAttackGUI(root)
    root.mainloop()
