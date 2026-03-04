import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import sys
import yaml
import glob
import threading

# Set appearance mode and color theme
ctk.set_appearance_mode("light")  # Changed to light mode
ctk.set_default_color_theme("blue")  # blue

class SmartCityDataTrainer:  # Renamed class
    def __init__(self, root):
        self.root = root
        self.root.title("Smart City Data Training")  # title
        self.root.geometry("1500x900")
        
        # Variables
        self.project_path = ctk.StringVar()
        self.yaml_file = ctk.StringVar()
        self.model_version = ctk.StringVar(value="yolov11n.pt")
        self.device = ctk.StringVar(value="0")
        self.conf_threshold = ctk.DoubleVar(value=0.25)
        self.train_from_scratch = ctk.BooleanVar(value=True)
        self.epochs = ctk.IntVar(value=200)
        
        self.processing = False
        self.training_thread = None
        self.help_visible = False
        
        # Create main container with padding
        main_frame = ctk.CTkFrame(root, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create sections
        self.create_header(main_frame)
        self.create_project_selection(main_frame)
        self.create_validation_section(main_frame)
        self.create_training_parameters(main_frame)
        self.create_console_section(main_frame)
        
    def create_header(self, parent):
        """Create header section"""
        header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 15))
        
        # Create a container for all header elements
        header_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_container.pack(fill="x")
        
        # Left side: Logo and Title/subtitle
        left_frame = ctk.CTkFrame(header_container, fg_color="transparent")
        left_frame.pack(side="left", fill="x", expand=True)
        
        # Try to load logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Smart City.jpg")
        if os.path.exists(logo_path):
            try:
                from PIL import Image
                logo_image = Image.open(logo_path)
                # Resize logo to appropriate size (e.g., 40x40)
                logo_image = logo_image.resize((40, 40), Image.Resampling.LANCZOS)
                logo_photo = ctk.CTkImage(light_image=logo_image, dark_image=logo_image, size=(40, 40))
                
                logo_label = ctk.CTkLabel(left_frame, image=logo_photo, text="")
                logo_label.image = logo_photo  # Keep a reference
                logo_label.pack(side="left", padx=(0, 15))
            except Exception as e:
                print(f"Could not load logo: {e}")
        
        # Title and subtitle container
        text_container = ctk.CTkFrame(left_frame, fg_color="transparent")
        text_container.pack(side="left", fill="x", expand=True)
        
        # Updated title
        title = ctk.CTkLabel(text_container, text="Smart City Data Training", 
                            font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(anchor="w")
        
        subtitle = ctk.CTkLabel(text_container, 
                               text="fine-tuning • Optimized parameters",
                               text_color="gray60")
        subtitle.pack(anchor="w")
        
        # Right side: Dependencies and About buttons
        buttons_frame = ctk.CTkFrame(header_container, fg_color="transparent")
        buttons_frame.pack(side="right", anchor="ne")
        
        about_btn = ctk.CTkButton(buttons_frame, text="ℹ About", command=self.show_about,
                                 width=100, height=32)
        about_btn.pack(side="left", padx=(0, 10))
        
        dependencies_btn = ctk.CTkButton(buttons_frame, text="🔧 Dependencies", 
                                        command=self.show_dependencies,
                                        width=140, height=32)
        dependencies_btn.pack(side="left")
        
    def create_project_selection(self, parent):
        """Create project path selection"""
        path_frame = ctk.CTkFrame(parent)
        path_frame.pack(fill="x", pady=(0, 15))
        
        # Title
        title = ctk.CTkLabel(path_frame, text="Project Configuration", 
                            font=ctk.CTkFont(size=14, weight="bold"))
        title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Project path row
        path_row = ctk.CTkFrame(path_frame, fg_color="transparent")
        path_row.pack(fill="x", padx=15, pady=(0, 10))
        
        path_label = ctk.CTkLabel(path_row, text="Project Folder:", width=120, anchor="w")
        path_label.pack(side="left", padx=(0, 10))
        
        path_entry = ctk.CTkEntry(path_row, textvariable=self.project_path, height=32)
        path_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        browse_btn = ctk.CTkButton(path_row, text="Browse", command=self.browse_project, width=100)
        browse_btn.pack(side="left")
        
        # Status label
        self.project_status = ctk.CTkLabel(path_frame, 
                                          text="Select project folder with images/, labels/, and .yaml file",
                                          text_color="gray60", anchor="w")
        self.project_status.pack(fill="x", padx=15, pady=(0, 10))
        
        # Help button
        self.help_btn = ctk.CTkButton(path_frame, text="? Show Expected Folder Structure",
                                     command=self.toggle_help, width=240, height=32,
                                     fg_color="gray30", hover_color="gray40")
        self.help_btn.pack(anchor="w", padx=15, pady=(0, 10))
        
        # Hidden info box
        self.info_frame = ctk.CTkFrame(path_frame, fg_color="gray20")
        
        info_text = """Expected folder structure:
  project_folder/
    ├── images/
    │   ├── train/  (your training images)
    │   ├── val/    (validation images)
    │   └── test/   (test images)
    ├── labels/
    │   ├── train/  (YOLO format labels)
    │   ├── val/
    │   └── test/
    └── data.yaml   (dataset configuration)"""
        
        self.info_label = ctk.CTkLabel(self.info_frame, text=info_text, 
                                      text_color="gray70", anchor="w", justify="left",
                                      font=ctk.CTkFont(family="Courier", size=11))
        self.info_label.pack(anchor="w", padx=15, pady=10)
        
    def create_validation_section(self, parent):
        """Create validation section"""
        val_frame = ctk.CTkFrame(parent)
        val_frame.pack(fill="x", pady=(0, 15))
        
        # Title
        title = ctk.CTkLabel(val_frame, text="Dataset Validation",
                            font=ctk.CTkFont(size=14, weight="bold"))
        title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Button and status row
        button_status_row = ctk.CTkFrame(val_frame, fg_color="transparent")
        button_status_row.pack(fill="x", padx=15, pady=(0, 15))
        
        validate_btn = ctk.CTkButton(button_status_row, text="Validate Dataset",
                                     command=self.validate_dataset, width=140)
        validate_btn.pack(side="left", padx=(0, 20))
        
        # Status variables
        self.val_images = ctk.StringVar(value="Not checked")
        self.val_labels = ctk.StringVar(value="Not checked")
        self.val_yaml = ctk.StringVar(value="Not checked")
        self.val_format = ctk.StringVar(value="Not checked")
        
        # Status indicators
        status_container = ctk.CTkFrame(button_status_row, fg_color="transparent")
        status_container.pack(side="left", fill="x", expand=True)
        
        # Images status
        img_frame = ctk.CTkFrame(status_container, fg_color="transparent")
        img_frame.pack(side="left", padx=(0, 15))
        ctk.CTkLabel(img_frame, text="Images:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 5))
        ctk.CTkLabel(img_frame, textvariable=self.val_images, text_color="gray60").pack(side="left")
        
        # Labels status
        lbl_frame = ctk.CTkFrame(status_container, fg_color="transparent")
        lbl_frame.pack(side="left", padx=(0, 15))
        ctk.CTkLabel(lbl_frame, text="Labels:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 5))
        ctk.CTkLabel(lbl_frame, textvariable=self.val_labels, text_color="gray60").pack(side="left")
        
        # YAML status
        yaml_frame = ctk.CTkFrame(status_container, fg_color="transparent")
        yaml_frame.pack(side="left", padx=(0, 15))
        ctk.CTkLabel(yaml_frame, text="YAML:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 5))
        ctk.CTkLabel(yaml_frame, textvariable=self.val_yaml, text_color="gray60").pack(side="left")
        
        # Format status
        fmt_frame = ctk.CTkFrame(status_container, fg_color="transparent")
        fmt_frame.pack(side="left")
        ctk.CTkLabel(fmt_frame, text="Format:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 5))
        ctk.CTkLabel(fmt_frame, textvariable=self.val_format, text_color="gray60").pack(side="left")
        
    def create_training_parameters(self, parent):
        """Create training parameters section"""
        params_frame = ctk.CTkFrame(parent)
        params_frame.pack(fill="x", pady=(0, 15))
        
        # Title
        title = ctk.CTkLabel(params_frame, text="Training Parameters",
                            font=ctk.CTkFont(size=14, weight="bold"))
        title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # First row: Model Version and Device
        row1 = ctk.CTkFrame(params_frame, fg_color="transparent")
        row1.pack(fill="x", padx=15, pady=(0, 10))
        
        # Model selection
        model_container = ctk.CTkFrame(row1, fg_color="transparent")
        model_container.pack(side="left", padx=(0, 30))
        
        ctk.CTkLabel(model_container, text="Model Version:", width=120, anchor="w").pack(side="left", padx=(0, 10))
        model_combo = ctk.CTkComboBox(model_container, variable=self.model_version, width=150,
                                      values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                                             "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
                                      state="readonly")
        model_combo.pack(side="left", padx=(0, 10))
        
        self.model_info = ctk.CTkLabel(model_container, text="(Nano - fastest, smallest)",
                                       text_color="gray60")
        self.model_info.pack(side="left")
        
        # Device selection
        device_container = ctk.CTkFrame(row1, fg_color="transparent")
        device_container.pack(side="left")
        
        ctk.CTkLabel(device_container, text="Device:", width=60, anchor="w").pack(side="left", padx=(0, 10))
        device_combo = ctk.CTkComboBox(device_container, variable=self.device, width=100,
                                       values=["0", "cpu"], state="readonly")
        device_combo.pack(side="left")
        
        # Update model info
        def update_model_info(*args):
            model = self.model_version.get()
            if 'n.pt' in model:
                info = "(Nano - fastest, best for learning)"
            elif 's.pt' in model:
                info = "(Small - balanced)"
            elif 'm.pt' in model:
                info = "(Medium - higher accuracy)"
            else:
                info = ""
            self.model_info.configure(text=info)
        
        self.model_version.trace('w', update_model_info)
        
        # Second row: Confidence and Epochs
        row2 = ctk.CTkFrame(params_frame, fg_color="transparent")
        row2.pack(fill="x", padx=15, pady=(0, 10))
        
        # Confidence threshold
        conf_container = ctk.CTkFrame(row2, fg_color="transparent")
        conf_container.pack(side="left", padx=(0, 30))
        
        ctk.CTkLabel(conf_container, text="Confidence:", width=120, anchor="w").pack(side="left", padx=(0, 10))
        conf_slider = ctk.CTkSlider(conf_container, from_=0.1, to=0.9, variable=self.conf_threshold,
                                    width=200, number_of_steps=80)
        conf_slider.pack(side="left", padx=(0, 10))
        
        self.conf_label = ctk.CTkLabel(conf_container, text=f"{self.conf_threshold.get():.2f}",
                                       width=50, anchor="w")
        self.conf_label.pack(side="left")
        
        # Update confidence label
        def update_conf_label(*args):
            self.conf_label.configure(text=f"{self.conf_threshold.get():.2f}")
        self.conf_threshold.trace('w', update_conf_label)
        
        # Epochs control
        epochs_container = ctk.CTkFrame(row2, fg_color="transparent")
        epochs_container.pack(side="left")
        
        ctk.CTkLabel(epochs_container, text="Epochs:", width=60, anchor="w").pack(side="left", padx=(0, 10))
        epochs_entry = ctk.CTkEntry(epochs_container, textvariable=self.epochs, width=100)
        epochs_entry.pack(side="left")
        
        # Third row: Train from scratch checkbox
        row3 = ctk.CTkFrame(params_frame, fg_color="transparent")
        row3.pack(fill="x", padx=15, pady=(0, 10))
        
        scratch_check = ctk.CTkCheckBox(row3, text="Train from Scratch (no pretrained weights)",
                                        variable=self.train_from_scratch)
        scratch_check.pack(side="left")
        
        # Fourth row: Fixed parameters info
        info_label = ctk.CTkLabel(params_frame,
                                 text="Fixed optimized: LR=0.01, Batch=1, No augmentation (for mini-datasets)",
                                 text_color="gray60")
        info_label.pack(anchor="w", padx=15, pady=(0, 10))
        
        # Training buttons
        button_row = ctk.CTkFrame(params_frame, fg_color="transparent")
        button_row.pack(fill="x", padx=15, pady=(0, 15))
        
        self.train_btn = ctk.CTkButton(button_row, text="▶ Start Fine-Tuning",
                                       command=self.start_training, width=180, height=36,
                                       fg_color="#2fa572", hover_color="#257a53")
        self.train_btn.pack(side="left", padx=(0, 10))
        
        self.stop_btn = ctk.CTkButton(button_row, text="⬛ Stop", command=self.stop_training,
                                      width=100, height=36, state="disabled",
                                      fg_color="#d63031", hover_color="#a82828")
        self.stop_btn.pack(side="left")
        
    def create_console_section(self, parent):
        """Create console and view buttons section"""
        console_frame = ctk.CTkFrame(parent)
        console_frame.pack(fill="both", expand=True)
        
        # Title with view buttons
        header = ctk.CTkFrame(console_frame, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(15, 10))
        
        ctk.CTkLabel(header, text="Training Console",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
        
        # View buttons
        btn_container = ctk.CTkFrame(header, fg_color="transparent")
        btn_container.pack(side="right")
        
        self.view_before_btn = ctk.CTkButton(btn_container, text="View Before", width=100,
                                            command=self.view_before, state="disabled",
                                            height=28)
        self.view_before_btn.pack(side="left", padx=2)
        
        self.view_after_btn = ctk.CTkButton(btn_container, text="View After", width=100,
                                           command=self.view_after, state="disabled",
                                           height=28)
        self.view_after_btn.pack(side="left", padx=2)
        
        self.view_val_btn = ctk.CTkButton(btn_container, text="View Val", width=100,
                                         command=self.view_val, state="disabled",
                                         height=28)
        self.view_val_btn.pack(side="left", padx=2)
        
        self.view_metrics_btn = ctk.CTkButton(btn_container, text="View Metrics", width=100,
                                             command=self.view_metrics, state="disabled",
                                             height=28)
        self.view_metrics_btn.pack(side="left", padx=2)
        
        # Console text area
        self.console = ctk.CTkTextbox(console_frame, height=300, wrap="word",
                                     font=ctk.CTkFont(family="Courier", size=11))
        self.console.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Updated log message
        self.log("Smart City Data Training - Ready")
        self.log("Select project folder and validate dataset to begin")
        
    def toggle_help(self):
        """Toggle help information visibility"""
        if self.help_visible:
            self.info_frame.pack_forget()
            self.help_btn.configure(text="? Show Expected Folder Structure")
            self.help_visible = False
        else:
            self.info_frame.pack(fill="x", padx=15, pady=(0, 10))
            self.help_btn.configure(text="✕ Hide Folder Structure")
            self.help_visible = True
    
    def browse_project(self):
        """Browse for project folder"""
        folder = filedialog.askdirectory(title="Select YOLO Project Folder")
        if folder:
            self.project_path.set(folder)
            self.project_status.configure(text=f"Selected: {folder}", text_color="gray60")
            self.log(f"\nProject folder selected: {folder}")
    
    def validate_dataset(self):
        """Validate the dataset structure"""
        path = self.project_path.get()
        if not path:
            messagebox.showwarning("No Path", "Please select a project folder first")
            return
        
        self.log("\n" + "=" * 60)
        self.log("VALIDATING DATASET...")
        self.log("=" * 60)
        
        issues = []
        
        # Check images
        img_train = os.path.join(path, "images", "train")
        img_val = os.path.join(path, "images", "val")
        img_test = os.path.join(path, "images", "test")
        
        train_imgs = len(glob.glob(os.path.join(img_train, "*.*"))) if os.path.exists(img_train) else 0
        val_imgs = len(glob.glob(os.path.join(img_val, "*.*"))) if os.path.exists(img_val) else 0
        test_imgs = len(glob.glob(os.path.join(img_test, "*.*"))) if os.path.exists(img_test) else 0
        
        if train_imgs > 0:
            self.val_images.set(f"✓ {train_imgs} train, {val_imgs} val, {test_imgs} test")
            self.log(f"✓ Images found: {train_imgs} train, {val_imgs} val, {test_imgs} test")
        else:
            self.val_images.set("✗ No images")
            issues.append("No training images found")
            self.log("✗ No training images found")
        
        # Check labels
        lbl_train = os.path.join(path, "labels", "train")
        lbl_val = os.path.join(path, "labels", "val")
        lbl_test = os.path.join(path, "labels", "test")
        
        train_lbls = len(glob.glob(os.path.join(lbl_train, "*.txt"))) if os.path.exists(lbl_train) else 0
        val_lbls = len(glob.glob(os.path.join(lbl_val, "*.txt"))) if os.path.exists(lbl_val) else 0
        test_lbls = len(glob.glob(os.path.join(lbl_test, "*.txt"))) if os.path.exists(lbl_test) else 0
        
        if train_lbls > 0:
            self.val_labels.set(f"✓ {train_lbls} train, {val_lbls} val, {test_lbls} test")
            self.log(f"✓ Labels found: {train_lbls} train, {val_lbls} val, {test_lbls} test")
        else:
            self.val_labels.set("✗ No labels")
            issues.append("No training labels found")
            self.log("✗ No training labels found")
        
        # Check YAML
        yaml_files = glob.glob(os.path.join(path, "*.yaml"))
        if yaml_files:
            self.yaml_file.set(yaml_files[0])
            self.val_yaml.set(f"✓ {os.path.basename(yaml_files[0])}")
            self.log(f"✓ YAML found: {os.path.basename(yaml_files[0])}")
            
            try:
                with open(yaml_files[0], 'r') as f:
                    yaml_content = yaml.safe_load(f)
                    if 'names' in yaml_content and 'nc' in yaml_content:
                        num_classes = yaml_content['nc']
                        class_names = yaml_content['names']
                        self.log(f"  Classes: {num_classes} - {class_names}")
            except:
                pass
        else:
            self.val_yaml.set("✗ Not found")
            issues.append("No .yaml file found")
            self.log("✗ No .yaml file found")
        
        # Check format
        if train_lbls > 0:
            sample_label = glob.glob(os.path.join(lbl_train, "*.txt"))[0]
            try:
                with open(sample_label, 'r') as f:
                    line = f.readline().strip()
                    parts = line.split()
                    if len(parts) == 5:
                        self.val_format.set("✓ YOLO format")
                        self.log("✓ Label format is valid YOLO")
                    else:
                        self.val_format.set("✗ Invalid")
                        issues.append("Label format invalid")
                        self.log("✗ Label format appears invalid")
            except:
                self.val_format.set("✗ Error")
                issues.append("Could not read label")
                self.log("✗ Could not read label file")
        else:
            self.val_format.set("Not checked")
        
        # Summary
        self.log("\n" + "=" * 60)
        if not issues:
            self.log("✓ VALIDATION PASSED - Dataset is ready!")
            self.project_status.configure(text="✓ Dataset validated - Ready to train", text_color="#2fa572")
        else:
            self.log("✗ VALIDATION FAILED:")
            for issue in issues:
                self.log(f"  - {issue}")
            self.project_status.configure(text="✗ Validation failed - Check console", text_color="#d63031")
        self.log("=" * 60)
    
    def log(self, message):
        """Log message to console"""
        self.console.insert("end", message + "\n")
        self.console.see("end")
        self.root.update_idletasks()
    
    def view_before(self):
        """Open before training results"""
        path = os.path.join(self.project_path.get(), "test_before_training")
        os.startfile(path) if os.name == 'nt' else os.system(f'open "{path}"')
    
    def view_after(self):
        """Open after training results"""
        path = os.path.join(self.project_path.get(), "test_after_training")
        os.startfile(path) if os.name == 'nt' else os.system(f'open "{path}"')
    
    def view_val(self):
        """Open validation results"""
        path = os.path.join(self.project_path.get(), "val_after_training")
        os.startfile(path) if os.name == 'nt' else os.system(f'open "{path}"')
    
    def view_metrics(self):
        """Open metrics/results"""
        path = os.path.join(self.project_path.get(), "finetuned_model")
        os.startfile(path) if os.name == 'nt' else os.system(f'open "{path}"')
    
    def start_training(self):
        """Start the training process"""
        path = self.project_path.get()
        yaml_path = self.yaml_file.get()
        
        if not path:
            messagebox.showwarning("No Project", "Please select a project folder")
            return
        
        if not yaml_path:
            messagebox.showwarning("No YAML", "Please validate dataset first")
            return
        
        # Disable buttons
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.processing = True
        
        # Disable view buttons
        self.view_before_btn.configure(state="disabled")
        self.view_after_btn.configure(state="disabled")
        self.view_val_btn.configure(state="disabled")
        self.view_metrics_btn.configure(state="disabled")
        
        def train_thread():
            try:
                from ultralytics import YOLO
                import torch
                
                self.log("\n" + "=" * 80)
                self.log("YOLO FINE-TUNING STARTED (MINI-DATASET MODE)")
                self.log("=" * 80)
                
                self.log(f"\nGPU Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    self.log(f"GPU Name: {torch.cuda.get_device_name(0)}")
                
                train_from_scratch = self.train_from_scratch.get()
                
                self.log(f"\nConfiguration:")
                self.log(f"  Project: {path}")
                self.log(f"  Model: {self.model_version.get()}")
                self.log(f"  Train from Scratch: {'YES' if train_from_scratch else 'NO (Using pretrained weights)'}")
                self.log(f"  Epochs: {self.epochs.get()}")
                self.log(f"  Fixed Learning Rate: 0.01")
                self.log(f"  Fixed Batch Size: 1")
                self.log(f"  Confidence: {self.conf_threshold.get():.2f}")
                self.log(f"  Device: {'GPU' if self.device.get() == '0' else 'CPU'}")
                
                # STEP 1
                self.log("\n" + "=" * 80)
                self.log("[STEP 1] Testing Base Model (before training)")
                self.log("=" * 80)
                
                if train_from_scratch:
                    self.log("Loading YOLOv8n architecture from scratch...")
                    base_model = YOLO("yolov8n.yaml")
                else:
                    self.log(f"Loading pretrained model: {self.model_version.get()}")
                    base_model = YOLO(self.model_version.get())

                test_path = os.path.join(path, "images", "test")
                if os.path.exists(test_path) and len(os.listdir(test_path)) > 0:
                    base_model.predict(
                        source=test_path,
                        conf=self.conf_threshold.get(),
                        save=True,
                        project=path,
                        name="test_before_training",
                        exist_ok=True,
                        verbose=False
                    )
                    self.log("→ Saved to: test_before_training/")
                    self.root.after(0, lambda: self.view_before_btn.configure(state="normal"))
                
                # STEP 2
                self.log("\n" + "=" * 80)
                self.log("[STEP 2] Fine-tuning on Training Images")
                self.log("=" * 80)
                
                results = base_model.train(
                    data=yaml_path,
                    epochs=self.epochs.get(),
                    pretrained=(not train_from_scratch),
                    imgsz=640,
                    batch=1,
                    lr0=0.01,
                    lrf=0.01,
                    freeze=None,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                    degrees=0.0, translate=0.0, scale=0.0,
                    fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
                    patience=0,
                    workers=0,
                    device=self.device.get(),
                    cache=False,
                    amp=True,
                    val=True,
                    plots=True,
                    project=path,
                    name="finetuned_model",
                    exist_ok=True,
                    verbose=True,
                    save=True,
                    save_period=-1,
                )
                
                self.log("\n→ Training complete!")
                
                # STEP 3
                self.log("\n" + "=" * 80)
                self.log("[STEP 3] Testing Fine-tuned Model")
                self.log("=" * 80)
                
                finetuned_model = YOLO(os.path.join(path, "finetuned_model", "weights", "best.pt"))
                
                if os.path.exists(test_path):
                    finetuned_model.predict(
                        source=test_path,
                        conf=self.conf_threshold.get(),
                        save=True,
                        project=path,
                        name="test_after_training",
                        exist_ok=True,
                        verbose=False
                    )
                    self.log("→ Saved to: test_after_training/")
                    self.root.after(0, lambda: self.view_after_btn.configure(state="normal"))
                
                val_path = os.path.join(path, "images", "val")
                if os.path.exists(val_path):
                    finetuned_model.predict(
                        source=val_path,
                        conf=self.conf_threshold.get(),
                        save=True,
                        project=path,
                        name="val_after_training",
                        exist_ok=True,
                        verbose=False
                    )
                    self.root.after(0, lambda: self.view_val_btn.configure(state="normal"))
                
                self.root.after(0, lambda: self.view_metrics_btn.configure(state="normal"))
                
                # Results
                self.log("\n" + "=" * 80)
                self.log("RESULTS SUMMARY")
                self.log("=" * 80)
                
                try:
                    import pandas as pd
                    results_csv = os.path.join(path, "finetuned_model", "results.csv")
                    df = pd.read_csv(results_csv)
                    
                    final_map50 = df['metrics/mAP50(B)'].iloc[-1]
                    final_precision = df['metrics/precision(B)'].iloc[-1]
                    final_recall = df['metrics/recall(B)'].iloc[-1]
                    
                    self.log(f"\nPerformance:")
                    self.log(f"  mAP50:     {final_map50:.4f} ({final_map50*100:.1f}%)")
                    self.log(f"  Precision: {final_precision:.4f} ({final_precision*100:.1f}%)")
                    self.log(f"  Recall:    {final_recall:.4f} ({final_recall*100:.1f}%)")
                    
                    if final_map50 > 0.8:
                        self.log("\n✓ SUCCESS! Model learned excellently!")
                    elif final_map50 > 0.5:
                        self.log("\n~ Good learning achieved")
                    else:
                        self.log("\n⚠ May need more data or epochs")
                        
                except Exception as e:
                    self.log(f"Could not read metrics: {e}")
                
                self.log("\n✓ Training complete! Use buttons above to view results")
                
            except Exception as e:
                self.log(f"\n✗ ERROR: {e}")
                import traceback
                self.log(traceback.format_exc())
                
            finally:
                self.processing = False
                self.root.after(0, lambda: self.train_btn.configure(state="normal"))
                self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
                
        self.training_thread = threading.Thread(target=train_thread)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def show_about(self):
        """Show About dialog"""
        about_window = ctk.CTkToplevel(self.root)
        about_window.title("About")
        about_window.geometry("400x280")  # Increased height for email
        about_window.resizable(False, False)
        
        # Center the window
        about_window.transient(self.root)
        about_window.grab_set()
        
        # Content frame with tighter padding
        content_frame = ctk.CTkFrame(about_window, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=25, pady=20)
        
        # Title
        title = ctk.CTkLabel(content_frame, text="About", 
                            font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(pady=(0, 12))
        
        # Developed by (updated)
        dev_label = ctk.CTkLabel(content_frame, text="developed by Khwaj Sulaiman siddiqi", 
                                font=ctk.CTkFont(size=13, weight="bold"),
                                text_color="lightblue")
        dev_label.pack(pady=(0, 5))
        
        # Email (new)
        email_label = ctk.CTkLabel(content_frame, text="khwajasulaimansiddiqi@gmail.com", 
                                  font=ctk.CTkFont(size=11),
                                  text_color="gray70")
        email_label.pack(pady=(0, 12))
        
        # Application info
        app_label = ctk.CTkLabel(content_frame, text="Application:", 
                                font=ctk.CTkFont(size=10),
                                text_color="gray60")
        app_label.pack(pady=(0, 2))
        
        app_name = ctk.CTkLabel(content_frame, text="Smart City Data Training",  # Updated
                               font=ctk.CTkFont(size=12, weight="bold"))
        app_name.pack(pady=(0, 8))
        
        # Version
        version_label = ctk.CTkLabel(content_frame, text="Version: 1.0", 
                                    font=ctk.CTkFont(size=11))
        version_label.pack(pady=(0, 8))
        
        # Description
        desc_label = ctk.CTkLabel(content_frame, 
                                 text="For educational fine-tuning of\nYOLO on small datasets",
                                 font=ctk.CTkFont(size=10),
                                 text_color="gray60",
                                 justify="center")
        desc_label.pack(pady=(0, 12))
        
        # Close button
        close_btn = ctk.CTkButton(content_frame, text="Close", 
                                 command=about_window.destroy, width=100, height=28)
        close_btn.pack()
    
    def show_dependencies(self):
        """Show Dependencies dialog"""
        dep_window = ctk.CTkToplevel(self.root)
        dep_window.title("Dependency Manager")
        dep_window.geometry("700x500")
        dep_window.resizable(False, False)
        
        # Center the window
        dep_window.transient(self.root)
        dep_window.grab_set()
        
        # Main content frame
        main_frame = ctk.CTkFrame(dep_window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 15))
        
        title = ctk.CTkLabel(header_frame, text="📦 Package Dependencies", 
                            font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(side="left")
        
        subtitle = ctk.CTkLabel(header_frame, 
                               text="Check and install required Python packages",
                               text_color="gray60")
        subtitle.pack(side="left", padx=(15, 0))
        
        # Console/Output area
        console_frame = ctk.CTkFrame(main_frame)
        console_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        console = ctk.CTkTextbox(console_frame, font=ctk.CTkFont(family="Courier", size=11),
                                wrap="word", state="normal")
        console.pack(fill="both", expand=True, padx=10, pady=10)
        
        def log_message(message):
            console.configure(state="normal")
            console.insert("end", message + "\n")
            console.see("end")
            console.configure(state="disabled")
            dep_window.update()
        
        # Progress bar
        progress_bar = ctk.CTkProgressBar(main_frame)
        progress_bar.pack(fill="x", pady=(0, 15))
        progress_bar.set(0)
        
        # Buttons frame
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x")
        
        check_btn = ctk.CTkButton(button_frame, text="🔍 Check Status", width=150)
        check_btn.pack(side="left", padx=(0, 10))
        
        install_btn = ctk.CTkButton(button_frame, text="⬇ Install Missing", width=150,
                                   fg_color="green", hover_color="darkgreen")
        install_btn.pack(side="left", padx=(0, 10))
        
        close_btn = ctk.CTkButton(button_frame, text="Close", width=100,
                                 command=dep_window.destroy)
        close_btn.pack(side="right")
        
        # Required packages
        REQUIRED_PACKAGES = {
            'customtkinter': 'customtkinter>=5.2.0',
            'PIL': 'pillow>=10.0.0',
            'cv2': 'opencv-python>=4.8.0',
            'numpy': 'numpy>=1.24.0',
            'pandas': 'pandas>=2.0.0',
            'ultralytics': 'ultralytics>=8.0.0',
            'torch': 'torch>=2.0.0',
        }
        
        def check_dependencies():
            """Check which dependencies are installed"""
            console.configure(state="normal")
            console.delete("1.0", "end")
            console.configure(state="disabled")
            
            log_message("Checking dependencies...\n")
            log_message("DEPENDENCY STATUS")
            log_message("=" * 50 + "\n")
            
            installed = []
            missing = []
            
            for import_name, pip_name in REQUIRED_PACKAGES.items():
                try:
                    __import__(import_name)
                    installed.append(pip_name.split('>=')[0])
                    log_message(f"✓ {pip_name.split('>=')[0]}")
                except ImportError:
                    missing.append(pip_name)
            
            if missing:
                log_message("\n✗ MISSING:")
                for pkg in missing:
                    log_message(f"  • {pkg}")
                log_message(f"\n⚠ Need to install {len(missing)} package(s)")
                install_btn.configure(state="normal")
            else:
                log_message("\n🎉 All dependencies are installed!")
                install_btn.configure(state="disabled")
            
            progress_bar.set(1.0)
            return installed, missing
        
        def install_missing():
            """Install missing packages"""
            install_btn.configure(state="disabled")
            check_btn.configure(state="disabled")
            
            log_message("\n" + "=" * 50)
            log_message("INSTALLING MISSING PACKAGES")
            log_message("=" * 50 + "\n")
            
            _, missing = check_dependencies_silent()
            
            if not missing:
                log_message("All packages already installed!")
                check_btn.configure(state="normal")
                return
            
            total = len(missing)
            for i, package in enumerate(missing, 1):
                progress_bar.set(i / total)
                log_message(f"Installing {i}/{total}: {package}...")
                
                try:
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        log_message(f"  ✓ {package} installed successfully")
                    else:
                        log_message(f"  ✗ Failed to install {package}")
                except Exception as e:
                    log_message(f"  ✗ Error: {str(e)}")
            
            log_message("\n" + "=" * 50)
            log_message("Verifying installation...")
            check_dependencies()
            
            check_btn.configure(state="normal")
            install_btn.configure(state="normal")
        
        def check_dependencies_silent():
            """Silent check for install function"""
            installed = []
            missing = []
            
            for import_name, pip_name in REQUIRED_PACKAGES.items():
                try:
                    __import__(import_name)
                    installed.append(pip_name.split('>=')[0])
                except ImportError:
                    missing.append(pip_name)
            
            return installed, missing
        
        # Connect button commands
        check_btn.configure(command=check_dependencies)
        install_btn.configure(command=lambda: threading.Thread(target=install_missing, daemon=True).start())
        
        # Initial check
        check_dependencies()
        
    def stop_training(self):
        """Stop training"""
        if self.processing:
            self.log("\n⚠ Stop requested...")
            self.processing = False

if __name__ == "__main__":
    root = ctk.CTk()
    app = SmartCityDataTrainer(root)  # Updated class instantiation
    root.mainloop()


#small