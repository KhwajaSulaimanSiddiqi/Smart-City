#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart City Data Training - Professional YOLO Fine-Tuning Platform
Version: 2.0 (Advanced)

Developed by: Khwaj Sulaiman Siddiqi
Email: khwajasulaimansiddiqi@gmail.com

This application provides a complete GUI for fine-tuning YOLO models on custom datasets,
with automatic dataset size detection, professional UI, and extensive configuration options.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox, Toplevel, Text, Scrollbar
import os
import sys
import yaml
import glob
import threading
import subprocess
import json
import time
import datetime
import shutil
import webbrowser
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import queue
import logging
import traceback
import zipfile
import tarfile
import requests
from io import BytesIO

# For advanced features
try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Set appearance
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Constants
APP_NAME = "Smart City Data Training"
APP_VERSION = "2.0.0"
AUTHOR = "Khwaj Sulaiman Siddiqi"
EMAIL = "khwajasulaimansiddiqi@gmail.com"
SUPPORTED_MODELS = [
    "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"
]
PRETRAINED_MODELS = SUPPORTED_MODELS  # For simplicity

# Default training parameters per dataset size
DATASET_PRESETS = {
    "tiny": {
        "name": "Tiny (< 100 images)",
        "epochs": 100,
        "batch": 1,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "patience": 0
    },
    "small": {
        "name": "Small (100-500 images)",
        "epochs": 150,
        "batch": 2,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "patience": 50
    },
    "medium": {
        "name": "Medium (500-2000 images)",
        "epochs": 200,
        "batch": 4,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "patience": 100
    },
    "large": {
        "name": "Large (2000-10000 images)",
        "epochs": 300,
        "batch": 8,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "patience": 100
    },
    "xlarge": {
        "name": "Extra Large (>10000 images)",
        "epochs": 500,
        "batch": 16,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "patience": 200
    }
}

class ScrollableFrame(ctk.CTkScrollableFrame):
    """A scrollable frame that can contain any widgets."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

class ConsoleHandler(logging.Handler):
    """Custom logging handler to redirect logs to GUI console."""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state="normal")
            self.text_widget.insert("end", msg + "\n")
            self.text_widget.see("end")
            self.text_widget.configure(state="disabled")
        self.text_widget.after(0, append)

class AdvancedYOLOTrainer:
    """Main application class for YOLO fine-tuning GUI."""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 700)
        
        # Variables
        self.project_path = ctk.StringVar()
        self.yaml_file = ctk.StringVar()
        self.model_version = ctk.StringVar(value="yolo11n.pt")
        self.device = ctk.StringVar(value="0")
        self.conf_threshold = ctk.DoubleVar(value=0.25)
        self.iou_threshold = ctk.DoubleVar(value=0.45)
        self.train_from_scratch = ctk.BooleanVar(value=False)
        self.epochs = ctk.IntVar(value=200)
        self.batch_size = ctk.IntVar(value=1)
        self.image_size = ctk.IntVar(value=640)
        self.workers = ctk.IntVar(value=0)
        self.lr0 = ctk.DoubleVar(value=0.01)
        self.lrf = ctk.DoubleVar(value=0.01)
        self.momentum = ctk.DoubleVar(value=0.937)
        self.weight_decay = ctk.DoubleVar(value=0.0005)
        self.warmup_epochs = ctk.IntVar(value=3)
        self.warmup_momentum = ctk.DoubleVar(value=0.8)
        self.warmup_bias_lr = ctk.DoubleVar(value=0.1)
        self.box_loss_gain = ctk.DoubleVar(value=7.5)
        self.cls_loss_gain = ctk.DoubleVar(value=0.5)
        self.dfl_loss_gain = ctk.DoubleVar(value=1.5)
        self.patience = ctk.IntVar(value=0)
        self.save_period = ctk.IntVar(value=-1)
        self.cache_images = ctk.BooleanVar(value=False)
        self.amp = ctk.BooleanVar(value=True)
        self.freeze_layers = ctk.IntVar(value=0)
        self.deterministic = ctk.BooleanVar(value=False)
        self.single_cls = ctk.BooleanVar(value=False)
        self.cos_lr = ctk.BooleanVar(value=False)
        self.multi_scale = ctk.BooleanVar(value=False)
        
        # Augmentation variables
        self.hsv_h = ctk.DoubleVar(value=0.015)
        self.hsv_s = ctk.DoubleVar(value=0.7)
        self.hsv_v = ctk.DoubleVar(value=0.4)
        self.degrees = ctk.DoubleVar(value=0.0)
        self.translate = ctk.DoubleVar(value=0.1)
        self.scale = ctk.DoubleVar(value=0.5)
        self.shear = ctk.DoubleVar(value=0.0)
        self.perspective = ctk.DoubleVar(value=0.0)
        self.flipud = ctk.DoubleVar(value=0.0)
        self.fliplr = ctk.DoubleVar(value=0.5)
        self.mosaic = ctk.DoubleVar(value=1.0)
        self.mixup = ctk.DoubleVar(value=0.0)
        self.copy_paste = ctk.DoubleVar(value=0.0)
        self.auto_augment = ctk.StringVar(value="randaugment")
        
        # Dataset size preset
        self.dataset_preset = ctk.StringVar(value="auto")
        self.dataset_size_estimate = ctk.StringVar(value="Unknown")
        self.dataset_class_count = ctk.StringVar(value="?")
        
        # UI state
        self.processing = False
        self.training_thread = None
        self.log_queue = queue.Queue()
        self.help_visible = False
        self.current_model = None
        self.training_results = {}
        self.validation_status = {"images": "Not checked", "labels": "Not checked", "yaml": "Not checked", "format": "Not checked"}
        
        # Create main container with scroll
        self.main_container = ScrollableFrame(root, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Build UI sections
        self.create_header()
        self.create_project_selection()
        self.create_dataset_analysis()
        self.create_validation_section()
        self.create_training_parameters_tabs()
        self.create_console_section()
        self.create_status_bar()
        
        # Start queue processing
        self.process_log_queue()
        
        # Load configuration if exists
        self.load_config()
        
        # Check dependencies
        self.check_dependencies_async()
    
    # ==================== UI Construction Methods ====================
    
    def create_header(self):
        """Create professional header with logo, title, and action buttons."""
        header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent", height=80)
        header_frame.pack(fill="x", pady=(0, 15))
        header_frame.pack_propagate(False)
        
        # Left side: Logo and title
        left_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        left_frame.pack(side="left", fill="y", padx=10)
        
        # Try to load logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Smart City.jpg")
        if os.path.exists(logo_path):
            try:
                logo_image = Image.open(logo_path)
                logo_image = logo_image.resize((50, 50), Image.Resampling.LANCZOS)
                logo_photo = ctk.CTkImage(light_image=logo_image, dark_image=logo_image, size=(50, 50))
                logo_label = ctk.CTkLabel(left_frame, image=logo_photo, text="")
                logo_label.image = logo_photo
                logo_label.pack(side="left", padx=(0, 15))
            except Exception as e:
                print(f"Logo load error: {e}")
        
        text_container = ctk.CTkFrame(left_frame, fg_color="transparent")
        text_container.pack(side="left")
        
        title = ctk.CTkLabel(text_container, text=APP_NAME, 
                            font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(anchor="w")
        
        subtitle = ctk.CTkLabel(text_container, 
                               text="Professional YOLO Fine-Tuning Platform",
                               text_color="gray60", font=ctk.CTkFont(size=12))
        subtitle.pack(anchor="w")
        
        # Right side buttons
        right_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        right_frame.pack(side="right", padx=10)
        
        # Theme toggle
        self.theme_btn = ctk.CTkButton(right_frame, text="🌙 Dark", command=self.toggle_theme,
                                       width=80, height=35)
        self.theme_btn.pack(side="left", padx=5)
        
        # Dependency check
        deps_btn = ctk.CTkButton(right_frame, text="🔧 Dependencies", 
                                 command=self.show_dependencies, width=120, height=35)
        deps_btn.pack(side="left", padx=5)
        
        # About
        about_btn = ctk.CTkButton(right_frame, text="ℹ About", command=self.show_about,
                                 width=80, height=35)
        about_btn.pack(side="left", padx=5)
        
        # Settings
        settings_btn = ctk.CTkButton(right_frame, text="⚙ Settings", command=self.show_settings,
                                     width=80, height=35)
        settings_btn.pack(side="left", padx=5)
    
    def create_project_selection(self):
        """Project folder selection with enhanced UI."""
        frame = ctk.CTkFrame(self.main_container)
        frame.pack(fill="x", pady=(0, 15))
        
        # Title with icon
        title_frame = ctk.CTkFrame(frame, fg_color="transparent")
        title_frame.pack(fill="x", padx=15, pady=(15, 10))
        
        ctk.CTkLabel(title_frame, text="📁 Project Configuration", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        
        # Project path row
        path_row = ctk.CTkFrame(frame, fg_color="transparent")
        path_row.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(path_row, text="Project Folder:", width=120, anchor="w",
                    font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 10))
        
        path_entry = ctk.CTkEntry(path_row, textvariable=self.project_path, height=35,
                                   placeholder_text="Select folder containing images/ and labels/")
        path_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        browse_btn = ctk.CTkButton(path_row, text="Browse", command=self.browse_project,
                                    width=100, height=35)
        browse_btn.pack(side="left")
        
        # Recent projects dropdown
        recent_btn = ctk.CTkButton(path_row, text="Recent", command=self.show_recent_projects,
                                    width=80, height=35)
        recent_btn.pack(side="left", padx=(5, 0))
        
        # Status and quick actions
        status_row = ctk.CTkFrame(frame, fg_color="transparent")
        status_row.pack(fill="x", padx=15, pady=(0, 10))
        
        self.project_status = ctk.CTkLabel(status_row, 
                                          text="Select project folder to begin",
                                          text_color="gray60", anchor="w")
        self.project_status.pack(side="left")
        
        # Help toggle
        self.help_btn = ctk.CTkButton(status_row, text="? Folder Structure",
                                     command=self.toggle_help, width=140, height=30,
                                     fg_color="gray30", hover_color="gray40")
        self.help_btn.pack(side="right")
        
        # Hidden info box
        self.info_frame = ctk.CTkFrame(frame, fg_color="gray20")
        info_text = """Expected folder structure:
  project_folder/
    ├── images/
    │   ├── train/  (training images)
    │   ├── val/    (validation images)
    │   └── test/   (test images - optional)
    ├── labels/
    │   ├── train/  (YOLO format .txt files)
    │   ├── val/
    │   └── test/
    └── data.yaml   (dataset configuration)"""
        
        self.info_label = ctk.CTkLabel(self.info_frame, text=info_text, 
                                      text_color="gray70", anchor="w", justify="left",
                                      font=ctk.CTkFont(family="Courier", size=11))
        self.info_label.pack(anchor="w", padx=15, pady=10)
    
    def create_dataset_analysis(self):
        """Analyze dataset and provide size-based presets."""
        frame = ctk.CTkFrame(self.main_container)
        frame.pack(fill="x", pady=(0, 15))
        
        # Title
        title_frame = ctk.CTkFrame(frame, fg_color="transparent")
        title_frame.pack(fill="x", padx=15, pady=(15, 10))
        
        ctk.CTkLabel(title_frame, text="📊 Dataset Analysis & Presets", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        
        # Analyze button
        analyze_btn = ctk.CTkButton(title_frame, text="Analyze Dataset", 
                                     command=self.analyze_dataset, width=150, height=30)
        analyze_btn.pack(side="right")
        
        # Preset selection
        preset_row = ctk.CTkFrame(frame, fg_color="transparent")
        preset_row.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(preset_row, text="Dataset Size Preset:", width=150, anchor="w",
                    font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 10))
        
        preset_combo = ctk.CTkComboBox(preset_row, variable=self.dataset_preset,
                                        values=["auto", "tiny", "small", "medium", "large", "xlarge", "custom"],
                                        width=200, state="readonly",
                                        command=self.apply_dataset_preset)
        preset_combo.pack(side="left")
        
        # Dataset info
        info_row = ctk.CTkFrame(frame, fg_color="transparent")
        info_row.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkLabel(info_row, text="Estimated Size:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 5))
        self.size_label = ctk.CTkLabel(info_row, textvariable=self.dataset_size_estimate, text_color="gray60")
        self.size_label.pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(info_row, text="Classes:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 5))
        self.class_label = ctk.CTkLabel(info_row, textvariable=self.dataset_class_count, text_color="gray60")
        self.class_label.pack(side="left")
    
    def create_validation_section(self):
        """Dataset validation with detailed status."""
        frame = ctk.CTkFrame(self.main_container)
        frame.pack(fill="x", pady=(0, 15))
        
        # Title
        title_frame = ctk.CTkFrame(frame, fg_color="transparent")
        title_frame.pack(fill="x", padx=15, pady=(15, 10))
        
        ctk.CTkLabel(title_frame, text="✅ Dataset Validation", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        
        validate_btn = ctk.CTkButton(title_frame, text="Validate Now", 
                                     command=self.validate_dataset, width=150, height=30)
        validate_btn.pack(side="right")
        
        # Status grid
        status_frame = ctk.CTkFrame(frame, fg_color="transparent")
        status_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        # Row 1
        row1 = ctk.CTkFrame(status_frame, fg_color="transparent")
        row1.pack(fill="x", pady=2)
        
        ctk.CTkLabel(row1, text="Images:", width=80, font=ctk.CTkFont(weight="bold")).pack(side="left")
        self.val_images = ctk.CTkLabel(row1, textvariable=self.get_val_var("images"), text_color="gray60", width=200, anchor="w")
        self.val_images.pack(side="left")
        
        ctk.CTkLabel(row1, text="Labels:", width=80, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(20,0))
        self.val_labels = ctk.CTkLabel(row1, textvariable=self.get_val_var("labels"), text_color="gray60", width=200, anchor="w")
        self.val_labels.pack(side="left")
        
        # Row 2
        row2 = ctk.CTkFrame(status_frame, fg_color="transparent")
        row2.pack(fill="x", pady=2)
        
        ctk.CTkLabel(row2, text="YAML:", width=80, font=ctk.CTkFont(weight="bold")).pack(side="left")
        self.val_yaml = ctk.CTkLabel(row2, textvariable=self.get_val_var("yaml"), text_color="gray60", width=200, anchor="w")
        self.val_yaml.pack(side="left")
        
        ctk.CTkLabel(row2, text="Format:", width=80, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(20,0))
        self.val_format = ctk.CTkLabel(row2, textvariable=self.get_val_var("format"), text_color="gray60", width=200, anchor="w")
        self.val_format.pack(side="left")
    
    def get_val_var(self, key):
        """Return a StringVar for validation status."""
        if not hasattr(self, '_val_vars'):
            self._val_vars = {
                'images': ctk.StringVar(value="Not checked"),
                'labels': ctk.StringVar(value="Not checked"),
                'yaml': ctk.StringVar(value="Not checked"),
                'format': ctk.StringVar(value="Not checked")
            }
        return self._val_vars[key]
    
    def create_training_parameters_tabs(self):
        """Create tabbed interface for training parameters."""
        # یک کانتینر برای تب‌ها و دکمه‌ها ایجاد می‌کنیم
        container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        container.pack(fill="x", pady=(0, 15))
        
        self.tab_view = ctk.CTkTabview(container)
        self.tab_view.pack(fill="x")
        
        # Create tabs
        self.tab_basic = self.tab_view.add("Basic")
        self.tab_advanced = self.tab_view.add("Advanced")
        self.tab_augmentation = self.tab_view.add("Augmentation")
        self.tab_system = self.tab_view.add("System")
        self.tab_logging = self.tab_view.add("Logging")
        
        # Fill tabs
        self.create_basic_tab()
        self.create_advanced_tab()
        self.create_augmentation_tab()
        self.create_system_tab()
        self.create_logging_tab()
        
        # Action buttons below tabs (در همان کانتینر)
        action_frame = ctk.CTkFrame(container, fg_color="transparent")
        action_frame.pack(fill="x", pady=(10, 0))
        
        self.train_btn = ctk.CTkButton(action_frame, text="🚀 Start Fine-Tuning",
                                       command=self.start_training, width=200, height=40,
                                       fg_color="#2fa572", hover_color="#257a53",
                                       font=ctk.CTkFont(size=14, weight="bold"))
        self.train_btn.pack(side="left", padx=(0, 10))
        
        self.stop_btn = ctk.CTkButton(action_frame, text="🛑 Stop", command=self.stop_training,
                                      width=100, height=40, state="disabled",
                                      fg_color="#d63031", hover_color="#a82828")
        self.stop_btn.pack(side="left")
        
        # Save/Load config
        ctk.CTkButton(action_frame, text="💾 Save Config", command=self.save_config,
                      width=120, height=40).pack(side="right", padx=5)
        ctk.CTkButton(action_frame, text="📂 Load Config", command=self.load_config_dialog,
                      width=120, height=40).pack(side="right", padx=5)
    
    def create_basic_tab(self):
        """Basic training parameters."""
        frame = self.tab_basic
        
        # Model selection
        row1 = ctk.CTkFrame(frame, fg_color="transparent")
        row1.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row1, text="Model Version:", width=150, anchor="w").pack(side="left")
        model_combo = ctk.CTkComboBox(row1, variable=self.model_version, width=200,
                                      values=SUPPORTED_MODELS, state="readonly")
        model_combo.pack(side="left", padx=(0, 10))
        
        self.model_info = ctk.CTkLabel(row1, text="(Nano - fastest)", text_color="gray60")
        self.model_info.pack(side="left")
        self.model_version.trace('w', self.update_model_info)
        
        # Device
        row2 = ctk.CTkFrame(frame, fg_color="transparent")
        row2.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row2, text="Device:", width=150, anchor="w").pack(side="left")
        device_combo = ctk.CTkComboBox(row2, variable=self.device, width=100,
                                       values=["0", "1", "2", "3", "cpu", "mps"], state="readonly")
        device_combo.pack(side="left")
        
        # Epochs and batch
        row3 = ctk.CTkFrame(frame, fg_color="transparent")
        row3.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row3, text="Epochs:", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row3, textvariable=self.epochs, width=100).pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(row3, text="Batch Size:", width=80, anchor="w").pack(side="left")
        ctk.CTkEntry(row3, textvariable=self.batch_size, width=100).pack(side="left")
        
        # Image size
        row4 = ctk.CTkFrame(frame, fg_color="transparent")
        row4.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row4, text="Image Size:", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row4, textvariable=self.image_size, width=100).pack(side="left")
        
        # Confidence and IoU
        row5 = ctk.CTkFrame(frame, fg_color="transparent")
        row5.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row5, text="Confidence Threshold:", width=150, anchor="w").pack(side="left")
        slider1 = ctk.CTkSlider(row5, from_=0.1, to=0.9, variable=self.conf_threshold, width=200)
        slider1.pack(side="left", padx=(0, 10))
        self.conf_label = ctk.CTkLabel(row5, text=f"{self.conf_threshold.get():.2f}", width=40)
        self.conf_label.pack(side="left")
        self.conf_threshold.trace('w', lambda *a: self.conf_label.configure(text=f"{self.conf_threshold.get():.2f}"))
        
        row6 = ctk.CTkFrame(frame, fg_color="transparent")
        row6.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row6, text="IoU Threshold:", width=150, anchor="w").pack(side="left")
        slider2 = ctk.CTkSlider(row6, from_=0.1, to=0.9, variable=self.iou_threshold, width=200)
        slider2.pack(side="left", padx=(0, 10))
        self.iou_label = ctk.CTkLabel(row6, text=f"{self.iou_threshold.get():.2f}", width=40)
        self.iou_label.pack(side="left")
        self.iou_threshold.trace('w', lambda *a: self.iou_label.configure(text=f"{self.iou_threshold.get():.2f}"))
        
        # Train from scratch
        row7 = ctk.CTkFrame(frame, fg_color="transparent")
        row7.pack(fill="x", pady=5)
        
        ctk.CTkCheckBox(row7, text="Train from Scratch (no pretrained weights)",
                        variable=self.train_from_scratch).pack(side="left")
    
    def create_advanced_tab(self):
        """Advanced hyperparameters."""
        frame = self.tab_advanced
        
        # Learning rates
        row1 = ctk.CTkFrame(frame, fg_color="transparent")
        row1.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row1, text="Initial LR (lr0):", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row1, textvariable=self.lr0, width=100).pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(row1, text="Final LR (lrf):", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row1, textvariable=self.lrf, width=100).pack(side="left")
        
        row2 = ctk.CTkFrame(frame, fg_color="transparent")
        row2.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row2, text="Momentum:", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row2, textvariable=self.momentum, width=100).pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(row2, text="Weight Decay:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row2, textvariable=self.weight_decay, width=100).pack(side="left")
        
        # Warmup
        row3 = ctk.CTkFrame(frame, fg_color="transparent")
        row3.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row3, text="Warmup Epochs:", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row3, textvariable=self.warmup_epochs, width=100).pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(row3, text="Warmup Momentum:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row3, textvariable=self.warmup_momentum, width=100).pack(side="left")
        
        row4 = ctk.CTkFrame(frame, fg_color="transparent")
        row4.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row4, text="Warmup Bias LR:", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row4, textvariable=self.warmup_bias_lr, width=100).pack(side="left")
        
        # Loss gains
        row5 = ctk.CTkFrame(frame, fg_color="transparent")
        row5.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row5, text="Box Loss Gain:", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row5, textvariable=self.box_loss_gain, width=100).pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(row5, text="Cls Loss Gain:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row5, textvariable=self.cls_loss_gain, width=100).pack(side="left")
        
        row6 = ctk.CTkFrame(frame, fg_color="transparent")
        row6.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row6, text="DFL Loss Gain:", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row6, textvariable=self.dfl_loss_gain, width=100).pack(side="left")
        
        # Other
        row7 = ctk.CTkFrame(frame, fg_color="transparent")
        row7.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row7, text="Patience (early stop):", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row7, textvariable=self.patience, width=100).pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(row7, text="Save Period:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row7, textvariable=self.save_period, width=100).pack(side="left")
        
        row8 = ctk.CTkFrame(frame, fg_color="transparent")
        row8.pack(fill="x", pady=5)
        
        ctk.CTkCheckBox(row8, text="Cosine LR Scheduler", variable=self.cos_lr).pack(side="left", padx=(0, 20))
        ctk.CTkCheckBox(row8, text="Multi-Scale Training", variable=self.multi_scale).pack(side="left", padx=(0, 20))
        ctk.CTkCheckBox(row8, text="Single Class", variable=self.single_cls).pack(side="left")
    
    def create_augmentation_tab(self):
        """Data augmentation parameters."""
        frame = self.tab_augmentation
        
        # Color augmentations
        row1 = ctk.CTkFrame(frame, fg_color="transparent")
        row1.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row1, text="HSV-Hue:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row1, textvariable=self.hsv_h, width=80).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(row1, text="HSV-Saturation:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row1, textvariable=self.hsv_s, width=80).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(row1, text="HSV-Value:", width=80, anchor="w").pack(side="left")
        ctk.CTkEntry(row1, textvariable=self.hsv_v, width=80).pack(side="left")
        
        # Geometric
        row2 = ctk.CTkFrame(frame, fg_color="transparent")
        row2.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row2, text="Rotation (deg):", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row2, textvariable=self.degrees, width=80).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(row2, text="Translate:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row2, textvariable=self.translate, width=80).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(row2, text="Scale:", width=80, anchor="w").pack(side="left")
        ctk.CTkEntry(row2, textvariable=self.scale, width=80).pack(side="left")
        
        row3 = ctk.CTkFrame(frame, fg_color="transparent")
        row3.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row3, text="Shear:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row3, textvariable=self.shear, width=80).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(row3, text="Perspective:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row3, textvariable=self.perspective, width=80).pack(side="left", padx=(0, 10))
        
        # Flip
        row4 = ctk.CTkFrame(frame, fg_color="transparent")
        row4.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row4, text="Flip Up-Down:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row4, textvariable=self.flipud, width=80).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(row4, text="Flip Left-Right:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row4, textvariable=self.fliplr, width=80).pack(side="left", padx=(0, 10))
        
        # Advanced augmentations
        row5 = ctk.CTkFrame(frame, fg_color="transparent")
        row5.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row5, text="Mosaic:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row5, textvariable=self.mosaic, width=80).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(row5, text="MixUp:", width=100, anchor="w").pack(side="left")
        ctk.CTkEntry(row5, textvariable=self.mixup, width=80).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(row5, text="Copy-Paste:", width=80, anchor="w").pack(side="left")
        ctk.CTkEntry(row5, textvariable=self.copy_paste, width=80).pack(side="left")
        
        # Auto augment
        row6 = ctk.CTkFrame(frame, fg_color="transparent")
        row6.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row6, text="Auto Augment:", width=100, anchor="w").pack(side="left")
        auto_combo = ctk.CTkComboBox(row6, variable=self.auto_augment,
                                      values=["randaugment", "autoaugment", "augmix", "None"],
                                      width=150, state="readonly")
        auto_combo.pack(side="left")
    
    def create_system_tab(self):
        """System and performance parameters."""
        frame = self.tab_system
        
        # Workers
        row1 = ctk.CTkFrame(frame, fg_color="transparent")
        row1.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row1, text="Number of Workers:", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row1, textvariable=self.workers, width=100).pack(side="left")
        ctk.CTkLabel(row1, text="(0 = main thread)", text_color="gray60").pack(side="left", padx=10)
        
        # Cache images
        row2 = ctk.CTkFrame(frame, fg_color="transparent")
        row2.pack(fill="x", pady=5)
        
        ctk.CTkCheckBox(row2, text="Cache Images (RAM)", variable=self.cache_images).pack(side="left", padx=(0, 20))
        ctk.CTkCheckBox(row2, text="Use AMP (mixed precision)", variable=self.amp).pack(side="left")
        
        # Freeze layers
        row3 = ctk.CTkFrame(frame, fg_color="transparent")
        row3.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row3, text="Freeze Layers:", width=150, anchor="w").pack(side="left")
        ctk.CTkEntry(row3, textvariable=self.freeze_layers, width=100).pack(side="left")
        ctk.CTkLabel(row3, text="(0 = none, 10 = backbone)", text_color="gray60").pack(side="left", padx=10)
        
        # Deterministic
        row4 = ctk.CTkFrame(frame, fg_color="transparent")
        row4.pack(fill="x", pady=5)
        
        ctk.CTkCheckBox(row4, text="Deterministic (reproducible)", variable=self.deterministic).pack(side="left")
    
    def create_logging_tab(self):
        """Logging and visualization options."""
        frame = self.tab_logging
        
        # Weights & Biases
        row1 = ctk.CTkFrame(frame, fg_color="transparent")
        row1.pack(fill="x", pady=5)
        
        self.wandb_enabled = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(row1, text="Enable Weights & Biases logging", variable=self.wandb_enabled,
                        command=self.toggle_wandb).pack(side="left")
        
        self.wandb_project = ctk.StringVar(value="yolo-finetune")
        self.wandb_entity = ctk.StringVar(value="")
        self.wandb_name = ctk.StringVar(value="")
        
        row2 = ctk.CTkFrame(frame, fg_color="transparent")
        row2.pack(fill="x", pady=5)
        ctk.CTkLabel(row2, text="W&B Project:", width=100).pack(side="left")
        ctk.CTkEntry(row2, textvariable=self.wandb_project, width=150).pack(side="left", padx=5)
        ctk.CTkLabel(row2, text="Entity:", width=50).pack(side="left")
        ctk.CTkEntry(row2, textvariable=self.wandb_entity, width=150).pack(side="left")
        
        row3 = ctk.CTkFrame(frame, fg_color="transparent")
        row3.pack(fill="x", pady=5)
        ctk.CTkLabel(row3, text="Run Name:", width=100).pack(side="left")
        ctk.CTkEntry(row3, textvariable=self.wandb_name, width=200).pack(side="left")
        
        # TensorBoard
        row4 = ctk.CTkFrame(frame, fg_color="transparent")
        row4.pack(fill="x", pady=5)
        
        self.tensorboard_enabled = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(row4, text="Enable TensorBoard logging", variable=self.tensorboard_enabled).pack(side="left")
        
        # Save directory
        row5 = ctk.CTkFrame(frame, fg_color="transparent")
        row5.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row5, text="Project Save Directory:", width=150).pack(side="left")
        self.save_dir = ctk.StringVar(value="runs/train")
        ctk.CTkEntry(row5, textvariable=self.save_dir, width=200).pack(side="left", padx=5)
    
    def create_console_section(self):
        """Console output with filtering and export."""
        frame = ctk.CTkFrame(self.main_container)
        frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Header with controls
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(header, text="📟 Training Console", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
        
        # Filter
        self.log_filter = ctk.StringVar()
        ctk.CTkEntry(header, textvariable=self.log_filter, placeholder_text="Filter...",
                     width=150).pack(side="right", padx=5)
        ctk.CTkButton(header, text="Clear", command=self.clear_console,
                      width=70).pack(side="right", padx=5)
        ctk.CTkButton(header, text="Export", command=self.export_logs,
                      width=70).pack(side="right", padx=5)
        
        # Console text
        self.console = ctk.CTkTextbox(frame, wrap="word", font=ctk.CTkFont(family="Courier", size=11))
        self.console.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Configure logging
        self.logger = logging.getLogger("YOLOTrainer")
        self.logger.setLevel(logging.INFO)
        handler = ConsoleHandler(self.console)
        self.logger.addHandler(handler)
        
        # View results buttons
        view_frame = ctk.CTkFrame(frame, fg_color="transparent")
        view_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.view_before_btn = ctk.CTkButton(view_frame, text="View Before", width=100,
                                            command=self.view_before, state="disabled")
        self.view_before_btn.pack(side="left", padx=2)
        
        self.view_after_btn = ctk.CTkButton(view_frame, text="View After", width=100,
                                           command=self.view_after, state="disabled")
        self.view_after_btn.pack(side="left", padx=2)
        
        self.view_val_btn = ctk.CTkButton(view_frame, text="View Val", width=100,
                                         command=self.view_val, state="disabled")
        self.view_val_btn.pack(side="left", padx=2)
        
        self.view_metrics_btn = ctk.CTkButton(view_frame, text="View Metrics", width=100,
                                             command=self.view_metrics, state="disabled")
        self.view_metrics_btn.pack(side="left", padx=2)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(frame)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.progress_bar.set(0)
    
    def create_status_bar(self):
        """Status bar at bottom."""
        status_bar = ctk.CTkFrame(self.root, height=25, fg_color="gray20")
        status_bar.pack(side="bottom", fill="x")
        
        # GPU info
        gpu_text = "GPU: " + (torch.cuda.get_device_name(0) if TORCH_AVAILABLE and torch.cuda.is_available() else "Not available")
        self.gpu_label = ctk.CTkLabel(status_bar, text=gpu_text, text_color="gray70", font=ctk.CTkFont(size=10))
        self.gpu_label.pack(side="left", padx=10)
        
        # Status
        self.status_label = ctk.CTkLabel(status_bar, text="Ready", text_color="gray70", font=ctk.CTkFont(size=10))
        self.status_label.pack(side="right", padx=10)
    
    # ==================== Event Handlers ====================
    
    def toggle_theme(self):
        """Switch between light and dark mode."""
        current = ctk.get_appearance_mode()
        if current == "Light":
            ctk.set_appearance_mode("dark")
            self.theme_btn.configure(text="☀ Light")
        else:
            ctk.set_appearance_mode("light")
            self.theme_btn.configure(text="🌙 Dark")
    
    def toggle_help(self):
        """Show/hide folder structure help."""
        if self.help_visible:
            self.info_frame.pack_forget()
            self.help_btn.configure(text="? Folder Structure")
            self.help_visible = False
        else:
            self.info_frame.pack(fill="x", padx=15, pady=(0, 10))
            self.help_btn.configure(text="✕ Hide")
            self.help_visible = True
    
    def browse_project(self):
        """Open folder selection dialog."""
        folder = filedialog.askdirectory(title="Select YOLO Project Folder")
        if folder:
            self.project_path.set(folder)
            self.project_status.configure(text=f"Selected: {folder}", text_color="gray60")
            self.log_info(f"Project folder selected: {folder}")
            # Auto-analyze if possible
            self.analyze_dataset()
    
    def show_recent_projects(self):
        """Show list of recently used projects."""
        # Simple implementation - could store in config
        recent = []  # TODO: load from config
        if not recent:
            messagebox.showinfo("Recent Projects", "No recent projects found.")
            return
        
        # Create popup menu
        popup = ctk.CTkToplevel(self.root)
        popup.title("Recent Projects")
        popup.geometry("400x300")
        popup.transient(self.root)
        
        listbox = ctk.CTkTextbox(popup)
        listbox.pack(fill="both", expand=True, padx=10, pady=10)
        for path in recent:
            listbox.insert("end", path + "\n")
    
    def update_model_info(self, *args):
        """Update model description based on selection."""
        model = self.model_version.get()
        if 'n' in model:
            info = "(Nano - fastest, smallest)"
        elif 's' in model:
            info = "(Small - balanced)"
        elif 'm' in model:
            info = "(Medium - higher accuracy)"
        elif 'l' in model:
            info = "(Large - accurate but slower)"
        elif 'x' in model:
            info = "(X-Large - maximum accuracy)"
        else:
            info = ""
        self.model_info.configure(text=info)
    
    def analyze_dataset(self):
        """Analyze dataset size and class distribution."""
        path = self.project_path.get()
        if not path:
            messagebox.showwarning("No Project", "Please select a project folder first.")
            return
        
        self.log_info("Analyzing dataset...")
        total_images = 0
        class_counts = {}
        
        # Check train images
        train_img_dir = os.path.join(path, "images", "train")
        if os.path.exists(train_img_dir):
            train_imgs = glob.glob(os.path.join(train_img_dir, "*.*"))
            total_images += len(train_imgs)
        
        # Also check val/test
        for sub in ["val", "test"]:
            img_dir = os.path.join(path, "images", sub)
            if os.path.exists(img_dir):
                total_images += len(glob.glob(os.path.join(img_dir, "*.*")))
        
        # Estimate size category
        if total_images < 100:
            category = "tiny"
        elif total_images < 500:
            category = "small"
        elif total_images < 2000:
            category = "medium"
        elif total_images < 10000:
            category = "large"
        else:
            category = "xlarge"
        
        self.dataset_size_estimate.set(f"{total_images} images ({category})")
        
        # Try to read classes from YAML
        yaml_files = glob.glob(os.path.join(path, "*.yaml"))
        if yaml_files:
            try:
                with open(yaml_files[0], 'r') as f:
                    data = yaml.safe_load(f)
                    nc = data.get('nc', 0)
                    names = data.get('names', [])
                    if nc > 0:
                        self.dataset_class_count.set(str(nc))
                    elif names:
                        self.dataset_class_count.set(str(len(names)))
            except:
                pass
        
        # Auto-select preset if set to auto
        if self.dataset_preset.get() == "auto":
            self.dataset_preset.set(category)
            self.apply_dataset_preset(category)
        
        self.log_info(f"Dataset analysis complete: {total_images} images, {self.dataset_class_count.get()} classes.")
    
    def apply_dataset_preset(self, choice):
        """Apply preset parameters based on dataset size."""
        if choice == "auto" or choice == "custom":
            return
        
        preset = DATASET_PRESETS.get(choice)
        if not preset:
            return
        
        self.log_info(f"Applying {preset['name']} preset...")
        
        # Apply parameters
        self.epochs.set(preset['epochs'])
        self.batch_size.set(preset['batch'])
        self.lr0.set(preset['lr0'])
        self.lrf.set(preset['lrf'])
        self.momentum.set(preset['momentum'])
        self.weight_decay.set(preset['weight_decay'])
        self.warmup_epochs.set(preset['warmup_epochs'])
        self.warmup_momentum.set(preset['warmup_momentum'])
        self.warmup_bias_lr.set(preset['warmup_bias_lr'])
        self.box_loss_gain.set(preset['box'])
        self.cls_loss_gain.set(preset['cls'])
        self.dfl_loss_gain.set(preset['dfl'])
        self.patience.set(preset['patience'])
        
        # Augmentations
        self.hsv_h.set(preset['hsv_h'])
        self.hsv_s.set(preset['hsv_s'])
        self.hsv_v.set(preset['hsv_v'])
        self.degrees.set(preset['degrees'])
        self.translate.set(preset['translate'])
        self.scale.set(preset['scale'])
        self.shear.set(preset['shear'])
        self.perspective.set(preset['perspective'])
        self.flipud.set(preset['flipud'])
        self.fliplr.set(preset['fliplr'])
        self.mosaic.set(preset['mosaic'])
        self.mixup.set(preset['mixup'])
        self.copy_paste.set(preset['copy_paste'])
        
        self.log_info(f"Preset applied.")
    
    def validate_dataset(self):
        """Perform comprehensive dataset validation."""
        path = self.project_path.get()
        if not path:
            messagebox.showwarning("No Path", "Please select a project folder first.")
            return
        
        self.log_info("\n" + "="*60)
        self.log_info("VALIDATING DATASET...")
        self.log_info("="*60)
        
        issues = []
        
        # Check images
        img_train = os.path.join(path, "images", "train")
        img_val = os.path.join(path, "images", "val")
        img_test = os.path.join(path, "images", "test")
        
        train_imgs = len(glob.glob(os.path.join(img_train, "*.*"))) if os.path.exists(img_train) else 0
        val_imgs = len(glob.glob(os.path.join(img_val, "*.*"))) if os.path.exists(img_val) else 0
        test_imgs = len(glob.glob(os.path.join(img_test, "*.*"))) if os.path.exists(img_test) else 0
        
        if train_imgs > 0:
            self._val_vars['images'].set(f"✓ {train_imgs} train, {val_imgs} val, {test_imgs} test")
            self.log_info(f"✓ Images found: {train_imgs} train, {val_imgs} val, {test_imgs} test")
        else:
            self._val_vars['images'].set("✗ No images")
            issues.append("No training images found")
            self.log_error("✗ No training images found")
        
        # Check labels
        lbl_train = os.path.join(path, "labels", "train")
        lbl_val = os.path.join(path, "labels", "val")
        lbl_test = os.path.join(path, "labels", "test")
        
        train_lbls = len(glob.glob(os.path.join(lbl_train, "*.txt"))) if os.path.exists(lbl_train) else 0
        val_lbls = len(glob.glob(os.path.join(lbl_val, "*.txt"))) if os.path.exists(lbl_val) else 0
        test_lbls = len(glob.glob(os.path.join(lbl_test, "*.txt"))) if os.path.exists(lbl_test) else 0
        
        if train_lbls > 0:
            self._val_vars['labels'].set(f"✓ {train_lbls} train, {val_lbls} val, {test_lbls} test")
            self.log_info(f"✓ Labels found: {train_lbls} train, {val_lbls} val, {test_lbls} test")
        else:
            self._val_vars['labels'].set("✗ No labels")
            issues.append("No training labels found")
            self.log_error("✗ No training labels found")
        
        # Check YAML
        yaml_files = glob.glob(os.path.join(path, "*.yaml"))
        if yaml_files:
            self.yaml_file.set(yaml_files[0])
            self._val_vars['yaml'].set(f"✓ {os.path.basename(yaml_files[0])}")
            self.log_info(f"✓ YAML found: {os.path.basename(yaml_files[0])}")
            
            try:
                with open(yaml_files[0], 'r') as f:
                    yaml_content = yaml.safe_load(f)
                    if 'names' in yaml_content and 'nc' in yaml_content:
                        num_classes = yaml_content['nc']
                        class_names = yaml_content['names']
                        self.log_info(f"  Classes: {num_classes} - {class_names}")
            except:
                pass
        else:
            self._val_vars['yaml'].set("✗ Not found")
            issues.append("No .yaml file found")
            self.log_error("✗ No .yaml file found")
        
        # Check format
        if train_lbls > 0:
            sample_label = glob.glob(os.path.join(lbl_train, "*.txt"))[0]
            try:
                with open(sample_label, 'r') as f:
                    line = f.readline().strip()
                    parts = line.split()
                    if len(parts) == 5:
                        self._val_vars['format'].set("✓ YOLO format")
                        self.log_info("✓ Label format is valid YOLO")
                    else:
                        self._val_vars['format'].set("✗ Invalid")
                        issues.append("Label format invalid (should be: class x_center y_center width height)")
                        self.log_error("✗ Label format appears invalid")
            except Exception as e:
                self._val_vars['format'].set("✗ Error")
                issues.append(f"Could not read label: {e}")
                self.log_error(f"✗ Could not read label file: {e}")
        else:
            self._val_vars['format'].set("Not checked")
        
        # Summary
        self.log_info("\n" + "="*60)
        if not issues:
            self.log_info("✓ VALIDATION PASSED - Dataset is ready!")
            self.project_status.configure(text="✓ Dataset validated - Ready to train", text_color="#2fa572")
        else:
            self.log_error("✗ VALIDATION FAILED:")
            for issue in issues:
                self.log_error(f"  - {issue}")
            self.project_status.configure(text="✗ Validation failed - Check console", text_color="#d63031")
        self.log_info("="*60)
        
        return issues
    
    def toggle_wandb(self):
        """Enable/disable wandb fields."""
        state = "normal" if self.wandb_enabled.get() else "disabled"
        # In a real implementation, we would enable/entry widgets
    
    def start_training(self):
        """Start the training process in a separate thread."""
        # Validate project and YAML
        path = self.project_path.get()
        yaml_path = self.yaml_file.get()
        
        if not path:
            messagebox.showwarning("No Project", "Please select a project folder.")
            return
        
        if not yaml_path:
            # Try to find YAML
            yaml_files = glob.glob(os.path.join(path, "*.yaml"))
            if yaml_files:
                self.yaml_file.set(yaml_files[0])
                yaml_path = yaml_files[0]
            else:
                messagebox.showwarning("No YAML", "Please validate dataset first or ensure a .yaml file exists.")
                return
        
        # Check if ultralytics is available
        if not ULTRALYTICS_AVAILABLE:
            messagebox.showerror("Missing Dependency", "Ultralytics package is not installed. Please install it via Dependencies.")
            return
        
        # Disable UI
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.processing = True
        self.status_label.configure(text="Training in progress...")
        
        # Start training thread
        self.training_thread = threading.Thread(target=self._training_worker, args=(path, yaml_path))
        self.training_thread.daemon = True
        self.training_thread.start()
        
        self.log_info("Training started in background thread.")
    
    def _training_worker(self, project_path, yaml_path):
        """Actual training logic running in thread."""
        try:
            from ultralytics import YOLO
            import torch
            
            self.log_info("\n" + "="*80)
            self.log_info("YOLO FINE-TUNING STARTED")
            self.log_info("="*80)
            
            # Log configuration
            self.log_info(f"\nConfiguration:")
            self.log_info(f"  Project: {project_path}")
            self.log_info(f"  Model: {self.model_version.get()}")
            self.log_info(f"  Pretrained: {not self.train_from_scratch.get()}")
            self.log_info(f"  Epochs: {self.epochs.get()}")
            self.log_info(f"  Batch: {self.batch_size.get()}")
            self.log_info(f"  Image size: {self.image_size.get()}")
            self.log_info(f"  Device: {self.device.get()}")
            self.log_info(f"  Workers: {self.workers.get()}")
            self.log_info(f"  LR0: {self.lr0.get()}")
            self.log_info(f"  LRF: {self.lrf.get()}")
            
            # STEP 1: Test base model (before training)
            self.log_info("\n" + "="*80)
            self.log_info("[STEP 1] Testing Base Model (before training)")
            self.log_info("="*80)
            
            if self.train_from_scratch.get():
                # Load architecture only
                model_name = self.model_version.get().replace('.pt', '.yaml')
                self.log_info(f"Loading architecture from {model_name}...")
                base_model = YOLO(model_name)
            else:
                self.log_info(f"Loading pretrained model: {self.model_version.get()}")
                base_model = YOLO(self.model_version.get())
            
            # Test on test set if exists
            test_path = os.path.join(project_path, "images", "test")
            if os.path.exists(test_path) and len(os.listdir(test_path)) > 0:
                self.log_info("Running predictions on test set with base model...")
                base_model.predict(
                    source=test_path,
                    conf=self.conf_threshold.get(),
                    iou=self.iou_threshold.get(),
                    save=True,
                    project=project_path,
                    name="test_before_training",
                    exist_ok=True,
                    verbose=False
                )
                self.log_info("→ Saved to: test_before_training/")
                self.root.after(0, lambda: self.view_before_btn.configure(state="normal"))
            
            # STEP 2: Fine-tuning
            self.log_info("\n" + "="*80)
            self.log_info("[STEP 2] Fine-tuning on Training Images")
            self.log_info("="*80)
            
            # Prepare training arguments
            train_args = {
                'data': yaml_path,
                'epochs': self.epochs.get(),
                'imgsz': self.image_size.get(),
                'batch': self.batch_size.get(),
                'workers': self.workers.get(),
                'device': self.device.get(),
                'pretrained': not self.train_from_scratch.get(),
                'lr0': self.lr0.get(),
                'lrf': self.lrf.get(),
                'momentum': self.momentum.get(),
                'weight_decay': self.weight_decay.get(),
                'warmup_epochs': self.warmup_epochs.get(),
                'warmup_momentum': self.warmup_momentum.get(),
                'warmup_bias_lr': self.warmup_bias_lr.get(),
                'box': self.box_loss_gain.get(),
                'cls': self.cls_loss_gain.get(),
                'dfl': self.dfl_loss_gain.get(),
                'patience': self.patience.get(),
                'save_period': self.save_period.get(),
                'cache': self.cache_images.get(),
                'amp': self.amp.get(),
                'freeze': self.freeze_layers.get() if self.freeze_layers.get() > 0 else None,
                'deterministic': self.deterministic.get(),
                'single_cls': self.single_cls.get(),
                'cos_lr': self.cos_lr.get(),
                'multi_scale': self.multi_scale.get(),
                # Augmentation
                'hsv_h': self.hsv_h.get(),
                'hsv_s': self.hsv_s.get(),
                'hsv_v': self.hsv_v.get(),
                'degrees': self.degrees.get(),
                'translate': self.translate.get(),
                'scale': self.scale.get(),
                'shear': self.shear.get(),
                'perspective': self.perspective.get(),
                'flipud': self.flipud.get(),
                'fliplr': self.fliplr.get(),
                'mosaic': self.mosaic.get(),
                'mixup': self.mixup.get(),
                'copy_paste': self.copy_paste.get(),
                'auto_augment': self.auto_augment.get() if self.auto_augment.get() != "None" else None,
                # Project
                'project': os.path.join(project_path, "finetuned_model"),
                'name': "train",
                'exist_ok': True,
                'verbose': True,
                'plots': True,
                'save': True,
            }
            
            # Start training
            results = base_model.train(**train_args)
            
            self.log_info("\n→ Training complete!")
            
            # STEP 3: Test fine-tuned model
            self.log_info("\n" + "="*80)
            self.log_info("[STEP 3] Testing Fine-tuned Model")
            self.log_info("="*80)
            
            # Load best model
            best_model_path = os.path.join(project_path, "finetuned_model", "train", "weights", "best.pt")
            if os.path.exists(best_model_path):
                finetuned_model = YOLO(best_model_path)
            else:
                finetuned_model = base_model  # fallback
            
            # Test on test set
            if os.path.exists(test_path):
                finetuned_model.predict(
                    source=test_path,
                    conf=self.conf_threshold.get(),
                    iou=self.iou_threshold.get(),
                    save=True,
                    project=project_path,
                    name="test_after_training",
                    exist_ok=True,
                    verbose=False
                )
                self.log_info("→ Saved to: test_after_training/")
                self.root.after(0, lambda: self.view_after_btn.configure(state="normal"))
            
            # Validate on val set
            val_path = os.path.join(project_path, "images", "val")
            if os.path.exists(val_path):
                finetuned_model.predict(
                    source=val_path,
                    conf=self.conf_threshold.get(),
                    iou=self.iou_threshold.get(),
                    save=True,
                    project=project_path,
                    name="val_after_training",
                    exist_ok=True,
                    verbose=False
                )
                self.log_info("→ Saved to: val_after_training/")
                self.root.after(0, lambda: self.view_val_btn.configure(state="normal"))
            
            # Enable metrics view
            self.root.after(0, lambda: self.view_metrics_btn.configure(state="normal"))
            
            # Parse results
            try:
                results_csv = os.path.join(project_path, "finetuned_model", "train", "results.csv")
                if os.path.exists(results_csv):
                    df = pd.read_csv(results_csv)
                    final_map50 = df['metrics/mAP50(B)'].iloc[-1]
                    final_precision = df['metrics/precision(B)'].iloc[-1]
                    final_recall = df['metrics/recall(B)'].iloc[-1]
                    
                    self.log_info("\n" + "="*80)
                    self.log_info("RESULTS SUMMARY")
                    self.log_info("="*80)
                    self.log_info(f"  mAP50:     {final_map50:.4f} ({final_map50*100:.1f}%)")
                    self.log_info(f"  Precision: {final_precision:.4f} ({final_precision*100:.1f}%)")
                    self.log_info(f"  Recall:    {final_recall:.4f} ({final_recall*100:.1f}%)")
                    
                    # Store for later
                    self.training_results = {
                        'map50': final_map50,
                        'precision': final_precision,
                        'recall': final_recall,
                        'csv': results_csv
                    }
            except Exception as e:
                self.log_error(f"Could not read metrics: {e}")
            
            self.log_info("\n✓ Training complete! Use the view buttons to see results.")
            
        except Exception as e:
            self.log_error(f"\n✗ ERROR: {e}")
            self.log_error(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        finally:
            self.processing = False
            self.root.after(0, lambda: self.train_btn.configure(state="normal"))
            self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
            self.root.after(0, lambda: self.status_label.configure(text="Ready"))
            self.root.after(0, lambda: self.progress_bar.set(0))
    
    def stop_training(self):
        """Request training to stop."""
        if self.processing:
            self.log_info("\n⚠ Stop requested...")
            # Ultralytics doesn't have a direct stop, but we can set a flag
            # For now, just note that we can't gracefully stop
            self.processing = False
            # In practice, we might need to kill the thread, but that's risky
    
    def view_before(self):
        """Open before training results folder."""
        path = os.path.join(self.project_path.get(), "test_before_training")
        if os.path.exists(path):
            self._open_folder(path)
        else:
            messagebox.showinfo("Info", "Before training results not found.")
    
    def view_after(self):
        """Open after training results folder."""
        path = os.path.join(self.project_path.get(), "test_after_training")
        if os.path.exists(path):
            self._open_folder(path)
        else:
            messagebox.showinfo("Info", "After training results not found.")
    
    def view_val(self):
        """Open validation results folder."""
        path = os.path.join(self.project_path.get(), "val_after_training")
        if os.path.exists(path):
            self._open_folder(path)
        else:
            messagebox.showinfo("Info", "Validation results not found.")
    
    def view_metrics(self):
        """Open metrics visualization."""
        if not self.training_results:
            messagebox.showinfo("Info", "No training metrics available yet.")
            return
        
        # Create a popup window with plots
        metrics_window = ctk.CTkToplevel(self.root)
        metrics_window.title("Training Metrics")
        metrics_window.geometry("800x600")
        
        # Read CSV
        csv_path = self.training_results.get('csv')
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("Error", "Metrics CSV not found.")
            return
        
        df = pd.read_csv(csv_path)
        
        # Create matplotlib figure
        fig = Figure(figsize=(10, 8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        # Plot metrics
        epochs = range(1, len(df)+1)
        if 'metrics/mAP50(B)' in df.columns:
            ax1.plot(epochs, df['metrics/mAP50(B)'], label='mAP50')
            ax1.set_title('mAP50')
            ax1.set_xlabel('Epoch')
            ax1.grid(True)
        
        if 'metrics/precision(B)' in df.columns:
            ax2.plot(epochs, df['metrics/precision(B)'], label='Precision', color='green')
            ax2.set_title('Precision')
            ax2.set_xlabel('Epoch')
            ax2.grid(True)
        
        if 'metrics/recall(B)' in df.columns:
            ax3.plot(epochs, df['metrics/recall(B)'], label='Recall', color='red')
            ax3.set_title('Recall')
            ax3.set_xlabel('Epoch')
            ax3.grid(True)
        
        if 'train/box_loss' in df.columns:
            ax4.plot(epochs, df['train/box_loss'], label='Box Loss')
            ax4.set_title('Box Loss')
            ax4.set_xlabel('Epoch')
            ax4.grid(True)
        
        fig.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=metrics_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, metrics_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _open_folder(self, path):
        """Open folder in file explorer."""
        if os.name == 'nt':
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', path])
        else:
            subprocess.run(['xdg-open', path])
    
    def clear_console(self):
        """Clear console text."""
        self.console.delete("1.0", "end")
    
    def export_logs(self):
        """Export console logs to file."""
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                  filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.console.get("1.0", "end-1c"))
                messagebox.showinfo("Export", f"Logs exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
    
    def log_info(self, msg):
        """Log info message."""
        self.logger.info(msg)
    
    def log_error(self, msg):
        """Log error message."""
        self.logger.error(msg)
    
    def log_warning(self, msg):
        """Log warning message."""
        self.logger.warning(msg)
    
    def process_log_queue(self):
        """Process any pending log messages (if using queue)."""
        # This is a placeholder; we use logger directly
        self.root.after(100, self.process_log_queue)
    
    def save_config(self):
        """Save current configuration to a JSON file."""
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                  filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return
        
        config = {
            'project_path': self.project_path.get(),
            'model_version': self.model_version.get(),
            'device': self.device.get(),
            'conf_threshold': self.conf_threshold.get(),
            'iou_threshold': self.iou_threshold.get(),
            'train_from_scratch': self.train_from_scratch.get(),
            'epochs': self.epochs.get(),
            'batch_size': self.batch_size.get(),
            'image_size': self.image_size.get(),
            'workers': self.workers.get(),
            'lr0': self.lr0.get(),
            'lrf': self.lrf.get(),
            'momentum': self.momentum.get(),
            'weight_decay': self.weight_decay.get(),
            'warmup_epochs': self.warmup_epochs.get(),
            'warmup_momentum': self.warmup_momentum.get(),
            'warmup_bias_lr': self.warmup_bias_lr.get(),
            'box_loss_gain': self.box_loss_gain.get(),
            'cls_loss_gain': self.cls_loss_gain.get(),
            'dfl_loss_gain': self.dfl_loss_gain.get(),
            'patience': self.patience.get(),
            'save_period': self.save_period.get(),
            'cache_images': self.cache_images.get(),
            'amp': self.amp.get(),
            'freeze_layers': self.freeze_layers.get(),
            'deterministic': self.deterministic.get(),
            'single_cls': self.single_cls.get(),
            'cos_lr': self.cos_lr.get(),
            'multi_scale': self.multi_scale.get(),
            'hsv_h': self.hsv_h.get(),
            'hsv_s': self.hsv_s.get(),
            'hsv_v': self.hsv_v.get(),
            'degrees': self.degrees.get(),
            'translate': self.translate.get(),
            'scale': self.scale.get(),
            'shear': self.shear.get(),
            'perspective': self.perspective.get(),
            'flipud': self.flipud.get(),
            'fliplr': self.fliplr.get(),
            'mosaic': self.mosaic.get(),
            'mixup': self.mixup.get(),
            'copy_paste': self.copy_paste.get(),
            'auto_augment': self.auto_augment.get(),
            'dataset_preset': self.dataset_preset.get(),
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4)
            messagebox.showinfo("Success", "Configuration saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
    
    def load_config_dialog(self):
        """Load configuration from a JSON file."""
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if file_path:
            self.load_config(file_path)
    
    def load_config(self, file_path=None):
        """Load configuration from file or default."""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                # Apply to variables
                for key, value in config.items():
                    var = getattr(self, key, None)
                    if var and isinstance(var, (ctk.StringVar, ctk.BooleanVar, ctk.IntVar, ctk.DoubleVar)):
                        try:
                            var.set(value)
                        except:
                            pass
                messagebox.showinfo("Success", "Configuration loaded.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")
    
    def show_about(self):
        """Display about dialog."""
        about = ctk.CTkToplevel(self.root)
        about.title("About")
        about.geometry("500x350")
        about.resizable(False, False)
        about.transient(self.root)
        
        frame = ctk.CTkFrame(about, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Logo or icon placeholder
        ctk.CTkLabel(frame, text="🏙️", font=ctk.CTkFont(size=48)).pack(pady=(0,10))
        
        ctk.CTkLabel(frame, text=APP_NAME, font=ctk.CTkFont(size=20, weight="bold")).pack()
        ctk.CTkLabel(frame, text=f"Version {APP_VERSION}", font=ctk.CTkFont(size=12)).pack(pady=(0,10))
        
        ctk.CTkLabel(frame, text=f"Developed by {AUTHOR}", font=ctk.CTkFont(size=12)).pack()
        ctk.CTkLabel(frame, text=EMAIL, font=ctk.CTkFont(size=10), text_color="gray60").pack(pady=(0,15))
        
        ctk.CTkLabel(frame, text="Professional YOLO Fine-Tuning Platform\n"
                                 "Supports YOLOv5, v8, v9, v10, v11\n"
                                 "Advanced dataset analysis and augmentation",
                     justify="center", text_color="gray70").pack(pady=(0,15))
        
        ctk.CTkButton(frame, text="Close", command=about.destroy, width=100).pack()
    
    def show_dependencies(self):
        """Show dependency manager window."""
        dep_window = ctk.CTkToplevel(self.root)
        dep_window.title("Dependency Manager")
        dep_window.geometry("700x500")
        dep_window.transient(self.root)
        
        main_frame = ctk.CTkFrame(dep_window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header = ctk.CTkFrame(main_frame, fg_color="transparent")
        header.pack(fill="x", pady=(0,15))
        
        ctk.CTkLabel(header, text="📦 Package Dependencies", 
                    font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        
        # Console
        console = ctk.CTkTextbox(main_frame, font=ctk.CTkFont(family="Courier", size=11), wrap="word")
        console.pack(fill="both", expand=True, pady=(0,15))
        
        # Progress
        progress = ctk.CTkProgressBar(main_frame)
        progress.pack(fill="x", pady=(0,15))
        progress.set(0)
        
        # Buttons
        btn_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        btn_frame.pack(fill="x")
        
        def check_deps():
            console.delete("1.0", "end")
            console.insert("end", "Checking dependencies...\n\n")
            
            required = {
                'customtkinter': 'customtkinter>=5.2.0',
                'PIL': 'pillow>=10.0.0',
                'cv2': 'opencv-python>=4.8.0',
                'numpy': 'numpy>=1.24.0',
                'pandas': 'pandas>=2.0.0',
                'ultralytics': 'ultralytics>=8.0.0',
                'torch': 'torch>=2.0.0',
                'matplotlib': 'matplotlib>=3.7.0',
                'yaml': 'pyyaml>=6.0',
            }
            
            installed = []
            missing = []
            
            for imp, pkg in required.items():
                try:
                    __import__(imp)
                    installed.append(pkg.split('>=')[0])
                    console.insert("end", f"✓ {pkg.split('>=')[0]}\n")
                except ImportError:
                    missing.append(pkg)
                    console.insert("end", f"✗ {pkg} (missing)\n")
            
            console.insert("end", "\n" + "="*50 + "\n")
            if missing:
                console.insert("end", f"⚠ {len(missing)} package(s) missing.\n")
            else:
                console.insert("end", "✓ All dependencies satisfied!\n")
            
            progress.set(1.0)
        
        def install_deps():
            # Simple pip install for missing
            console.insert("end", "\nInstalling missing packages...\n")
            # This would require subprocess and admin rights, but we'll simulate
            console.insert("end", "Please use terminal: pip install -r requirements.txt\n")
        
        ctk.CTkButton(btn_frame, text="🔍 Check", command=check_deps, width=100).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="⬇ Install Missing", command=install_deps, width=150).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Close", command=dep_window.destroy, width=100).pack(side="right", padx=5)
        
        # Initial check
        check_deps()
    
    def show_settings(self):
        """Show settings dialog."""
        settings = ctk.CTkToplevel(self.root)
        settings.title("Settings")
        settings.geometry("400x300")
        settings.transient(self.root)
        
        frame = ctk.CTkFrame(settings)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(frame, text="Application Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(0,15))
        
        # Auto-check updates
        self.auto_update = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(frame, text="Check for updates on startup", variable=self.auto_update).pack(anchor="w", pady=5)
        
        # Default project path
        ctk.CTkLabel(frame, text="Default project folder:").pack(anchor="w", pady=(10,0))
        default_path_frame = ctk.CTkFrame(frame, fg_color="transparent")
        default_path_frame.pack(fill="x", pady=5)
        self.default_path = ctk.StringVar(value=os.path.expanduser("~"))
        ctk.CTkEntry(default_path_frame, textvariable=self.default_path).pack(side="left", fill="x", expand=True, padx=(0,5))
        ctk.CTkButton(default_path_frame, text="Browse", command=lambda: self.default_path.set(filedialog.askdirectory()), width=70).pack(side="right")
        
        # Save on exit
        self.save_on_exit = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(frame, text="Save configuration on exit", variable=self.save_on_exit).pack(anchor="w", pady=5)
        
        ctk.CTkButton(frame, text="Save", command=settings.destroy, width=100).pack(pady=15)
    
    def check_dependencies_async(self):
        """Check dependencies in background."""
        # Simple version check
        if not ULTRALYTICS_AVAILABLE:
            self.log_warning("Ultralytics not installed. Training will not work.")
        if not TORCH_AVAILABLE:
            self.log_warning("PyTorch not installed. GPU acceleration may not work.")
    
    # ==================== Utility ====================
    
    def run(self):
        """Start the main loop."""
        self.root.mainloop()

# ==================== Main Entry Point ====================

def main():
    root = ctk.CTk()
    app = AdvancedYOLOTrainer(root)
    app.run()

if __name__ == "__main__":
    main()