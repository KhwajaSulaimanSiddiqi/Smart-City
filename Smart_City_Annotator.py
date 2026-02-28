import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas, Toplevel, Listbox, Scrollbar, colorchooser
import cv2
import os
import glob
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json

# تنظیمات ظاهری: روشن و آبی
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class ClassManager:
    """مدیریت کلاس‌ها (نام و رنگ)"""
    def __init__(self):
        self.classes = {
            0: {"name": "people", "color": "#FF3232"},
            1: {"name": "bicycle", "color": "#32FF32"},
            2: {"name": "car", "color": "#FFFF32"},
            3: {"name": "van", "color": "#3232FF"},
            4: {"name": "truck", "color": "#FF32FF"},
            5: {"name": "tricycle", "color": "#32FFFF"},
            6: {"name": "rikshaw", "color": "#803280"},
            7: {"name": "bus", "color": "#FFA532"},
            8: {"name": "motorcycle", "color": "#328080"},
        }
        self.next_id = max(self.classes.keys()) + 1

    def add_class(self, name, color=None):
        if not color:
            color = "#%06x" % (hash(name) % 0xFFFFFF)
        self.classes[self.next_id] = {"name": name, "color": color}
        self.next_id += 1
        return self.next_id - 1

    def remove_class(self, class_id):
        if class_id in self.classes:
            del self.classes[class_id]

    def rename_class(self, class_id, new_name):
        if class_id in self.classes:
            self.classes[class_id]["name"] = new_name

    def change_color(self, class_id, new_color):
        if class_id in self.classes:
            self.classes[class_id]["color"] = new_color

    def get_class_name(self, class_id):
        return self.classes.get(class_id, {}).get("name", "unknown")

    def get_class_color(self, class_id):
        return self.classes.get(class_id, {}).get("color", "#FFFFFF")

    def load_from_file(self, filepath):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    cid = int(parts[0])
                    name = parts[1]
                    color = parts[2] if len(parts) > 2 else "#%06x" % (hash(name) % 0xFFFFFF)
                    self.classes[cid] = {"name": name, "color": color}
        self.next_id = max(self.classes.keys()) + 1 if self.classes else 0

    def save_to_file(self, filepath):
        with open(filepath, 'w') as f:
            for cid, info in self.classes.items():
                f.write(f"{cid} {info['name']} {info['color']}\n")


class UndoRedoStack:
    def __init__(self, max_size=20):
        self.undo_stack = []
        self.redo_stack = []
        self.max_size = max_size

    def push(self, state):
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return None
        state = self.undo_stack.pop()
        self.redo_stack.append(state)
        return state

    def redo(self):
        if not self.redo_stack:
            return None
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        return state

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()


class AdvancedYOLOAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart City Annotator")  
        self.root.geometry("1500x900")
       
      
        self.root.attributes('-fullscreen', False)  # Optional, set True for true fullscreen

        # متغیرهای اصلی
        self.image_dir = None
        self.output_dir = None
        self.image_files = []
        self.current_img_index = 0
        self.original_img = None
        self.display_img = None
        self.rectangles = []  # [xmin, ymin, xmax, ymax, class_id]
        self.class_manager = ClassManager()
        self.undo_redo = UndoRedoStack()

        # حالت‌های تعامل
        self.drawing = False
        self.selected_rect_idx = None
        self.interaction_mode = "draw"
        self.draw_start_x, self.draw_start_y = 0, 0
        self.last_mouse_x, self.last_mouse_y = 0, 0

        # hover برای snap
        self.hovered_rect_idx = None
        self.hover_mode = None

        # زوم و پن
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.panning = False
        self.pan_start_x, self.pan_start_y = 0, 0

        self.current_class_id = 3
        self.vline = None
        self.hline = None

        self.output_dir_customized = False
        self.auto_save = True
        self.show_labels = True
        self.current_format = "YOLO"  # یا "VOC"

        # Theme state
        self.current_theme = "Light Blue"  # default متناسب با light

        self.setup_ui()
        self.setup_bindings()
        self.update_stats()

    def setup_ui(self):
        # نوار منو
        menubar = ctk.CTkFrame(self.root, height=30, corner_radius=0)
        menubar.pack(fill="x", padx=0, pady=0)

        # منوی File
        file_menu = ctk.CTkOptionMenu(menubar, values=["Open Image", "Open Dir", "Set Output", "Save", "Save As", "Export COCO", "Exit"],
                                       command=self.menu_file, width=100)
        file_menu.pack(side="left", padx=2)
        ctk.CTkLabel(menubar, text="File").pack(side="left", padx=(5,0))

        # منوی Edit
        edit_menu = ctk.CTkOptionMenu(menubar, values=["Undo", "Redo", "Auto Save On/Off", "Clear All", "Delete"],
                                       command=self.menu_edit, width=100)
        edit_menu.pack(side="left", padx=2)
        ctk.CTkLabel(menubar, text="Edit").pack(side="left", padx=(5,0))

        # منوی View
        view_menu = ctk.CTkOptionMenu(menubar, values=["Zoom In", "Zoom Out", "Fit Window", "100%", "Show/Hide Labels"],
                                       command=self.menu_view, width=100)
        view_menu.pack(side="left", padx=2)
        ctk.CTkLabel(menubar, text="View").pack(side="left", padx=(5,0))

        # منوی Help
        help_menu = ctk.CTkOptionMenu(menubar, values=["About", "Class Manager"],
                                       command=self.menu_help, width=100)
        help_menu.pack(side="left", padx=2)
        ctk.CTkLabel(menubar, text="Help").pack(side="left", padx=(5,0))

        # منوی Themes (جدید)
        themes = ["Light Blue", "Light Green", "Light Dark-Blue", "Dark Blue", "Dark Green", "Dark Dark-Blue"]
        self.themes_menu = ctk.CTkOptionMenu(menubar, values=themes,
                                             command=self.change_theme, width=100)
        self.themes_menu.pack(side="left", padx=2)
        ctk.CTkLabel(menubar, text="Themes").pack(side="left", padx=(5,0))
        self.themes_menu.set(self.current_theme)  # نمایش تم فعلی

        # بدنه اصلی
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True)

        # نوار کناری (لیست فایل‌ها)
        self.sidebar = ctk.CTkFrame(main_container, width=280, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=(0, 10))
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(self.sidebar, text="📂 Image List", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15)
        self.file_listbox = ctk.CTkScrollableFrame(self.sidebar)
        self.file_listbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # آمار
        self.stats_label = ctk.CTkLabel(self.sidebar, text="", font=ctk.CTkFont(size=12), justify="left")
        self.stats_label.pack(pady=10, padx=10, anchor="w")

        # نوار ابزار کناری (دکمه‌های اضافی)
        ctk.CTkButton(self.sidebar, text="Manage Classes", command=self.open_class_manager).pack(pady=5, padx=10, fill="x")

        # قسمت راست (نمایش تصویر و کنترل پنل)
        right_container = ctk.CTkFrame(main_container, fg_color="transparent")
        right_container.pack(side="right", fill="both", expand=True)

        # کنترل پنل بالایی
        self.create_control_panel(right_container)

        # کانواس
        self.create_canvas_frame(right_container)

        # نوار وضعیت
        self.status_bar = ctk.CTkFrame(self.root, height=25, corner_radius=0)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_label = ctk.CTkLabel(self.status_bar, text="Ready", anchor="w")
        self.status_label.pack(side="left", padx=10)

    def create_control_panel(self, parent):
        control_frame = ctk.CTkFrame(parent, height=80, corner_radius=0)
        control_frame.pack(fill="x", padx=0, pady=(0, 10))

        btn_container = ctk.CTkFrame(control_frame, fg_color="transparent")
        btn_container.pack(expand=True, pady=10)

        # دکمه‌های اصلی (بدون fg_color سفارشی تا آبی تم گرفته شود)
        self.btn_load_img = ctk.CTkButton(btn_container, text="🖼️ Open Image", command=self.load_single_image, width=100)
        self.btn_load_img.pack(side="left", padx=5)

        self.btn_load_dir = ctk.CTkButton(btn_container, text="📂 Open Dir", command=self.load_directory, width=100)
        self.btn_load_dir.pack(side="left", padx=5)

        self.btn_output = ctk.CTkButton(btn_container, text="🎯 Set Output", command=self.set_output_directory, width=100)
        self.btn_output.pack(side="left", padx=5)

        # ناوبری
        self.btn_prev = ctk.CTkButton(btn_container, text="◀ Prev (A)", command=self.prev_image, width=80, state="disabled")
        self.btn_prev.pack(side="left", padx=5)

        self.img_counter = ctk.CTkLabel(btn_container, text="0 / 0", width=60)
        self.img_counter.pack(side="left", padx=5)

        self.btn_next = ctk.CTkButton(btn_container, text="Next (D) ▶", command=self.next_image, width=80, state="disabled")
        self.btn_next.pack(side="left", padx=5)

        # انتخاب کلاس
        self.class_var = ctk.StringVar(value="3 - car")
        self.class_dropdown = ctk.CTkComboBox(btn_container, variable=self.class_var,
                                              values=self.get_class_list(), state="readonly",
                                              width=120, command=self.on_class_change)
        self.class_dropdown.pack(side="left", padx=10)

        # ویرایش (اکنون همه دکمه‌ها آبی هستند)
        self.btn_delete = ctk.CTkButton(btn_container, text="🗑️ Delete (Del)", command=self.delete_selected, width=100)
        self.btn_delete.pack(side="left", padx=5)

        self.btn_clear = ctk.CTkButton(btn_container, text="❌ Clear All", command=self.clear_all_boxes, width=90)
        self.btn_clear.pack(side="left", padx=5)

        self.btn_save = ctk.CTkButton(btn_container, text="💾 Save (Ctrl+S)", command=self.save_labels, width=110)
        self.btn_save.pack(side="left", padx=5)

        self.btn_toggle_labels = ctk.CTkButton(btn_container, text="👁️ Hide", command=self.toggle_labels, width=80)
        self.btn_toggle_labels.pack(side="left", padx=5)

    def create_canvas_frame(self, parent):
        canvas_frame = ctk.CTkFrame(parent, corner_radius=0)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # تنظیم رنگ پس‌زمینه کانواس بر اساس حالت ظاهری
        if ctk.get_appearance_mode().lower() == "light":
            canvas_bg = "#f0f0f0"
        else:
            canvas_bg = "#1a1a1a"

        self.canvas = Canvas(canvas_frame, bg=canvas_bg, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)

        # اتصالات موس
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.remove_crosshair)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)

    def setup_bindings(self):
        self.root.bind('d', lambda e: self.next_image())
        self.root.bind('a', lambda e: self.prev_image())
        self.root.bind('<Control-s>', lambda e: self.save_labels())
        self.root.bind('<Control-S>', lambda e: self.save_as())
        self.root.bind('<Delete>', lambda e: self.delete_selected())
        self.root.bind('<Escape>', lambda e: self.deselect_all())
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        self.root.bind('<Control-plus>', lambda e: self.zoom_in())
        self.root.bind('<Control-minus>', lambda e: self.zoom_out())
        self.root.bind('<Control-0>', lambda e: self.zoom_fit())

    # --- Theme Management ---
    def change_theme(self, theme_name):
        """تغییر تم و بازسازی رابط کاربری"""
        self.current_theme = theme_name
        # استخراج حالت ظاهری و رنگ تم
        if theme_name.startswith("Dark"):
            appearance = "dark"
        else:
            appearance = "light"

        if "Blue" in theme_name:
            color_theme = "blue"
        elif "Green" in theme_name:
            color_theme = "green"
        elif "Dark-Blue" in theme_name:
            color_theme = "dark-blue"
        else:
            color_theme = "blue"  # fallback

        # اعمال تنظیمات
        ctk.set_appearance_mode(appearance)
        ctk.set_default_color_theme(color_theme)

        # بازسازی رابط کاربری با حفظ وضعیت
        self.rebuild_ui()

    def rebuild_ui(self):
        """ذخیره وضعیت، نابود کردن ویجت‌ها و بازسازی کامل UI"""
        # ذخیره وضعیت فعلی
        saved_state = {
            'image_dir': self.image_dir,
            'output_dir': self.output_dir,
            'image_files': self.image_files[:],
            'current_img_index': self.current_img_index,
            'original_img': self.original_img,
            'rectangles': [rect[:] for rect in self.rectangles],
            'class_manager': self.class_manager,  # object preserved
            'current_class_id': self.current_class_id,
            'show_labels': self.show_labels,
            'current_format': self.current_format,
            'auto_save': self.auto_save,
            'output_dir_customized': self.output_dir_customized,
        }

        # نابود کردن همه ویجت‌های ریشه
        for widget in self.root.winfo_children():
            widget.destroy()

        # بازسازی UI
        self.setup_ui()
        self.setup_bindings()

        # بازیابی وضعیت
        self.image_dir = saved_state['image_dir']
        self.output_dir = saved_state['output_dir']
        self.image_files = saved_state['image_files']
        self.current_img_index = saved_state['current_img_index']
        self.original_img = saved_state['original_img']
        self.rectangles = saved_state['rectangles']
        self.class_manager = saved_state['class_manager']
        self.current_class_id = saved_state['current_class_id']
        self.show_labels = saved_state['show_labels']
        self.current_format = saved_state['current_format']
        self.auto_save = saved_state['auto_save']
        self.output_dir_customized = saved_state['output_dir_customized']

        # به‌روزرسانی نمایش
        self.update_class_dropdown()
        self.themes_menu.set(self.current_theme)  # تنظیم مجدد منوی تم

        if self.image_files and self.current_img_index < len(self.image_files):
            self.load_current_image()
            self.rectangles = saved_state['rectangles']  # بازنویسی با حالت ذخیره‌شده
            self.update_image_display()
            self.update_ui_state()
            self.update_stats()
        else:
            self.update_image_display()
            self.update_ui_state()
            self.update_stats()

        self.deselect_all()

    def get_class_list(self):
        return [f"{cid} - {info['name']}" for cid, info in self.class_manager.classes.items()]

    def on_class_change(self, choice):
        self.current_class_id = int(choice.split(" - ")[0])
        if self.selected_rect_idx is not None:
            self.rectangles[self.selected_rect_idx][4] = self.current_class_id
            self.save_state_for_undo()
            self.draw_annotations()

    def toggle_labels(self):
        self.show_labels = not self.show_labels
        self.btn_toggle_labels.configure(text="👁️ Show" if not self.show_labels else "👁️ Hide")
        self.draw_annotations()

    # --- منوها ---
    def menu_file(self, choice):
        if choice == "Open Image":
            self.load_single_image()
        elif choice == "Open Dir":
            self.load_directory()
        elif choice == "Set Output":
            self.set_output_directory()
        elif choice == "Save":
            self.save_labels()
        elif choice == "Save As":
            self.save_as()
        elif choice == "Export COCO":
            self.export_coco()
        elif choice == "Exit":
            self.root.quit()

    def menu_edit(self, choice):
        if choice == "Undo":
            self.undo()
        elif choice == "Redo":
            self.redo()
        elif choice == "Auto Save On/Off":
            self.auto_save = not self.auto_save
            messagebox.showinfo("Auto Save", f"Auto save {'enabled' if self.auto_save else 'disabled'}")
        elif choice == "Clear All":
            self.clear_all_boxes()
        elif choice == "Delete":
            self.delete_selected()

    def menu_view(self, choice):
        if choice == "Zoom In":
            self.zoom_in()
        elif choice == "Zoom Out":
            self.zoom_out()
        elif choice == "Fit Window":
            self.zoom_fit()
        elif choice == "100%":
            self.zoom_100()
        elif choice == "Show/Hide Labels":
            self.toggle_labels()

    def menu_help(self, choice):
        if choice == "About":
            
            messagebox.showinfo("About", "Smart City Annotator\nVersion 2.0\nDeveloped by Engineer Khwaja Sulaiman Siddiqi\nEmail: khwajasulaimansiddiqi@gmail.com")
        elif choice == "Class Manager":
            self.open_class_manager()

    # --- مدیریت کلاس‌ها ---
    def open_class_manager(self):
        dialog = Toplevel(self.root)
        dialog.title("Class Manager")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ctk.CTkFrame(dialog)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # لیست کلاس‌ها
        list_frame = ctk.CTkFrame(frame)
        list_frame.pack(fill="both", expand=True, pady=5)

        scrollbar = Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.class_listbox = Listbox(list_frame, yscrollcommand=scrollbar.set, bg="#2b2b2b", fg="white",
                                      selectbackground="#1f538d", font=("Arial", 11))
        self.class_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.class_listbox.yview)

        # پر کردن لیست
        self.refresh_class_listbox()

        # دکمه‌های مدیریت
        btn_frame = ctk.CTkFrame(frame)
        btn_frame.pack(fill="x", pady=5)

        ctk.CTkButton(btn_frame, text="Add", command=lambda: self.add_class_dialog(dialog)).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Edit", command=lambda: self.edit_class_dialog(dialog)).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Delete", command=self.delete_class).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Change Color", command=self.change_class_color).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Load from File", command=self.load_classes_from_file).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Save to File", command=self.save_classes_to_file).pack(side="left", padx=5)

    def refresh_class_listbox(self):
        self.class_listbox.delete(0, "end")
        for cid, info in self.class_manager.classes.items():
            self.class_listbox.insert("end", f"{cid}: {info['name']} [{info['color']}]")

    def add_class_dialog(self, parent):
        dialog = Toplevel(parent)
        dialog.title("Add Class")
        dialog.geometry("300x150")
        dialog.transient(parent)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Class Name:").pack(pady=5)
        name_entry = ctk.CTkEntry(dialog)
        name_entry.pack(pady=5)

        def add():
            name = name_entry.get().strip()
            if name:
                self.class_manager.add_class(name)
                self.refresh_class_listbox()
                self.update_class_dropdown()
                dialog.destroy()

        ctk.CTkButton(dialog, text="Add", command=add).pack(pady=10)

    def edit_class_dialog(self, parent):
        selection = self.class_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        cid = list(self.class_manager.classes.keys())[idx]

        dialog = Toplevel(parent)
        dialog.title("Edit Class")
        dialog.geometry("300x150")
        dialog.transient(parent)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="New Name:").pack(pady=5)
        name_entry = ctk.CTkEntry(dialog)
        name_entry.insert(0, self.class_manager.classes[cid]["name"])
        name_entry.pack(pady=5)

        def edit():
            new_name = name_entry.get().strip()
            if new_name:
                self.class_manager.rename_class(cid, new_name)
                self.refresh_class_listbox()
                self.update_class_dropdown()
                dialog.destroy()

        ctk.CTkButton(dialog, text="Update", command=edit).pack(pady=10)

    def delete_class(self):
        selection = self.class_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        cid = list(self.class_manager.classes.keys())[idx]
        if messagebox.askyesno("Confirm", f"Delete class {self.class_manager.classes[cid]['name']}?"):
            self.class_manager.remove_class(cid)
            self.refresh_class_listbox()
            self.update_class_dropdown()

    def change_class_color(self):
        selection = self.class_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        cid = list(self.class_manager.classes.keys())[idx]
        color = colorchooser.askcolor(title="Choose color", color=self.class_manager.classes[cid]["color"])[1]
        if color:
            self.class_manager.change_color(cid, color)
            self.refresh_class_listbox()
            self.update_class_dropdown()

    def load_classes_from_file(self):
        filepath = filedialog.askopenfilename(title="Select classes file", filetypes=[("Text files", "*.txt")])
        if filepath:
            self.class_manager.load_from_file(filepath)
            self.refresh_class_listbox()
            self.update_class_dropdown()

    def save_classes_to_file(self):
        filepath = filedialog.asksaveasfilename(title="Save classes file", defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")])
        if filepath:
            self.class_manager.save_to_file(filepath)

    def update_class_dropdown(self):
        self.class_dropdown.configure(values=self.get_class_list())
        # به‌روزرسانی مقدار فعلی
        current_text = f"{self.current_class_id} - {self.class_manager.get_class_name(self.current_class_id)}"
        if current_text in self.get_class_list():
            self.class_var.set(current_text)

    # --- عملیات فایل ---
    def load_single_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_files = [file_path]
            self.image_dir = os.path.dirname(file_path)
            if not self.output_dir:
                self.output_dir = self.image_dir
                self.output_dir_customized = False
            self.current_img_index = 0
            self.update_sidebar()
            self.load_current_image()
            self.update_stats()

    def load_directory(self):
        dir_path = filedialog.askdirectory(title="Select Image Directory")
        if dir_path:
            self.image_dir = dir_path
            if not self.output_dir:
                self.output_dir = dir_path
                self.output_dir_customized = False

            self.image_files = []
            for file in glob.glob(os.path.join(dir_path, '*.*')):
                ext = os.path.splitext(file)[1].lower()
                if ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                    self.image_files.append(file)
            self.image_files.sort()

            if not self.image_files:
                messagebox.showerror("Error", "No images found in directory!")
                return

            self.current_img_index = 0
            self.update_sidebar()
            self.load_current_image()
            self.update_stats()

    def set_output_directory(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_dir_customized = True
            messagebox.showinfo("Success", f"Output set to:\n{directory}")
            self.update_stats()

    def save_as(self):
        """ذخیره با انتخاب فرمت"""
        formats = ["YOLO", "Pascal VOC", "COCO"]
        choice = ctk.CTkInputDialog(text="Choose format (YOLO, VOC or COCO):", title="Save As")
        if choice:
            self.current_format = choice.get()
            if self.current_format not in formats:
                self.current_format = "YOLO"
            self.save_labels()

    # --- ناوبری تصاویر ---
    def next_image(self):
        if self.current_img_index < len(self.image_files) - 1:
            if self.auto_save:
                self.save_labels(silent=True)
            else:
                self.save_state_for_undo()
            self.current_img_index += 1
            self.load_current_image()

    def prev_image(self):
        if self.current_img_index > 0:
            if self.auto_save:
                self.save_labels(silent=True)
            else:
                self.save_state_for_undo()
            self.current_img_index -= 1
            self.load_current_image()

    def jump_to_image(self, index):
        if self.auto_save:
            self.save_labels(silent=True)
        else:
            self.save_state_for_undo()
        self.current_img_index = index
        self.load_current_image()

    def update_sidebar(self):
        for widget in self.file_listbox.winfo_children():
            widget.destroy()
        for i, file_path in enumerate(self.image_files):
            filename = os.path.basename(file_path)
            # بررسی وجود فایل برچسب
            base_dir = self.output_dir if self.output_dir else self.image_dir
            label_path = os.path.join(base_dir, os.path.splitext(filename)[0] + ".txt")
            labeled = os.path.exists(label_path)
            color = "#1f538d" if i == self.current_img_index else "transparent"
            text = f"{'✅ ' if labeled else '❌ '}{filename}"
            btn = ctk.CTkButton(self.file_listbox, text=text, fg_color=color, anchor="w",
                                 command=lambda idx=i: self.jump_to_image(idx))
            btn.pack(fill="x", pady=2)

    def load_current_image(self):
        if not self.image_files:
            return
        self.deselect_all()
        self.undo_redo.clear()
        img_path = self.image_files[self.current_img_index]
        self.original_img = cv2.imread(img_path)

        self.rectangles = []
        self.load_existing_labels(img_path)

        self.fit_image_to_canvas()
        self.update_ui_state()
        self.update_image_display()
        self.update_status_bar()

    def load_existing_labels(self, img_path):
        h, w = self.original_img.shape[:2]
        label_path = os.path.join(self.output_dir if self.output_dir else self.image_dir,
                                   os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, bw, bh = map(float, parts)
                        xmin = int((x_center - bw/2) * w)
                        ymin = int((y_center - bh/2) * h)
                        xmax = int((x_center + bw/2) * w)
                        ymax = int((y_center + bh/2) * h)
                        self.rectangles.append([xmin, ymin, xmax, ymax, int(class_id)])

    def fit_image_to_canvas(self):
        self.root.update()
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1 and self.original_img is not None:
            img_h, img_w = self.original_img.shape[:2]
            scale_w = canvas_width / img_w
            scale_h = canvas_height / img_h
            self.scale_factor = min(scale_w, scale_h, 1.0) * 0.95
            new_w = int(img_w * self.scale_factor)
            new_h = int(img_h * self.scale_factor)
            self.offset_x = (canvas_width - new_w) / 2
            self.offset_y = (canvas_height - new_h) / 2

    def update_ui_state(self):
        total = len(self.image_files)
        self.img_counter.configure(text=f"{self.current_img_index + 1} / {total}")
        self.btn_prev.configure(state="normal" if self.current_img_index > 0 else "disabled")
        self.btn_next.configure(state="normal" if self.current_img_index < total - 1 else "disabled")
        self.update_sidebar()

    def update_status_bar(self):
        if self.original_img is not None:
            h, w = self.original_img.shape[:2]
            text = f"Image: {w}x{h} | Zoom: {self.scale_factor*100:.1f}% | Boxes: {len(self.rectangles)}"
        else:
            text = "Ready"
        self.status_label.configure(text=text)

    # --- نمایش تصویر و جعبه‌ها ---
    def update_image_display(self):
        if self.original_img is None:
            return
        new_w = int(self.original_img.shape[1] * self.scale_factor)
        new_h = int(self.original_img.shape[0] * self.scale_factor)
        if new_w > 0 and new_h > 0:
            img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.display_img = ImageTk.PhotoImage(pil_img)
            self.canvas.delete("img")
            self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.display_img, tags="img")
            self.canvas.tag_lower("img")
        self.draw_annotations()

    def draw_annotations(self):
        self.canvas.delete("annotation")
        if not self.show_labels:
            return

        for i, rect in enumerate(self.rectangles):
            xmin, ymin, xmax, ymax, cls_id = rect
            cx1 = xmin * self.scale_factor + self.offset_x
            cy1 = ymin * self.scale_factor + self.offset_y
            cx2 = xmax * self.scale_factor + self.offset_x
            cy2 = ymax * self.scale_factor + self.offset_y

            color = self.class_manager.get_class_color(cls_id)
            is_selected = (i == self.selected_rect_idx)
            is_hovered = (i == self.hovered_rect_idx and not is_selected)

            width = 3 if is_selected else 2
            dash = (4, 4) if is_selected else ()

            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=width, dash=dash, tags="annotation")

            # برچسب
            label = self.class_manager.get_class_name(cls_id)
            self.canvas.create_rectangle(cx1, cy1 - 20, cx1 + (len(label) * 8) + 5, cy1,
                                         fill=color, outline=color, tags="annotation")
            self.canvas.create_text(cx1 + 3, cy1 - 10, text=label, fill="black",
                                    anchor="w", font=("Arial", 10, "bold"), tags="annotation")

            if is_selected:
                r = 4
                self.canvas.create_oval(cx1-r, cy1-r, cx1+r, cy1+r, fill="white", outline=color, tags="annotation")
                self.canvas.create_oval(cx2-r, cy2-r, cx2+r, cy2+r, fill="white", outline=color, tags="annotation")
                self.canvas.create_oval(cx2-r, cy1-r, cx2+r, cy1+r, fill="white", outline=color, tags="annotation")
                self.canvas.create_oval(cx1-r, cy2-r, cx1+r, cy2+r, fill="white", outline=color, tags="annotation")
            elif is_hovered:
                # نمایش دستگیره‌های گوشه برای باکس hover شده
                r = 4
                self.canvas.create_oval(cx1-r, cy1-r, cx1+r, cy1+r, fill="yellow", outline=color, tags="annotation")
                self.canvas.create_oval(cx2-r, cy2-r, cx2+r, cy2+r, fill="yellow", outline=color, tags="annotation")
                self.canvas.create_oval(cx2-r, cy1-r, cx2+r, cy1+r, fill="yellow", outline=color, tags="annotation")
                self.canvas.create_oval(cx1-r, cy2-r, cx1+r, cy2+r, fill="yellow", outline=color, tags="annotation")

        if self.drawing and self.interaction_mode == "draw":
            cx1 = self.draw_start_x * self.scale_factor + self.offset_x
            cy1 = self.draw_start_y * self.scale_factor + self.offset_y
            cx2 = self.last_mouse_x * self.scale_factor + self.offset_x
            cy2 = self.last_mouse_y * self.scale_factor + self.offset_y
            color = self.class_manager.get_class_color(self.current_class_id)
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2, tags="annotation")

    # --- زوم و پن ---
    def on_mouse_wheel(self, event):
        if self.original_img is None:
            return
        factor = 0.9 if (event.num == 5 or event.delta < 0) else 1.1
        x, y = event.x, event.y
        img_x = (x - self.offset_x) / self.scale_factor
        img_y = (y - self.offset_y) / self.scale_factor
        new_scale = self.scale_factor * factor
        if 0.1 < new_scale < 15.0:
            self.scale_factor = new_scale
            self.offset_x = x - (img_x * self.scale_factor)
            self.offset_y = y - (img_y * self.scale_factor)
            self.update_image_display()
            self.update_status_bar()

    def zoom_in(self):
        if self.original_img:
            self.scale_factor *= 1.2
            self.update_image_display()
            self.update_status_bar()

    def zoom_out(self):
        if self.original_img:
            self.scale_factor *= 0.8
            self.update_image_display()
            self.update_status_bar()

    def zoom_fit(self):
        self.fit_image_to_canvas()
        self.update_image_display()
        self.update_status_bar()

    def zoom_100(self):
        if self.original_img:
            self.scale_factor = 1.0
            canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
            img_w, img_h = self.original_img.shape[1], self.original_img.shape[0]
            self.offset_x = (canvas_width - img_w) / 2
            self.offset_y = (canvas_height - img_h) / 2
            self.update_image_display()
            self.update_status_bar()

    def start_pan(self, event):
        self.panning = True
        self.pan_start_x, self.pan_start_y = event.x, event.y
        self.canvas.config(cursor="fleur")

    def do_pan(self, event):
        if self.panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start_x, self.pan_start_y = event.x, event.y
            self.canvas.move("img", dx, dy)
            self.canvas.move("annotation", dx, dy)

    def end_pan(self, event):
        self.panning = False
        self.canvas.config(cursor="crosshair")

    # --- تعامل با موس ---
    def on_mouse_move(self, event):
        self.draw_crosshair(event)
        if self.original_img:
            ix, iy = self.get_image_coords(event.x, event.y)
            self.status_label.configure(text=f"X: {ix}, Y: {iy} | Zoom: {self.scale_factor*100:.1f}% | Boxes: {len(self.rectangles)}")
            # بررسی hover برای snap
            self.check_hover(ix, iy)

    def check_hover(self, img_x, img_y):
        """تشخیص hover روی bounding boxها و تغییر cursor"""
        hit_idx, hit_mode = self.get_hit_target(img_x, img_y, for_hover=True)
        if hit_idx != self.hovered_rect_idx or hit_mode != self.hover_mode:
            self.hovered_rect_idx = hit_idx
            self.hover_mode = hit_mode
            self.draw_annotations()

        # تغییر شکل موس بر اساس حالت
        if hit_idx is not None:
            if hit_mode.startswith("resize"):
                if hit_mode in ("resize_tl", "resize_br"):
                    self.canvas.config(cursor="size_nw_se")
                elif hit_mode in ("resize_tr", "resize_bl"):
                    self.canvas.config(cursor="size_ne_sw")
                elif hit_mode in ("resize_l", "resize_r"):
                    self.canvas.config(cursor="size_we")
                elif hit_mode in ("resize_t", "resize_b"):
                    self.canvas.config(cursor="size_ns")
                else:
                    self.canvas.config(cursor="crosshair")
            elif hit_mode == "move":
                self.canvas.config(cursor="fleur")
            else:
                self.canvas.config(cursor="crosshair")
        else:
            self.canvas.config(cursor="crosshair")

    def draw_crosshair(self, event):
        self.remove_crosshair(None)
        # انتخاب رنگ crosshair بر اساس حالت ظاهری
        if ctk.get_appearance_mode().lower() == "light":
            crosshair_color = "black"
        else:
            crosshair_color = "white"
        self.vline = self.canvas.create_line(event.x, 0, event.x, self.canvas.winfo_height(),
                                              fill=crosshair_color, dash=(2, 2), tags="crosshair")
        self.hline = self.canvas.create_line(0, event.y, self.canvas.winfo_width(), event.y,
                                              fill=crosshair_color, dash=(2, 2), tags="crosshair")

    def remove_crosshair(self, event):
        if self.vline:
            self.canvas.delete(self.vline)
        if self.hline:
            self.canvas.delete(self.hline)

    def get_image_coords(self, canvas_x, canvas_y):
        ix = int((canvas_x - self.offset_x) / self.scale_factor)
        iy = int((canvas_y - self.offset_y) / self.scale_factor)
        if self.original_img is not None:
            ix = max(0, min(ix, self.original_img.shape[1]))
            iy = max(0, min(iy, self.original_img.shape[0]))
        return ix, iy

    def get_hit_target(self, img_x, img_y, for_hover=False):
        """تشخیص برخورد با bounding boxها (برای کلیک یا hover)"""
        tolerance = 8 / self.scale_factor
        for i in range(len(self.rectangles)-1, -1, -1):
            rx1, ry1, rx2, ry2, _ = self.rectangles[i]
            # برای hover می‌خواهیم فقط روی گوشه‌ها حساس باشیم، ولی در کلیک کل باکس
            if for_hover:
                # فقط گوشه‌ها
                if abs(img_x - rx1) < tolerance and abs(img_y - ry1) < tolerance:
                    return i, "resize_tl"
                if abs(img_x - rx2) < tolerance and abs(img_y - ry2) < tolerance:
                    return i, "resize_br"
                if abs(img_x - rx2) < tolerance and abs(img_y - ry1) < tolerance:
                    return i, "resize_tr"
                if abs(img_x - rx1) < tolerance and abs(img_y - ry2) < tolerance:
                    return i, "resize_bl"
                if abs(img_x - rx1) < tolerance and ry1 <= img_y <= ry2:
                    return i, "resize_l"
                if abs(img_x - rx2) < tolerance and ry1 <= img_y <= ry2:
                    return i, "resize_r"
                if abs(img_y - ry1) < tolerance and rx1 <= img_x <= rx2:
                    return i, "resize_t"
                if abs(img_y - ry2) < tolerance and rx1 <= img_x <= rx2:
                    return i, "resize_b"
                # اگر داخل باکس باشد، فقط حرکت
                if rx1 <= img_x <= rx2 and ry1 <= img_y <= ry2:
                    return i, "move"
            else:
                # برای کلیک: تمام حالات (گوشه‌ها و داخل)
                if abs(img_x - rx1) < tolerance and abs(img_y - ry1) < tolerance:
                    return i, "resize_tl"
                if abs(img_x - rx2) < tolerance and abs(img_y - ry2) < tolerance:
                    return i, "resize_br"
                if abs(img_x - rx2) < tolerance and abs(img_y - ry1) < tolerance:
                    return i, "resize_tr"
                if abs(img_x - rx1) < tolerance and abs(img_y - ry2) < tolerance:
                    return i, "resize_bl"
                if abs(img_x - rx1) < tolerance and ry1 <= img_y <= ry2:
                    return i, "resize_l"
                if abs(img_x - rx2) < tolerance and ry1 <= img_y <= ry2:
                    return i, "resize_r"
                if abs(img_y - ry1) < tolerance and rx1 <= img_x <= rx2:
                    return i, "resize_t"
                if abs(img_y - ry2) < tolerance and rx1 <= img_x <= rx2:
                    return i, "resize_b"
                if rx1 <= img_x <= rx2 and ry1 <= img_y <= ry2:
                    return i, "move"
        return None, "draw"

    def on_mouse_down(self, event):
        if self.original_img is None:
            return
        img_x, img_y = self.get_image_coords(event.x, event.y)
        hit_idx, hit_mode = self.get_hit_target(img_x, img_y, for_hover=False)

        if hit_idx is not None:
            self.selected_rect_idx = hit_idx
            self.interaction_mode = hit_mode
            self.class_var.set(f"{self.rectangles[hit_idx][4]} - {self.class_manager.get_class_name(self.rectangles[hit_idx][4])}")
            self.btn_delete.configure(state="normal")
        else:
            self.deselect_all()
            self.interaction_mode = "draw"
            self.drawing = True
            self.draw_start_x, self.draw_start_y = img_x, img_y

        self.last_mouse_x, self.last_mouse_y = img_x, img_y
        self.save_state_for_undo()
        self.draw_annotations()

    def on_mouse_drag(self, event):
        if self.original_img is None:
            return
        self.draw_crosshair(event)
        current_x, current_y = self.get_image_coords(event.x, event.y)
        dx = current_x - self.last_mouse_x
        dy = current_y - self.last_mouse_y

        if self.selected_rect_idx is not None:
            idx = self.selected_rect_idx
            rect = self.rectangles[idx]
            if self.interaction_mode == "move":
                rect[0] += dx; rect[1] += dy; rect[2] += dx; rect[3] += dy
            elif self.interaction_mode == "resize_tl":
                rect[0] += dx; rect[1] += dy
            elif self.interaction_mode == "resize_br":
                rect[2] += dx; rect[3] += dy
            elif self.interaction_mode == "resize_tr":
                rect[2] += dx; rect[1] += dy
            elif self.interaction_mode == "resize_bl":
                rect[0] += dx; rect[3] += dy
            elif self.interaction_mode == "resize_l":
                rect[0] += dx
            elif self.interaction_mode == "resize_r":
                rect[2] += dx
            elif self.interaction_mode == "resize_t":
                rect[1] += dy
            elif self.interaction_mode == "resize_b":
                rect[3] += dy

        self.last_mouse_x, self.last_mouse_y = current_x, current_y
        self.draw_annotations()

    def on_mouse_up(self, event):
        if self.original_img is None:
            return
        if self.interaction_mode == "draw" and self.drawing:
            xmin, ymin = min(self.draw_start_x, self.last_mouse_x), min(self.draw_start_y, self.last_mouse_y)
            xmax, ymax = max(self.draw_start_x, self.last_mouse_x), max(self.draw_start_y, self.last_mouse_y)
            if xmax - xmin > 5 and ymax - ymin > 5:
                self.rectangles.append([xmin, ymin, xmax, ymax, self.current_class_id])
                self.selected_rect_idx = len(self.rectangles) - 1
                self.btn_delete.configure(state="normal")
                self.save_state_for_undo()
            self.drawing = False
        elif self.selected_rect_idx is not None:
            rect = self.rectangles[self.selected_rect_idx]
            xmin, ymin = min(rect[0], rect[2]), min(rect[1], rect[3])
            xmax, ymax = max(rect[0], rect[2]), max(rect[1], rect[3])
            self.rectangles[self.selected_rect_idx] = [xmin, ymin, xmax, ymax, rect[4]]
            self.save_state_for_undo()
        self.interaction_mode = "idle"
        self.draw_annotations()
        self.update_status_bar()

    # --- Undo/Redo ---
    def save_state_for_undo(self):
        # یک کپی عمیق از rectangles ذخیره می‌کنیم
        state = [rect[:] for rect in self.rectangles]
        self.undo_redo.push(state)

    def undo(self):
        state = self.undo_redo.undo()
        if state is not None:
            self.rectangles = [rect[:] for rect in state]
            self.deselect_all()
            self.draw_annotations()
            self.update_status_bar()

    def redo(self):
        state = self.undo_redo.redo()
        if state is not None:
            self.rectangles = [rect[:] for rect in state]
            self.deselect_all()
            self.draw_annotations()
            self.update_status_bar()

    # --- عملیات روی جعبه‌ها ---
    def delete_selected(self):
        if self.selected_rect_idx is not None:
            self.rectangles.pop(self.selected_rect_idx)
            self.save_state_for_undo()
            self.deselect_all()
            self.draw_annotations()
            self.update_status_bar()

    def clear_all_boxes(self):
        if not self.rectangles:
            return
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear ALL bounding boxes for this image?"):
            self.rectangles = []
            self.save_state_for_undo()
            self.deselect_all()
            self.draw_annotations()
            self.update_status_bar()

    def deselect_all(self):
        self.selected_rect_idx = None
        self.btn_delete.configure(state="disabled")
        self.draw_annotations()

    # --- ذخیره برچسب‌ها ---
    def save_labels(self, silent=False):
        if self.original_img is None:
            return

        if not self.output_dir_customized:
            response = messagebox.askyesno(
                "Output Folder",
                "Output folder not set. Would you like to select an output folder?\n"
                "(If not, labels will be saved in the image folder.)"
            )
            if response:
                directory = filedialog.askdirectory(title="Select Output Directory")
                if directory:
                    self.output_dir = directory
                    self.output_dir_customized = True
                else:
                    return
        base = os.path.splitext(os.path.basename(self.image_files[self.current_img_index]))[0]
        if self.current_format == "YOLO":
            self.save_yolo(base)
        elif self.current_format == "Pascal VOC":
            self.save_voc(base)
        elif self.current_format == "COCO":
            self.save_coco_single(base)

        if not silent:
            self._flash_save_btn("Saved!")
        self.update_stats()
        self.update_sidebar()  # برای به‌روزرسانی آیکون ✅

    def save_yolo(self, base):
        h, w = self.original_img.shape[:2]
        label_path = os.path.join(self.output_dir, f"{base}.txt")
        if not self.rectangles:
            if os.path.exists(label_path):
                os.remove(label_path)
            return
        with open(label_path, 'w') as f:
            for rect in self.rectangles:
                xmin, ymin, xmax, ymax, class_id = rect
                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    def save_voc(self, base):
        h, w = self.original_img.shape[:2]
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = os.path.basename(self.output_dir)
        ET.SubElement(annotation, "filename").text = os.path.basename(self.image_files[self.current_img_index])
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"

        for rect in self.rectangles:
            xmin, ymin, xmax, ymax, class_id = rect
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = self.class_manager.get_class_name(class_id)
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

        xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
        label_path = os.path.join(self.output_dir, f"{base}.xml")
        with open(label_path, 'w') as f:
            f.write(xml_str)

    def save_coco_single(self, base):
        """ذخیره یک تصویر به فرمت COCO (برای سازگاری با Save As)"""
        h, w = self.original_img.shape[:2]
        coco_data = {
            "images": [{
                "id": 1,
                "file_name": os.path.basename(self.image_files[self.current_img_index]),
                "width": w,
                "height": h
            }],
            "categories": [],
            "annotations": []
        }
        # دسته‌بندی‌ها
        cat_id_map = {}
        for cid, info in self.class_manager.classes.items():
            cat_id_map[cid] = cid
            coco_data["categories"].append({
                "id": cid,
                "name": info["name"]
            })

        # annotion‌ها
        for i, rect in enumerate(self.rectangles):
            xmin, ymin, xmax, ymax, cls_id = rect
            width = xmax - xmin
            height = ymax - ymin
            coco_data["annotations"].append({
                "id": i + 1,
                "image_id": 1,
                "category_id": cls_id,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0
            })

        label_path = os.path.join(self.output_dir, f"{base}.json")
        with open(label_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

    def _flash_save_btn(self, text):
        original_color = self.btn_save.cget("fg_color")
        self.btn_save.configure(text=text, fg_color="#059669")
        self.root.after(1000, lambda: self.btn_save.configure(text="💾 Save (Ctrl+S)", fg_color=original_color))

    # --- آمار ---
    def update_stats(self):
        if not self.image_files or not (self.output_dir or self.image_dir):
            self.stats_label.configure(text="No data")
            return
        base_dir = self.output_dir if self.output_dir else self.image_dir
        labeled = 0
        total_boxes = 0
        for img_path in self.image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(base_dir, base + ".txt")
            if os.path.exists(label_path):
                labeled += 1
                try:
                    with open(label_path, 'r') as f:
                        total_boxes += sum(1 for line in f if line.strip())
                except:
                    pass
        self.stats_label.configure(text=f"Labeled: {labeled}/{len(self.image_files)} images\nTotal boxes: {total_boxes}")

    # --- خروجی COCO کلی ---
    def export_coco(self):
        """ایجاد یک فایل COCO JSON برای تمام تصاویر دارای لیبل"""
        if not self.image_files:
            messagebox.showerror("Error", "No images loaded.")
            return
        if not self.output_dir_customized:
            messagebox.showwarning("Warning", "Please set output directory first.")
            return

        # جمع‌آوری اطلاعات
        images = []
        annotations = []
        categories = []
        cat_id_map = {}
        ann_id = 1
        img_id = 1

        # دسته‌بندی‌ها
        for cid, info in self.class_manager.classes.items():
            cat_id_map[cid] = cid
            categories.append({"id": cid, "name": info["name"]})

        # پردازش هر تصویر
        for img_path in self.image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.output_dir, base + ".txt")
            if not os.path.exists(label_path):
                continue  # تصاویر بدون لیبل نادیده گرفته می‌شوند

            # خواندن ابعاد تصویر
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            images.append({
                "id": img_id,
                "file_name": os.path.basename(img_path),
                "width": w,
                "height": h
            })

            # خواندن لیبل‌ها
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, x_center, y_center, bw, bh = map(float, parts)
                    xmin = (x_center - bw/2) * w
                    ymin = (y_center - bh/2) * h
                    width = bw * w
                    height = bh * h
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls_id),
                        "bbox": [xmin, ymin, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    ann_id += 1

            img_id += 1

        if not annotations:
            messagebox.showinfo("Info", "No annotations found.")
            return

        coco_data = {
            "images": images,
            "categories": categories,
            "annotations": annotations
        }

        # انتخاب مسیر ذخیره
        save_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")],
                                                 title="Save COCO JSON")
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            messagebox.showinfo("Success", f"COCO annotations saved to:\n{save_path}")


if __name__ == "__main__":
    root = ctk.CTk()
    app = AdvancedYOLOAnnotator(root)
    root.mainloop()