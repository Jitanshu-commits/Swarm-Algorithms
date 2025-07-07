"""
An advanced pathfinding visualization and analysis tool using Tkinter.

This application serves as an experimental testbed for comparing three metaheuristic
algorithms, Ant Colony Optimization (ACO), Particle Swarm Optimization (PSO),
and Artificial Bee Colony (ABC), for solving the TSP on an intermodal grid.

Features:
- Interactive grid for designing complex, intermodal delivery networks.
- Intermodal A* pathfinding with caching for high performance.
- Control over key algorithm parameters for all three algorithms, with tooltips.
- Control over intermodal travel and transfer costs.
- An "Experiment Mode" to automatically run and compare algorithms multiple
  times, with a live progress bar, exporting the results to a CSV file.
- Textbook-accurate implementations of ACO, DPSO, and ABC.
- A modern, themed UI with a scrollable control panel.

Bugs:
- You may experience sudden crashes if there are less then 4 transfer point during ABC visualization.
- The application can crash during Experiment run. 
Required library: numpy (pip install numpy)
"""

import tkinter as tk
from tkinter import messagebox, ttk, simpledialog, filedialog
import random
import heapq
import time
import threading
from platform import system
from typing import List, Tuple, Optional, Dict, Any
import csv

# I need numpy for this to work, especially for some math operations in the algorithms.
# You can install it with: pip install numpy
import numpy as np


# --- Configuration Constants and Enums ---

# These are just some default values to set up the grid and window size.
_GRID_ROWS = 20
_GRID_COLS = 25
_INITIAL_CELL_SIZE = 28
_CONTROL_PANEL_WIDTH = 300
_PADDING = 10


# Using a class for enums is a clean way to avoid "magic numbers" in the code.
# So instead of grid[r][c] == 1, I can write grid[r][c] == CellType.WALL, which is much clearer.
class CellType:
    """Enumeration for the different types of cells on the grid."""
    EMPTY = 0
    WALL = 1
    START = 2
    END = 3
    PATH = 4
    HIGHWAY = 5
    DELIVERY = 7
    AIRPORT = 8
    RAIL_TERMINAL = 9


# Same idea here, for the different algorithms I'm comparing.
class Algorithm:
    """Enumeration for the available pathfinding algorithms."""
    ACO = "Ant Colony Optimization"
    PSO = "Discrete Particle Swarm Optimization"
    ABC = "Artificial Bee Colony"

# A helper class to create hover-over tooltips for the UI.
class Tooltip:
    """Creates a tooltip for a given widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        # Bind mouse events to the widget.
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        # When the mouse enters, create a small pop-up window.
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True) # Removes the window border.
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background="#3a3f4b", relief='solid', borderwidth=1,
                         wraplength=200, fg="white", font=("Segoe UI", 8))
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        # When the mouse leaves, destroy the pop-up.
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

# --- Main Application Class ---

class PathfindingUI(tk.Tk):
    # The __init__ method is where everything gets set up.
    def __init__(self) -> None:
        super().__init__()
        self.title("Intermodal Delivery Network Simulation")
        # Make the application start in a maximized window state.
        self.state('zoomed')

        # --- UI and Grid Configuration ---
        self.grid_rows = _GRID_ROWS
        self.grid_cols = _GRID_COLS
        self.cell_size = float(_INITIAL_CELL_SIZE)

        # --- Core Application State ---
        # This is the main 2D list that holds the state of every cell on the grid.
        self.grid_data: List[List[int]] = [[CellType.EMPTY for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        # Variables to keep track of where the special points are.
        self.start: Optional[Tuple[int, int]] = None
        self.end: Optional[Tuple[int, int]] = None
        self.delivery_points: List[Tuple[int, int]] = []
        # Lists to store hub locations for easy access later.
        self.airports: List[Tuple[int, int]] = []
        self.rail_terminals: List[Tuple[int, int]] = []
        # This is a big performance boost! We store paths we've already found so we don't have to recalculate them.
        self.path_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Optional[List[Tuple[int, int]]]] = {}

        # --- Algorithm Control State ---
        # A flag to check if an algorithm is running, used to stop it gracefully.
        self.is_running = False
        # These are the variables that the UI sliders will control.
        # Algorithm parameters
        self.abc_limit_var = tk.IntVar(value=20)
        # Intermodal cost parameters - gives the user control over the simulation's economy.
        self.cost_transfer_var = tk.DoubleVar(value=5.0)
        self.cost_highway_var = tk.DoubleVar(value=0.4)
        self.cost_rail_var = tk.DoubleVar(value=0.1)
        self.cost_air_var = tk.DoubleVar(value=0.05)

        # --- Window and Layout Setup ---
        # Now, call the methods to actually build the window and create the widgets.
        self._setup_window()
        main_frame = self._setup_main_layout()
        self.create_widgets(main_frame)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.update_idletasks()
        self.on_canvas_resize(None)
        self._update_algo_params_ui()

    # This method just sets up the main window size and properties.
    def _setup_window(self) -> None:
        initial_canvas_width = self.grid_cols * self.cell_size
        initial_canvas_height = self.grid_rows * self.cell_size
        # Increased window height to accommodate new sliders.
        initial_window_width = int(initial_canvas_width + _CONTROL_PANEL_WIDTH + 3 * _PADDING)
        min_control_panel_height = 800 
        initial_window_height = int(max(initial_canvas_height, min_control_panel_height) + 2 * _PADDING)
        self.geometry(f"{initial_window_width}x{initial_window_height}")
        min_win_width = int(self.grid_cols * 10 + _CONTROL_PANEL_WIDTH + 3 * _PADDING)
        min_win_height = max(int(self.grid_rows * 10 + 2 * _PADDING), 600)
        self.minsize(min_win_width, min_win_height)
        self.configure(bg='#282c34')

    # This sets up the main frames: the canvas on the left, controls on the right.
    def _setup_main_layout(self) -> tk.Frame:
        main_frame = tk.Frame(self, bg='#282c34')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=_PADDING, pady=_PADDING)
        main_frame.grid_columnconfigure(0, weight=3) # Canvas takes up more space.
        main_frame.grid_columnconfigure(1, weight=1, minsize=_CONTROL_PANEL_WIDTH)
        main_frame.grid_rowconfigure(0, weight=1)
        return main_frame

    # This is a big method where all the buttons, sliders, and labels are created.
    def create_widgets(self, parent_frame: tk.Frame) -> None:
        # The main canvas where the grid is drawn.
        self.canvas = tk.Canvas(parent_frame, bg='#1c1f26', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, _PADDING // 2))
        self.canvas.bind("<Button-1>", self.left_click)
        self.canvas.bind("<B1-Motion>", self.left_click)
        self.canvas.bind("<Button-3>", self.right_click)
        
        # This whole block is for making the right-side control panel scrollable.
        controls_container = tk.Frame(parent_frame, width=_CONTROL_PANEL_WIDTH, bg='#21252b')
        controls_container.grid(row=0, column=1, sticky="nsew", padx=(_PADDING // 2, 0))
        controls_container.grid_propagate(False)
        controls_canvas = tk.Canvas(controls_container, bg='#21252b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(controls_container, orient="vertical", command=controls_canvas.yview)
        controls_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        controls_canvas.pack(side="left", fill="both", expand=True)
        self.ctrl_frame = tk.Frame(controls_canvas, bg='#21252b')
        canvas_window_id = controls_canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")
        def on_inner_frame_configure(e): controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))
        def on_canvas_configure(e): controls_canvas.itemconfig(canvas_window_id, width=e.width)
        self.ctrl_frame.bind("<Configure>", on_inner_frame_configure)
        controls_canvas.bind("<Configure>", on_canvas_configure)
        def _on_mousewheel(e):
            # Platform-specific mouse wheel scrolling.
            d = 0
            if system()=='Windows': d=int(-1*(e.delta/120))
            elif system()=='Darwin': d=int(-1*e.delta)
            else: d=-1 if e.num==4 else (1 if e.num==5 else 0)
            controls_canvas.yview_scroll(d, "units")
        def _bind_mouse(e): self.bind_all("<MouseWheel>", _on_mousewheel); self.bind_all("<Button-4>", _on_mousewheel); self.bind_all("<Button-5>", _on_mousewheel)
        def _unbind_mouse(e): self.unbind_all("<MouseWheel>"); self.unbind_all("<Button-4>"); self.unbind_all("<Button-5>")
        controls_container.bind('<Enter>', _bind_mouse); controls_container.bind('<Leave>', _unbind_mouse)
        
        # --- Now for the actual widgets inside the control panel ---
        tk.Label(self.ctrl_frame, text="Controls", font=("Segoe UI", 16, 'bold'), fg='white', bg='#21252b').pack(pady=(10,15), fill='x', padx=10)
        
        # Frame for the drawing mode radio buttons.
        mode_frame = tk.LabelFrame(self.ctrl_frame, text="Drawing Mode", fg='white', bg='#2c313a', font=("Segoe UI", 10, 'bold'), padx=8, pady=8, labelanchor="n")
        mode_frame.pack(pady=5, fill='x', padx=10)
        self.mode_var = tk.StringVar(value='wall')
        modes = [("Draw/Remove Walls", 'wall'), ("Draw/Remove Highways", 'highway'), ("Add/Remove Airports", 'airport'), ("Add/Remove Rail Terminals", 'rail_terminal'), ("Set Start Point", 'start'), ("Set End Point", 'end'), ("Add Delivery Points", 'delivery')]
        for text, val in modes: tk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=val, bg='#2c313a', fg='white', selectcolor='#61afef', font=("Segoe UI", 9), activebackground='#3a3f4b', activeforeground='white', indicatoron=1).pack(anchor='w', pady=1)
        self.delivery_info = tk.Label(self.ctrl_frame, text="Delivery Points: 0", font=("Segoe UI", 10), fg='#f39c12', bg='#21252b')
        self.delivery_info.pack(pady=5)
        
        # Frame for algorithm selection.
        algo_frame = tk.LabelFrame(self.ctrl_frame, text="Algorithm", fg='white', bg='#2c313a', font=("Segoe UI", 10, 'bold'), padx=8, pady=8, labelanchor="n")
        algo_frame.pack(pady=10, fill='x', padx=10)
        self.algo_var = tk.StringVar(value=Algorithm.ACO)
        for algo in [Algorithm.ACO, Algorithm.PSO, Algorithm.ABC]: tk.Radiobutton(algo_frame, text=algo, variable=self.algo_var, value=algo, command=self._update_algo_params_ui, bg='#2c313a', fg='white', selectcolor='#98c379', font=("Segoe UI", 9), activebackground='#3a3f4b', activeforeground='white', indicatoron=1).pack(anchor='w', pady=2)
        
        # NEW: Frame for the intermodal cost sliders.
        cost_settings_frame = tk.LabelFrame(self.ctrl_frame, text="Intermodal Cost Settings", fg='white', bg='#2c313a', font=("Segoe UI", 10, 'bold'), padx=8, pady=8, labelanchor="n")
        cost_settings_frame.pack(pady=10, fill='x', padx=10)
        self._create_slider(cost_settings_frame, "Transfer Penalty:", self.cost_transfer_var, 0, 20, "Fixed cost for loading/unloading at a hub.", 0.5)
        self._create_slider(cost_settings_frame, "Highway Cost:", self.cost_highway_var, 0.1, 1.0, "Travel cost multiplier for highways (lower is faster).", 0.05)
        self._create_slider(cost_settings_frame, "Rail Cost:", self.cost_rail_var, 0.01, 0.5, "Travel cost multiplier for rail (lower is faster).", 0.01)
        self._create_slider(cost_settings_frame, "Air Cost:", self.cost_air_var, 0.01, 0.5, "Travel cost multiplier for air (lower is faster).", 0.01)
        
        # Frame for the main algorithm parameters.
        settings_frame = tk.LabelFrame(self.ctrl_frame, text="Algorithm Parameters", fg='white', bg='#2c313a', font=("Segoe UI", 10, 'bold'), padx=8, pady=8, labelanchor="n")
        settings_frame.pack(pady=10, fill='x', padx=10)
        self.num_iter_var = tk.IntVar(value=50); self.population_var = tk.IntVar(value=30)
        self._create_slider(settings_frame, "Iterations:", self.num_iter_var, 10, 200, "Number of generations the algorithm will run.")
        self._create_slider(settings_frame, "Colony/Swarm Size:", self.population_var, 10, 100, "Number of agents (ants, bees, particles) in the population.")
        # These frames will be shown/hidden depending on which algorithm is selected.
        self.aco_params_frame = tk.Frame(settings_frame, bg='#2c313a'); self.pso_params_frame = tk.Frame(settings_frame, bg='#2c313a'); self.abc_params_frame = tk.Frame(settings_frame, bg='#2c313a')
        self.aco_alpha = tk.DoubleVar(value=1.0); self.aco_beta = tk.DoubleVar(value=2.0); self.aco_decay = tk.DoubleVar(value=0.5)
        self._create_slider(self.aco_params_frame, "Alpha (α):", self.aco_alpha, 0.1, 5.0, "Pheromone influence. Higher values prioritize popular paths.", .1)
        self._create_slider(self.aco_params_frame, "Beta (β):", self.aco_beta, 0.1, 5.0, "Heuristic influence. Higher values prioritize shorter-distance paths.", .1)
        self._create_slider(self.aco_params_frame, "Decay (ρ):", self.aco_decay, 0.1, 1.0, "Pheromone evaporation rate. Higher values encourage exploration.", .05)
        self.pso_w = tk.DoubleVar(value=0.7); self.pso_c1 = tk.DoubleVar(value=1.5); self.pso_c2 = tk.DoubleVar(value=1.5)
        self._create_slider(self.pso_params_frame, "Inertia (w):", self.pso_w, 0.1, 1.0, "How much a particle maintains its previous direction.", .05)
        self._create_slider(self.pso_params_frame, "Cognitive (c1):", self.pso_c1, 0.1, 3.0, "How much a particle is attracted to its own best solution.", .1)
        self._create_slider(self.pso_params_frame, "Social (c2):", self.pso_c2, 0.1, 3.0, "How much a particle is attracted to the swarm's best solution.", .1)
        self._create_slider(self.abc_params_frame, "Abandonment Limit:", self.abc_limit_var, 1, 100, "Number of failed attempts before a bee abandons a food source.")
        
        # Frame for live statistics during a run.
        stats_frame = tk.LabelFrame(self.ctrl_frame, text="Live Statistics", fg='white', bg='#2c313a', font=("Segoe UI", 10, 'bold'), padx=8, pady=8, labelanchor="n")
        stats_frame.pack(pady=10, fill='x', padx=10)
        self.iteration_label = tk.Label(stats_frame, text="Status: Idle", font=("Segoe UI", 9), fg='#abb2bf', bg='#2c313a'); self.iteration_label.pack(anchor='w')
        self.best_path_label = tk.Label(stats_frame, text="Best Path Cost: N/A", font=("Segoe UI", 9), fg='#abb2bf', bg='#2c313a'); self.best_path_label.pack(anchor='w')
        self.progress = ttk.Progressbar(stats_frame, mode='determinate'); self.progress.pack(fill='x', pady=5)
        
        # Finally, the main action buttons.
        btn_font = ("Segoe UI", 10, 'bold'); action_frame = tk.Frame(self.ctrl_frame, bg='#21252b'); action_frame.pack(pady=5, fill='x', padx=10)
        self.run_btn = tk.Button(action_frame, text="Run Simulation", font=btn_font, bg='#61afef', fg='white', relief='flat', command=self.run_pathfinding, cursor='hand2'); self.run_btn.pack(side='left', fill='x', expand=True, padx=(0,2))
        self.run_exp_btn = tk.Button(action_frame, text="Run Experiment", font=btn_font, bg='#98c379', fg='white', relief='flat', command=self.run_experiment, cursor='hand2'); self.run_exp_btn.pack(side='left', fill='x', expand=True, padx=(2,0))
        self.stop_btn = tk.Button(self.ctrl_frame, text="Stop Algorithm", font=btn_font, bg='#e06c75', fg='white', relief='flat', command=self.stop_algorithm, cursor='hand2', state='disabled'); self.stop_btn.pack(pady=2, fill='x', padx=10)
        clear_frame = tk.Frame(self.ctrl_frame, bg='#21252b'); clear_frame.pack(pady=5, fill='x', padx=10)
        self.clear_path_btn = tk.Button(clear_frame, text="Clear Path", font=btn_font, bg='#e5c07b', fg='black', relief='flat', command=lambda: self.clear_path(redraw=True), cursor='hand2'); self.clear_path_btn.pack(side='left', fill='x', expand=True, padx=(0,2))
        self.clear_all_btn = tk.Button(clear_frame, text="Reset Grid", font=btn_font, bg='#c678dd', fg='white', relief='flat', command=self.clear_all, cursor='hand2'); self.clear_all_btn.pack(side='left', fill='x', expand=True, padx=(2,0))

    # A helper to make creating sliders less repetitive.
    def _create_slider(self, parent, text, var, from_, to, tooltip_text, res=1):
        frame = tk.Frame(parent, bg='#2c313a'); frame.pack(fill='x', pady=2)
        label = tk.Label(frame, text=text, font=("Segoe UI", 9), fg='white', bg='#2c313a'); label.pack(side='left')
        s = tk.Scale(frame, variable=var, from_=from_, to=to, resolution=res, orient='horizontal', bg='#2c313a', fg='white', troughcolor='#3a3f4b', highlightthickness=0); s.pack(side='right', fill='x', expand=True)
        # Attach our cool tooltip helper class to the slider's frame.
        Tooltip(frame, tooltip_text)
        return s

    # This just shows/hides the specific parameter sliders for the selected algorithm.
    def _update_algo_params_ui(self):
        self.aco_params_frame.pack_forget()
        self.pso_params_frame.pack_forget()
        self.abc_params_frame.pack_forget()
        algo = self.algo_var.get()
        if algo == Algorithm.ACO: self.aco_params_frame.pack(fill='x', pady=5)
        elif algo == Algorithm.PSO: self.pso_params_frame.pack(fill='x', pady=5)
        elif algo == Algorithm.ABC: self.abc_params_frame.pack(fill='x', pady=5)

    # Disables/enables UI controls when an algorithm is running.
    def _set_ui_state(self, state: str):
        for child in self.ctrl_frame.winfo_children():
            if isinstance(child, tk.Button) and child != self.stop_btn: child.config(state=state)
            elif isinstance(child, tk.LabelFrame):
                for widget in child.winfo_children():
                    try: widget.config(state=state)
                    except tk.TclError: pass
        self.run_btn.config(state=state); self.run_exp_btn.config(state=state)

    def on_canvas_resize(self, event): self._redraw_canvas()
    def draw_grid(self): self._redraw_canvas()
    def _redraw_canvas(self):
        self.canvas.delete("all")
        if self.canvas.winfo_width() <= 1 or self.canvas.winfo_height() <= 1: return
        self.cell_size = float(min(self.canvas.winfo_width() / self.grid_cols, self.canvas.winfo_height() / self.grid_rows))
        self.grid_draw_width, self.grid_draw_height = self.grid_cols * self.cell_size, self.grid_rows * self.cell_size
        self.offset_x, self.offset_y = (self.canvas.winfo_width() - self.grid_draw_width) / 2, (self.canvas.winfo_height() - self.grid_draw_height) / 2
        cs = self.cell_size
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cell, color = self.grid_data[r][c], self.get_color(self.grid_data[r][c])
                x0, y0, x1, y1 = self.offset_x + c * cs, self.offset_y + r * cs, self.offset_x + (c+1) * cs, self.offset_y + (r+1) * cs
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='#3b4048')
                if cell == CellType.DELIVERY:
                    delivery_idx = self.delivery_points.index((r, c)) + 1 if (r, c) in self.delivery_points else '?'
                    font_size = max(6, int(cs * 0.35))
                    self.canvas.create_text(x0 + cs/2, y0 + cs/2, text=str(delivery_idx), fill='white', font=("Arial", font_size, 'bold'))

    def get_color(self, cell_type: int) -> str:
        colors = {CellType.EMPTY: '#1c2333', CellType.WALL: '#22252a', CellType.START: '#98c379', CellType.END: '#e06c75', CellType.PATH: '#61afef', CellType.DELIVERY: '#f39c12', CellType.HIGHWAY: '#4b5263', CellType.AIRPORT: '#56b6c2', CellType.RAIL_TERMINAL: '#d19a66'}
        return colors.get(cell_type, '#000000')

    def update_delivery_info(self): self.delivery_info.config(text=f"Delivery Points: {len(self.delivery_points)}")
    def _get_cell_from_event(self, event: tk.Event) -> Optional[Tuple[int, int]]:
        if self.cell_size == 0: return None
        col, row = int((event.x - self.offset_x) / self.cell_size), int((event.y - self.offset_y) / self.cell_size)
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols: return row, col
        return None

    # Handles left-clicks on the canvas for drawing.
    def left_click(self, event: tk.Event):
        if self.is_running: return
        cell = self._get_cell_from_event(event)
        if not cell: return
        row, col = cell
        mode, cell_type = self.mode_var.get(), self.grid_data[row][col]
        allowed_placement = [CellType.EMPTY, CellType.HIGHWAY, CellType.AIRPORT, CellType.RAIL_TERMINAL]
        if mode == "wall": self.grid_data[row][col] = CellType.EMPTY if cell_type == CellType.WALL else CellType.WALL
        elif mode == "highway": self.grid_data[row][col] = CellType.EMPTY if cell_type == CellType.HIGHWAY else CellType.HIGHWAY
        elif mode == "airport":
            if cell_type == CellType.AIRPORT: self.grid_data[row][col] = CellType.EMPTY; self.airports.remove((row,col))
            elif self.grid_data[row][col] in [CellType.EMPTY, CellType.HIGHWAY]: self.grid_data[row][col] = CellType.AIRPORT; self.airports.append((row,col))
        elif mode == "rail_terminal":
            if cell_type == CellType.RAIL_TERMINAL: self.grid_data[row][col] = CellType.EMPTY; self.rail_terminals.remove((row,col))
            elif self.grid_data[row][col] in [CellType.EMPTY, CellType.HIGHWAY]: self.grid_data[row][col] = CellType.RAIL_TERMINAL; self.rail_terminals.append((row,col))
        elif mode == "start":
            if self.start: self.grid_data[self.start[0]][self.start[1]] = CellType.EMPTY
            self.start = (row, col) if self.grid_data[row][col] in allowed_placement else None
            if self.start: self.grid_data[row][col] = CellType.START
        elif mode == "end":
            if self.end: self.grid_data[self.end[0]][self.end[1]] = CellType.EMPTY
            self.end = (row, col) if self.grid_data[row][col] in allowed_placement else None
            if self.end: self.grid_data[row][col] = CellType.END
        elif mode == "delivery":
            if cell_type == CellType.DELIVERY: self.delivery_points.remove((row, col)); self.grid_data[row][col] = CellType.EMPTY
            elif self.grid_data[row][col] in allowed_placement and len(self.delivery_points) < 10: self.delivery_points.append((row, col)); self.grid_data[row][col] = CellType.DELIVERY
            self.update_delivery_info()
        self.clear_path(redraw=True)

    # Handles right-clicks for erasing.
    def right_click(self, event: tk.Event):
        if self.is_running: return
        cell = self._get_cell_from_event(event)
        if not cell: return
        row, col = cell
        if (row, col) == self.start: self.start = None
        elif (row, col) == self.end: self.end = None
        elif self.grid_data[row][col] == CellType.DELIVERY: self.delivery_points.remove((row, col)); self.update_delivery_info()
        elif self.grid_data[row][col] == CellType.AIRPORT: self.airports.remove((row,col))
        elif self.grid_data[row][col] == CellType.RAIL_TERMINAL: self.rail_terminals.remove((row,col))
        self.grid_data[row][col] = CellType.EMPTY
        self.clear_path(redraw=True)

    def stop_algorithm(self): self.is_running = False
    def clear_path(self, redraw: bool = True):
        self.canvas.delete("path_line")
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if self.grid_data[r][c] == CellType.PATH: self.grid_data[r][c] = CellType.EMPTY
        if redraw: self.draw_grid()

    def reset_for_new_run(self): self.clear_path(redraw=False); self.path_cache = {}
    def clear_all(self):
        self.grid_data = [[CellType.EMPTY for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        self.start = self.end = None
        self.delivery_points = []; self.airports = []; self.rail_terminals = []
        self.update_delivery_info()
        self.reset_for_new_run(); self.draw_grid()

    def neighbors(self, node: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        COST_TRUCK = 1.0
        COST_HIGHWAY = self.cost_highway_var.get()
        COST_RAIL = self.cost_rail_var.get()
        COST_AIR = self.cost_air_var.get()
        TRANSFER_PENALTY = self.cost_transfer_var.get()
        
        r, c = node
        result: List[Tuple[Tuple[int, int], float]] = []
        current_cell_type = self.grid_data[r][c]
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and self.grid_data[nr][nc] != CellType.WALL:
                neighbor_cell_type = self.grid_data[nr][nc]
                cost = COST_HIGHWAY if neighbor_cell_type == CellType.HIGHWAY else COST_TRUCK
                if neighbor_cell_type in [CellType.AIRPORT, CellType.RAIL_TERMINAL] and current_cell_type not in [CellType.AIRPORT, CellType.RAIL_TERMINAL]:
                    cost += TRANSFER_PENALTY
                result.append(((nr, nc), cost))
                
        if current_cell_type == CellType.AIRPORT:
            for other_hub in self.airports:
                if other_hub != node:
                    dist = self.heuristic(node, other_hub)
                    cost = (dist * COST_AIR) + TRANSFER_PENALTY
                    result.append((other_hub, cost))
        if current_cell_type == CellType.RAIL_TERMINAL:
            for other_hub in self.rail_terminals:
                if other_hub != node:
                    dist = self.heuristic(node, other_hub)
                    cost = (dist * COST_RAIL) + TRANSFER_PENALTY
                    result.append((other_hub, cost))
        return result

    def heuristic(self, a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_shortest_path_a_star(self, start_node, end_node) -> Optional[Tuple[List[Tuple[int, int]], float]]:
        cache_key = tuple(sorted((start_node, end_node)))
        if cache_key in self.path_cache: return self.path_cache[cache_key]
        
        open_set = [(0, start_node)]
        came_from = {}
        g_score = { (r,c): float('inf') for r in range(self.grid_rows) for c in range(self.grid_cols) }
        g_score[start_node] = 0
        f_score = { (r,c): float('inf') for r in range(self.grid_rows) for c in range(self.grid_cols) }
        f_score[start_node] = self.heuristic(start_node, end_node)
        
        open_set_hash = {start_node}
        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            if current == end_node:
                path = [current]; temp = current
                while temp in came_from: temp = came_from[temp]; path.append(temp)
                path.reverse(); path_cost = g_score[end_node]
                self.path_cache[cache_key] = (path, path_cost)
                return path, path_cost
            
            for neighbor, cost in self.neighbors(current):
                tentative_g_score = g_score[current] + cost
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end_node)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        self.path_cache[cache_key] = None; return None

    def _precompute_distance_matrix(self, progress_var=None) -> Optional[Tuple[np.ndarray, Dict, Dict]]:
        if not self.start or not self.end: return None
        points = [self.start] + self.delivery_points + [self.end]; num_points = len(points)
        
        if progress_var:
            max_val = (num_points * (num_points - 1)) / 2
            self.after(0, progress_var.config, {'value': 0, 'maximum': max_val if max_val > 0 else 1})

        idx_to_point, point_to_idx = {i: p for i, p in enumerate(points)}, {p: i for i, p in enumerate(points)}
        dist_matrix = np.full((num_points, num_points), float('inf'))
        
        progress_counter = 0
        for i in range(num_points):
            for j in range(i, num_points):
                if i == j: dist_matrix[i, j] = 0; continue
                path_data = self.find_shortest_path_a_star(idx_to_point[i], idx_to_point[j])
                if path_data: _, cost = path_data; dist_matrix[i, j] = dist_matrix[j, i] = cost
                
                if progress_var:
                    progress_counter += 1
                    self.after(0, progress_var.config, {'value': progress_counter})
        
        if np.any(dist_matrix == float('inf')): messagebox.showwarning("Graph Error", "Not all points are reachable. Cannot solve TSP."); return None
        return dist_matrix, idx_to_point, point_to_idx

    # --- Core Algorithm Engines (MODIFIED) ---
    # These functions now return the initial best score, the final best score, and the final best tour.

    def _aco_engine(self, dist_matrix, delivery_indices, start_idx, end_idx, params, iteration_callback=None):
        """Core engine for the Ant Colony Optimization algorithm."""
        num_ants, num_iter, alpha, beta, decay = params['pop'], params['iter'], params['alpha'], params['beta'], params['decay']
        phero_matrix = np.ones(dist_matrix.shape)
        g_best_tour = None
        g_best_score = float('inf')
        initial_g_best_score = float('inf') # Will store the result from the first iteration

        for i in range(num_iter):
            if not self.is_running: break
            all_ant_tours = []
            for _ in range(num_ants):
                rem = list(delivery_indices); random.shuffle(rem)
                tour, city = [], start_idx
                while rem:
                    probs = [(phero_matrix[city, next_c]**alpha) * ((1.0/(dist_matrix[city, next_c]+1e-9))**beta) for next_c in rem]
                    total_prob = sum(probs)
                    chosen = random.choice(rem) if total_prob == 0 else np.random.choice(rem, p=np.array(probs)/total_prob)
                    tour.append(chosen); rem.remove(chosen); city = chosen
                full_tour = [start_idx] + tour + [end_idx]
                tour_len = sum(dist_matrix[full_tour[k], full_tour[k+1]] for k in range(len(full_tour)-1))
                all_ant_tours.append((full_tour, tour_len))

            phero_matrix *= (1.0 - decay)
            for tour, length in all_ant_tours:
                if length > 0:
                    for k in range(len(tour) - 1):
                        phero_matrix[tour[k], tour[k+1]] += 100 / length
                        phero_matrix[tour[k+1], tour[k]] += 100 / length

            best_tour, best_len = min(all_ant_tours, key=lambda x: x[1])
            
            # Capture the best score from the very first iteration
            if i == 0:
                initial_g_best_score = best_len

            if best_len < g_best_score:
                g_best_score, g_best_tour = best_len, best_tour

            if iteration_callback:
                iteration_callback(i + 1, g_best_score, g_best_tour)
        
        return initial_g_best_score, g_best_score, g_best_tour

    def _pso_engine(self, dist_matrix, delivery_indices, start_idx, end_idx, params, iteration_callback=None):
        """Core engine for the Discrete Particle Swarm Optimization algorithm."""
        swarm_size, num_iter, w, c1, c2 = params['pop'], params['iter'], params['w'], params['c1'], params['c2']

        def fitness(tour_perm):
            full_tour = [start_idx] + tour_perm + [end_idx]
            return sum(dist_matrix[full_tour[i], full_tour[i+1]] for i in range(len(full_tour)-1))
        def get_swaps(current, best):
            v, temp = [], list(current)
            for i in range(len(temp)):
                if temp[i] != best[i]:
                    j = temp.index(best[i]); v.append((i, j)); temp[i], temp[j] = temp[j], temp[i]
            return v
        def apply_swaps(pos, vel):
            new_pos = list(pos); 
            for i, j in vel: new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
            return new_pos

        swarm = [{'pos': random.sample(delivery_indices, len(delivery_indices)), 'vel': []} for _ in range(swarm_size)]
        for p in swarm: p.update({'p_best_pos': p['pos'], 'p_best_score': fitness(p['pos'])})
        
        g_best_pos = min(swarm, key=lambda p: p['p_best_score'])['p_best_pos']
        g_best_score = fitness(g_best_pos)
        
        # Capture the initial best score from the random starting swarm
        initial_g_best_score = g_best_score
        
        for i in range(num_iter):
            if not self.is_running: break
            for p in swarm:
                inertia_v = [s for s in p['vel'] if random.random() < w]
                cognitive_v = [s for s in get_swaps(p['pos'], p['p_best_pos']) if random.random() < c1]
                social_v = [s for s in get_swaps(p['pos'], g_best_pos) if random.random() < c2]
                p['vel'] = inertia_v + cognitive_v + social_v
                p['pos'] = apply_swaps(p['pos'], p['vel'])
                current_score = fitness(p['pos'])
                if current_score < p['p_best_score']: p['p_best_score'], p['p_best_pos'] = current_score, p['pos']
                if current_score < g_best_score: g_best_score, g_best_pos = current_score, p['pos']
            
            if iteration_callback:
                iteration_callback(i + 1, g_best_score, [start_idx] + g_best_pos + [end_idx])

        return initial_g_best_score, g_best_score, [start_idx] + g_best_pos + [end_idx]

    def _abc_engine(self, dist_matrix, delivery_indices, start_idx, end_idx, params, iteration_callback=None):
        """Core engine for the Artificial Bee Colony algorithm."""
        colony_size, num_iter, limit = params['pop'], params['iter'], params['limit']
        num_food_sources = colony_size // 2

        def _calculate_fitness(cost): return 1 / (1 + cost) if cost >= 0 else 1 + abs(cost)
        def _tour_cost(tour_perm):
            tour = [start_idx] + tour_perm + [end_idx]
            return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1))
        def _create_neighbor(tour):
            neighbor = list(tour); i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            return neighbor

        foods = [{'tour': random.sample(delivery_indices, len(delivery_indices))} for _ in range(num_food_sources)]
        for f in foods: f.update({'cost': _tour_cost(f['tour']), 'trials': 0})
        
        g_best_food = min(foods, key=lambda x: x['cost'])
        g_best_score = g_best_food['cost']
        g_best_tour = g_best_food['tour']

        # Capture the initial best score from the random starting food sources
        initial_g_best_score = g_best_score
        
        for i in range(num_iter):
            if not self.is_running: break
            # Employed bees
            for j in range(num_food_sources):
                neighbor_tour = _create_neighbor(foods[j]['tour']); neighbor_cost = _tour_cost(neighbor_tour)
                if neighbor_cost < foods[j]['cost']: foods[j] = {'tour': neighbor_tour, 'cost': neighbor_cost, 'trials': 0}
                else: foods[j]['trials'] += 1
            # Onlooker bees
            fitnesses = np.array([_calculate_fitness(f['cost']) for f in foods]); total_fitness = np.sum(fitnesses)
            probs = fitnesses / total_fitness if total_fitness > 0 else np.ones(num_food_sources) / num_food_sources
            for _ in range(num_food_sources):
                chosen_idx = np.random.choice(num_food_sources, p=probs); neighbor_tour = _create_neighbor(foods[chosen_idx]['tour']); neighbor_cost = _tour_cost(neighbor_tour)
                if neighbor_cost < foods[chosen_idx]['cost']: foods[chosen_idx] = {'tour': neighbor_tour, 'cost': neighbor_cost, 'trials': 0}
                else: foods[chosen_idx]['trials'] += 1
            # Scout bees
            for j in range(num_food_sources):
                if foods[j]['trials'] > limit:
                    new_tour = random.sample(delivery_indices, len(delivery_indices))
                    foods[j] = {'tour': new_tour, 'cost': _tour_cost(new_tour), 'trials': 0}
            
            current_best_food = min(foods, key=lambda x: x['cost'])
            if current_best_food['cost'] < g_best_score:
                g_best_score, g_best_tour = current_best_food['cost'], current_best_food['tour']
            
            if iteration_callback:
                iteration_callback(i + 1, g_best_score, [start_idx] + g_best_tour + [end_idx])

        return initial_g_best_score, g_best_score, [start_idx] + g_best_tour + [end_idx]

    def run_aco_visual(self):
        """Runs the ACO algorithm with full visualization."""
        precomputation_result = self._precompute_distance_matrix()
        if not precomputation_result: return
        dist_matrix, idx_to_point, point_to_idx = precomputation_result
        delivery_indices, start_idx, end_idx = [point_to_idx[p] for p in self.delivery_points], point_to_idx[self.start], point_to_idx[self.end]
        
        params = {'iter': self.num_iter_var.get(), 'pop': self.population_var.get(), 'alpha': self.aco_alpha.get(), 'beta': self.aco_beta.get(), 'decay': self.aco_decay.get()}
        self.best_path_len_algo = float('inf')
        self.is_running = True
        self.after(0, self._set_ui_state, 'disabled')
        self.stop_btn.config(state='normal')
        self.progress['maximum'] = params['iter']
        
        def visual_callback(iteration, best_cost, best_tour):
            self.iteration_label.config(text=f"Iteration: {iteration}/{params['iter']}")
            self.progress['value'] = iteration
            if best_cost < self.best_path_len_algo:
                self.best_path_len_algo = best_cost
                self.best_path_label.config(text=f"Best Path Cost: {best_cost:.2f}")
            if best_tour:
                self._visualize_tour(best_tour, idx_to_point)
            self.update()
            time.sleep(0.05)
        
        self._aco_engine(dist_matrix, delivery_indices, start_idx, end_idx, params, visual_callback)
        
        self.stop_algorithm()
        self.after(0, self._set_ui_state, 'normal')
        self.iteration_label.config(text="Status: Finished")

    def run_pso_visual(self):
        """Runs the PSO algorithm with full visualization."""
        precomputation_result = self._precompute_distance_matrix()
        if not precomputation_result: return
        dist_matrix, idx_to_point, point_to_idx = precomputation_result
        delivery_indices, start_idx, end_idx = [point_to_idx[p] for p in self.delivery_points], point_to_idx[self.start], point_to_idx[self.end]

        params = {'iter': self.num_iter_var.get(), 'pop': self.population_var.get(), 'w': self.pso_w.get(), 'c1': self.pso_c1.get(), 'c2': self.pso_c2.get()}
        self.best_path_len_algo = float('inf')
        self.is_running = True
        self.after(0, self._set_ui_state, 'disabled')
        self.stop_btn.config(state='normal')
        self.progress['maximum'] = params['iter']
        
        def visual_callback(iteration, best_cost, best_tour):
            self.iteration_label.config(text=f"Iteration: {iteration}/{params['iter']}")
            self.progress['value'] = iteration
            if best_cost < self.best_path_len_algo:
                self.best_path_len_algo = best_cost
                self.best_path_label.config(text=f"Best Path Cost: {best_cost:.2f}")
            if best_tour:
                self._visualize_tour(best_tour, idx_to_point)
            self.update()
            time.sleep(0.05)

        self._pso_engine(dist_matrix, delivery_indices, start_idx, end_idx, params, visual_callback)
        
        self.stop_algorithm()
        self.after(0, self._set_ui_state, 'normal')
        self.iteration_label.config(text="Status: Finished")

    def run_abc_visual(self):
        """Runs the ABC algorithm with full visualization."""
        precomputation_result = self._precompute_distance_matrix()
        if not precomputation_result: return
        dist_matrix, idx_to_point, point_to_idx = precomputation_result
        delivery_indices, start_idx, end_idx = [point_to_idx[p] for p in self.delivery_points], point_to_idx[self.start], point_to_idx[self.end]
        
        params = {'iter': self.num_iter_var.get(), 'pop': self.population_var.get(), 'limit': self.abc_limit_var.get()}
        self.best_path_len_algo = float('inf')
        self.is_running = True
        self.after(0, self._set_ui_state, 'disabled')
        self.stop_btn.config(state='normal')
        self.progress['maximum'] = params['iter']
        
        def visual_callback(iteration, best_cost, best_tour):
            self.iteration_label.config(text=f"Iteration: {iteration}/{params['iter']}")
            self.progress['value'] = iteration
            if best_cost < self.best_path_len_algo:
                self.best_path_len_algo = best_cost
                self.best_path_label.config(text=f"Best Path Cost: {best_cost:.2f}")
            if best_tour:
                self._visualize_tour(best_tour, idx_to_point)
            self.update()
            time.sleep(0.05)
        
        self._abc_engine(dist_matrix, delivery_indices, start_idx, end_idx, params, visual_callback)

        self.stop_algorithm()
        self.after(0, self._set_ui_state, 'normal')
        self.iteration_label.config(text="Status: Finished")


    def run_pathfinding(self) -> None:
        self.reset_for_new_run()
        self.best_path_len_algo = float('inf') 
        self.iteration_label.config(text="Status: Preparing..."); self.best_path_label.config(text="Best Path Cost: N/A")
        self.progress['value'] = 0; self.update()
        if not (self.start and self.end and self.delivery_points):
            messagebox.showwarning("Input Error", "Start, end, and at least one delivery point are required.")
            return
        algo = self.algo_var.get()
        if algo == Algorithm.ACO: target_func = self.run_aco_visual
        elif algo == Algorithm.PSO: target_func = self.run_pso_visual
        elif algo == Algorithm.ABC: target_func = self.run_abc_visual
        else: return
        threading.Thread(target=target_func, daemon=True).start()

    def run_experiment(self):
        if not (self.start and self.end and self.delivery_points): messagebox.showwarning("Input Error", "Cannot run experiment. Set start, end, and delivery points first."); return
        num_runs = simpledialog.askinteger("Run Experiment", "How many runs per algorithm?", parent=self, minvalue=1, maxvalue=100)
        if not num_runs: return
        threading.Thread(target=self._experiment_worker, args=(num_runs,), daemon=True).start()

    def _experiment_worker(self, num_runs: int):
        self.is_running = True
        self.after(0, self._set_ui_state, 'disabled')
        self.after(0, self.iteration_label.config, {'text': "Status: Pre-computing distances..."})

        precomputation_result = self._precompute_distance_matrix(self.progress)
        
        if not precomputation_result:
            self.after(0, lambda: [self.iteration_label.config(text="Status: Idle"), self._set_ui_state('normal'), self.progress.config(value=0)])
            return
            
        dist_matrix, idx_to_point, point_to_idx = precomputation_result
        delivery_indices, start_idx, end_idx = [point_to_idx[p] for p in self.delivery_points], point_to_idx[self.start], point_to_idx[self.end]
        results = []
        total_algos = 3
        
        self.after(0, self.progress.config, {'maximum': num_runs * total_algos, 'value': 0})
        
        algorithms_to_run = [
            ('ACO', self._aco_engine, {'iter': self.num_iter_var.get(), 'pop': self.population_var.get(), 'alpha': self.aco_alpha.get(), 'beta': self.aco_beta.get(), 'decay': self.aco_decay.get()}),
            ('PSO', self._pso_engine, {'iter': self.num_iter_var.get(), 'pop': self.population_var.get(), 'w': self.pso_w.get(), 'c1': self.pso_c1.get(), 'c2': self.pso_c2.get()}),
            ('ABC', self._abc_engine, {'iter': self.num_iter_var.get(), 'pop': self.population_var.get(), 'limit': self.abc_limit_var.get()})
        ]
        
        for algo_idx, (algo_name, engine_func, params) in enumerate(algorithms_to_run):
            for i in range(num_runs):
                if not self.is_running:
                    self.after(0, lambda: [self.iteration_label.config(text="Status: Stopped"), self._set_ui_state('normal')])
                    return
                
                progress_value = (algo_idx * num_runs) + i + 1
                status_text = f"Running {algo_name}: {i+1}/{num_runs}"
                self.after(0, self.iteration_label.config, {'text': status_text})
                self.after(0, self.progress.config, {'value': progress_value})
                
                start_time = time.perf_counter()
                
                # The engine now returns initial and final costs
                initial_cost, final_cost, tour = engine_func(dist_matrix, delivery_indices, start_idx, end_idx, params, iteration_callback=None)
                
                end_time = time.perf_counter()

                # Record both costs in the results dictionary
                results.append({
                    'Algorithm': algo_name,
                    'Run': i + 1,
                    'Initial_Cost': f"{initial_cost:.2f}",
                    'Final_Best_Cost': f"{final_cost:.2f}",
                    'Time_Seconds': f"{end_time - start_time:.4f}"
                })
        
        self.after(0, self.iteration_label.config, {'text': "Status: Saving results..."})
        self.after(0, self._save_results_to_csv, results)
        self.is_running = False
        self.after(0, lambda: [self.iteration_label.config(text="Status: Experiment Finished"), self._set_ui_state('normal')])

    def _save_results_to_csv(self, results: List[Dict]):
        if not results: return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")], title="Save Experiment Results")
        if not filepath: return
        try:
            with open(filepath, 'w', newline='') as csvfile:
                # DictWriter automatically gets headers from the dictionary keys
                writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            messagebox.showinfo("Success", f"Results successfully saved to\n{filepath}")
        except IOError as e: messagebox.showerror("Save Error", f"Could not save file: {e}")

    def _visualize_tour(self, tour_indices, idx_to_point):
        self.clear_path(redraw=True)
        waypoints_coords = [idx_to_point[idx] for idx in tour_indices]
        
        for j in range(len(waypoints_coords) - 1):
            start_node = waypoints_coords[j]
            end_node = waypoints_coords[j+1]
            start_type = self.grid_data[start_node[0]][start_node[1]]
            end_type = self.grid_data[end_node[0]][end_node[1]]

            is_hub_jump = (start_type == end_type and start_type in [CellType.AIRPORT, CellType.RAIL_TERMINAL])

            if is_hub_jump:
                cs = self.cell_size
                x0 = self.offset_x + start_node[1] * cs + cs / 2
                y0 = self.offset_y + start_node[0] * cs + cs / 2
                x1 = self.offset_x + end_node[1] * cs + cs / 2
                y1 = self.offset_y + end_node[0] * cs + cs / 2
                self.canvas.create_line(x0, y0, x1, y1, fill='#61afef', width=2, dash=(4, 4), tags="path_line")
            else:
                path_data = self.find_shortest_path_a_star(start_node, end_node)
                if path_data:
                    segment, _ = path_data
                    for r_idx, c_idx in segment:
                         if self.grid_data[r_idx][c_idx] not in [CellType.START, CellType.END, CellType.DELIVERY, CellType.AIRPORT, CellType.RAIL_TERMINAL]:
                            cs = self.cell_size
                            x0, y0 = self.offset_x + c_idx * cs, self.offset_y + r_idx * cs
                            self.canvas.create_rectangle(x0, y0, x0 + cs, y0 + cs, fill=self.get_color(CellType.PATH), outline='#3b4048', tags="path_line")

if __name__ == '__main__':
    app = PathfindingUI()
    app.mainloop()
