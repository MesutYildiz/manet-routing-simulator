import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import math
import threading
import time
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging

from simulator import MANETSimulator
from models import MobilityParameters

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MANETSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced MANET Routing Simulator - Mobility Enabled")
        
        # Get screen dimensions and set responsive window size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size to 90% of screen size, with minimum and maximum limits
        window_width = min(max(int(screen_width * 0.9), 1200), 1600)
        window_height = min(max(int(screen_height * 0.9), 700), 1000)
        
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(1000, 600)  # Minimum size
        
        self.simulator = MANETSimulator()
        self.selected_nodes = []
        self.animation_active = False
        self.simulation_thread = None
        
        self.setup_gui()
        self.create_sample_network()
        self.start_simulation_loop()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls (more compact with scrollbar)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", width=280)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_frame.pack_propagate(False)
        
        # Create scrollable frame for controls
        control_canvas = tk.Canvas(control_frame, width=260)
        control_scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=control_canvas.yview)
        scrollable_frame = ttk.Frame(control_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        
        control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Network management
        ttk.Label(scrollable_frame, text="Network Management:", font=("Arial", 9, "bold")).pack(pady=(10, 3))
        ttk.Button(scrollable_frame, text="Random Node", command=self.add_random_node, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Remove Selected", command=self.remove_selected_node, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Clear", command=self.clear_network, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Sample Network", command=self.create_sample_network, width=18).pack(pady=1)
        
        # MOBILITY CONTROLS - NEW SECTION
        ttk.Label(scrollable_frame, text="Mobility:", font=("Arial", 9, "bold")).pack(pady=(15, 3))
        
        # Mobility enable/disable
        self.mobility_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(scrollable_frame, text="Mobility Active", 
                       variable=self.mobility_enabled_var, 
                       command=self.toggle_mobility).pack(pady=1)
        
        # Mobility model selection
        ttk.Label(scrollable_frame, text="Model:").pack(pady=(5, 1))
        self.mobility_model_var = tk.StringVar(value="random_waypoint")
        mobility_models = ["random_waypoint", "random_walk", "group_mobility", "highway"]
        
        for model in mobility_models:
            ttk.Radiobutton(scrollable_frame, text=model.replace("_", " ")[:12], 
                           variable=self.mobility_model_var, value=model,
                           command=self.change_mobility_model).pack(anchor="w", padx=10, pady=0)
        
        # Speed controls
        speed_frame = ttk.LabelFrame(scrollable_frame, text="Speed")
        speed_frame.pack(fill=tk.X, pady=3, padx=5)
        
        ttk.Label(speed_frame, text="Min (m/s):").pack()
        self.min_speed_var = tk.StringVar(value="1.0")
        ttk.Entry(speed_frame, textvariable=self.min_speed_var, width=15).pack(pady=1)
        
        ttk.Label(speed_frame, text="Max (m/s):").pack()
        self.max_speed_var = tk.StringVar(value="5.0")
        ttk.Entry(speed_frame, textvariable=self.max_speed_var, width=15).pack(pady=1)
        
        ttk.Label(speed_frame, text="Pause (s):").pack()
        self.pause_time_var = tk.StringVar(value="2.0")
        ttk.Entry(speed_frame, textvariable=self.pause_time_var, width=15).pack(pady=1)
        
        ttk.Button(speed_frame, text="Apply", 
                  command=self.apply_mobility_params, width=15).pack(pady=2)
        
        # Protocol selection
        ttk.Label(scrollable_frame, text="Protocol:", font=("Arial", 9, "bold")).pack(pady=(15, 3))
        self.protocol_var = tk.StringVar(value="AODV")
        protocol_frame = ttk.Frame(scrollable_frame)
        protocol_frame.pack(pady=2)
        
        protocols = ["AODV", "OLSR", "DSR", "FLOODING", "ZRP"]
        for protocol in protocols:
            ttk.Radiobutton(protocol_frame, text=protocol, variable=self.protocol_var, 
                           value=protocol, command=self.change_protocol).pack(anchor="w")
        
        # Packet sending
        ttk.Label(scrollable_frame, text="Packet:", font=("Arial", 9, "bold")).pack(pady=(15, 3))
        self.source_var = tk.StringVar()
        self.dest_var = tk.StringVar()
        
        ttk.Label(scrollable_frame, text="Source:").pack(pady=0)
        source_combo = ttk.Combobox(scrollable_frame, textvariable=self.source_var, width=18)
        source_combo.pack(pady=1)
        
        ttk.Label(scrollable_frame, text="Destination:").pack(pady=0)
        dest_combo = ttk.Combobox(scrollable_frame, textvariable=self.dest_var, width=18)
        dest_combo.pack(pady=1)
        
        ttk.Label(scrollable_frame, text="Count:").pack(pady=0)
        self.packet_count_var = tk.StringVar(value="1")
        ttk.Entry(scrollable_frame, textvariable=self.packet_count_var, width=18).pack(pady=1)
        
        self.source_combo = source_combo
        self.dest_combo = dest_combo
        
        ttk.Button(scrollable_frame, text="Send", command=self.send_packet, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Batch Send", command=self.send_multiple_packets, width=18).pack(pady=1)
        
        # Discrete-Event Simulation Controls
        ttk.Label(scrollable_frame, text="Discrete-Event:", font=("Arial", 9, "bold")).pack(pady=(15, 3))
        
        ttk.Button(scrollable_frame, text="Start", command=self.start_discrete_simulation, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Stop", command=self.stop_discrete_simulation, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Step-by-Step", command=self.step_simulation, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Clear Events", command=self.clear_events, width=18).pack(pady=1)
        
        # Simulation duration control
        ttk.Label(scrollable_frame, text="Duration (s):").pack(pady=(5, 0))
        self.simulation_duration_var = tk.StringVar(value="10.0")
        duration_entry = ttk.Entry(scrollable_frame, textvariable=self.simulation_duration_var, width=18)
        duration_entry.pack(pady=1)
        
        # Performance testing
        ttk.Label(scrollable_frame, text="Test:", font=("Arial", 9, "bold")).pack(pady=(15, 3))
        ttk.Button(scrollable_frame, text="Compare Protocols", command=self.run_protocol_comparison, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Reset Metrics", command=self.reset_metrics, width=18).pack(pady=1)
        
        # Right panel - Simulation and stats
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Top: Simulation area
        sim_frame = ttk.LabelFrame(right_frame, text="Simulation Area")
        sim_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Canvas with adaptive sizing
        canvas_frame = ttk.Frame(sim_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack canvas and scrollbars
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Configure canvas scrolling region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Animation controls
        anim_frame = ttk.Frame(sim_frame)
        anim_frame.pack(pady=5)
        
        ttk.Button(anim_frame, text="Last Packet Route", command=self.animate_last_packet).pack(side=tk.LEFT, padx=5)
        ttk.Button(anim_frame, text="Refresh Network", command=self.redraw_network).pack(side=tk.LEFT, padx=5)
        ttk.Button(anim_frame, text="Show Topology", command=self.show_topology_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(anim_frame, text="Mobility Paths", command=self.show_mobility_paths).pack(side=tk.LEFT, padx=5)
        
        # Bottom: Statistics
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics and Performance Metrics")
        stats_frame.pack(fill=tk.BOTH, expand=False)
        stats_frame.configure(height=250)
        
        # Notebook for different stats views
        self.stats_notebook = ttk.Notebook(stats_frame)
        self.stats_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current stats tab
        current_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(current_tab, text="Current Statistics")
        
        self.stats_text = tk.Text(current_tab, height=8, width=80, font=("Courier", 10))
        scrollbar1 = ttk.Scrollbar(current_tab, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar1.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Protocol comparison tab
        comparison_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(comparison_tab, text="Protocol Comparison")
        
        self.comparison_text = tk.Text(comparison_tab, height=8, width=80, font=("Courier", 10))
        scrollbar2 = ttk.Scrollbar(comparison_tab, orient="vertical", command=self.comparison_text.yview)
        self.comparison_text.configure(yscrollcommand=scrollbar2.set)
        self.comparison_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mobility stats tab - NEW
        mobility_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(mobility_tab, text="Mobility Statistics")
        
        self.mobility_text = tk.Text(mobility_tab, height=8, width=80, font=("Courier", 10))
        scrollbar3 = ttk.Scrollbar(mobility_tab, orient="vertical", command=self.mobility_text.yview)
        self.mobility_text.configure(yscrollcommand=scrollbar3.set)
        self.mobility_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar3.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Node details tab
        details_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(details_tab, text="Node Details")
        
        self.details_text = tk.Text(details_tab, height=8, width=80, font=("Courier", 10))
        scrollbar4 = ttk.Scrollbar(details_tab, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=scrollbar4.set)
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar4.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Update button
        ttk.Button(stats_frame, text="Update Statistics", command=self.update_all_stats).pack(pady=5)
    
    # MOBILITY CONTROL METHODS - NEW
    def toggle_mobility(self):
        """Enable/disable mobility"""
        enabled = self.mobility_enabled_var.get()
        self.simulator.enable_mobility(enabled, self.get_mobility_params())
        
        if enabled:
            messagebox.showinfo("Mobility", "Mobility system activated!")
        else:
            messagebox.showinfo("Mobility", "Mobility system deactivated!")
    
    def change_mobility_model(self):
        """Change mobility model"""
        if self.mobility_enabled_var.get():
            params = self.get_mobility_params()
            self.simulator.enable_mobility(True, params)
    
    def get_mobility_params(self):
        """Get current mobility parameters from GUI"""
        try:
            return MobilityParameters(
                model_type=self.mobility_model_var.get(),
                min_speed=float(self.min_speed_var.get()),
                max_speed=float(self.max_speed_var.get()),
                pause_time=float(self.pause_time_var.get())
            )
        except ValueError:
            messagebox.showerror("Error", "Enter valid mobility parameters!")
            return MobilityParameters()
    
    def apply_mobility_params(self):
        """Apply new mobility parameters"""
        if self.mobility_enabled_var.get():
            params = self.get_mobility_params()
            self.simulator.mobility_params = params
            
            # Re-initialize mobility for all nodes
            for node in self.simulator.nodes.values():
                if hasattr(node, 'mobility'):
                    self.simulator.mobility_model.initialize_node_mobility(node, params)
            
            messagebox.showinfo("Success", "Mobility parameters updated!")
    
    def show_mobility_paths(self):
        """Show mobility paths of nodes"""
        if not self.mobility_enabled_var.get():
            messagebox.showinfo("Info", "Mobility system is not active!")
            return
        
        self.redraw_network()
        
        # Draw paths for all nodes
        for node_id, node in self.simulator.nodes.items():
            if hasattr(node, 'mobility') and len(node.mobility['path_history']) > 1:
                path = node.mobility['path_history']
                
                # Draw path
                for i in range(1, len(path)):
                    x1, y1, _ = path[i-1]
                    x2, y2, _ = path[i]
                    
                    # Color gradient based on time (newer = darker)
                    alpha = i / len(path)
                    color_val = int(255 * (1 - alpha * 0.7))
                    color = f"#{color_val:02x}{color_val:02x}ff"
                    
                    self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, dash=(2, 2))
                
                # Draw target if exists
                if (hasattr(node, 'mobility') and node.mobility['target_x'] is not None and 
                    node.mobility['model'] in ['random_waypoint']):
                    self.canvas.create_oval(
                        node.mobility['target_x'] - 8, node.mobility['target_y'] - 8,
                        node.mobility['target_x'] + 8, node.mobility['target_y'] + 8,
                        outline="red", width=2, dash=(3, 3))
                    
                    # Draw line to target
                    self.canvas.create_line(
                        node.x, node.y, 
                        node.mobility['target_x'], node.mobility['target_y'],
                        fill="red", width=1, dash=(5, 5))
    
    def start_simulation_loop(self):
        """Start the main simulation loop"""
        self.update_simulation()
    
    def update_simulation(self):
        """Update simulation state"""
        self.simulator.update_simulation()
        
        # Redraw network periodically if mobility is enabled
        if self.mobility_enabled_var.get():
            self.redraw_network()
        
        # Schedule next update
        self.root.after(100, self.update_simulation)  # Update every 100ms
    
    def update_mobility_stats(self):
        """Update mobility statistics display"""
        if not self.mobility_enabled_var.get():
            self.mobility_text.delete(1.0, tk.END)
            self.mobility_text.insert(1.0, "Mobility system is not active.")
            return
        
        mobility_stats = self.simulator.get_mobility_stats()
        
        stats_text = f"""
MOBILITY STATISTICS
{'='*40}

General Information:
  Mobility Active: {'Yes' if self.mobility_enabled_var.get() else 'No'}
  Active Model: {self.mobility_model_var.get().replace('_', ' ').title()}
  Moving Nodes Count: {mobility_stats['nodes_moving']}
  Average Speed: {mobility_stats['average_speed']:.2f} m/s

Model Distribution:
"""
        
        for model, count in mobility_stats['mobility_models'].items():
            stats_text += f"  {model.replace('_', ' ').title()}: {count} node\n"
        
        stats_text += f"""
Movement Statistics:
  Total Distance Traveled: {mobility_stats['total_distance_traveled']:.1f} m

Mobility Parameters:
  Min Speed: {self.min_speed_var.get()} m/s
  Max Speed: {self.max_speed_var.get()} m/s  
  Pause Time: {self.pause_time_var.get()} s
"""
        
        # Add node-specific mobility info
        stats_text += "\nNode Details:\n" + "-"*30 + "\n"
        
        for node_id, node in self.simulator.nodes.items():
            if hasattr(node, 'mobility'):
                mobility = node.mobility
                current_speed = mobility['speed']
                model = mobility['model']
                
                stats_text += f"{node_id}: {model} - {current_speed:.1f} m/s"
                
                if mobility['model'] == 'random_waypoint' and mobility['target_x']:
                    distance_to_target = math.sqrt(
                        (mobility['target_x'] - node.x)**2 + 
                        (mobility['target_y'] - node.y)**2
                    )
                    stats_text += f" (To Target: {distance_to_target:.1f}m)"
                
                stats_text += "\n"
        
        self.mobility_text.delete(1.0, tk.END)
        self.mobility_text.insert(1.0, stats_text)
    
    def create_sample_network(self):
        # Create sample network with strategic positioning
        self.simulator.nodes.clear()
        self.simulator.packet_history.clear()
        
        positions = [
            (100, 100), (200, 80), (300, 120), (450, 100), (600, 90),
            (80, 200), (180, 220), (280, 200), (380, 210), (480, 190), (580, 220),
            (120, 300), (220, 320), (320, 310), (420, 300), (520, 320),
            (200, 380), (350, 370), (500, 380)
        ]
        
        for i, (x, y) in enumerate(positions):
            self.simulator.add_node(f"N{i}", x, y)
        
        self.update_node_combos()
        self.redraw_network()
        self.update_all_stats()
    
    def add_random_node(self):
        node_id = f"N{len(self.simulator.nodes)}"
        self.simulator.add_node(node_id)
        self.update_node_combos()
        self.redraw_network()
        self.update_all_stats()
    
    def remove_selected_node(self):
        if self.selected_nodes:
            node_id = self.selected_nodes[0]
            self.simulator.remove_node(node_id)
            self.selected_nodes.clear()
            self.update_node_combos()
            self.redraw_network()
            self.update_all_stats()
    
    def clear_network(self):
        self.simulator.nodes.clear()
        self.simulator.packet_history.clear()
        self.selected_nodes.clear()
        self.simulator.reset_metrics()
        self.update_node_combos()
        self.redraw_network()
        self.update_all_stats()
    
    def change_protocol(self):
        self.simulator.routing_protocol = self.protocol_var.get()
        self.update_all_stats()
    
    def update_node_combos(self):
        node_ids = list(self.simulator.nodes.keys())
        self.source_combo['values'] = node_ids
        self.dest_combo['values'] = node_ids
    
    def send_packet(self):
        """GUI'den paket gönderme"""
        source = self.source_var.get()
        dest = self.dest_var.get()
        
        if source and dest and source != dest:
            # Schedule packet
            success = self.simulator.send_packet(source, dest)
            
            # Process events until packet is delivered or dropped
            max_events = 1000  # Prevent infinite loop
            events_processed = 0
            
            while events_processed < max_events:
                if not self.simulator.discrete_simulator.step():
                    break
                events_processed += 1
            
            # Update GUI
            self.redraw_network()
            self.update_all_stats()
            
            if success:
                messagebox.showinfo("Success", f"Packet {source} → {dest} processed {events_processed} events!")
            else:
                messagebox.showerror("Error", "Packet could not be sent!")
    
    def send_multiple_packets(self):
        source = self.source_var.get()
        dest = self.dest_var.get()
        
        try:
            count = int(self.packet_count_var.get())
        except ValueError:
            messagebox.showerror("Error", "Enter a valid packet count!")
            return
        
        if source and dest and source != dest and count > 0:
            success_count = 0
            for i in range(count):
                if self.simulator.send_packet(source, dest, f"Test Data {i+1}"):
                    success_count += 1
                time.sleep(0.001)  # Small delay to vary timestamps
            
            messagebox.showinfo("Completed", 
                              f"{success_count} out of {count} packets sent successfully!")
            self.update_all_stats()
    
    def run_protocol_comparison(self):
        """Run performance comparison between all protocols"""
        if len(self.simulator.nodes) < 3:
            messagebox.showwarning("Warning", "At least 3 nodes required for comparison!")
            return
        
        # Reset metrics
        self.simulator.reset_metrics()
        
        protocols = ["AODV", "OLSR", "DSR", "FLOODING", "ZRP"]
        test_pairs = []
        
        # Generate test node pairs
        node_ids = list(self.simulator.nodes.keys())
        for i in range(min(10, len(node_ids))):
            for j in range(i+1, min(i+6, len(node_ids))):
                test_pairs.append((node_ids[i], node_ids[j]))
        
        # Test each protocol
        for protocol in protocols:
            self.simulator.routing_protocol = protocol
            self.protocol_var.set(protocol)
            
            for source, dest in test_pairs:
                self.simulator.send_packet(source, dest, f"Test-{protocol}")
                time.sleep(0.001)
        
        self.update_all_stats()
        messagebox.showinfo("Completed", "Protocol comparison completed!")
    
    def reset_metrics(self):
        self.simulator.reset_metrics()
        self.update_all_stats()
        messagebox.showinfo("Completed", "All metrics reset!")
    
    # Discrete-Event Simulation Methods
    def start_discrete_simulation(self):
        """Start discrete-event simulation"""
        try:
            duration = float(self.simulation_duration_var.get())
            if duration <= 0:
                messagebox.showerror("Error", "Simulation duration must be greater than 0!")
                return
            
            events_processed = self.simulator.start_discrete_simulation(duration)
            messagebox.showinfo("Simulation", 
                              f"Discrete-event simulation completed!\n"
                              f"Duration: {duration}s\n"
                              f"Events Processed: {events_processed}")
            self.update_all_stats()
        except ValueError:
            messagebox.showerror("Error", "Enter a valid duration value!")
    
    def stop_discrete_simulation(self):
        """Stop discrete-event simulation"""
        self.simulator.discrete_simulator.stop_simulation()
        messagebox.showinfo("Simulation", "Simulation stopped!")
    
    def step_simulation(self):
        """Step through simulation one event at a time"""
        if self.simulator.discrete_simulator.step():
            self.redraw_network()
            self.update_all_stats()
            next_event_time = self.simulator.discrete_simulator.get_next_event_time()
            if next_event_time:
                messagebox.showinfo("Simulation", 
                                  f"One event processed.\n"
                                  f"Time: {self.simulator.discrete_simulator.current_simulation_time:.2f}s\n"
                                  f"Next Event: {next_event_time:.2f}s")
            else:
                messagebox.showinfo("Simulation", 
                                  f"One event processed.\n"
                                  f"Time: {self.simulator.discrete_simulator.current_simulation_time:.2f}s\n"
                                  f"No pending events.")
        else:
            messagebox.showinfo("Simulation", "No events to process!")
    
    def clear_events(self):
        """Clear all pending events"""
        self.simulator.discrete_simulator.clear_events()
        messagebox.showinfo("Simulation", "All pending events cleared!")
    
    def show_topology_info(self):
        """Show detailed topology information"""
        info = "NETWORK TOPOLOGY INFORMATION\n" + "="*50 + "\n\n"
        
        for node_id, node in self.simulator.nodes.items():
            info += f"Node {node_id}:\n"
            info += f"  Location: ({node.x}, {node.y})\n"
            info += f"  Neighbor Count: {len(node.neighbors)}\n"
            info += f"  Neighbors: {', '.join(sorted(node.neighbors))}\n"
            
            if self.simulator.routing_protocol == "OLSR":
                info += f"  MPR Set: {', '.join(sorted(node.mpr_set))}\n"
                info += f"  Routing Table: {len(node.routing_table)} routes\n"
            elif self.simulator.routing_protocol == "AODV":
                active_routes = sum(1 for route in node.routing_table.values() 
                                  if self.simulator.discrete_simulator.current_simulation_time < route.expiry_time)
                info += f"  Active Routes: {active_routes}\n"
            
            info += "\n"
        
        # Show in a new window
        info_window = tk.Toplevel(self.root)
        info_window.title("Network Topology Information")
        info_window.geometry("600x500")
        
        text_widget = tk.Text(info_window, font=("Courier", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, info)
        text_widget.config(state=tk.DISABLED)
    
    def on_canvas_click(self, event):
        # Convert canvas coordinates to actual coordinates (account for scrolling)
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        clicked_node = None
        
        # Find clicked node
        for node_id, node in self.simulator.nodes.items():
            distance_squared = (node.x - x)**2 + (node.y - y)**2
            if distance_squared <= 15**2:  # 15 pixel radius
                clicked_node = node_id
                break
        
        if clicked_node:
            if clicked_node in self.selected_nodes:
                self.selected_nodes.remove(clicked_node)
            else:
                self.selected_nodes = [clicked_node]
            self.redraw_network()
            self.update_node_details()
            # Switch to node details tab to show the selected node info
            self.stats_notebook.select(3)  # Select the Node Details tab
    
    def redraw_network(self):
        """Enhanced network drawing with mobility visualization"""
        # Clear canvas (this also clears packet path visualizations)
        self.canvas.delete("all")
        
        # Draw connections with dynamic coloring
        for node_id, node in self.simulator.nodes.items():
            for neighbor_id in node.neighbors:
                if neighbor_id in self.simulator.nodes:
                    neighbor = self.simulator.nodes[neighbor_id]
                    # Color code by connection strength
                    distance = node.distance_to(neighbor)
                    strength = max(0, (node.transmission_range - distance) / node.transmission_range)
                    color_intensity = int(255 * (1 - strength))
                    color = f"#{color_intensity:02x}{color_intensity:02x}{color_intensity:02x}"
                    
                    self.canvas.create_line(node.x, node.y, neighbor.x, neighbor.y, 
                                          fill=color, width=1)
        
        # Draw transmission range for selected nodes
        for node_id in self.selected_nodes:
            if node_id in self.simulator.nodes:
                node = self.simulator.nodes[node_id]
                self.canvas.create_oval(node.x - node.transmission_range, 
                                      node.y - node.transmission_range,
                                      node.x + node.transmission_range, 
                                      node.y + node.transmission_range,
                                      outline="blue", width=2, dash=(5, 5))
        
        # Draw nodes with mobility-aware coloring
        for node_id, node in self.simulator.nodes.items():
            # Color based on mobility and activity
            if hasattr(node, 'mobility'):
                if node.mobility['model'] == 'random_waypoint':
                    color = "lightgreen"
                elif node.mobility['model'] == 'random_walk':
                    color = "lightblue"
                elif node.mobility['model'] == 'group_mobility':
                    color = "lightyellow"
                elif node.mobility['model'] == 'highway':
                    color = "lightcoral"
                else:
                    color = "white"
            else:
                color = "lightgray"  # Static nodes
            
            if node_id in self.selected_nodes:
                outline_color = "red"
                outline_width = 3
            else:
                outline_color = "black"
                outline_width = 2
            
            self.canvas.create_oval(node.x - 15, node.y - 15, node.x + 15, node.y + 15, 
                                  fill=color, outline=outline_color, width=outline_width)
            self.canvas.create_text(node.x, node.y, text=node_id, font=("Arial", 8, "bold"))
            
            # Show speed for mobile nodes
            if hasattr(node, 'mobility') and self.mobility_enabled_var.get():
                speed = node.mobility['speed']
                self.canvas.create_text(node.x, node.y + 25, text=f"{speed:.1f}m/s", 
                                      font=("Arial", 7), fill="darkgreen")
        
        # Update canvas scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def animate_last_packet(self):
        if not self.simulator.packet_history:
            messagebox.showinfo("Info", "No packet sent yet!")
            return
        
        last_packet = self.simulator.packet_history[-1]
        
        # Debug: Check if path exists and has nodes
        if not last_packet.path or len(last_packet.path) == 0:
            messagebox.showwarning("Warning", "Packet route information not found!")
            return
        
        # Show packet information
        info = f"Last Packet:\n"
        info += f"Source: {last_packet.source}\n"
        info += f"Destination: {last_packet.destination}\n"
        info += f"Route: {' → '.join(last_packet.path)}\n"
        info += f"Hop Count: {last_packet.hop_count}\n"
        info += f"Status: {'Delivered' if not last_packet.dropped else 'Dropped'}\n"
        
        if last_packet.delivery_time:
            info += f"Delivery Time: {last_packet.delivery_time*1000:.2f} ms\n"
        
        # First draw the static path, then animate
        self.redraw_network()
        
        # Draw the path
        if len(last_packet.path) >= 1:
            self.draw_packet_path_static(last_packet.path)
            if len(last_packet.path) >= 2:
                self.animate_packet_path(last_packet.path)
        
        # Show info after starting animation
        messagebox.showinfo("Last Packet Info", info)
    
    def draw_packet_path_static(self, path):
        """Draw the packet path as a static highlighted route"""
        if len(path) < 2:
            return
        
        for i in range(len(path) - 1):
            if path[i] in self.simulator.nodes and path[i+1] in self.simulator.nodes:
                current_node = self.simulator.nodes[path[i]]
                next_node = self.simulator.nodes[path[i + 1]]
                
                # Draw thick red line for the path
                self.canvas.create_line(
                    current_node.x, current_node.y, 
                    next_node.x, next_node.y,
                    fill="red", width=3, arrow=tk.LAST, arrowshape=(10, 12, 5),
                    tags="packet_path"
                )
        
        # Highlight source and destination nodes
        if path[0] in self.simulator.nodes:
            source = self.simulator.nodes[path[0]]
            self.canvas.create_oval(
                source.x - 20, source.y - 20,
                source.x + 20, source.y + 20,
                outline="green", width=3, tags="packet_path"
            )
            self.canvas.create_text(
                source.x, source.y - 30, 
                text="Source", fill="green", font=("Arial", 10, "bold"),
                tags="packet_path"
            )
        
        if path[-1] in self.simulator.nodes:
            dest = self.simulator.nodes[path[-1]]
            self.canvas.create_oval(
                dest.x - 20, dest.y - 20,
                dest.x + 20, dest.y + 20,
                outline="blue", width=3, tags="packet_path"
            )
            self.canvas.create_text(
                dest.x, dest.y - 30,
                text="Destination", fill="blue", font=("Arial", 10, "bold"),
                tags="packet_path"
            )
    
    def animate_packet_path(self, path):
        if len(path) < 2:
            return
        
        def animate_step(step):
            if step >= len(path) - 1:
                return
            
            if path[step] not in self.simulator.nodes or path[step+1] not in self.simulator.nodes:
                return
            
            current_node = self.simulator.nodes[path[step]]
            next_node = self.simulator.nodes[path[step + 1]]
            
            # Draw moving packet as a pulsing circle
            packet_dot = self.canvas.create_oval(
                next_node.x - 8, next_node.y - 8,
                next_node.x + 8, next_node.y + 8,
                fill="orange", outline="red", width=2,
                tags="packet_animation"
            )
            
            # Next step
            self.root.after(600, lambda: animate_step(step + 1))
            
            # Clean up animation
            self.root.after(1000, lambda: self.canvas.delete(packet_dot))
        
        animate_step(0)
    
    def update_all_stats(self):
        """Update all statistics including mobility"""
        self.update_current_stats()
        self.update_protocol_comparison()
        self.update_mobility_stats()  # NEW
        self.update_node_details()
    
    def update_current_stats(self):
        stats = self.simulator.get_network_stats()
        
        # Get discrete-event simulator statistics
        discrete_stats = self.simulator.discrete_simulator.statistics
        
        stats_text = f"""
NETWORK GENERAL STATISTICS
{'='*40}

Network Information:
  Node Count: {stats['total_nodes']}
  Connection Count: {stats['total_connections']}
  Network Connectivity: {stats['network_connectivity']:.1f}%
  Active Protocol: {stats['current_protocol']}

DISCRETE-EVENT SIMULATION
{'='*40}
  Simulation Time: {self.simulator.discrete_simulator.current_simulation_time:.2f}s
  Events Processed: {discrete_stats['events_processed']}
  Events/Second: {discrete_stats['events_per_second']:.2f}
  Collision Count: {discrete_stats['collision_count']}
  Pending Events: {len(self.simulator.discrete_simulator.event_queue)}

Packet Statistics:
  Packets Sent: {stats['packets_sent']}
  Packets Delivered: {stats['packets_delivered']}
  Packets Dropped: {stats['packets_dropped']}
  
PERFORMANCE METRICS
{'='*40}

  Packet Delivery Ratio: {stats['delivery_ratio']:.2f}%
  Average Latency: {stats['average_delay']*1000:.2f} ms
  Routing Overhead: {stats['routing_overhead']:.2f}
  Average Hop Count: {stats['average_hop_count']:.2f}
  Routing Messages: {stats['routing_messages_sent']}
"""
        
        if self.simulator.packet_history:
            last_packet = self.simulator.packet_history[-1]
            stats_text += f"""
Last Packet Information:
  Route: {' → '.join(last_packet.path)}
  Hop Count: {last_packet.hop_count}
  Delivery Time: {(last_packet.delivery_time or 0)*1000:.2f} ms
  Status: {'Delivered' if not last_packet.dropped else f'Dropped ({last_packet.drop_reason})'}
"""
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def update_protocol_comparison(self):
        comparison = self.simulator.get_protocol_comparison()
        
        comparison_text = f"""
PROTOCOL COMPARISON
{'='*50}

"""
        
        if not comparison:
            comparison_text += "No comparison data yet.\nRun comparison test by clicking 'Compare Protocols' button."
        else:
            # Header
            comparison_text += f"{'Protocol':<12} {'Delivery%':<10} {'Latency':<12} {'Overhead':<10} {'Hop':<8} {'Packets':<8}\n"
            comparison_text += "-" * 70 + "\n"
            
            # Data
            for protocol, data in comparison.items():
                comparison_text += f"{protocol:<12} "
                comparison_text += f"{data['delivery_ratio']:<10.2f} "
                comparison_text += f"{data['average_delay']:<12.2f} "
                comparison_text += f"{data['routing_overhead']:<10.2f} "
                comparison_text += f"{data['average_hop_count']:<8.2f} "
                comparison_text += f"{data['packets_sent']:<8}\n"
            
            comparison_text += "\n" + "="*50 + "\n"
            comparison_text += "Metric Descriptions:\n"
            comparison_text += "• Delivery%: Successfully delivered packet percentage\n"
            comparison_text += "• Latency: Average packet delivery time (ms)\n"
            comparison_text += "• Overhead: Routing messages / Delivered packets\n"
            comparison_text += "• Hop: Average hop count\n"
            comparison_text += "• Packets: Total packets sent\n"
        
        self.comparison_text.delete(1.0, tk.END)
        self.comparison_text.insert(1.0, comparison_text)
    
    def update_node_details(self):
        if not self.selected_nodes:
            details_text = "Select a node to view details."
        else:
            node_id = self.selected_nodes[0]
            node = self.simulator.nodes[node_id]
            
            details_text = f"""
NODE DETAIL INFORMATION: {node_id}
{'='*40}

Position Information:
  Coordinates: ({node.x}, {node.y})
  Transmission Range: {node.transmission_range}
  Neighbor Count: {len(node.neighbors)}
  Neighbors: {', '.join(sorted(node.neighbors)) if node.neighbors else 'None'}

Packet Statistics:
  Sent: {node.stats['packets_sent']}
  Received: {node.stats['packets_received']}
  Forwarded: {node.stats['packets_forwarded']}
  Dropped: {node.stats['packets_dropped']}
  
Routing Statistics:
  Hello Messages: {node.stats['hello_messages_sent']}
  RREQ Messages: {node.stats['rreq_messages_sent']}
  RREP Messages: {node.stats['rrep_messages_sent']}
  Total Routing Messages: {node.stats['routing_messages_sent']}

Protocol-Specific Information:
"""
            
            if self.simulator.routing_protocol == "AODV":
                active_routes = sum(1 for route in node.routing_table.values() 
                                  if self.simulator.discrete_simulator.current_simulation_time < route.expiry_time)
                details_text += f"  Active Routes: {active_routes}\n"
                details_text += f"  Total Route Entries: {len(node.routing_table)}\n"
                details_text += f"  Sequence Number: {node.sequence_number}\n"
                
                if node.routing_table:
                    details_text += "\n  Routing Table:\n"
                    for dest, route in node.routing_table.items():
                        status = "Active" if self.simulator.discrete_simulator.current_simulation_time < route.expiry_time else "Expired"
                        details_text += f"    {dest} → {route.next_hop} (Hop: {route.hop_count}, {status})\n"
            
            elif self.simulator.routing_protocol == "OLSR":
                details_text += f"  MPR Set: {', '.join(sorted(node.mpr_set)) if node.mpr_set else 'None'}\n"
                details_text += f"  MPR Selector Count: {len(node.mpr_selector_set)}\n"
                details_text += f"  LSA Database: {len(node.lsa_database)} entries\n"
                details_text += f"  Last Hello: {self.simulator.discrete_simulator.current_simulation_time - node.last_hello_time:.1f}s ago\n"
            
            # Add mobility information if available
            if hasattr(node, 'mobility'):
                details_text += f"\nMobility Information:\n"
                mobility = node.mobility
                details_text += f"  Model: {mobility['model']}\n"
                details_text += f"  Current Speed: {mobility['speed']:.2f} m/s\n"
                if mobility['model'] == 'random_waypoint' and mobility['target_x']:
                    distance_to_target = math.sqrt(
                        (mobility['target_x'] - node.x)**2 + 
                        (mobility['target_y'] - node.y)**2
                    )
                    details_text += f"  Distance to Target: {distance_to_target:.1f} m\n"
                    details_text += f"  Target: ({mobility['target_x']:.0f}, {mobility['target_y']:.0f})\n"
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details_text)