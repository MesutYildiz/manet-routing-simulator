import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
import threading
import time
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
import json

@dataclass
class RouteEntry:
    destination: str
    next_hop: str
    hop_count: int
    sequence_number: int
    expiry_time: float
    
@dataclass
class HelloMessage:
    sender: str
    neighbors: Set[str]
    timestamp: float

@dataclass
class RREQMessage:
    source: str
    destination: str
    sequence_number: int
    hop_count: int
    path: List[str]
    broadcast_id: int

@dataclass
class RREPMessage:
    source: str
    destination: str
    sequence_number: int
    hop_count: int
    path: List[str]

@dataclass
class LSAMessage:
    originator: str
    sequence_number: int
    neighbors: Set[str]
    timestamp: float

@dataclass
class MobilityParameters:
    """Mobility model parameters"""
    model_type: str = "random_waypoint"
    min_speed: float = 1.0  # m/s
    max_speed: float = 5.0  # m/s
    pause_time: float = 2.0  # seconds
    direction_change_prob: float = 0.1  # for random walk
    boundary_behavior: str = "bounce"  # bounce, wrap, stop

@dataclass
class LSAEntry:
    """Link State Advertisement entry for OLSR"""
    originator: str
    neighbors: Set[str]
    sequence_number: int
    timestamp: float

@dataclass
class RERRMessage:
    """Route Error message for AODV"""
    source: str
    destinations: List[str]
    sequence_number: int
    hop_count: int

@dataclass
class EnergyModel:
    """Energy model for nodes"""
    initial_energy: float = 100.0
    current_energy: float = 100.0
    tx_power: float = 0.66  # Watts
    rx_power: float = 0.395  # Watts
    idle_power: float = 0.035  # Watts
    
    def consume_tx(self, packet_size, duration):
        """Consume energy for transmission"""
        energy = self.tx_power * duration
        self.current_energy -= energy
    
    def consume_rx(self, duration):
        """Consume energy for reception"""
        energy = self.rx_power * duration
        self.current_energy -= energy
    
    def consume_idle(self, duration):
        """Consume energy for idle state"""
        energy = self.idle_power * duration
        self.current_energy -= energy
    
    def is_alive(self):
        """Check if node has energy"""
        return self.current_energy > 0
    
    def get_energy_percentage(self):
        """Get remaining energy percentage"""
        return (self.current_energy / self.initial_energy) * 100
    
    def check_and_handle_death(self, node_id, simulator):
        """Check if node is dead and handle death if so"""
        if self.current_energy <= 0 and self.current_energy != -1:  # -1 indicates already dead
            self.current_energy = -1  # Mark as dead
            simulator._handle_node_death(node_id)
            return True
        return False

@dataclass
class Event:
    """Discrete event for simulation"""
    event_type: str  # PACKET_SEND, PACKET_RECEIVE, ROUTE_UPDATE, HELLO_BROADCAST, TC_BROADCAST, LINK_BREAK
    timestamp: float
    source_node: str
    target_node: Optional[str] = None
    data: Optional[Dict] = None
    priority: int = 0  # Lower number = higher priority
    
    def __lt__(self, other):
        """For priority queue ordering"""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority < other.priority

class DiscreteEventSimulator:
    """Discrete-event simulation engine"""
    
    def __init__(self):
        self.event_queue = []  # Priority queue for events
        self.current_simulation_time = 0.0
        self.simulation_running = False
        self.event_handlers = {}
        self.statistics = {
            'events_processed': 0,
            'total_simulation_time': 0.0,
            'events_per_second': 0.0,
            'collision_count': 0
        }
    
    def schedule_event(self, event: Event):
        """Schedule a new event"""
        heapq.heappush(self.event_queue, event)
    
    def register_handler(self, event_type: str, handler_func):
        """Register event handler for specific event type"""
        self.event_handlers[event_type] = handler_func
    
    def process_event(self):
        """Process the next event in queue"""
        if not self.event_queue:
            return False
        
        event = heapq.heappop(self.event_queue)
        self.current_simulation_time = event.timestamp
        
        # Call appropriate handler
        if event.event_type in self.event_handlers:
            self.event_handlers[event.event_type](event)
        
        self.statistics['events_processed'] += 1
        return True
    
    def run_until(self, end_time: float):
        """Run simulation until specified time"""
        self.simulation_running = True
        events_processed = 0
        
        while self.event_queue and self.simulation_running:
            if self.current_simulation_time >= end_time:
                break
            
            if self.process_event():
                events_processed += 1
            else:
                break
        
        self.statistics['total_simulation_time'] = self.current_simulation_time
        if self.current_simulation_time > 0:
            self.statistics['events_per_second'] = events_processed / self.current_simulation_time
        
        return events_processed
    
    def step(self):
        """Process one event (for step-by-step simulation)"""
        return self.process_event()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.simulation_running = False
    
    def clear_events(self):
        """Clear all pending events"""
        self.event_queue.clear()
    
    def get_next_event_time(self):
        """Get timestamp of next event"""
        if self.event_queue:
            return self.event_queue[0].timestamp
        return None

class PerformanceMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_packets_sent = 0
        self.total_packets_delivered = 0
        self.total_packets_dropped = 0
        self.total_delay = 0.0
        self.total_routing_overhead = 0
        self.packet_delivery_times = []
        self.routing_messages_sent = 0
        self.hop_counts = []
        
    def calculate_delivery_ratio(self) -> float:
        if self.total_packets_sent == 0:
            return 0.0
        return (self.total_packets_delivered / self.total_packets_sent) * 100
    
    def calculate_average_delay(self) -> float:
        if not self.packet_delivery_times:
            return 0.0
        return sum(self.packet_delivery_times) / len(self.packet_delivery_times)
    
    def calculate_routing_overhead(self) -> float:
        if self.total_packets_delivered == 0:
            return float('inf')
        return self.routing_messages_sent / self.total_packets_delivered
    
    def calculate_average_hop_count(self) -> float:
        if not self.hop_counts:
            return 0.0
        return sum(self.hop_counts) / len(self.hop_counts)

class MobilityModel:
    """Handles different mobility models for MANET nodes"""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.simulation_time = 0
        self.time_step = 0.1  # seconds
        
    def initialize_node_mobility(self, node, params: MobilityParameters):
        """Initialize mobility state for a node"""
        node.mobility = {
            'model': params.model_type,
            'speed': random.uniform(params.min_speed, params.max_speed),
            'direction': random.uniform(0, 2 * math.pi),
            'target_x': None,
            'target_y': None,
            'pause_until': 0,
            'last_update': 0,  # Will be set by simulator
            'path_history': [(node.x, node.y, 0)],  # Will be updated with simulation time
            'params': params
        }
        
        if params.model_type == "random_waypoint":
            self._set_random_destination(node)
        elif params.model_type == "group_mobility":
            node.mobility['group_id'] = random.randint(0, 2)  # 3 groups
            node.mobility['group_center_x'] = random.uniform(100, self.width-100)
            node.mobility['group_center_y'] = random.uniform(100, self.height-100)
    
    def update_positions(self, nodes, current_time):
        """Update positions of all mobile nodes"""
        dt = current_time - self.simulation_time
        self.simulation_time = current_time
        
        for node_id, node in nodes.items():
            if not hasattr(node, 'mobility'):
                continue
                
            mobility = node.mobility
            
            # Check if node is in pause state
            if current_time < mobility['pause_until']:
                continue
            
            old_x, old_y = node.x, node.y
            
            if mobility['model'] == "random_waypoint":
                self._update_random_waypoint(node, dt)
            elif mobility['model'] == "random_walk":
                self._update_random_walk(node, dt)
            elif mobility['model'] == "group_mobility":
                self._update_group_mobility(node, dt)
            elif mobility['model'] == "highway":
                self._update_highway_mobility(node, dt)
            
            # Update path history
            if (node.x, node.y) != (old_x, old_y):
                mobility['path_history'].append((node.x, node.y, current_time))
                # Keep only last 50 positions
                if len(mobility['path_history']) > 50:
                    mobility['path_history'] = mobility['path_history'][-50:]
    
    def _update_random_waypoint(self, node, dt):
        """Random Waypoint Model implementation"""
        mobility = node.mobility
        
        # If no target or reached target, set new destination
        if (mobility['target_x'] is None or 
            self._distance_to_target(node) < 5):
            
            # Pause at destination
            if mobility['target_x'] is not None:
                mobility['pause_until'] = self.simulation_time + mobility['params'].pause_time
                return
            
            self._set_random_destination(node)
        
        # Move towards target
        self._move_towards_target(node, dt)
    
    def _update_random_walk(self, node, dt):
        """Random Walk Model implementation"""
        mobility = node.mobility
        
        # Randomly change direction
        if random.random() < mobility['params'].direction_change_prob:
            mobility['direction'] = random.uniform(0, 2 * math.pi)
        
        # Calculate new position
        distance = mobility['speed'] * dt
        new_x = node.x + distance * math.cos(mobility['direction'])
        new_y = node.y + distance * math.sin(mobility['direction'])
        
        # Handle boundaries
        new_x, new_y, new_direction = self._handle_boundaries(
            new_x, new_y, mobility['direction'], mobility['params'].boundary_behavior)
        
        node.x, node.y = new_x, new_y
        mobility['direction'] = new_direction
    
    def _update_group_mobility(self, node, dt):
        """Group Mobility Model - nodes move in groups"""
        mobility = node.mobility
        
        # Update group center (group leader movement)
        if not hasattr(mobility, 'group_target_x'):
            mobility['group_target_x'] = random.uniform(50, self.width-50)
            mobility['group_target_y'] = random.uniform(50, self.height-50)
        
        # Move group center
        group_dx = mobility['group_target_x'] - mobility['group_center_x']
        group_dy = mobility['group_target_y'] - mobility['group_center_y']
        group_distance = math.sqrt(group_dx**2 + group_dy**2)
        
        if group_distance > 5:
            group_speed = 2.0  # Group movement speed
            mobility['group_center_x'] += (group_dx / group_distance) * group_speed * dt
            mobility['group_center_y'] += (group_dy / group_distance) * group_speed * dt
        else:
            # Set new group target
            mobility['group_target_x'] = random.uniform(50, self.width-50)
            mobility['group_target_y'] = random.uniform(50, self.height-50)
        
        # Individual node movement within group
        group_radius = 80
        target_x = mobility['group_center_x'] + random.uniform(-group_radius, group_radius)
        target_y = mobility['group_center_y'] + random.uniform(-group_radius, group_radius)
        
        # Move towards group position
        dx = target_x - node.x
        dy = target_y - node.y
        distance_to_group = math.sqrt(dx**2 + dy**2)
        
        if distance_to_group > 5:
            move_distance = mobility['speed'] * dt
            node.x += (dx / distance_to_group) * move_distance
            node.y += (dy / distance_to_group) * move_distance
    
    def _update_highway_mobility(self, node, dt):
        """Highway Mobility Model - nodes move along predefined paths"""
        mobility = node.mobility
        
        if not hasattr(mobility, 'lane'):
            # Assign to a highway lane
            mobility['lane'] = random.choice(['horizontal_top', 'horizontal_bottom', 
                                           'vertical_left', 'vertical_right'])
            mobility['highway_speed'] = random.uniform(10, 20)  # Higher speed for highway
        
        distance = mobility['highway_speed'] * dt
        
        if mobility['lane'] == 'horizontal_top':
            node.x += distance
            node.y = 100 + random.uniform(-20, 20)  # Lane width variation
            if node.x > self.width:
                node.x = 0
        elif mobility['lane'] == 'horizontal_bottom':
            node.x -= distance
            node.y = self.height - 100 + random.uniform(-20, 20)
            if node.x < 0:
                node.x = self.width
        elif mobility['lane'] == 'vertical_left':
            node.y += distance
            node.x = 100 + random.uniform(-20, 20)
            if node.y > self.height:
                node.y = 0
        elif mobility['lane'] == 'vertical_right':
            node.y -= distance
            node.x = self.width - 100 + random.uniform(-20, 20)
            if node.y < 0:
                node.y = self.height
    
    def _set_random_destination(self, node):
        """Set a random destination for the node"""
        mobility = node.mobility
        mobility['target_x'] = random.uniform(50, self.width - 50)
        mobility['target_y'] = random.uniform(50, self.height - 50)
        mobility['speed'] = random.uniform(
            mobility['params'].min_speed, 
            mobility['params'].max_speed
        )
    
    def _move_towards_target(self, node, dt):
        """Move node towards its target destination"""
        mobility = node.mobility
        
        dx = mobility['target_x'] - node.x
        dy = mobility['target_y'] - node.y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        
        if distance_to_target > 0:
            move_distance = min(mobility['speed'] * dt, distance_to_target)
            node.x += (dx / distance_to_target) * move_distance
            node.y += (dy / distance_to_target) * move_distance
    
    def _distance_to_target(self, node):
        """Calculate distance to target"""
        mobility = node.mobility
        if mobility['target_x'] is None:
            return float('inf')
        
        dx = mobility['target_x'] - node.x
        dy = mobility['target_y'] - node.y
        return math.sqrt(dx**2 + dy**2)
    
    def _handle_boundaries(self, x, y, direction, behavior):
        """Handle boundary conditions"""
        new_direction = direction
        
        if behavior == "bounce":
            if x <= 0 or x >= self.width:
                new_direction = math.pi - direction
                x = max(0, min(self.width, x))
            if y <= 0 or y >= self.height:
                new_direction = -direction
                y = max(0, min(self.height, y))
        elif behavior == "wrap":
            x = x % self.width
            y = y % self.height
        elif behavior == "stop":
            x = max(0, min(self.width, x))
            y = max(0, min(self.height, y))
            if x <= 0 or x >= self.width or y <= 0 or y >= self.height:
                new_direction = random.uniform(0, 2 * math.pi)
        
        return x, y, new_direction
    
    def get_mobility_stats(self, nodes):
        """Get mobility statistics"""
        stats = {
            'total_distance_traveled': 0,
            'average_speed': 0,
            'nodes_moving': 0,
            'mobility_models': {}
        }
        
        speeds = []
        for node in nodes.values():
            if hasattr(node, 'mobility'):
                mobility = node.mobility
                model = mobility['model']
                if model not in stats['mobility_models']:
                    stats['mobility_models'][model] = 0
                stats['mobility_models'][model] += 1
                
                if len(mobility['path_history']) > 1:
                    # Calculate distance traveled
                    total_distance = 0
                    for i in range(1, len(mobility['path_history'])):
                        prev_x, prev_y, _ = mobility['path_history'][i-1]
                        curr_x, curr_y, _ = mobility['path_history'][i]
                        total_distance += math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    
                    stats['total_distance_traveled'] += total_distance
                    speeds.append(mobility['speed'])
                    stats['nodes_moving'] += 1
        
        if speeds:
            stats['average_speed'] = sum(speeds) / len(speeds)
        
        return stats

class Node:
    def __init__(self, node_id, x, y, transmission_range=100):
        self.id = node_id
        self.x = x
        self.y = y
        self.transmission_range = transmission_range
        self.neighbors = set()
        self.packet_queue = deque()
        self.routing_table = {}  # For AODV
        self.lsa_database = {}   # For OLSR
        self.sequence_number = 0
        self.hello_interval = 2.0  # seconds
        self.last_hello_time = 0
        self.neighbor_expiry = {}  # neighbor_id -> expiry_time
        self.broadcast_ids = set()
        self.mpr_set = set()  # Multipoint Relays (OLSR)
        self.mpr_selector_set = set()
        
        # Packet buffering for AODV route discovery
        self.packet_buffer = {}  # destination_id -> list of packets waiting for route
        self.buffer_timeouts = {}  # destination_id -> timeout timestamp
        
        # Loop prevention for flooding protocol
        self.seen_packets = {}  # packet_id -> timestamp when first seen
        
        # DSR route cache
        self.route_cache = {}  # destination_id -> complete path
        
        # Performance tracking
        self.stats = {
            'packets_sent': 0,
            'packets_received': 0,
            'packets_forwarded': 0,
            'packets_dropped': 0,
            'routing_messages_sent': 0,
            'hello_messages_sent': 0,
            'rreq_messages_sent': 0,
            'rrep_messages_sent': 0,
            'routes_expired': 0,
            'route_rediscoveries': 0,
            'dsr_cache_hits': 0,
            'dsr_cache_misses': 0
        }
        
        # Energy model
        self.energy = EnergyModel()
    
    def distance_to(self, other_node):
        return math.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)
    
    def can_communicate_with(self, other_node):
        return self.distance_to(other_node) <= self.transmission_range
    
    def add_neighbor(self, neighbor_id, expiry_time=None):
        self.neighbors.add(neighbor_id)
        if expiry_time:
            self.neighbor_expiry[neighbor_id] = expiry_time
    
    def remove_neighbor(self, neighbor_id):
        self.neighbors.discard(neighbor_id)
        if neighbor_id in self.neighbor_expiry:
            del self.neighbor_expiry[neighbor_id]
    
    def cleanup_expired_neighbors(self, current_time):
        expired = []
        for neighbor_id, expiry_time in self.neighbor_expiry.items():
            if current_time > expiry_time:
                expired.append(neighbor_id)
        
        for neighbor_id in expired:
            self.remove_neighbor(neighbor_id)
    
    def select_mpr_set(self, all_nodes):
        """Select Multipoint Relays for OLSR"""
        if len(self.neighbors) <= 1:
            return
        
        # Two-hop neighbors
        two_hop_neighbors = set()
        for neighbor_id in self.neighbors:
            if neighbor_id in all_nodes:
                neighbor_node = all_nodes[neighbor_id]
                for two_hop in neighbor_node.neighbors:
                    if two_hop != self.id and two_hop not in self.neighbors:
                        two_hop_neighbors.add(two_hop)
        
        # Simple MPR selection algorithm
        self.mpr_set.clear()
        uncovered_two_hop = two_hop_neighbors.copy()
        
        # Select neighbors that cover the most uncovered two-hop neighbors
        while uncovered_two_hop:
            best_neighbor = None
            best_coverage = 0
            
            for neighbor_id in self.neighbors:
                if neighbor_id in all_nodes:
                    neighbor_node = all_nodes[neighbor_id]
                    coverage = len(uncovered_two_hop.intersection(neighbor_node.neighbors))
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_neighbor = neighbor_id
            
            if best_neighbor:
                self.mpr_set.add(best_neighbor)
                neighbor_node = all_nodes[best_neighbor]
                uncovered_two_hop -= neighbor_node.neighbors
            else:
                break

class Packet:
    def __init__(self, source, destination, data, packet_type="DATA", size=64):
        self.source = source
        self.destination = destination
        self.data = data
        self.packet_type = packet_type
        self.path = [source]
        self.hop_count = 0
        self.timestamp = 0  # Will be set by simulator
        self.packet_id = f"{source}_{destination}_{int(self.timestamp * 1000000)}"  # Will be updated when timestamp is set
        self.size = size  # bytes
        self.delivery_time = None
        self.dropped = False
        self.drop_reason = ""
    
    def update_packet_id(self):
        """Update packet_id with current timestamp"""
        self.packet_id = f"{self.source}_{self.destination}_{int(self.timestamp * 1000000)}"

class MANETSimulator:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.nodes = {}
        self.simulation_running = False
        self.routing_protocol = "AODV"
        self.packet_history = []
        self.animation_speed = 500  # ms
        self.simulation_time = 0
        self.control_messages = []
        
        # Add discrete-event simulator
        self.discrete_simulator = DiscreteEventSimulator()
        self.setup_event_handlers()
        
        # Add mobility system
        self.mobility_model = MobilityModel(width, height)
        self.mobility_enabled = False
        self.mobility_params = MobilityParameters()
        self.last_topology_update = 0
        
        # Performance metrics for each protocol
        self.metrics = {
            'AODV': PerformanceMetrics(),
            'OLSR': PerformanceMetrics(),
            'DSR': PerformanceMetrics(),
            'FLOODING': PerformanceMetrics(),
            'ZRP': PerformanceMetrics()
        }
        
        # Protocol-specific parameters
        self.aodv_params = {
            'route_timeout': 10.0,
            'hello_interval': 1.0,
            'rreq_retries': 3,
            'net_diameter': 35,
            'node_traversal_time': 0.04
        }
        
        self.olsr_params = {
            'hello_interval': 2.0,
            'tc_interval': 5.0,
            'neighbor_hold_time': 6.0,
            'topology_hold_time': 15.0
        }
        
        # Network parameters for discrete-event simulation
        self.network_params = {
            'propagation_delay': 0.001,  # 1ms per hop
            'transmission_delay': 0.0001,  # 0.1ms per byte
            'processing_delay': 0.0005,  # 0.5ms processing time
            'collision_probability': 0.01  # 1% collision probability
        }
    
    def setup_event_handlers(self):
        """Setup event handlers for discrete-event simulation"""
        self.discrete_simulator.register_handler("PACKET_DELIVERED", self.handle_packet_delivered)
        self.discrete_simulator.register_handler("PACKET_FORWARD", self.handle_packet_forward)
        self.discrete_simulator.register_handler("PACKET_RECEIVE", self.handle_packet_receive)
        self.discrete_simulator.register_handler("ROUTE_UPDATE", self.handle_route_update_event)
        self.discrete_simulator.register_handler("HELLO_BROADCAST", self.handle_hello_broadcast_event)
        self.discrete_simulator.register_handler("TC_BROADCAST", self.handle_tc_broadcast_event)
        self.discrete_simulator.register_handler("LINK_BREAK", self.handle_link_break_event)
        self.discrete_simulator.register_handler("MOBILITY_UPDATE", self.handle_mobility_update)
        self.discrete_simulator.register_handler("RREQ_BROADCAST", self.handle_rreq_broadcast)
        self.discrete_simulator.register_handler("RREP_UNICAST", self.handle_rrep_unicast)
        self.discrete_simulator.register_handler("ROUTE_FOUND", self.handle_route_found)
        self.discrete_simulator.register_handler("CSMA_RETRY", self.handle_csma_retry)
        self.discrete_simulator.register_handler("BUFFER_TIMEOUT", self.handle_buffer_timeout)
        self.discrete_simulator.register_handler("OLSR_TOPOLOGY_UPDATE", self.handle_olsr_topology_update)
        self.discrete_simulator.register_handler("OLSR_ROUTING_UPDATE", self.handle_olsr_routing_update)
        self.discrete_simulator.register_handler("FLOODING_CLEANUP", self.handle_flooding_cleanup)
        self.discrete_simulator.register_handler("RERR_BROADCAST", self.handle_rerr_broadcast)
        self.discrete_simulator.register_handler("ROUTE_MAINTENANCE", self.handle_route_maintenance)
        self.discrete_simulator.register_handler("DSR_RREQ", self.handle_dsr_rreq)
        self.discrete_simulator.register_handler("DSR_RREP", self.handle_dsr_rrep)
        self.discrete_simulator.register_handler("ZRP_BORDERCAST", self.handle_zrp_bordercast)
    
    def handle_packet_delivered(self, event):
        """Handle packet delivery at destination"""
        packet = event.data['packet']
        
        # Calculate delivery time
        packet.delivery_time = self.discrete_simulator.current_simulation_time - packet.timestamp
        
        # Update metrics
        current_metrics = self.metrics[self.routing_protocol]
        current_metrics.total_packets_delivered += 1
        current_metrics.packet_delivery_times.append(packet.delivery_time)
        current_metrics.hop_counts.append(packet.hop_count)
        
        if event.source_node in self.nodes:
            self.nodes[event.source_node].stats['packets_received'] += 1
        
        self.packet_history.append(packet)
    
    def handle_packet_forward(self, event):
        """Handle packet forwarding at intermediate node"""
        packet = event.data['packet']
        node_id = event.source_node
        
        if node_id in self.nodes:
            self.nodes[node_id].stats['packets_forwarded'] += 1
    
    def handle_packet_receive(self, event):
        """Handle packet arrival at a node"""
        packet = event.data['packet']
        node_id = event.source_node
        node = self.nodes[node_id]
        
        # Update packet_id with current timestamp if not already set
        if packet.timestamp > 0:
            packet.update_packet_id()
        
        # Loop prevention for FLOODING protocol
        if self.routing_protocol == "FLOODING":
            current_time = self.discrete_simulator.current_simulation_time
            
            # Check if we've already seen this packet
            if packet.packet_id in node.seen_packets:
                # Drop packet - already processed
                current_metrics = self.metrics[self.routing_protocol]
                current_metrics.total_packets_dropped += 1
                node.stats['packets_dropped'] += 1
                return
            
            # Add packet to seen_packets with current timestamp
            node.seen_packets[packet.packet_id] = current_time
        
        # Energy consumption for reception
        if hasattr(node, 'energy') and node.energy.is_alive():
            reception_time = self.calculate_hop_delay(packet.size)
            node.energy.consume_rx(reception_time)
        else:
            # Node dead - drop packet
            current_metrics = self.metrics[self.routing_protocol]
            current_metrics.total_packets_dropped += 1
            node.stats['packets_dropped'] += 1
            return
        
        # Check if this is destination
        if node_id == packet.destination:
            # Schedule delivery event
            delivery_event = Event(
                event_type="PACKET_DELIVERED",
                timestamp=self.discrete_simulator.current_simulation_time,
                source_node=node_id,
                data={'packet': packet}
            )
            self.discrete_simulator.schedule_event(delivery_event)
        else:
            # Forward to next hop
            self.forward_packet_discrete(packet, node_id)
    
    def handle_route_update_event(self, event):
        """Handle route update events"""
        if event.source_node in self.nodes:
            node = self.nodes[event.source_node]
            # Update routing tables based on protocol
            if self.routing_protocol == "AODV":
                self.aodv_route_maintenance(node)
            elif self.routing_protocol == "OLSR":
                self.olsr_route_maintenance(node)
    
    def handle_hello_broadcast_event(self, event):
        """Handle hello broadcast events"""
        if event.source_node in self.nodes:
            node = self.nodes[event.source_node]
            
            # Check node energy before sending hello
            if not node.energy.is_alive():
                # Node is dead - don't send hello
                return
            
            # Consume energy for hello message transmission
            hello_size = 64  # Hello message size in bytes
            transmission_time = self.calculate_hop_delay(hello_size)
            node.energy.consume_tx(hello_size, transmission_time)
            
            # Check if node died from energy consumption
            if node.energy.check_and_handle_death(event.source_node, self):
                return
            
            # Send hello message
            if self.routing_protocol == "OLSR":
                self.olsr_send_hello_discrete(node)
            elif self.routing_protocol == "AODV":
                self.aodv_send_hello_discrete(node)
            
            # Schedule next hello message
            interval = self.olsr_params['hello_interval'] if self.routing_protocol == "OLSR" else self.aodv_params['hello_interval']
            next_hello_time = self.discrete_simulator.current_simulation_time + interval
            next_event = Event(
                event_type="HELLO_BROADCAST",
                timestamp=next_hello_time,
                source_node=event.source_node
            )
            self.discrete_simulator.schedule_event(next_event)
    
    def handle_tc_broadcast_event(self, event):
        """Handle topology control broadcast events"""
        if event.source_node in self.nodes and self.routing_protocol == "OLSR":
            node = self.nodes[event.source_node]
            
            # Check node energy before sending TC
            if not node.energy.is_alive():
                # Node is dead - don't send TC
                return
            
            # Consume energy for TC message transmission
            tc_size = 64  # TC message size in bytes
            transmission_time = self.calculate_hop_delay(tc_size)
            node.energy.consume_tx(tc_size, transmission_time)
            
            # Process TC message and update LSA database
            if 'originator' in event.data and 'mpr_set' in event.data:
                # This is a forwarded TC message - update LSA database
                originator = event.data['originator']
                mpr_set = event.data['mpr_set']
                
                # Update LSA database for all nodes that receive this TC
                for node_id, receiving_node in self.nodes.items():
                    if receiving_node.id != originator:  # Don't update originator's own database
                        # Create or update LSA entry
                        lsa_entry = LSAEntry(
                            originator=originator,
                            neighbors=mpr_set,
                            sequence_number=0,  # Simplified - in real OLSR this would be tracked
                            timestamp=self.discrete_simulator.current_simulation_time
                        )
                        receiving_node.lsa_database[originator] = lsa_entry
                        
                        # Schedule routing table update for this node
                        routing_update_event = Event(
                            event_type="OLSR_ROUTING_UPDATE",
                            timestamp=self.discrete_simulator.current_simulation_time + 0.1,
                            source_node=node_id
                        )
                        self.discrete_simulator.schedule_event(routing_update_event)
            else:
                # This is the original TC broadcast - send TC message
                self.olsr_send_tc_discrete(node)
            
            # Schedule next TC broadcast
            next_tc_time = self.discrete_simulator.current_simulation_time + self.olsr_params['tc_interval']
            next_event = Event(
                event_type="TC_BROADCAST",
                timestamp=next_tc_time,
                source_node=event.source_node
            )
            self.discrete_simulator.schedule_event(next_event)
    
    def handle_link_break_event(self, event):
        """Handle link break events with protocol-specific route invalidation"""
        if event.data and 'broken_links' in event.data:
            broken_links = event.data['broken_links']
            routing_protocol = event.data.get('routing_protocol', self.routing_protocol)
            
            # Update topology and invalidate routes
            self.update_network_topology()
            
            # Protocol-specific route invalidation
            if routing_protocol == "AODV":
                # AODV: Remove routes that use broken links as next hop
                for node_id, node in self.nodes.items():
                    routes_to_remove = []
                    for dest, route in node.routing_table.items():
                        if route.next_hop in broken_links:
                            routes_to_remove.append(dest)
                    
                    for dest in routes_to_remove:
                        del node.routing_table[dest]
                        
            elif routing_protocol == "OLSR":
                # OLSR: Update LSA database and trigger routing table updates
                for node_id, node in self.nodes.items():
                    # Remove broken links from LSA database
                    for originator, lsa in node.lsa_database.items():
                        if originator in broken_links:
                            lsa.neighbors.discard(node_id)
                            if not lsa.neighbors:  # Remove empty LSAs
                                del node.lsa_database[originator]
                    
                    # Trigger immediate routing table update
                    self.olsr_update_routing_table(node)
                    
            elif routing_protocol == "DSR":
                # DSR: Remove routes that contain broken links in their path
                for node_id, node in self.nodes.items():
                    routes_to_remove = []
                    for dest, route in node.routing_table.items():
                        # Check if any node in the route path is a broken link
                        if any(hop in broken_links for hop in route.path if hasattr(route, 'path')):
                            routes_to_remove.append(dest)
                    
                    for dest in routes_to_remove:
                        del node.routing_table[dest]
                        
            else:
                # Generic route invalidation for other protocols
                for node_id, node in self.nodes.items():
                    routes_to_remove = []
                    for dest, route in node.routing_table.items():
                        if route.next_hop in broken_links:
                            routes_to_remove.append(dest)
                    
                    for dest in routes_to_remove:
                        del node.routing_table[dest]
    
    def handle_mobility_update(self, event):
        """Update node positions in discrete-event simulation"""
        # Update all node positions
        current_time = self.discrete_simulator.current_simulation_time
        
        for node_id, node in self.nodes.items():
            if hasattr(node, 'mobility'):
                mobility = node.mobility
                
                # Check pause state
                if current_time < mobility['pause_until']:
                    continue
                
                old_x, old_y = node.x, node.y
                dt = 0.1  # Time step
                
                # Update position based on model
                if mobility['model'] == "random_waypoint":
                    self.mobility_model._update_random_waypoint(node, dt)
                elif mobility['model'] == "random_walk":
                    self.mobility_model._update_random_walk(node, dt)
                elif mobility['model'] == "group_mobility":
                    self.mobility_model._update_group_mobility(node, dt)
                elif mobility['model'] == "highway":
                    self.mobility_model._update_highway_mobility(node, dt)
                
                # Update path history with simulation time
                if (node.x, node.y) != (old_x, old_y):
                    mobility['path_history'].append((node.x, node.y, current_time))
                    if len(mobility['path_history']) > 50:
                        mobility['path_history'] = mobility['path_history'][-50:]
        
        # Check for topology changes
        topology_changed = self.update_network_topology_discrete()
        
        if topology_changed:
            # Schedule link break events
            link_break_event = Event(
                event_type="LINK_BREAK",
                timestamp=current_time,
                source_node="SYSTEM",
                data={'topology_changed': True}
            )
            self.discrete_simulator.schedule_event(link_break_event)
        
        # Schedule next mobility update
        next_update_time = current_time + 0.1  # 100ms updates
        next_event = Event(
            event_type="MOBILITY_UPDATE",
            timestamp=next_update_time,
            source_node="SYSTEM"
        )
        self.discrete_simulator.schedule_event(next_event)
    
    def handle_rreq_broadcast(self, event):
        """Handle RREQ broadcast event"""
        rreq = event.data['rreq']
        node_id = event.source_node
        node = self.nodes[node_id]
        
        # Energy consumption for RREQ processing
        if hasattr(node, 'energy') and node.energy.is_alive():
            node.energy.consume_rx(0.001)  # Processing time
        else:
            # Node dead - drop packet
            current_metrics = self.metrics[self.routing_protocol]
            current_metrics.total_packets_dropped += 1
            return
        
        # Check if destination
        if node_id == rreq.destination:
            # Schedule RREP back
            self.schedule_rrep(rreq, event.data.get('original_packet'))
            return
        
        # Forward RREQ to neighbors
        for neighbor_id in node.neighbors:
            if neighbor_id not in rreq.path:
                # Consume TX energy for RREQ forwarding
                rreq_size = 64  # RREQ message size
                transmission_time = self.calculate_hop_delay(rreq_size)
                node.energy.consume_tx(rreq_size, transmission_time)
                
                new_rreq = RREQMessage(
                    source=rreq.source,
                    destination=rreq.destination,
                    sequence_number=rreq.sequence_number,
                    hop_count=rreq.hop_count + 1,
                    path=rreq.path + [neighbor_id],
                    broadcast_id=rreq.broadcast_id
                )
                
                delay = self.calculate_hop_delay(64)  # RREQ size
                forward_event = Event(
                    event_type="RREQ_BROADCAST",
                    timestamp=self.discrete_simulator.current_simulation_time + delay,
                    source_node=neighbor_id,
                    data={
                        'rreq': new_rreq,
                        'original_packet': event.data.get('original_packet')  # Pass original packet through
                    }
                )
                self.discrete_simulator.schedule_event(forward_event)
    
    def handle_rrep_unicast(self, event):
        """Handle RREP unicast back to source"""
        rrep = event.data['rrep']
        node_id = event.source_node
        node = self.nodes[node_id]
        
        # Energy consumption for RREP processing
        if hasattr(node, 'energy') and node.energy.is_alive():
            node.energy.consume_rx(0.001)  # Processing time
        else:
            # Node dead - drop RREP
            current_metrics = self.metrics[self.routing_protocol]
            current_metrics.total_packets_dropped += 1
            return
        
        # Install route in routing table
        if len(rrep.path) > 1:
            prev_hop = rrep.path[-2]
            node.routing_table[rrep.source] = RouteEntry(
                destination=rrep.source,
                next_hop=prev_hop,
                hop_count=rrep.hop_count,
                sequence_number=rrep.sequence_number,
                expiry_time=self.discrete_simulator.current_simulation_time + self.aodv_params['route_timeout']
            )
        
        # Check if reached source
        if node_id == rrep.source:
            # Route discovery complete - schedule route found event
            route_found_event = Event(
                event_type="ROUTE_FOUND",
                timestamp=self.discrete_simulator.current_simulation_time,
                source_node=node_id,
                data={'packet': event.data.get('original_packet'), 'path': rrep.path}
            )
            self.discrete_simulator.schedule_event(route_found_event)
        else:
            # Forward RREP to next hop
            if len(rrep.path) > 1:
                # Consume TX energy for RREP forwarding
                rrep_size = 64  # RREP message size
                transmission_time = self.calculate_hop_delay(rrep_size)
                node.energy.consume_tx(rrep_size, transmission_time)
                
                next_hop = rrep.path[-2]
                delay = self.calculate_hop_delay(64)
                forward_event = Event(
                    event_type="RREP_UNICAST",
                    timestamp=self.discrete_simulator.current_simulation_time + delay,
                    source_node=next_hop,
                    data={
                        'rrep': rrep,
                        'original_packet': event.data.get('original_packet')  # Preserve original packet
                    }
                )
                self.discrete_simulator.schedule_event(forward_event)
    
    def handle_route_found(self, event):
        """Handle route discovery completion"""
        packet = event.data['packet']
        path = event.data['path']
        source_id = event.source_node
        
        # Send the original packet if provided
        if packet:
            self._schedule_packet_forwarding(packet, path)
        
        # Send all buffered packets for this destination
        if source_id in self.nodes:
            source_node = self.nodes[source_id]
            destination_id = path[-1] if path else None  # Last element is destination
            
            if destination_id and destination_id in source_node.packet_buffer:
                buffered_packets = source_node.packet_buffer[destination_id]
                for buffered_packet in buffered_packets:
                    # Update packet timestamp to current time
                    buffered_packet.timestamp = self.discrete_simulator.current_simulation_time
                    self._schedule_packet_forwarding(buffered_packet, path)
                
                # Clear the buffer and timeout for this destination
                del source_node.packet_buffer[destination_id]
                if destination_id in source_node.buffer_timeouts:
                    del source_node.buffer_timeouts[destination_id]
    
    def handle_csma_retry(self, event):
        """Handle CSMA/CA retry after backoff"""
        packet = event.data['packet']
        node_id = event.source_node
        
        # Try transmission again
        self.schedule_transmission_with_csma(node_id, packet)
    
    def handle_buffer_timeout(self, event):
        """Handle buffer timeout - drop buffered packets for destination"""
        node_id = event.source_node
        destination_id = event.data['destination_id']
        
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if destination_id in node.packet_buffer:
                # Drop all buffered packets for this destination
                buffered_packets = node.packet_buffer[destination_id]
                current_metrics = self.metrics[self.routing_protocol]
                current_metrics.total_packets_dropped += len(buffered_packets)
                node.stats['packets_dropped'] += len(buffered_packets)
                
                # Clear buffer and timeout
                del node.packet_buffer[destination_id]
                if destination_id in node.buffer_timeouts:
                    del node.buffer_timeouts[destination_id]
    
    def handle_olsr_topology_update(self, event):
        """Periyodik OLSR topology güncelleme"""
        node_id = event.source_node
        node = self.nodes[node_id]
        
        # Dijkstra çalıştır ve routing table'ı güncelle
        self.olsr_update_routing_table(node)
        
        # Schedule next update
        next_update = self.discrete_simulator.current_simulation_time + 5.0
        update_event = Event(
            event_type="OLSR_TOPOLOGY_UPDATE",
            timestamp=next_update,
            source_node=node_id
        )
        self.discrete_simulator.schedule_event(update_event)
    
    def handle_olsr_routing_update(self, event):
        """Handle OLSR routing table update"""
        node_id = event.source_node
        node = self.nodes[node_id]
        
        # Update routing table using Dijkstra algorithm
        self.olsr_update_routing_table(node)
        
        # Schedule next routing update (every 5 seconds)
        next_update = self.discrete_simulator.current_simulation_time + 5.0
        routing_update_event = Event(
            event_type="OLSR_ROUTING_UPDATE",
            timestamp=next_update,
            source_node=node_id
        )
        self.discrete_simulator.schedule_event(routing_update_event)
    
    def handle_flooding_cleanup(self, event):
        """Handle periodic cleanup of old packet IDs for flooding protocol"""
        current_time = self.discrete_simulator.current_simulation_time
        cleanup_threshold = 10.0  # seconds
        
        for node_id, node in self.nodes.items():
            if hasattr(node, 'seen_packets'):
                # Remove old packet IDs
                old_packets = []
                for packet_id, timestamp in node.seen_packets.items():
                    if current_time - timestamp > cleanup_threshold:
                        old_packets.append(packet_id)
                
                for packet_id in old_packets:
                    del node.seen_packets[packet_id]
        
        # Schedule next cleanup (every 5 seconds)
        next_cleanup = current_time + 5.0
        cleanup_event = Event(
            event_type="FLOODING_CLEANUP",
            timestamp=next_cleanup,
            source_node="SYSTEM"
        )
        self.discrete_simulator.schedule_event(cleanup_event)
    
    def handle_rerr_broadcast(self, event):
        """Handle RERR broadcast for AODV route invalidation"""
        rerr = event.data['rerr']
        sender = event.data['sender']
        node_id = event.source_node
        node = self.nodes[node_id]
        
        # Check if we have routes to any of the invalidated destinations
        routes_to_remove = []
        for dest in rerr.destinations:
            if dest in node.routing_table:
                route = node.routing_table[dest]
                # Remove route if it goes through the sender or uses invalidated destinations
                if route.next_hop == sender or dest in rerr.destinations:
                    routes_to_remove.append(dest)
        
        # Remove invalid routes
        for dest in routes_to_remove:
            del node.routing_table[dest]
        
        # Forward RERR to other neighbors (not the sender)
        for neighbor_id in node.neighbors:
            if neighbor_id != sender:
                delay = self.calculate_hop_delay(64)
                forward_event = Event(
                    event_type="RERR_BROADCAST",
                    timestamp=self.discrete_simulator.current_simulation_time + delay,
                    source_node=neighbor_id,
                    data={'rerr': rerr, 'sender': node_id}
                )
                self.discrete_simulator.schedule_event(forward_event)
    
    def _handle_node_death(self, node_id):
        """Handle node death when energy reaches 0"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Clear all routing table entries
            node.routing_table.clear()
            
            # Clear packet buffers
            if hasattr(node, 'packet_buffer'):
                node.packet_buffer.clear()
            if hasattr(node, 'buffer_timeouts'):
                node.buffer_timeouts.clear()
            
            # Clear LSA database for OLSR
            if hasattr(node, 'lsa_database'):
                node.lsa_database.clear()
            
            # Clear neighbor relationships
            node.neighbors.clear()
            node.neighbor_expiry.clear()
            
            # Update statistics
            current_metrics = self.metrics[self.routing_protocol]
            current_metrics.total_packets_dropped += len(node.packet_queue)
            node.stats['packets_dropped'] += len(node.packet_queue)
            
            # Clear packet queue
            node.packet_queue.clear()
            
            # Schedule link break events for all neighbors
            for other_node_id, other_node in self.nodes.items():
                if other_node_id != node_id and node_id in other_node.neighbors:
                    other_node.remove_neighbor(node_id)
                    
                    # Schedule link break event
                    link_break_event = Event(
                        event_type="LINK_BREAK",
                        timestamp=self.discrete_simulator.current_simulation_time,
                        source_node="SYSTEM",
                        data={
                            'broken_links': [node_id],
                            'topology_changed': True,
                            'routing_protocol': self.routing_protocol
                        }
                    )
                    self.discrete_simulator.schedule_event(link_break_event)
    
    def handle_route_maintenance(self, event):
        """Handle periodic route maintenance for AODV nodes"""
        node_id = event.source_node
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        current_time = self.discrete_simulator.current_simulation_time
        
        # Track expired routes for statistics
        expired_routes = []
        routes_to_remove = []
        
        # Check all routes for expiry
        for dest, route in node.routing_table.items():
            if current_time >= route.expiry_time:
                expired_routes.append(dest)
                routes_to_remove.append(dest)
        
        # Remove expired routes
        for dest in routes_to_remove:
            del node.routing_table[dest]
        
        # Update statistics
        if expired_routes:
            node.stats['routes_expired'] += len(expired_routes)
        
        # Check for active transmissions using expired routes
        for dest in expired_routes:
            # Check if there are buffered packets for this destination
            if hasattr(node, 'packet_buffer') and dest in node.packet_buffer:
                buffered_packets = node.packet_buffer[dest]
                if buffered_packets:
                    # Trigger route rediscovery for buffered packets
                    for packet in buffered_packets:
                        # Clear the buffer for this destination
                        del node.packet_buffer[dest]
                        if dest in node.buffer_timeouts:
                            del node.buffer_timeouts[dest]
                        
                        # Initiate new route discovery
                        self.initiate_route_discovery(node_id, dest, packet)
                        node.stats['route_rediscoveries'] += 1
        
        # Schedule next route maintenance (every 1 second)
        next_maintenance = current_time + 1.0
        maintenance_event = Event(
            event_type="ROUTE_MAINTENANCE",
            timestamp=next_maintenance,
            source_node=node_id
        )
        self.discrete_simulator.schedule_event(maintenance_event)
    
    def olsr_update_routing_table(self, node):
        """Update OLSR routing table using dijkstra"""
        # Dijkstra's algorithm to update routing table
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[node.id] = 0
        previous = {}
        unvisited = set(self.nodes.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            if distances[current] == float('inf'):
                break
            
            unvisited.remove(current)
            
            # Check neighbors from topology database
            current_node = self.nodes[current]
            neighbors = current_node.neighbors.copy()
            
            # Add neighbors from LSA database
            for lsa in current_node.lsa_database.values():
                if lsa.originator == current:
                    neighbors.update(lsa.neighbors)
            
            for neighbor in neighbors:
                if neighbor in unvisited:
                    new_distance = distances[current] + 1
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        # Update routing table
        for dest_id in self.nodes:
            if dest_id != node.id and distances[dest_id] != float('inf'):
                # Reconstruct path
                path = []
                current = dest_id
                while current is not None:
                    path.append(current)
                    current = previous.get(current)
                path = path[::-1]
                
                if len(path) > 1:
                    node.routing_table[dest_id] = RouteEntry(
                        destination=dest_id,
                        next_hop=path[1],
                        hop_count=len(path) - 1,
                        sequence_number=0,
                        expiry_time=self.discrete_simulator.current_simulation_time + 30.0
                    )
    
    def initiate_route_discovery(self, source_id, destination_id, packet):
        """Schedule RREQ broadcast for route discovery"""
        source_node = self.nodes[source_id]
        broadcast_id = len(source_node.broadcast_ids)
        source_node.broadcast_ids.add(broadcast_id)
        
        rreq = RREQMessage(
            source=source_id,
            destination=destination_id,
            sequence_number=source_node.sequence_number,
            hop_count=0,
            path=[source_id],
            broadcast_id=broadcast_id
        )
        
        # Track routing overhead
        self.metrics[self.routing_protocol].routing_messages_sent += 1
        source_node.stats['rreq_messages_sent'] += 1
        
        # Schedule RREQ broadcast
        rreq_event = Event(
            event_type="RREQ_BROADCAST",
            timestamp=self.discrete_simulator.current_simulation_time,
            source_node=source_id,
            data={
                'rreq': rreq,
                'original_packet': packet  # Store packet to send after route found
            }
        )
        self.discrete_simulator.schedule_event(rreq_event)
    
    def schedule_rrep(self, rreq, original_packet=None):
        """Schedule RREP back to source"""
        dest_node = self.nodes[rreq.destination]
        dest_node.sequence_number += 1
        
        rrep = RREPMessage(
            source=rreq.destination,
            destination=rreq.source,
            sequence_number=dest_node.sequence_number,
            hop_count=rreq.hop_count,
            path=rreq.path[::-1]  # Reverse path
        )
        
        # Track routing overhead
        self.metrics[self.routing_protocol].routing_messages_sent += 1
        dest_node.stats['rrep_messages_sent'] += 1
        
        # Schedule RREP unicast
        delay = self.calculate_hop_delay(64)
        rrep_event = Event(
            event_type="RREP_UNICAST",
            timestamp=self.discrete_simulator.current_simulation_time + delay,
            source_node=rreq.path[-2] if len(rreq.path) > 1 else rreq.source,
            data={
                'rrep': rrep,
                'original_packet': original_packet  # Include original packet in RREP event
            }
        )
        self.discrete_simulator.schedule_event(rrep_event)
    
    def _schedule_packet_transmission(self, packet, next_hop):
        """Schedule packet transmission to next hop"""
        delay = self.calculate_hop_delay(packet.size)
        arrival_time = self.discrete_simulator.current_simulation_time + delay
        
        # Check for collision
        if self.check_collision(packet.source, next_hop, arrival_time):
            self.discrete_simulator.statistics['collision_count'] += 1
            arrival_time += random.uniform(0.001, 0.01)  # Random backoff
        
        # Schedule packet receive event
        receive_event = Event(
            event_type="PACKET_RECEIVE",
            timestamp=arrival_time,
            source_node=next_hop,
            data={'packet': packet}
        )
        self.discrete_simulator.schedule_event(receive_event)
    
    def _is_route_discovery_in_progress(self, source_id, destination_id):
        """Check if route discovery is already in progress for this destination"""
        # This is a simple implementation - in a real system you'd track active RREQs
        # For now, we'll assume route discovery is not in progress if no recent RREQ
        # In a more sophisticated implementation, you'd maintain a set of active route discoveries
        return False  # Simplified - always allow new route discovery
    
    def schedule_transmission_with_csma(self, node_id, packet):
        """CSMA/CA medium access"""
        # Carrier sense
        if self.is_channel_busy(node_id):
            # Backoff
            backoff_time = random.uniform(0.001, 0.010)
            retry_event = Event(
                event_type="CSMA_RETRY",
                timestamp=self.discrete_simulator.current_simulation_time + backoff_time,
                source_node=node_id,
                data={'packet': packet}
            )
            self.discrete_simulator.schedule_event(retry_event)
        else:
            # Transmit
            self.transmit_packet(node_id, packet)
    
    def is_channel_busy(self, node_id):
        """Check if channel is busy for CSMA/CA"""
        current_time = self.discrete_simulator.current_simulation_time
        sensing_window = 0.001  # 1ms sensing window
        
        # Check for ongoing transmissions
        for event in self.discrete_simulator.event_queue:
            if event.event_type in ["PACKET_SEND", "PACKET_FORWARD", "PACKET_RECEIVE"]:
                if abs(event.timestamp - current_time) < sensing_window:
                    # Check if transmission is in range
                    if event.source_node in self.nodes:
                        sender = self.nodes[event.source_node]
                        receiver = self.nodes[node_id]
                        if receiver.can_communicate_with(sender):
                            return True
        
        return False
    
    def transmit_packet(self, node_id, packet):
        """Transmit packet with energy consumption"""
        node = self.nodes[node_id]
        
        # Check energy if available
        if hasattr(node, 'energy') and not node.energy.is_alive():
            # Node is dead, drop packet
            current_metrics = self.metrics[self.routing_protocol]
            current_metrics.total_packets_dropped += 1
            node.stats['packets_dropped'] += 1
            return False
        
        # Calculate transmission time
        transmission_time = packet.size * 8 / 250000  # 250 kbps
        
        # Consume energy if available
        if hasattr(node, 'energy'):
            node.energy.consume_tx(packet.size, transmission_time)
        
        # Schedule transmission completion
        completion_event = Event(
            event_type="TRANSMISSION_COMPLETE",
            timestamp=self.discrete_simulator.current_simulation_time + transmission_time,
            source_node=node_id,
            data={'packet': packet}
        )
        self.discrete_simulator.schedule_event(completion_event)
        
        return True
    
    def handle_dsr_rreq(self, event):
        """Handle DSR RREQ"""
        node_id = event.source_node
        destination = event.data['destination']
        path = event.data['path']
        original_packet = event.data['original_packet']
        
        # Energy consumption for RREQ processing
        node = self.nodes[node_id]
        if hasattr(node, 'energy') and node.energy.is_alive():
            node.energy.consume_rx(0.001)  # Processing time
        else:
            # Node dead - drop packet
            current_metrics = self.metrics[self.routing_protocol]
            current_metrics.total_packets_dropped += 1
            return
        
        if node_id == destination:
            # Send RREP back
            self.schedule_dsr_rrep(path, original_packet)
        else:
            # Broadcast to neighbors
            node = self.nodes[node_id]
            for neighbor_id in node.neighbors:
                if neighbor_id not in path:
                    delay = self.calculate_hop_delay(64)
                    forward_event = Event(
                        event_type="DSR_RREQ",
                        timestamp=self.discrete_simulator.current_simulation_time + delay,
                        source_node=neighbor_id,
                        data={
                            'destination': destination,
                            'path': path + [neighbor_id],
                            'original_packet': original_packet
                        }
                    )
                    self.discrete_simulator.schedule_event(forward_event)
    
    def schedule_dsr_rrep(self, path, original_packet):
        """Schedule DSR RREP back to source"""
        # Reverse path for RREP
        rrep_path = path[::-1]
        
        # Schedule RREP unicast
        delay = self.calculate_hop_delay(64)
        rrep_event = Event(
            event_type="DSR_RREP",
            timestamp=self.discrete_simulator.current_simulation_time + delay,
            source_node=rrep_path[-2] if len(rrep_path) > 1 else rrep_path[0],
            data={'path': rrep_path, 'original_packet': original_packet}
        )
        self.discrete_simulator.schedule_event(rrep_event)
    
    def handle_dsr_rrep(self, event):
        """Handle DSR RREP"""
        path = event.data['path']
        original_packet = event.data['original_packet']
        node_id = event.source_node
        
        # Energy consumption
        node = self.nodes[node_id]
        if hasattr(node, 'energy') and node.energy.is_alive():
            node.energy.consume_rx(0.001)
        else:
            current_metrics = self.metrics[self.routing_protocol]
            current_metrics.total_packets_dropped += 1
            return
        
        # Check if reached source
        if node_id == original_packet.source:
            # Route discovery complete - store the complete path in the packet
            original_packet.path = path.copy()  # Store the complete source route
            original_packet.hop_count = 0  # Reset hop count to start of path
            
            # Store route in route cache for future use
            source_node = self.nodes[node_id]
            source_node.route_cache[original_packet.destination] = path.copy()
            
            # Route discovery complete
            route_found_event = Event(
                event_type="ROUTE_FOUND",
                timestamp=self.discrete_simulator.current_simulation_time,
                source_node=node_id,
                data={'packet': original_packet, 'path': path}
            )
            self.discrete_simulator.schedule_event(route_found_event)
        else:
            # Forward RREP to next hop
            if len(path) > 1:
                next_hop = path[-2]
                delay = self.calculate_hop_delay(64)
                forward_event = Event(
                    event_type="DSR_RREP",
                    timestamp=self.discrete_simulator.current_simulation_time + delay,
                    source_node=next_hop,
                    data={'path': path, 'original_packet': original_packet}
                )
                self.discrete_simulator.schedule_event(forward_event)
    
    def initiate_dsr_route_discovery(self, source_id, destination_id, packet):
        """DSR route request with route cache checking"""
        source_node = self.nodes[source_id]
        
        # Check route cache first
        if destination_id in source_node.route_cache:
            # Use cached route
            cached_path = source_node.route_cache[destination_id]
            packet.path = cached_path.copy()
            packet.hop_count = 0
            
            # Update cache hit statistics
            source_node.stats['dsr_cache_hits'] += 1
            
            # Schedule route found event
            route_found_event = Event(
                event_type="ROUTE_FOUND",
                timestamp=self.discrete_simulator.current_simulation_time,
                source_node=source_id,
                data={'packet': packet, 'path': cached_path}
            )
            self.discrete_simulator.schedule_event(route_found_event)
            return
        
        # No cached route - initiate route discovery
        source_node.stats['dsr_cache_misses'] += 1
        
        rreq_event = Event(
            event_type="DSR_RREQ",
            timestamp=self.discrete_simulator.current_simulation_time,
            source_node=source_id,
            data={
                'destination': destination_id,
                'path': [source_id],
                'original_packet': packet
            }
        )
        self.discrete_simulator.schedule_event(rreq_event)
    
    def handle_zrp_bordercast(self, event):
        """Handle ZRP bordercast to zone border nodes"""
        source_id = event.source_node
        destination_id = event.data['destination']
        original_packet = event.data['original_packet']
        
        # Find border nodes and use AODV-like discovery
        border_nodes = self.find_border_nodes(source_id, 2)
        
        for border_node in border_nodes:
            # Use AODV route discovery from border node
            self.initiate_route_discovery(border_node, destination_id, original_packet)
    
    def initiate_zrp_bordercast(self, source_id, destination_id, packet):
        """ZRP bordercast to zone border nodes"""
        bordercast_event = Event(
            event_type="ZRP_BORDERCAST",
            timestamp=self.discrete_simulator.current_simulation_time,
            source_node=source_id,
            data={
                'destination': destination_id,
                'original_packet': packet
            }
        )
        self.discrete_simulator.schedule_event(bordercast_event)
    
    def enable_mobility(self, enabled=True, params=None):
        """Enable or disable mobility"""
        self.mobility_enabled = enabled
        if params:
            self.mobility_params = params
        
        if enabled:
            # Initialize mobility for existing nodes
            for node in self.nodes.values():
                if not hasattr(node, 'mobility'):
                    self.mobility_model.initialize_node_mobility(node, self.mobility_params)
    
    def add_node(self, node_id, x=None, y=None):
        if x is None:
            x = random.randint(50, self.width - 50)
        if y is None:
            y = random.randint(50, self.height - 50)
        
        node = Node(node_id, x, y)
        self.nodes[node_id] = node
        
        # Initialize mobility if enabled
        if self.mobility_enabled:
            self.mobility_model.initialize_node_mobility(node, self.mobility_params)
        
        self.update_network_topology()
        return node
    
    def update_simulation(self):
        """Main simulation update loop - now using discrete-event simulation"""
        # Process discrete events
        if not self.discrete_simulator.simulation_running:
            # Process a few events at a time for smooth GUI updates
            for _ in range(5):  # Process up to 5 events per update
                if not self.discrete_simulator.step():
                    break
        
        # Update mobility if enabled (keep real-time mobility for GUI)
        if self.mobility_enabled:
            current_time = self.discrete_simulator.current_simulation_time
            self.mobility_model.update_positions(self.nodes, current_time)
            
            # Update topology less frequently to reduce overhead
            if current_time - self.last_topology_update > 0.5:  # Update every 500ms
                self.update_network_topology()
                self.last_topology_update = current_time
        
        # Update simulation time
        self.simulation_time = self.discrete_simulator.current_simulation_time
    
    def start_discrete_simulation(self, duration=10.0):
        """Start discrete-event simulation for specified duration"""
        # Schedule initial events
        self.schedule_initial_events()
        
        # Run simulation
        events_processed = self.discrete_simulator.run_until(duration)
        return events_processed
    
    def schedule_initial_events(self):
        """Schedule initial events for simulation"""
        current_time = self.discrete_simulator.current_simulation_time
        
        # Schedule hello messages for all nodes
        for i, node_id in enumerate(self.nodes.keys()):
            # Stagger hello messages slightly
            hello_time = current_time + 0.1 + (i * 0.01)
            hello_event = Event(
                event_type="HELLO_BROADCAST",
                timestamp=hello_time,
                source_node=node_id
            )
            self.discrete_simulator.schedule_event(hello_event)
            
            # Schedule TC broadcasts for OLSR
            if self.routing_protocol == "OLSR":
                tc_time = current_time + 0.5 + (i * 0.01)
                tc_event = Event(
                    event_type="TC_BROADCAST",
                    timestamp=tc_time,
                    source_node=node_id
                )
                self.discrete_simulator.schedule_event(tc_event)
        
        # Schedule OLSR topology updates if OLSR is active
        if self.routing_protocol == "OLSR":
            for node_id in self.nodes.keys():
                # Schedule initial topology update
                update_event = Event(
                    event_type="OLSR_TOPOLOGY_UPDATE",
                    timestamp=current_time + 1.0,
                    source_node=node_id
                )
                self.discrete_simulator.schedule_event(update_event)
                
                # Schedule initial routing update
                routing_event = Event(
                    event_type="OLSR_ROUTING_UPDATE",
                    timestamp=current_time + 2.0,
                    source_node=node_id
                )
                self.discrete_simulator.schedule_event(routing_event)
        
        # Schedule mobility updates if enabled
        if self.mobility_enabled:
            mobility_event = Event(
                event_type="MOBILITY_UPDATE",
                timestamp=current_time + 0.1,
                source_node="SYSTEM"
            )
            self.discrete_simulator.schedule_event(mobility_event)
        
        # Schedule flooding cleanup if FLOODING is active
        if self.routing_protocol == "FLOODING":
            cleanup_event = Event(
                event_type="FLOODING_CLEANUP",
                timestamp=current_time + 5.0,
                source_node="SYSTEM"
            )
            self.discrete_simulator.schedule_event(cleanup_event)
        
        # Schedule route maintenance for AODV nodes
        if self.routing_protocol == "AODV":
            for node_id in self.nodes.keys():
                maintenance_event = Event(
                    event_type="ROUTE_MAINTENANCE",
                    timestamp=current_time + 1.0,
                    source_node=node_id
                )
                self.discrete_simulator.schedule_event(maintenance_event)
    
    def aodv_send_hello_discrete(self, node):
        """Send AODV hello message in discrete-event simulation"""
        # Track routing overhead
        self.metrics[self.routing_protocol].routing_messages_sent += 1
        node.stats['hello_messages_sent'] += 1
        
        # Process hello at neighbors
        for neighbor_id in node.neighbors:
            if neighbor_id in self.nodes:
                neighbor = self.nodes[neighbor_id]
                neighbor.add_neighbor(node.id, self.discrete_simulator.current_simulation_time + self.aodv_params['hello_interval'])
    
    def olsr_send_hello_discrete(self, node):
        """Send OLSR hello message in discrete-event simulation"""
        # Track routing overhead
        self.metrics[self.routing_protocol].routing_messages_sent += 1
        node.stats['hello_messages_sent'] += 1
        
        # Process hello at neighbors
        for neighbor_id in node.neighbors:
            if neighbor_id in self.nodes:
                neighbor = self.nodes[neighbor_id]
                neighbor.add_neighbor(node.id, self.discrete_simulator.current_simulation_time + self.olsr_params['neighbor_hold_time'])
    
    def olsr_send_tc_discrete(self, node):
        """Send OLSR TC message in discrete-event simulation"""
        # Track routing overhead
        self.metrics[self.routing_protocol].routing_messages_sent += 1
        
        # Flood TC message through MPRs
        self.olsr_flood_tc_discrete(node)
    
    def olsr_flood_tc_discrete(self, node):
        """Flood TC message through MPR network in discrete-event simulation"""
        # Schedule TC message to all MPRs
        for mpr_id in node.mpr_set:
            if mpr_id in self.nodes:
                # Calculate transmission delay
                delay = self.calculate_hop_delay(64)  # TC message size
                arrival_time = self.discrete_simulator.current_simulation_time + delay
                
                tc_event = Event(
                    event_type="TC_BROADCAST",
                    timestamp=arrival_time,
                    source_node=mpr_id,
                    data={'originator': node.id, 'mpr_set': node.mpr_set}
                )
                self.discrete_simulator.schedule_event(tc_event)
    
    def aodv_route_maintenance(self, node):
        """AODV route maintenance in discrete-event simulation"""
        current_time = self.discrete_simulator.current_simulation_time
        
        # Clean up expired routes
        expired_routes = []
        for dest, route in node.routing_table.items():
            if current_time >= route.expiry_time:
                expired_routes.append(dest)
        
        for dest in expired_routes:
            del node.routing_table[dest]
    
    def olsr_route_maintenance(self, node):
        """OLSR route maintenance in discrete-event simulation"""
        current_time = self.discrete_simulator.current_simulation_time
        
        # Clean up expired neighbors
        node.cleanup_expired_neighbors(current_time)
        
        # Update MPR sets
        node.select_mpr_set(self.nodes)
    
    def remove_node(self, node_id):
        if node_id in self.nodes:
            # Clean up routing tables and neighbor lists
            for other_id, other_node in self.nodes.items():
                if other_id != node_id:
                    other_node.remove_neighbor(node_id)
                    # Remove routes through this node
                    to_remove = []
                    for dest, route in other_node.routing_table.items():
                        if route.next_hop == node_id:
                            to_remove.append(dest)
                    for dest in to_remove:
                        del other_node.routing_table[dest]
            
            del self.nodes[node_id]
            self.update_network_topology()
    
    def update_network_topology(self):
        """Updated topology update with mobility considerations"""
        return self.update_network_topology_discrete()
    
    def update_network_topology_discrete(self):
        """Update topology using discrete simulation time"""
        current_time = self.discrete_simulator.current_simulation_time
        topology_changed = False
        all_broken_links = set()  # Track all broken links across the network
        
        for node_id, node in self.nodes.items():
            # Clean up expired neighbors with simulation time
            expired = []
            for neighbor_id, expiry_time in node.neighbor_expiry.items():
                if current_time > expiry_time:
                    expired.append(neighbor_id)
            
            for neighbor_id in expired:
                node.remove_neighbor(neighbor_id)
            
            # Update current neighbors
            old_neighbors = node.neighbors.copy()
            node.neighbors.clear()
            
            for other_id, other_node in self.nodes.items():
                if node_id != other_id and node.can_communicate_with(other_node):
                    expiry_time = current_time + self.olsr_params['neighbor_hold_time']
                    node.add_neighbor(other_id, expiry_time)
            
            # Detect broken links for this node
            if old_neighbors != node.neighbors:
                topology_changed = True
                broken_links = old_neighbors - node.neighbors
                
                # Add broken links to global set
                for broken_neighbor in broken_links:
                    all_broken_links.add(broken_neighbor)
                
                # Protocol-specific handling
                if self.routing_protocol == "AODV":
                    self._invalidate_broken_routes_discrete(node_id, old_neighbors)
                    # Send RERR messages for broken links
                    self._send_rerr_for_broken_links(node_id, broken_links)
                elif self.routing_protocol == "OLSR":
                    # Schedule immediate TC broadcast for topology changes
                    self._schedule_immediate_tc_broadcast(node_id)
                
                if self.routing_protocol == "OLSR":
                    node.select_mpr_set(self.nodes)
        
        # Create comprehensive link break event if topology changed
        if topology_changed and all_broken_links:
            link_break_event = Event(
                event_type="LINK_BREAK",
                timestamp=current_time,
                source_node="SYSTEM",
                data={
                    'broken_links': list(all_broken_links),
                    'topology_changed': True,
                    'routing_protocol': self.routing_protocol
                }
            )
            self.discrete_simulator.schedule_event(link_break_event)
        
        return topology_changed
    
    def _send_rerr_for_broken_links(self, node_id, broken_links):
        """Send RERR messages for broken links in AODV"""
        current_time = self.discrete_simulator.current_simulation_time
        node = self.nodes[node_id]
        
        # Find destinations that use broken links as next hop
        destinations_to_invalidate = []
        for dest, route in node.routing_table.items():
            if route.next_hop in broken_links:
                destinations_to_invalidate.append(dest)
        
        if destinations_to_invalidate:
            # Create RERR message
            rerr = RERRMessage(
                source=node_id,
                destinations=destinations_to_invalidate,
                sequence_number=node.sequence_number + 1,
                hop_count=0
            )
            
            # Update node sequence number
            node.sequence_number += 1
            
            # Remove invalid routes from routing table
            for dest in destinations_to_invalidate:
                del node.routing_table[dest]
            
            # Broadcast RERR to all neighbors
            for neighbor_id in node.neighbors:
                if neighbor_id not in broken_links:  # Don't send to broken neighbors
                    delay = self.calculate_hop_delay(64)  # RERR message size
                    rerr_event = Event(
                        event_type="RERR_BROADCAST",
                        timestamp=current_time + delay,
                        source_node=neighbor_id,
                        data={'rerr': rerr, 'sender': node_id}
                    )
                    self.discrete_simulator.schedule_event(rerr_event)
    
    def _schedule_immediate_tc_broadcast(self, node_id):
        """Schedule immediate TC broadcast for OLSR topology changes"""
        current_time = self.discrete_simulator.current_simulation_time
        
        # Schedule immediate TC broadcast (not wait for periodic)
        tc_event = Event(
            event_type="TC_BROADCAST",
            timestamp=current_time + 0.01,  # Very small delay
            source_node=node_id
        )
        self.discrete_simulator.schedule_event(tc_event)
    
    def _invalidate_broken_routes(self, node_id, old_neighbors):
        """Invalidate routes when links break due to mobility"""
        self._invalidate_broken_routes_discrete(node_id, old_neighbors)
    
    def _invalidate_broken_routes_discrete(self, node_id, old_neighbors):
        """Invalidate routes when links break due to mobility - DISCRETE VERSION"""
        node = self.nodes[node_id]
        broken_links = old_neighbors - node.neighbors
        
        if broken_links:
            # Remove routes that use broken links
            routes_to_remove = []
            for dest, route in node.routing_table.items():
                if route.next_hop in broken_links:
                    routes_to_remove.append(dest)
            
            for dest in routes_to_remove:
                del node.routing_table[dest]
    
    def get_mobility_stats(self):
        """Get mobility statistics"""
        return self.mobility_model.get_mobility_stats(self.nodes)
    
    # AODV Routing Protocol Implementation - EVENT-BASED ONLY
    def aodv_find_route(self, source_id, destination_id):
        """AODV Route Lookup - Only checks routing table"""
        source_node = self.nodes[source_id]
        
        # Check if route exists in routing table
        if destination_id in source_node.routing_table:
            route = source_node.routing_table[destination_id]
            if self.discrete_simulator.current_simulation_time < route.expiry_time:
                return self.aodv_build_path(source_id, destination_id)
        
        # No route in table - return None (route discovery will be event-based)
        return None
    
    def aodv_send_rrep(self, rreq):
        """Send Route Reply (RREP) back to source - DISCRETE VERSION"""
        dest_node = self.nodes[rreq.destination]
        dest_node.sequence_number += 1
        
        # Install reverse route with SIMULATION TIME
        if len(rreq.path) > 1:
            prev_hop = rreq.path[-2]
            dest_node.routing_table[rreq.source] = RouteEntry(
                destination=rreq.source,
                next_hop=prev_hop,
                hop_count=rreq.hop_count,
                sequence_number=rreq.sequence_number,
                expiry_time=self.discrete_simulator.current_simulation_time + self.aodv_params['route_timeout']
            )
        
        # Track routing overhead
        self.metrics[self.routing_protocol].routing_messages_sent += 1
        dest_node.stats['rrep_messages_sent'] += 1
        
        return [rreq.path]
    
    def aodv_build_path(self, source_id, destination_id):
        """Build path using routing table - DISCRETE VERSION with comprehensive expiry checking"""
        path = [source_id]
        current = source_id
        visited = set()
        current_time = self.discrete_simulator.current_simulation_time
        
        while current != destination_id and current not in visited:
            visited.add(current)
            node = self.nodes[current]
            
            if destination_id in node.routing_table:
                route = node.routing_table[destination_id]
                if current_time < route.expiry_time:  # Route is still valid
                    # Additional check: ensure next hop is still a neighbor
                    if route.next_hop in node.neighbors:
                        current = route.next_hop
                        path.append(current)
                    else:
                        # Next hop is no longer a neighbor - route is stale
                        return None
                else:
                    # Route expired - remove it from routing table
                    del node.routing_table[destination_id]
                    return None
            else:
                return None  # No route
        
        # Final check: ensure the complete path doesn't contain any expired routes
        for i in range(len(path) - 1):
            current_node = self.nodes[path[i]]
            next_hop = path[i + 1]
            if next_hop in current_node.routing_table:
                route = current_node.routing_table[next_hop]
                if current_time >= route.expiry_time:
                    return None  # Path contains expired route
        
        return [path] if current == destination_id else None
    
    # OLSR Routing Protocol Implementation (existing code)
    def olsr_find_route(self, source_id, destination_id):
        """OLSR routing using topology table"""
        # Update topology database
        self.olsr_update_topology()
        
        # Use Dijkstra's algorithm on topology table
        return self.olsr_dijkstra(source_id, destination_id)
    
    def olsr_update_topology(self):
        """Update OLSR topology database"""
        current_time = self.discrete_simulator.current_simulation_time
        
        for node_id, node in self.nodes.items():
            # Generate Hello messages
            if current_time - node.last_hello_time > self.olsr_params['hello_interval']:
                self.olsr_send_hello(node_id)
                node.last_hello_time = current_time
            
            # Update MPR sets
            node.select_mpr_set(self.nodes)
            
            # Generate TC messages (only MPRs)
            if node_id in self.get_all_mpr_selectors():
                self.olsr_send_tc(node_id)
    
    def olsr_send_hello(self, node_id):
        """Send OLSR Hello message - DISCRETE VERSION"""
        node = self.nodes[node_id]
        current_time = self.discrete_simulator.current_simulation_time
        
        hello = HelloMessage(
            sender=node_id,
            neighbors=node.neighbors.copy(),
            timestamp=current_time  # Use simulation time
        )
        
        # Track routing overhead
        self.metrics[self.routing_protocol].routing_messages_sent += 1
        node.stats['hello_messages_sent'] += 1
        
        # Process hello at neighbors
        for neighbor_id in node.neighbors:
            if neighbor_id in self.nodes:
                neighbor = self.nodes[neighbor_id]
                neighbor.add_neighbor(node_id, current_time + self.olsr_params['neighbor_hold_time'])
    
    def olsr_send_tc(self, node_id):
        """Send OLSR Topology Control message"""
        node = self.nodes[node_id]
        node.sequence_number += 1
        
        lsa = LSAMessage(
            originator=node_id,
            sequence_number=node.sequence_number,
            neighbors=node.mpr_selector_set.copy(),
            timestamp=self.discrete_simulator.current_simulation_time
        )
        
        # Flood TC message through MPRs
        self.olsr_flood_tc(lsa, node_id, set())
    
    def olsr_flood_tc(self, lsa, current_node_id, processed):
        """Flood TC message through MPR network"""
        processed.add(current_node_id)
        current_node = self.nodes[current_node_id]
        
        # Store LSA in topology database
        current_node.lsa_database[lsa.originator] = lsa
        
        # Forward to MPRs
        for mpr_id in current_node.mpr_set:
            if mpr_id not in processed and mpr_id in self.nodes:
                self.metrics[self.routing_protocol].routing_messages_sent += 1
                self.olsr_flood_tc(lsa, mpr_id, processed.copy())
    
    def get_all_mpr_selectors(self):
        """Get all nodes that have selected this node as MPR"""
        mpr_selectors = set()
        for node_id, node in self.nodes.items():
            mpr_selectors.update(node.mpr_selector_set)
        return mpr_selectors
    
    def olsr_dijkstra(self, source_id, destination_id):
        """Dijkstra's algorithm using OLSR topology database"""
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[source_id] = 0
        previous = {}
        unvisited = set(self.nodes.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            if distances[current] == float('inf'):
                break
            
            unvisited.remove(current)
            
            if current == destination_id:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = previous.get(current)
                return [path[::-1]]
            
            # Check neighbors from topology database
            current_node = self.nodes[current]
            neighbors = current_node.neighbors.copy()
            
            # Add neighbors from LSA database
            for lsa in current_node.lsa_database.values():
                if lsa.originator == current:
                    neighbors.update(lsa.neighbors)
            
            for neighbor in neighbors:
                if neighbor in unvisited:
                    new_distance = distances[current] + 1
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        return None
    
    # DSR Routing Protocol Implementation (existing code)
    def dsr_find_route(self, source_id, destination_id):
        """DSR Source Routing"""
        # Simple DSR implementation using source routing
        return self.dsr_route_discovery(source_id, destination_id)
    
    def dsr_route_discovery(self, source_id, destination_id):
        """DSR Route Discovery using flooding with route caching"""
        paths = []
        visited = set()
        
        def dsr_flood(current_id, path, target_id):
            if current_id in visited:
                return
            visited.add(current_id)
            
            if current_id == target_id:
                paths.append(path)
                return
            
            current_node = self.nodes[current_id]
            for neighbor_id in current_node.neighbors:
                if neighbor_id not in path:  # Avoid loops
                    self.metrics[self.routing_protocol].routing_messages_sent += 1
                    dsr_flood(neighbor_id, path + [neighbor_id], target_id)
        
        dsr_flood(source_id, [source_id], destination_id)
        return paths if paths else None
    
    # ZRP Routing Protocol Implementation
    def zrp_find_route(self, source_id, destination_id):
        """ZRP - Hybrid protocol combining proactive and reactive"""
        zone_radius = 2  # hops
        
        # Check if destination in zone (proactive)
        if self.is_in_zone(source_id, destination_id, zone_radius):
            return self.proactive_route_lookup(source_id, destination_id)
        else:
            # Use reactive discovery (bordercasting)
            return self.zrp_bordercast(source_id, destination_id)
    
    def is_in_zone(self, source_id, destination_id, zone_radius):
        """Check if destination is within zone radius"""
        if source_id == destination_id:
            return True
        
        # Use BFS to find shortest path
        visited = set()
        queue = [(source_id, 0)]
        
        while queue:
            current_id, hop_count = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id == destination_id:
                return hop_count <= zone_radius
            
            if hop_count >= zone_radius:
                continue
            
            current_node = self.nodes[current_id]
            for neighbor_id in current_node.neighbors:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, hop_count + 1))
        
        return False
    
    def proactive_route_lookup(self, source_id, destination_id):
        """Proactive route lookup within zone"""
        # Use Dijkstra's algorithm within zone
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[source_id] = 0
        previous = {}
        unvisited = set(self.nodes.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            if distances[current] == float('inf'):
                break
            
            unvisited.remove(current)
            
            if current == destination_id:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = previous.get(current)
                return [path[::-1]]
            
            # Check neighbors
            current_node = self.nodes[current]
            for neighbor in current_node.neighbors:
                if neighbor in unvisited:
                    new_distance = distances[current] + 1
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        return None
    
    def zrp_bordercast(self, source_id, destination_id):
        """ZRP bordercast to zone border nodes"""
        # Find border nodes (nodes at zone boundary)
        border_nodes = self.find_border_nodes(source_id, 2)
        
        # Use reactive discovery through border nodes
        paths = []
        for border_node in border_nodes:
            # Use AODV-like discovery from border node
            sub_paths = self.aodv_route_discovery(border_node, destination_id)
            if sub_paths:
                # Combine paths
                for path in sub_paths:
                    full_path = [source_id] + path[1:]  # Remove duplicate border node
                    paths.append(full_path)
        
        return paths if paths else None
    
    def find_border_nodes(self, source_id, zone_radius):
        """Find nodes at zone boundary"""
        border_nodes = set()
        visited = set()
        queue = [(source_id, 0)]
        
        while queue:
            current_id, hop_count = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if hop_count == zone_radius:
                border_nodes.add(current_id)
                continue
            
            current_node = self.nodes[current_id]
            for neighbor_id in current_node.neighbors:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, hop_count + 1))
        
        return border_nodes
    
    # Flooding Implementation (baseline)
    def flooding_routing(self, packet, visited_nodes=None):
        """Simple flooding for comparison"""
        if visited_nodes is None:
            visited_nodes = set()
        
        current_node = self.nodes[packet.path[-1]]
        visited_nodes.add(current_node.id)
        
        if packet.destination in current_node.neighbors:
            packet.path.append(packet.destination)
            packet.hop_count += 1
            return [packet.path]
        
        paths = []
        for neighbor_id in current_node.neighbors:
            if neighbor_id not in visited_nodes:
                new_packet = Packet(packet.source, packet.destination, packet.data)
                new_packet.path = packet.path + [neighbor_id]
                new_packet.hop_count = packet.hop_count + 1
                
                if neighbor_id == packet.destination:
                    paths.append(new_packet.path)
                else:
                    self.metrics[self.routing_protocol].routing_messages_sent += 1
                    sub_paths = self.flooding_routing(new_packet, visited_nodes.copy())
                    if sub_paths:
                        paths.extend(sub_paths)
        
        return paths
    
    def send_packet(self, source_id, destination_id, data="Test Data", packet_size=64):
        """Send packet - initiate route discovery if needed"""
        if source_id not in self.nodes or destination_id not in self.nodes:
            return False
        
        # Check source node energy before sending
        source_node = self.nodes[source_id]
        if not source_node.energy.is_alive():
            # Node is dead - drop packet
            current_metrics = self.metrics[self.routing_protocol]
            current_metrics.total_packets_dropped += 1
            source_node.stats['packets_dropped'] += 1
            return False
        
        # Create packet with current simulation time
        packet = Packet(source_id, destination_id, data, size=packet_size)
        packet.timestamp = self.discrete_simulator.current_simulation_time
        
        # Update metrics
        current_metrics = self.metrics[self.routing_protocol]
        current_metrics.total_packets_sent += 1
        source_node.stats['packets_sent'] += 1
        
        # Check if route exists in routing table
        if destination_id in source_node.routing_table:
            route = source_node.routing_table[destination_id]
            if self.discrete_simulator.current_simulation_time < route.expiry_time:
                # Route exists
                self._schedule_packet_transmission(packet, route.next_hop)
                return True
        
        # No route - protocol-specific route discovery
        if self.routing_protocol == "AODV":
            # Buffer the packet for this destination
            if destination_id not in source_node.packet_buffer:
                source_node.packet_buffer[destination_id] = []
                # Set timeout for this destination (5 seconds from now)
                source_node.buffer_timeouts[destination_id] = self.discrete_simulator.current_simulation_time + 5.0
                # Schedule timeout event
                timeout_event = Event(
                    event_type="BUFFER_TIMEOUT",
                    timestamp=source_node.buffer_timeouts[destination_id],
                    source_node=source_id,
                    data={'destination_id': destination_id}
                )
                self.discrete_simulator.schedule_event(timeout_event)
            
            source_node.packet_buffer[destination_id].append(packet)
            
            # Check if route discovery is already in progress
            if not self._is_route_discovery_in_progress(source_id, destination_id):
                self.initiate_route_discovery(source_id, destination_id, packet)
        elif self.routing_protocol == "OLSR":
            # OLSR proactive - routing table zaten dolu olmalı
            # Eğer yoksa packet drop et
            current_metrics.total_packets_dropped += 1
            self.nodes[source_id].stats['packets_dropped'] += 1
            return False
        elif self.routing_protocol == "DSR":
            self.initiate_dsr_route_discovery(source_id, destination_id, packet)
        elif self.routing_protocol == "ZRP":
            if self.is_in_zone(source_id, destination_id, 2):
                # Zone içinde - routing table kullan
                if destination_id in source_node.routing_table:
                    route = source_node.routing_table[destination_id]
                    self._schedule_packet_transmission(packet, route.next_hop)
                    return True
                else:
                    # Zone içinde route yok - drop
                    current_metrics.total_packets_dropped += 1
                    self.nodes[source_id].stats['packets_dropped'] += 1
                    return False
            else:
                self.initiate_zrp_bordercast(source_id, destination_id, packet)
        elif self.routing_protocol == "FLOODING":
            # Flooding - direct transmission
            self._schedule_packet_forwarding(packet, [source_id, destination_id])
        
        return True
    
    def _find_route_for_protocol(self, source_id, destination_id):
        """Find route based on current protocol"""
        if self.routing_protocol == "AODV":
            return self.aodv_find_route(source_id, destination_id)
        elif self.routing_protocol == "OLSR":
            return self.olsr_find_route(source_id, destination_id)
        elif self.routing_protocol == "DSR":
            return self.dsr_find_route(source_id, destination_id)
        elif self.routing_protocol == "FLOODING":
            packet = Packet(source_id, destination_id, "temp")
            return self.flooding_routing(packet)
        elif self.routing_protocol == "ZRP":
            return self.zrp_find_route(source_id, destination_id)
        return None
    
    def _schedule_packet_forwarding(self, packet, path):
        """Schedule packet delivery events along the path"""
        current_time = self.discrete_simulator.current_simulation_time
        
        # Schedule arrival at each hop
        for i in range(len(path)):
            node_id = path[i]
            
            # Calculate cumulative delay
            if i == 0:
                arrival_time = current_time
            else:
                delay = self.calculate_hop_delay(packet.size)
                arrival_time = current_time + (i * delay)
            
            # Check for collision
            if i > 0 and self.check_collision(path[i-1], node_id, arrival_time):
                self.discrete_simulator.statistics['collision_count'] += 1
                arrival_time += random.uniform(0.001, 0.01)
            
            # Schedule receive event
            if i == len(path) - 1:
                # Final destination - mark as delivered
                event = Event(
                    event_type="PACKET_DELIVERED",
                    timestamp=arrival_time,
                    source_node=node_id,
                    data={'packet': packet}
                )
            else:
                # Intermediate hop - forward
                event = Event(
                    event_type="PACKET_FORWARD",
                    timestamp=arrival_time,
                    source_node=node_id,
                    data={'packet': packet, 'next_hop': path[i+1]}
                )
            
            self.discrete_simulator.schedule_event(event)
    
    
    
    def calculate_hop_delay(self, packet_size):
        """Calculate delay for one hop"""
        propagation_delay = self.network_params['propagation_delay']
        transmission_delay = self.network_params['transmission_delay'] * packet_size
        processing_delay = self.network_params['processing_delay']
        
        return propagation_delay + transmission_delay + processing_delay
    
    def check_collision(self, source, destination, arrival_time):
        """Check for collision at destination with spatial and temporal proximity"""
        # Get source and destination node objects
        source_node = self.nodes[source]
        destination_node = self.nodes[destination]
        collision_window = 0.001  # 1ms
        
        # Find all nodes within transmission range of destination (potential interferers)
        potential_interferers = set()
        for node_id, node in self.nodes.items():
            if node_id != destination and destination_node.can_communicate_with(node):
                potential_interferers.add(node_id)
        
        # Check event queue for transmissions in the time window
        for event in self.discrete_simulator.event_queue:
            if event.event_type in ["PACKET_SEND", "PACKET_FORWARD", "PACKET_RECEIVE"]:
                # Check temporal proximity (within ±1ms)
                if abs(event.timestamp - arrival_time) < collision_window:
                    other_sender = event.source_node
                    
                    # Check spatial proximity - sender must be within interfering range of destination
                    if (other_sender in self.nodes and 
                        other_sender in potential_interferers and
                        other_sender != source):  # Don't collide with self
                        
                        # Additional check: ensure the other sender can actually reach the destination
                        other_node = self.nodes[other_sender]
                        if destination_node.can_communicate_with(other_node):
                            return True
        
        # Also check with probability for realistic simulation (1% random collision)
        return random.random() < self.network_params['collision_probability']
    
    def forward_packet_discrete(self, packet, current_node):
        """Forward packet to next hop in discrete-event simulation"""
        if current_node not in self.nodes:
            return
        
        node = self.nodes[current_node]
        
        # Find next hop based on routing protocol
        next_hop = None
        if self.routing_protocol == "AODV" and packet.destination in node.routing_table:
            route = node.routing_table[packet.destination]
            if self.discrete_simulator.current_simulation_time < route.expiry_time:
                next_hop = route.next_hop
        elif self.routing_protocol == "OLSR":
            # Use OLSR routing table
            if packet.destination in node.routing_table:
                route = node.routing_table[packet.destination]
                next_hop = route.next_hop
        elif self.routing_protocol == "DSR":
            # DSR uses source routing - check if path is valid and not exhausted
            if packet.hop_count >= len(packet.path):
                # Path exhausted - drop packet
                packet.dropped = True
                packet.drop_reason = "DSR path exhausted"
                current_metrics = self.metrics[self.routing_protocol]
                current_metrics.total_packets_dropped += 1
                node.stats['packets_dropped'] += 1
                return
            
            # Use next hop from source route
            next_hop = packet.path[packet.hop_count]
        elif self.routing_protocol == "FLOODING":
            # Flood to all neighbors, excluding:
            # 1. Sender (packet.source)
            # 2. Nodes already in packet.path
            # 3. Nodes that have already processed this packet_id
            for neighbor_id in node.neighbors:
                neighbor_node = self.nodes[neighbor_id]
                
                # Check if neighbor has already seen this packet
                has_seen_packet = (hasattr(neighbor_node, 'seen_packets') and 
                                 packet.packet_id in neighbor_node.seen_packets)
                
                if (neighbor_id != packet.source and 
                    neighbor_id not in packet.path and 
                    not has_seen_packet):
                    
                    # Schedule packet to each neighbor
                    delay = self.calculate_hop_delay(packet.size)
                    arrival_time = self.discrete_simulator.current_simulation_time + delay
                    
                    receive_event = Event(
                        event_type="PACKET_RECEIVE",
                        timestamp=arrival_time,
                        source_node=neighbor_id,
                        data={'packet': packet}
                    )
                    self.discrete_simulator.schedule_event(receive_event)
            return
        
        if next_hop and next_hop in node.neighbors:
            # Check node energy before forwarding
            if not node.energy.is_alive():
                # Node is dead - drop packet
                packet.dropped = True
                packet.drop_reason = "Node out of energy"
                current_metrics = self.metrics[self.routing_protocol]
                current_metrics.total_packets_dropped += 1
                node.stats['packets_dropped'] += 1
                return
            
            # Consume TX energy for forwarding
            transmission_time = self.calculate_hop_delay(packet.size)
            node.energy.consume_tx(packet.size, transmission_time)
            
            # Check if node died from energy consumption
            if node.energy.check_and_handle_death(current_node, self):
                return
            
            # Update packet path and hop count
            packet.path.append(next_hop)
            packet.hop_count += 1
            
            # Calculate delay and schedule next hop
            delay = self.calculate_hop_delay(packet.size)
            arrival_time = self.discrete_simulator.current_simulation_time + delay
            
            # Check for collision
            if self.check_collision(current_node, next_hop, arrival_time):
                self.discrete_simulator.statistics['collision_count'] += 1
                arrival_time += random.uniform(0.001, 0.01)  # Random backoff
            
            # Schedule packet receive event
            receive_event = Event(
                event_type="PACKET_RECEIVE",
                timestamp=arrival_time,
                source_node=next_hop,
                data={'packet': packet}
            )
            self.discrete_simulator.schedule_event(receive_event)
            
            # Update forwarding stats
            node.stats['packets_forwarded'] += 1
        else:
            # No route found - packet dropped
            packet.dropped = True
            packet.drop_reason = "No route to next hop"
            current_metrics = self.metrics[self.routing_protocol]
            current_metrics.total_packets_dropped += 1
            node.stats['packets_dropped'] += 1
    
    def get_network_stats(self):
        """Get comprehensive network statistics"""
        total_nodes = len(self.nodes)
        total_connections = sum(len(node.neighbors) for node in self.nodes.values()) // 2
        
        # Calculate network connectivity
        connectivity = 0
        if total_nodes > 1:
            max_connections = total_nodes * (total_nodes - 1) // 2
            connectivity = (total_connections / max_connections) * 100
        
        stats = {
            'total_nodes': total_nodes,
            'total_connections': total_connections,
            'network_connectivity': connectivity,
            'current_protocol': self.routing_protocol
        }
        
        # Add protocol-specific metrics
        current_metrics = self.metrics[self.routing_protocol]
        stats.update({
            'packets_sent': current_metrics.total_packets_sent,
            'packets_delivered': current_metrics.total_packets_delivered,
            'packets_dropped': current_metrics.total_packets_dropped,
            'delivery_ratio': current_metrics.calculate_delivery_ratio(),
            'average_delay': current_metrics.calculate_average_delay(),
            'routing_overhead': current_metrics.calculate_routing_overhead(),
            'average_hop_count': current_metrics.calculate_average_hop_count(),
            'routing_messages_sent': current_metrics.routing_messages_sent
        })
        
        return stats
    
    def get_protocol_comparison(self):
        """Get performance comparison between protocols"""
        comparison = {}
        for protocol, metrics in self.metrics.items():
            if metrics.total_packets_sent > 0:
                comparison[protocol] = {
                    'delivery_ratio': metrics.calculate_delivery_ratio(),
                    'average_delay': metrics.calculate_average_delay() * 1000,  # ms
                    'routing_overhead': metrics.calculate_routing_overhead(),
                    'average_hop_count': metrics.calculate_average_hop_count(),
                    'packets_sent': metrics.total_packets_sent,
                    'packets_delivered': metrics.total_packets_delivered
                }
        return comparison
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        for metrics in self.metrics.values():
            metrics.reset()
        
        for node in self.nodes.values():
            for key in node.stats:
                node.stats[key] = 0
    
    def run_test_suite(self):
        """Run comprehensive test suite to verify simulator functionality"""
        print("=" * 60)
        print("MANET SIMULATOR TEST SUITE")
        print("=" * 60)
        
        test_results = []
        
        # Test 1: AODV Single Packet
        print("\n1. Testing AODV Single Packet Delivery...")
        result = self.test_aodv_single_packet()
        test_results.append(("AODV Single Packet", result))
        
        # Test 2: AODV Multiple Paths
        print("\n2. Testing AODV Multiple Paths...")
        result = self.test_aodv_multiple_paths()
        test_results.append(("AODV Multiple Paths", result))
        
        # Test 3: OLSR After Convergence
        print("\n3. Testing OLSR After Convergence...")
        result = self.test_olsr_after_convergence()
        test_results.append(("OLSR After Convergence", result))
        
        # Test 4: Flooding Loop Prevention
        print("\n4. Testing Flooding Loop Prevention...")
        result = self.test_flooding_loop_prevention()
        test_results.append(("Flooding Loop Prevention", result))
        
        # Test 5: Mobility Route Break
        print("\n5. Testing Mobility Route Break...")
        result = self.test_mobility_route_break()
        test_results.append(("Mobility Route Break", result))
        
        # Test 6: Energy Depletion
        print("\n6. Testing Energy Depletion...")
        result = self.test_energy_depletion()
        test_results.append(("Energy Depletion", result))
        
        # Test 7: Route Expiry
        print("\n7. Testing Route Expiry...")
        result = self.test_route_expiry()
        test_results.append(("Route Expiry", result))
        
        # Test 8: DSR Source Routing
        print("\n8. Testing DSR Source Routing...")
        result = self.test_dsr_source_routing()
        test_results.append(("DSR Source Routing", result))
        
        # Print Summary
        print("\n" + "=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, result in test_results:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"{test_name}: {status}")
            if result['passed']:
                passed += 1
            else:
                failed += 1
                print(f"  Reason: {result['reason']}")
        
        print(f"\nTotal: {passed} passed, {failed} failed")
        print("=" * 60)
        
        return test_results
    
    def test_aodv_single_packet(self):
        """Test AODV single packet delivery"""
        # Setup: 5 nodes in line topology
        self.setup_network(5, "AODV")
        self.setup_line_topology()
        
        # Send single packet
        self.send_packet("node_0", "node_4", "Test Data", 64)
        
        # Run simulation for 10 seconds
        self.run_discrete_simulation(10.0)
        
        # Check results
        metrics = self.metrics["AODV"]
        delivery_ratio = metrics.get_delivery_ratio()
        
        result = {
            'passed': delivery_ratio > 0.8,  # Expect high delivery ratio
            'delivery_ratio': delivery_ratio,
            'packets_sent': metrics.total_packets_sent,
            'packets_delivered': metrics.total_packets_delivered,
            'packets_dropped': metrics.total_packets_dropped,
            'routing_overhead': metrics.routing_messages_sent
        }
        
        if not result['passed']:
            result['reason'] = f"Low delivery ratio: {delivery_ratio:.2%}"
        
        print(f"  Delivery Ratio: {delivery_ratio:.2%}")
        print(f"  Packets Sent: {metrics.total_packets_sent}")
        print(f"  Packets Delivered: {metrics.total_packets_delivered}")
        print(f"  Routing Overhead: {metrics.routing_messages_sent}")
        
        return result
    
    def test_aodv_multiple_paths(self):
        """Test AODV with multiple node pairs"""
        # Setup: 6 nodes in grid topology
        self.setup_network(6, "AODV")
        self.setup_grid_topology()
        
        # Send packets between multiple pairs
        pairs = [("node_0", "node_5"), ("node_1", "node_4"), ("node_2", "node_3")]
        for source, dest in pairs:
            self.send_packet(source, dest, f"Data from {source} to {dest}", 64)
        
        # Run simulation
        self.run_discrete_simulation(15.0)
        
        # Check results
        metrics = self.metrics["AODV"]
        delivery_ratio = metrics.get_delivery_ratio()
        
        result = {
            'passed': delivery_ratio > 0.7,  # Expect good delivery ratio
            'delivery_ratio': delivery_ratio,
            'packets_sent': metrics.total_packets_sent,
            'packets_delivered': metrics.total_packets_delivered,
            'routing_overhead': metrics.routing_messages_sent
        }
        
        if not result['passed']:
            result['reason'] = f"Low delivery ratio: {delivery_ratio:.2%}"
        
        print(f"  Delivery Ratio: {delivery_ratio:.2%}")
        print(f"  Packets Sent: {metrics.total_packets_sent}")
        print(f"  Packets Delivered: {metrics.total_packets_delivered}")
        
        return result
    
    def test_olsr_after_convergence(self):
        """Test OLSR after TC message convergence"""
        # Setup: 5 nodes in mesh topology
        self.setup_network(5, "OLSR")
        self.setup_mesh_topology()
        
        # Wait for convergence (TC messages)
        self.run_discrete_simulation(5.0)
        
        # Send packets after convergence
        self.send_packet("node_0", "node_4", "OLSR Test Data", 64)
        self.send_packet("node_1", "node_3", "OLSR Test Data 2", 64)
        
        # Run simulation
        self.run_discrete_simulation(10.0)
        
        # Check results
        metrics = self.metrics["OLSR"]
        delivery_ratio = metrics.get_delivery_ratio()
        
        result = {
            'passed': delivery_ratio > 0.8,
            'delivery_ratio': delivery_ratio,
            'packets_sent': metrics.total_packets_sent,
            'packets_delivered': metrics.total_packets_delivered,
            'tc_messages': sum(node.stats.get('tc_messages_sent', 0) for node in self.nodes.values())
        }
        
        if not result['passed']:
            result['reason'] = f"Low delivery ratio: {delivery_ratio:.2%}"
        
        print(f"  Delivery Ratio: {delivery_ratio:.2%}")
        print(f"  TC Messages: {result['tc_messages']}")
        
        return result
    
    def test_flooding_loop_prevention(self):
        """Test flooding loop prevention with packet_id"""
        # Setup: 6 nodes in dense topology
        self.setup_network(6, "FLOODING")
        self.setup_dense_topology()
        
        # Send packet
        self.send_packet("node_0", "node_5", "Flooding Test", 64)
        
        # Run simulation
        self.run_discrete_simulation(8.0)
        
        # Check for loop prevention
        metrics = self.metrics["FLOODING"]
        total_transmissions = sum(node.stats['packets_forwarded'] for node in self.nodes.values())
        expected_max = len(self.nodes) * 2  # Should not exceed reasonable limit
        
        result = {
            'passed': total_transmissions < expected_max,
            'total_transmissions': total_transmissions,
            'expected_max': expected_max,
            'delivery_ratio': metrics.get_delivery_ratio()
        }
        
        if not result['passed']:
            result['reason'] = f"Too many transmissions: {total_transmissions} (expected < {expected_max})"
        
        print(f"  Total Transmissions: {total_transmissions}")
        print(f"  Delivery Ratio: {metrics.get_delivery_ratio():.2%}")
        
        return result
    
    def test_mobility_route_break(self):
        """Test route invalidation due to mobility"""
        # Setup: 4 nodes in line
        self.setup_network(4, "AODV")
        self.setup_line_topology()
        
        # Send packet
        self.send_packet("node_0", "node_3", "Mobility Test", 64)
        
        # Run for 2 seconds
        self.run_discrete_simulation(2.0)
        
        # Move nodes apart to break route
        self.nodes["node_1"].x = 1000  # Move far away
        self.nodes["node_2"].x = 1000
        
        # Continue simulation
        self.run_discrete_simulation(5.0)
        
        # Check for route invalidation
        metrics = self.metrics["AODV"]
        rerr_messages = sum(node.stats.get('rerr_messages_sent', 0) for node in self.nodes.values())
        
        result = {
            'passed': rerr_messages > 0,  # Should have RERR messages
            'rerr_messages': rerr_messages,
            'delivery_ratio': metrics.get_delivery_ratio()
        }
        
        if not result['passed']:
            result['reason'] = "No RERR messages sent for route invalidation"
        
        print(f"  RERR Messages: {rerr_messages}")
        print(f"  Delivery Ratio: {metrics.get_delivery_ratio():.2%}")
        
        return result
    
    def test_energy_depletion(self):
        """Test energy depletion and node death"""
        # Setup: 3 nodes with low energy
        self.setup_network(3, "AODV")
        self.setup_line_topology()
        
        # Set low energy for middle node
        self.nodes["node_1"].energy.current_energy = 0.1
        
        # Send packet through middle node
        self.send_packet("node_0", "node_2", "Energy Test", 64)
        
        # Run simulation
        self.run_discrete_simulation(5.0)
        
        # Check if middle node died
        middle_node_dead = not self.nodes["node_1"].energy.is_alive()
        metrics = self.metrics["AODV"]
        
        result = {
            'passed': middle_node_dead and metrics.total_packets_dropped > 0,
            'middle_node_dead': middle_node_dead,
            'packets_dropped': metrics.total_packets_dropped,
            'delivery_ratio': metrics.get_delivery_ratio()
        }
        
        if not result['passed']:
            result['reason'] = f"Middle node not dead: {middle_node_dead}, Packets dropped: {metrics.total_packets_dropped}"
        
        print(f"  Middle Node Dead: {middle_node_dead}")
        print(f"  Packets Dropped: {metrics.total_packets_dropped}")
        
        return result
    
    def test_route_expiry(self):
        """Test route expiry and rediscovery"""
        # Setup: 4 nodes with short route expiry
        self.setup_network(4, "AODV")
        self.setup_line_topology()
        
        # Set short route expiry time
        for node in self.nodes.values():
            for route in node.routing_table.values():
                route.expiry_time = self.discrete_simulator.current_simulation_time + 2.0
        
        # Send packet
        self.send_packet("node_0", "node_3", "Route Expiry Test", 64)
        
        # Run simulation
        self.run_discrete_simulation(8.0)
        
        # Check for route expiry and rediscovery
        metrics = self.metrics["AODV"]
        routes_expired = sum(node.stats.get('routes_expired', 0) for node in self.nodes.values())
        route_rediscoveries = sum(node.stats.get('route_rediscoveries', 0) for node in self.nodes.values())
        
        result = {
            'passed': routes_expired > 0 and route_rediscoveries > 0,
            'routes_expired': routes_expired,
            'route_rediscoveries': route_rediscoveries,
            'delivery_ratio': metrics.get_delivery_ratio()
        }
        
        if not result['passed']:
            result['reason'] = f"No route expiry: {routes_expired}, No rediscovery: {route_rediscoveries}"
        
        print(f"  Routes Expired: {routes_expired}")
        print(f"  Route Rediscoveries: {route_rediscoveries}")
        
        return result
    
    def test_dsr_source_routing(self):
        """Test DSR source routing"""
        # Setup: 5 nodes in line
        self.setup_network(5, "DSR")
        self.setup_line_topology()
        
        # Send packet
        self.send_packet("node_0", "node_4", "DSR Test", 64)
        
        # Run simulation
        self.run_discrete_simulation(10.0)
        
        # Check DSR-specific metrics
        metrics = self.metrics["DSR"]
        cache_hits = sum(node.stats.get('dsr_cache_hits', 0) for node in self.nodes.values())
        cache_misses = sum(node.stats.get('dsr_cache_misses', 0) for node in self.nodes.values())
        
        result = {
            'passed': metrics.get_delivery_ratio() > 0.8 and cache_misses > 0,
            'delivery_ratio': metrics.get_delivery_ratio(),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'packets_sent': metrics.total_packets_sent
        }
        
        if not result['passed']:
            result['reason'] = f"Low delivery ratio: {metrics.get_delivery_ratio():.2%} or no cache misses: {cache_misses}"
        
        print(f"  Delivery Ratio: {metrics.get_delivery_ratio():.2%}")
        print(f"  Cache Hits: {cache_hits}")
        print(f"  Cache Misses: {cache_misses}")
        
        return result
    
    def setup_line_topology(self):
        """Setup nodes in a line"""
        for i, (node_id, node) in enumerate(self.nodes.items()):
            node.x = i * 100
            node.y = 0
    
    def setup_grid_topology(self):
        """Setup nodes in a 2x3 grid"""
        positions = [(0, 0), (100, 0), (200, 0), (0, 100), (100, 100), (200, 100)]
        for i, (node_id, node) in enumerate(self.nodes.items()):
            if i < len(positions):
                node.x, node.y = positions[i]
    
    def setup_mesh_topology(self):
        """Setup nodes in a mesh (all connected)"""
        for i, (node_id, node) in enumerate(self.nodes.items()):
            node.x = i * 50
            node.y = i * 50
    
    def setup_dense_topology(self):
        """Setup nodes in a dense topology"""
        for i, (node_id, node) in enumerate(self.nodes.items()):
            node.x = i * 30
            node.y = i * 30

class MANETSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gelişmiş MANET Routing Simulator - Mobility Destekli")
        
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
        control_frame = ttk.LabelFrame(main_frame, text="Kontroller", width=280)
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
        ttk.Label(scrollable_frame, text="Ağ Yönetimi:", font=("Arial", 9, "bold")).pack(pady=(10, 3))
        ttk.Button(scrollable_frame, text="Rastgele Node", command=self.add_random_node, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Seçili Sil", command=self.remove_selected_node, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Temizle", command=self.clear_network, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Örnek Ağ", command=self.create_sample_network, width=18).pack(pady=1)
        
        # MOBILITY CONTROLS - NEW SECTION
        ttk.Label(scrollable_frame, text="Mobility:", font=("Arial", 9, "bold")).pack(pady=(15, 3))
        
        # Mobility enable/disable
        self.mobility_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(scrollable_frame, text="Mobility Aktif", 
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
        speed_frame = ttk.LabelFrame(scrollable_frame, text="Hız")
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
        
        ttk.Button(speed_frame, text="Uygula", 
                  command=self.apply_mobility_params, width=15).pack(pady=2)
        
        # Protocol selection
        ttk.Label(scrollable_frame, text="Protokol:", font=("Arial", 9, "bold")).pack(pady=(15, 3))
        self.protocol_var = tk.StringVar(value="AODV")
        protocol_frame = ttk.Frame(scrollable_frame)
        protocol_frame.pack(pady=2)
        
        protocols = ["AODV", "OLSR", "DSR", "FLOODING", "ZRP"]
        for protocol in protocols:
            ttk.Radiobutton(protocol_frame, text=protocol, variable=self.protocol_var, 
                           value=protocol, command=self.change_protocol).pack(anchor="w")
        
        # Packet sending
        ttk.Label(scrollable_frame, text="Paket:", font=("Arial", 9, "bold")).pack(pady=(15, 3))
        self.source_var = tk.StringVar()
        self.dest_var = tk.StringVar()
        
        ttk.Label(scrollable_frame, text="Kaynak:").pack(pady=0)
        source_combo = ttk.Combobox(scrollable_frame, textvariable=self.source_var, width=18)
        source_combo.pack(pady=1)
        
        ttk.Label(scrollable_frame, text="Hedef:").pack(pady=0)
        dest_combo = ttk.Combobox(scrollable_frame, textvariable=self.dest_var, width=18)
        dest_combo.pack(pady=1)
        
        ttk.Label(scrollable_frame, text="Sayı:").pack(pady=0)
        self.packet_count_var = tk.StringVar(value="1")
        ttk.Entry(scrollable_frame, textvariable=self.packet_count_var, width=18).pack(pady=1)
        
        self.source_combo = source_combo
        self.dest_combo = dest_combo
        
        ttk.Button(scrollable_frame, text="Gönder", command=self.send_packet, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Toplu Gönder", command=self.send_multiple_packets, width=18).pack(pady=1)
        
        # Discrete-Event Simulation Controls
        ttk.Label(scrollable_frame, text="Discrete-Event:", font=("Arial", 9, "bold")).pack(pady=(15, 3))
        
        ttk.Button(scrollable_frame, text="Başlat", command=self.start_discrete_simulation, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Durdur", command=self.stop_discrete_simulation, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Adım Adım", command=self.step_simulation, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Event Temizle", command=self.clear_events, width=18).pack(pady=1)
        
        # Simulation duration control
        ttk.Label(scrollable_frame, text="Süre (s):").pack(pady=(5, 0))
        self.simulation_duration_var = tk.StringVar(value="10.0")
        duration_entry = ttk.Entry(scrollable_frame, textvariable=self.simulation_duration_var, width=18)
        duration_entry.pack(pady=1)
        
        # Performance testing
        ttk.Label(scrollable_frame, text="Test:", font=("Arial", 9, "bold")).pack(pady=(15, 3))
        ttk.Button(scrollable_frame, text="Protokol Karşılaştır", command=self.run_protocol_comparison, width=18).pack(pady=1)
        ttk.Button(scrollable_frame, text="Metrikleri Sıfırla", command=self.reset_metrics, width=18).pack(pady=1)
        
        # Right panel - Simulation and stats
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Top: Simulation area
        sim_frame = ttk.LabelFrame(right_frame, text="Simülasyon Alanı")
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
        
        ttk.Button(anim_frame, text="Son Paket Rotası", command=self.animate_last_packet).pack(side=tk.LEFT, padx=5)
        ttk.Button(anim_frame, text="Ağı Yenile", command=self.redraw_network).pack(side=tk.LEFT, padx=5)
        ttk.Button(anim_frame, text="Topoloji Göster", command=self.show_topology_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(anim_frame, text="Mobility Paths", command=self.show_mobility_paths).pack(side=tk.LEFT, padx=5)
        
        # Bottom: Statistics
        stats_frame = ttk.LabelFrame(right_frame, text="İstatistikler ve Performans Metrikleri")
        stats_frame.pack(fill=tk.BOTH, expand=False)
        stats_frame.configure(height=250)
        
        # Notebook for different stats views
        self.stats_notebook = ttk.Notebook(stats_frame)
        self.stats_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current stats tab
        current_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(current_tab, text="Mevcut İstatistikler")
        
        self.stats_text = tk.Text(current_tab, height=8, width=80, font=("Courier", 10))
        scrollbar1 = ttk.Scrollbar(current_tab, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar1.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Protocol comparison tab
        comparison_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(comparison_tab, text="Protokol Karşılaştırması")
        
        self.comparison_text = tk.Text(comparison_tab, height=8, width=80, font=("Courier", 10))
        scrollbar2 = ttk.Scrollbar(comparison_tab, orient="vertical", command=self.comparison_text.yview)
        self.comparison_text.configure(yscrollcommand=scrollbar2.set)
        self.comparison_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mobility stats tab - NEW
        mobility_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(mobility_tab, text="Mobility İstatistikleri")
        
        self.mobility_text = tk.Text(mobility_tab, height=8, width=80, font=("Courier", 10))
        scrollbar3 = ttk.Scrollbar(mobility_tab, orient="vertical", command=self.mobility_text.yview)
        self.mobility_text.configure(yscrollcommand=scrollbar3.set)
        self.mobility_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar3.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Node details tab
        details_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(details_tab, text="Node Detayları")
        
        self.details_text = tk.Text(details_tab, height=8, width=80, font=("Courier", 10))
        scrollbar4 = ttk.Scrollbar(details_tab, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=scrollbar4.set)
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar4.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Update button
        ttk.Button(stats_frame, text="İstatistikleri Güncelle", command=self.update_all_stats).pack(pady=5)
    
    # MOBILITY CONTROL METHODS - NEW
    def toggle_mobility(self):
        """Enable/disable mobility"""
        enabled = self.mobility_enabled_var.get()
        self.simulator.enable_mobility(enabled, self.get_mobility_params())
        
        if enabled:
            messagebox.showinfo("Mobility", "Mobility sistemi aktif edildi!")
        else:
            messagebox.showinfo("Mobility", "Mobility sistemi devre dışı bırakıldı!")
    
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
            messagebox.showerror("Hata", "Geçerli mobility parametreleri girin!")
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
            
            messagebox.showinfo("Başarılı", "Mobility parametreleri güncellendi!")
    
    def show_mobility_paths(self):
        """Show mobility paths of nodes"""
        if not self.mobility_enabled_var.get():
            messagebox.showinfo("Bilgi", "Mobility sistemi aktif değil!")
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
            self.mobility_text.insert(1.0, "Mobility sistemi aktif değil.")
            return
        
        mobility_stats = self.simulator.get_mobility_stats()
        
        stats_text = f"""
MOBILITY İSTATİSTİKLERİ
{'='*40}

Genel Bilgiler:
  Mobility Aktif: {'Evet' if self.mobility_enabled_var.get() else 'Hayır'}
  Aktif Model: {self.mobility_model_var.get().replace('_', ' ').title()}
  Hareketli Node Sayısı: {mobility_stats['nodes_moving']}
  Ortalama Hız: {mobility_stats['average_speed']:.2f} m/s

Model Dağılımı:
"""
        
        for model, count in mobility_stats['mobility_models'].items():
            stats_text += f"  {model.replace('_', ' ').title()}: {count} node\n"
        
        stats_text += f"""
Hareket İstatistikleri:
  Toplam Kat Edilen Mesafe: {mobility_stats['total_distance_traveled']:.1f} m

Mobility Parametreleri:
  Min Hız: {self.min_speed_var.get()} m/s
  Max Hız: {self.max_speed_var.get()} m/s  
  Pause Süresi: {self.pause_time_var.get()} s
"""
        
        # Add node-specific mobility info
        stats_text += "\nNode Detayları:\n" + "-"*30 + "\n"
        
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
                    stats_text += f" (Hedefe: {distance_to_target:.1f}m)"
                
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
                messagebox.showinfo("Başarılı", f"Paket {source} → {dest} için {events_processed} event işlendi!")
            else:
                messagebox.showerror("Hata", "Paket gönderilemedi!")
    
    def send_multiple_packets(self):
        source = self.source_var.get()
        dest = self.dest_var.get()
        
        try:
            count = int(self.packet_count_var.get())
        except ValueError:
            messagebox.showerror("Hata", "Geçerli bir paket sayısı girin!")
            return
        
        if source and dest and source != dest and count > 0:
            success_count = 0
            for i in range(count):
                if self.simulator.send_packet(source, dest, f"Test Data {i+1}"):
                    success_count += 1
                time.sleep(0.001)  # Small delay to vary timestamps
            
            messagebox.showinfo("Tamamlandı", 
                              f"{count} paketten {success_count} tanesi başarıyla gönderildi!")
            self.update_all_stats()
    
    def run_protocol_comparison(self):
        """Run performance comparison between all protocols"""
        if len(self.simulator.nodes) < 3:
            messagebox.showwarning("Uyarı", "Karşılaştırma için en az 3 node gerekli!")
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
        messagebox.showinfo("Tamamlandı", "Protokol karşılaştırması tamamlandı!")
    
    def reset_metrics(self):
        self.simulator.reset_metrics()
        self.update_all_stats()
        messagebox.showinfo("Tamamlandı", "Tüm metrikler sıfırlandı!")
    
    # Discrete-Event Simulation Methods
    def start_discrete_simulation(self):
        """Start discrete-event simulation"""
        try:
            duration = float(self.simulation_duration_var.get())
            if duration <= 0:
                messagebox.showerror("Hata", "Simülasyon süresi 0'dan büyük olmalı!")
                return
            
            events_processed = self.simulator.start_discrete_simulation(duration)
            messagebox.showinfo("Simülasyon", 
                              f"Discrete-event simülasyonu tamamlandı!\n"
                              f"Süre: {duration}s\n"
                              f"İşlenen Event: {events_processed}")
            self.update_all_stats()
        except ValueError:
            messagebox.showerror("Hata", "Geçerli bir süre değeri girin!")
    
    def stop_discrete_simulation(self):
        """Stop discrete-event simulation"""
        self.simulator.discrete_simulator.stop_simulation()
        messagebox.showinfo("Simülasyon", "Simülasyon durduruldu!")
    
    def step_simulation(self):
        """Step through simulation one event at a time"""
        if self.simulator.discrete_simulator.step():
            self.redraw_network()
            self.update_all_stats()
            next_event_time = self.simulator.discrete_simulator.get_next_event_time()
            if next_event_time:
                messagebox.showinfo("Simülasyon", 
                                  f"Bir event işlendi.\n"
                                  f"Zaman: {self.simulator.discrete_simulator.current_simulation_time:.2f}s\n"
                                  f"Sonraki Event: {next_event_time:.2f}s")
            else:
                messagebox.showinfo("Simülasyon", 
                                  f"Bir event işlendi.\n"
                                  f"Zaman: {self.simulator.discrete_simulator.current_simulation_time:.2f}s\n"
                                  f"Bekleyen event yok.")
        else:
            messagebox.showinfo("Simülasyon", "İşlenecek event yok!")
    
    def clear_events(self):
        """Clear all pending events"""
        self.simulator.discrete_simulator.clear_events()
        messagebox.showinfo("Simülasyon", "Tüm bekleyen eventler temizlendi!")
    
    def show_topology_info(self):
        """Show detailed topology information"""
        info = "AĞ TOPOLOJİSİ BİLGİSİ\n" + "="*50 + "\n\n"
        
        for node_id, node in self.simulator.nodes.items():
            info += f"Node {node_id}:\n"
            info += f"  Konum: ({node.x}, {node.y})\n"
            info += f"  Komşu Sayısı: {len(node.neighbors)}\n"
            info += f"  Komşular: {', '.join(sorted(node.neighbors))}\n"
            
            if self.simulator.routing_protocol == "OLSR":
                info += f"  MPR Kümesi: {', '.join(sorted(node.mpr_set))}\n"
                info += f"  Routing Table: {len(node.routing_table)} rota\n"
            elif self.simulator.routing_protocol == "AODV":
                active_routes = sum(1 for route in node.routing_table.values() 
                                  if self.simulator.discrete_simulator.current_simulation_time < route.expiry_time)
                info += f"  Aktif Rotalar: {active_routes}\n"
            
            info += "\n"
        
        # Show in a new window
        info_window = tk.Toplevel(self.root)
        info_window.title("Ağ Topolojisi Bilgisi")
        info_window.geometry("600x500")
        
        text_widget = tk.Text(info_window, font=("Courier", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, info)
        text_widget.config(state=tk.DISABLED)
    
    def on_canvas_click(self, event):
        x, y = event.x, event.y
        clicked_node = None
        
        # Find clicked node
        for node_id, node in self.simulator.nodes.items():
            if (node.x - x)**2 + (node.y - y)**2 <= 15**2:
                clicked_node = node_id
                break
        
        if clicked_node:
            if clicked_node in self.selected_nodes:
                self.selected_nodes.remove(clicked_node)
            else:
                self.selected_nodes = [clicked_node]
            self.redraw_network()
            self.update_node_details()
    
    def redraw_network(self):
        """Enhanced network drawing with mobility visualization"""
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
            messagebox.showinfo("Bilgi", "Henüz paket gönderilmedi!")
            return
        
        last_packet = self.simulator.packet_history[-1]
        self.animate_packet_path(last_packet.path)
    
    def animate_packet_path(self, path):
        if len(path) < 2:
            return
        
        def animate_step(step):
            if step >= len(path) - 1:
                return
            
            current_node = self.simulator.nodes[path[step]]
            next_node = self.simulator.nodes[path[step + 1]]
            
            # Draw animated line
            line = self.canvas.create_line(current_node.x, current_node.y, 
                                         next_node.x, next_node.y,
                                         fill="red", width=4)
            
            # Draw moving packet
            packet_dot = self.canvas.create_oval(next_node.x - 5, next_node.y - 5,
                                               next_node.x + 5, next_node.y + 5,
                                               fill="orange", outline="red")
            
            # Next step
            self.root.after(800, lambda: animate_step(step + 1))
            
            # Clean up
            self.root.after(1200, lambda: self.canvas.delete(line))
            self.root.after(1200, lambda: self.canvas.delete(packet_dot))
        
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
AĞIN GENEL İSTATİSTİKLERİ
{'='*40}

Ağ Bilgileri:
  Node Sayısı: {stats['total_nodes']}
  Bağlantı Sayısı: {stats['total_connections']}
  Ağ Bağlantısallığı: {stats['network_connectivity']:.1f}%
  Aktif Protokol: {stats['current_protocol']}

DISCRETE-EVENT SİMÜLASYON
{'='*40}
  Simülasyon Zamanı: {self.simulator.discrete_simulator.current_simulation_time:.2f}s
  İşlenen Event Sayısı: {discrete_stats['events_processed']}
  Event/Saniye: {discrete_stats['events_per_second']:.2f}
  Collision Sayısı: {discrete_stats['collision_count']}
  Bekleyen Event: {len(self.simulator.discrete_simulator.event_queue)}

Paket İstatistikleri:
  Gönderilen Paketler: {stats['packets_sent']}
  Teslim Edilen Paketler: {stats['packets_delivered']}
  Düşürülen Paketler: {stats['packets_dropped']}
  
PERFORMANS METRİKLERİ
{'='*40}

  Paket Teslim Oranı: {stats['delivery_ratio']:.2f}%
  Ortalama Gecikme: {stats['average_delay']*1000:.2f} ms
  Routing Overhead: {stats['routing_overhead']:.2f}
  Ortalama Hop Sayısı: {stats['average_hop_count']:.2f}
  Routing Mesajları: {stats['routing_messages_sent']}
"""
        
        if self.simulator.packet_history:
            last_packet = self.simulator.packet_history[-1]
            stats_text += f"""
Son Paket Bilgileri:
  Rota: {' → '.join(last_packet.path)}
  Hop Sayısı: {last_packet.hop_count}
  Teslim Süresi: {(last_packet.delivery_time or 0)*1000:.2f} ms
  Durum: {'Teslim Edildi' if not last_packet.dropped else f'Düşürüldü ({last_packet.drop_reason})'}
"""
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def update_protocol_comparison(self):
        comparison = self.simulator.get_protocol_comparison()
        
        comparison_text = f"""
PROTOKOL KARŞILAŞTIRMASI
{'='*50}

"""
        
        if not comparison:
            comparison_text += "Henüz karşılaştırma verisi yok.\n'Protokol Karşılaştırması' butonuna basarak test çalıştırın."
        else:
            # Header
            comparison_text += f"{'Protokol':<12} {'Teslim%':<10} {'Gecikme':<12} {'Overhead':<10} {'Hop':<8} {'Paket':<8}\n"
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
            comparison_text += "Metrik Açıklamaları:\n"
            comparison_text += "• Teslim%: Başarıyla teslim edilen paket yüzdesi\n"
            comparison_text += "• Gecikme: Ortalama paket teslim süresi (ms)\n"
            comparison_text += "• Overhead: Routing mesajı / Teslim edilen paket\n"
            comparison_text += "• Hop: Ortalama hop sayısı\n"
            comparison_text += "• Paket: Toplam gönderilen paket sayısı\n"
        
        self.comparison_text.delete(1.0, tk.END)
        self.comparison_text.insert(1.0, comparison_text)
    
    def update_node_details(self):
        if not self.selected_nodes:
            details_text = "Detayları görmek için bir node seçin."
        else:
            node_id = self.selected_nodes[0]
            node = self.simulator.nodes[node_id]
            
            details_text = f"""
NODE DETAY BİLGİLERİ: {node_id}
{'='*40}

Pozisyon Bilgileri:
  Koordinatlar: ({node.x}, {node.y})
  Transmission Range: {node.transmission_range}
  Komşu Sayısı: {len(node.neighbors)}
  Komşular: {', '.join(sorted(node.neighbors)) if node.neighbors else 'Yok'}

Paket İstatistikleri:
  Gönderilen: {node.stats['packets_sent']}
  Alınan: {node.stats['packets_received']}
  İletilen: {node.stats['packets_forwarded']}
  Düşürülen: {node.stats['packets_dropped']}
  
Routing İstatistikleri:
  Hello Mesajları: {node.stats['hello_messages_sent']}
  RREQ Mesajları: {node.stats['rreq_messages_sent']}
  RREP Mesajları: {node.stats['rrep_messages_sent']}
  Toplam Routing Mesajı: {node.stats['routing_messages_sent']}

Protokol Özel Bilgiler:
"""
            
            if self.simulator.routing_protocol == "AODV":
                active_routes = sum(1 for route in node.routing_table.values() 
                                  if self.simulator.discrete_simulator.current_simulation_time < route.expiry_time)
                details_text += f"  Aktif Rotalar: {active_routes}\n"
                details_text += f"  Toplam Rota Girişi: {len(node.routing_table)}\n"
                details_text += f"  Sequence Number: {node.sequence_number}\n"
                
                if node.routing_table:
                    details_text += "\n  Routing Table:\n"
                    for dest, route in node.routing_table.items():
                        status = "Aktif" if self.simulator.discrete_simulator.current_simulation_time < route.expiry_time else "Süresi Dolmuş"
                        details_text += f"    {dest} → {route.next_hop} (Hop: {route.hop_count}, {status})\n"
            
            elif self.simulator.routing_protocol == "OLSR":
                details_text += f"  MPR Kümesi: {', '.join(sorted(node.mpr_set)) if node.mpr_set else 'Yok'}\n"
                details_text += f"  MPR Selector Sayısı: {len(node.mpr_selector_set)}\n"
                details_text += f"  LSA Database: {len(node.lsa_database)} giriş\n"
                details_text += f"  Son Hello: {self.simulator.discrete_simulator.current_simulation_time - node.last_hello_time:.1f}s önce\n"
            
            # Add mobility information if available
            if hasattr(node, 'mobility'):
                details_text += f"\nMobility Bilgileri:\n"
                mobility = node.mobility
                details_text += f"  Model: {mobility['model']}\n"
                details_text += f"  Mevcut Hız: {mobility['speed']:.2f} m/s\n"
                if mobility['model'] == 'random_waypoint' and mobility['target_x']:
                    distance_to_target = math.sqrt(
                        (mobility['target_x'] - node.x)**2 + 
                        (mobility['target_y'] - node.y)**2
                    )
                    details_text += f"  Hedefe Mesafe: {distance_to_target:.1f} m\n"
                    details_text += f"  Hedef: ({mobility['target_x']:.0f}, {mobility['target_y']:.0f})\n"
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details_text)



        

def run_test_suite():
    """Run the test suite independently"""
    print("Starting MANET Simulator Test Suite...")
    simulator = MANETSimulator()
    results = simulator.run_test_suite()
    return results

if __name__ == "__main__":
    import sys
    
    # Check if test suite should be run
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test_suite()
    else:
        # Run GUI normally
        root = tk.Tk()
        app = MANETSimulatorGUI(root)
        root.mainloop()