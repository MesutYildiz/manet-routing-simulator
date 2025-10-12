"""
Data models and structures for MANET simulator
"""
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from simulator import MANETSimulator

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
    
    def check_and_handle_death(self, node_id, simulator: 'MANETSimulator'):
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