"""
Node class for MANET simulator
"""
import math
from collections import deque
from models import EnergyModel

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
