import random
import math
import threading
import time
import heapq
from collections import defaultdict, deque
from typing import Dict, List, Set, Optional, Tuple
import json

from models import (
    Event, PerformanceMetrics, Packet, RouteEntry, HelloMessage, 
    RREQMessage, RREPMessage, LSAMessage, LSAEntry, RERRMessage,
    EnergyModel, MobilityParameters
)
from discrete_simulator import DiscreteEventSimulator
from mobility import MobilityModel
from node import Node
from mac_layer import IEEE80211MAC, CSMACAHandler, integrate_mac_layer

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
        
        # Add MAC layer
        self.mac_layer = IEEE80211MAC()
        self.csma_handler = CSMACAHandler(self.mac_layer)
        
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
                
                # After LSA database update, immediately update routing tables
                for node_id in self.nodes.keys():
                    if node_id != event.source_node:
                        self.olsr_update_routing_table(self.nodes[node_id])
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
        
        # Safety check: node might have been removed
        if node_id not in self.nodes:
            return
        
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
                    # Update timestamp and initialize path for visualization
                    buffered_packet.timestamp = self.discrete_simulator.current_simulation_time
                    buffered_packet.path = [source_id]  # Initialize with source
                    buffered_packet.hop_count = 0  # Reset hop count
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
            source_node=rreq.destination,  # RREP starts from destination node
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
        
        # Safety check: node might have been removed
        if node_id not in self.nodes:
            return
        
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
        
        # Safety check: node might have been removed
        if source_id not in self.nodes:
            return
        
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
            # Advance simulation time by 0.1 seconds for GUI updates
            # This ensures mobility continues to work even without discrete events
            self.discrete_simulator.current_simulation_time += 0.1
            current_time = self.discrete_simulator.current_simulation_time
            
            # Update positions
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
            
            # Safety check: node might have been removed
            if current_id not in self.nodes:
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
            
            # Safety check: node might have been removed during simulation
            if node_id not in self.nodes:
                # Node doesn't exist, drop packet
                packet.dropped = True
                packet.drop_reason = f"Node {node_id} no longer exists"
                return
            
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
        # Safety check: nodes might have been removed
        if source not in self.nodes or destination not in self.nodes:
            return False
        
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
    
    def setup_network(self, num_nodes, protocol):
        """Setup a test network with specified number of nodes and protocol"""
        # Clear existing network
        self.nodes.clear()
        self.packet_history.clear()
        self.reset_metrics()
        
        # Set routing protocol
        self.routing_protocol = protocol
        
        # Create nodes in a grid pattern
        import math
        cols = math.ceil(math.sqrt(num_nodes))
        rows = math.ceil(num_nodes / cols)
        
        node_id = 0
        for row in range(rows):
            for col in range(cols):
                if node_id >= num_nodes:
                    break
                
                x = 100 + col * 150
                y = 100 + row * 150
                
                node = Node(f"Node_{node_id}", x, y, transmission_range=120)
                self.nodes[f"Node_{node_id}"] = node
                
                # Initialize routing protocol
                self._initialize_routing_protocol(node)
                
                node_id += 1
        
        # Update topology
        self.update_topology()
        
        # Schedule initial events
        self.schedule_initial_events()
    
    def _initialize_routing_protocol(self, node):
        """Initialize routing protocol specific data structures"""
        if self.routing_protocol == "AODV":
            node.routing_table = {}
            node.packet_buffer = {}
            node.buffer_timeouts = {}
        elif self.routing_protocol == "OLSR":
            node.lsa_database = {}
            node.mpr_set = set()
            node.mpr_selector_set = set()
        elif self.routing_protocol == "DSR":
            node.route_cache = {}
        elif self.routing_protocol == "FLOODING":
            node.seen_packets = {}
    
    def update_topology(self):
        """Update network topology based on node positions"""
        current_time = self.discrete_simulator.current_simulation_time
        
        # Update neighbor relationships
        for node_id, node in self.nodes.items():
            old_neighbors = node.neighbors.copy()
            node.neighbors.clear()
            
            for other_id, other_node in self.nodes.items():
                if node_id != other_id and node.can_communicate_with(other_node):
                    node.add_neighbor(other_id, current_time + 5.0)  # 5 second expiry
            
            # Detect new and lost neighbors
            new_neighbors = node.neighbors - old_neighbors
            lost_neighbors = old_neighbors - node.neighbors
            
            # Handle neighbor changes based on routing protocol
            if new_neighbors or lost_neighbors:
                self._handle_topology_change(node_id, new_neighbors, lost_neighbors)
        
        self.last_topology_update = current_time
    
    def _handle_topology_change(self, node_id, new_neighbors, lost_neighbors):
        """Handle topology changes for different routing protocols"""
        if self.routing_protocol == "AODV":
            # Invalidate routes through lost neighbors
            for lost_neighbor in lost_neighbors:
                self._invalidate_routes_through_neighbor(node_id, lost_neighbor)
        elif self.routing_protocol == "OLSR":
            # Update MPR sets and immediately recalculate routing table
            node = self.nodes[node_id]
            node.select_mpr_set(self.nodes)
            # Trigger immediate routing table update for topology changes
            self.olsr_update_routing_table(node)
    
    def _invalidate_routes_through_neighbor(self, node_id, lost_neighbor):
        """Invalidate routes that go through a lost neighbor"""
        node = self.nodes[node_id]
        routes_to_remove = []
        
        for dest, route in node.routing_table.items():
            if route.next_hop == lost_neighbor:
                routes_to_remove.append(dest)
        
        for dest in routes_to_remove:
            del node.routing_table[dest]
    
    def run_discrete_simulation(self, duration):
        """Run discrete simulation for specified duration"""
        self.discrete_simulator.run_until(duration)
    
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
        self.send_packet("Node_0", "Node_4", "Test Data", 64)
        
        # Run simulation for 10 seconds
        self.run_discrete_simulation(10.0)
        
        # Check results
        metrics = self.metrics["AODV"]
        delivery_ratio = metrics.calculate_delivery_ratio()
        
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
        pairs = [("Node_0", "Node_5"), ("Node_1", "Node_4"), ("Node_2", "Node_3")]
        for source, dest in pairs:
            self.send_packet(source, dest, f"Data from {source} to {dest}", 64)
        
        # Run simulation
        self.run_discrete_simulation(15.0)
        
        # Check results
        metrics = self.metrics["AODV"]
        delivery_ratio = metrics.calculate_delivery_ratio()
        
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
        self.send_packet("Node_0", "Node_4", "OLSR Test Data", 64)
        self.send_packet("Node_1", "Node_3", "OLSR Test Data 2", 64)
        
        # Run simulation
        self.run_discrete_simulation(10.0)
        
        # Check results
        metrics = self.metrics["OLSR"]
        delivery_ratio = metrics.calculate_delivery_ratio()
        
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
        self.send_packet("Node_0", "Node_5", "Flooding Test", 64)
        
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
            'delivery_ratio': metrics.calculate_delivery_ratio()
        }
        
        if not result['passed']:
            result['reason'] = f"Too many transmissions: {total_transmissions} (expected < {expected_max})"
        
        print(f"  Total Transmissions: {total_transmissions}")
        print(f"  Delivery Ratio: {metrics.calculate_delivery_ratio():.2%}")
        
        return result
    
    def test_mobility_route_break(self):
        """Test route invalidation due to mobility"""
        # Setup: 4 nodes in line
        self.setup_network(4, "AODV")
        self.setup_line_topology()
        
        # Send packet
        self.send_packet("Node_0", "Node_3", "Mobility Test", 64)
        
        # Run for 2 seconds
        self.run_discrete_simulation(2.0)
        
        # Move nodes apart to break route
        self.nodes["Node_1"].x = 1000  # Move far away
        self.nodes["Node_2"].x = 1000
        
        # Continue simulation
        self.run_discrete_simulation(5.0)
        
        # Check for route invalidation
        metrics = self.metrics["AODV"]
        rerr_messages = sum(node.stats.get('rerr_messages_sent', 0) for node in self.nodes.values())
        
        result = {
            'passed': rerr_messages > 0,  # Should have RERR messages
            'rerr_messages': rerr_messages,
            'delivery_ratio': metrics.calculate_delivery_ratio()
        }
        
        if not result['passed']:
            result['reason'] = "No RERR messages sent for route invalidation"
        
        print(f"  RERR Messages: {rerr_messages}")
        print(f"  Delivery Ratio: {metrics.calculate_delivery_ratio():.2%}")
        
        return result
    
    def test_energy_depletion(self):
        """Test energy depletion and node death"""
        # Setup: 3 nodes with low energy
        self.setup_network(3, "AODV")
        self.setup_line_topology()
        
        # Set low energy for middle node
        self.nodes["Node_1"].energy.current_energy = 0.1
        
        # Send packet through middle node
        self.send_packet("Node_0", "Node_2", "Energy Test", 64)
        
        # Run simulation
        self.run_discrete_simulation(5.0)
        
        # Check if middle node died
        middle_node_dead = not self.nodes["Node_1"].energy.is_alive()
        metrics = self.metrics["AODV"]
        
        result = {
            'passed': middle_node_dead and metrics.total_packets_dropped > 0,
            'middle_node_dead': middle_node_dead,
            'packets_dropped': metrics.total_packets_dropped,
            'delivery_ratio': metrics.calculate_delivery_ratio()
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
        self.send_packet("Node_0", "Node_3", "Route Expiry Test", 64)
        
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
            'delivery_ratio': metrics.calculate_delivery_ratio()
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
        self.send_packet("Node_0", "Node_4", "DSR Test", 64)
        
        # Run simulation
        self.run_discrete_simulation(10.0)
        
        # Check DSR-specific metrics
        metrics = self.metrics["DSR"]
        cache_hits = sum(node.stats.get('dsr_cache_hits', 0) for node in self.nodes.values())
        cache_misses = sum(node.stats.get('dsr_cache_misses', 0) for node in self.nodes.values())
        
        result = {
            'passed': metrics.calculate_delivery_ratio() > 0.8 and cache_misses > 0,
            'delivery_ratio': metrics.calculate_delivery_ratio(),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'packets_sent': metrics.total_packets_sent
        }
        
        if not result['passed']:
            result['reason'] = f"Low delivery ratio: {metrics.calculate_delivery_ratio():.2%} or no cache misses: {cache_misses}"
        
        print(f"  Delivery Ratio: {metrics.calculate_delivery_ratio():.2%}")
        print(f"  Cache Hits: {cache_hits}")
        print(f"  Cache Misses: {cache_misses}")
        
        return result
    
    def setup_line_topology(self):
        """Setup nodes in a line"""
        for i, (node_id, node) in enumerate(self.nodes.items()):
            node.x = i * 100
            node.y = 0
        
        # Update topology after changing positions
        self.update_topology()
    
    def setup_grid_topology(self):
        """Setup nodes in a 2x3 grid"""
        positions = [(0, 0), (100, 0), (200, 0), (0, 100), (100, 100), (200, 100)]
        for i, (node_id, node) in enumerate(self.nodes.items()):
            if i < len(positions):
                node.x, node.y = positions[i]
        
        # Update topology after changing positions
        self.update_topology()
    
    def setup_mesh_topology(self):
        """Setup nodes in a mesh (all connected)"""
        for i, (node_id, node) in enumerate(self.nodes.items()):
            node.x = i * 50
            node.y = i * 50
        
        # Update topology after changing positions
        self.update_topology()
    
    def setup_dense_topology(self):
        """Setup nodes in a dense topology"""
        for i, (node_id, node) in enumerate(self.nodes.items()):
            node.x = i * 30
            node.y = i * 30
        
        # Update topology after changing positions
        self.update_topology()