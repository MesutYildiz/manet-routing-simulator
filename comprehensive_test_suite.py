"""
Comprehensive Test Suite for MANET Simulator
Tests all components to ensure everything works correctly
"""

import sys
import time
from typing import List, Dict, Tuple


class ComprehensiveTestSuite:
    """Tüm simulator bileşenlerini test eden kapsamlı test suite"""
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = []
    
    def print_section(self, title: str):
        """Print formatted section header"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)
    
    def print_test(self, test_name: str, passed: bool, details: str = ""):
        """Print test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"  # Green or Red
        reset = "\033[0m"
        
        print(f"{color}{status}{reset} | {test_name}")
        if details:
            print(f"       {details}")
        
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'details': details
        })
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    def test_1_imports(self) -> bool:
        """Test 1: Can all modules be imported?"""
        self.print_section("TEST 1: MODULE IMPORTS")
        
        all_passed = True
        
        # Test each module
        modules = [
            ('models', 'Event, Packet, RouteEntry, MobilityParameters'),
            ('discrete_simulator', 'DiscreteEventSimulator'),
            ('node', 'Node'),
            ('mobility', 'MobilityModel'),
            ('simulator', 'MANETSimulator'),
            ('gui', 'MANETSimulatorGUI')
        ]
        
        for module_name, classes in modules:
            try:
                module = __import__(module_name)
                self.print_test(f"Import {module_name}", True, f"Classes: {classes}")
            except ImportError as e:
                self.print_test(f"Import {module_name}", False, f"Error: {e}")
                all_passed = False
        
        return all_passed
    
    def test_2_discrete_simulator(self) -> bool:
        """Test 2: Discrete Event Simulator functionality"""
        self.print_section("TEST 2: DISCRETE EVENT SIMULATOR")
        
        try:
            from discrete_simulator import DiscreteEventSimulator
            from models import Event
            
            # Create simulator
            des = DiscreteEventSimulator()
            self.print_test("Create DiscreteEventSimulator", True)
            
            # Test attributes
            required_attrs = ['event_queue', 'current_simulation_time', 'simulation_running', 
                            'event_handlers', 'statistics']
            has_attrs = all(hasattr(des, attr) for attr in required_attrs)
            self.print_test("Has required attributes", has_attrs, 
                          f"Attributes: {', '.join(required_attrs)}")
            
            # Test event scheduling
            event_handled = False
            def test_handler(event):
                nonlocal event_handled
                event_handled = True
            
            des.register_handler("TEST_EVENT", test_handler)
            
            test_event = Event(
                event_type="TEST_EVENT",
                timestamp=1.0,
                source_node="TestNode"
            )
            
            des.schedule_event(test_event)
            self.print_test("Schedule event", len(des.event_queue) == 1,
                          f"Queue size: {len(des.event_queue)}")
            
            # Test event processing
            processed = des.process_event()
            self.print_test("Process event", processed and event_handled,
                          f"Event handled: {event_handled}")
            
            # Test run_until
            des2 = DiscreteEventSimulator()
            count = 0
            def count_handler(event):
                nonlocal count
                count += 1
            
            des2.register_handler("COUNT", count_handler)
            for i in range(10):
                des2.schedule_event(Event("COUNT", float(i), "Node"))
            
            events = des2.run_until(15.0)
            self.print_test("Run until time", events == 10 and count == 10,
                          f"Processed: {events}, Handler calls: {count}")
            
            return has_attrs and processed and event_handled and (events == 10)
            
        except Exception as e:
            self.print_test("Discrete Simulator Test", False, f"Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_3_node_creation(self) -> bool:
        """Test 3: Node creation and properties"""
        self.print_section("TEST 3: NODE CREATION")
        
        try:
            from node import Node
            
            # Create node
            node = Node("TestNode", 100, 200, transmission_range=150)
            self.print_test("Create Node", True, f"ID: {node.id}, Pos: ({node.x}, {node.y})")
            
            # Test attributes
            has_id = node.id == "TestNode"
            has_pos = node.x == 100 and node.y == 200
            has_range = node.transmission_range == 150
            
            self.print_test("Node attributes", has_id and has_pos and has_range,
                          f"ID: {has_id}, Pos: {has_pos}, Range: {has_range}")
            
            # Test distance calculation
            node2 = Node("Node2", 400, 200)
            distance = node.distance_to(node2)
            expected_distance = 300.0
            
            self.print_test("Distance calculation", abs(distance - expected_distance) < 0.1,
                          f"Distance: {distance:.2f}, Expected: {expected_distance}")
            
            # Test neighbor management
            node.add_neighbor("Neighbor1", 10.0)
            has_neighbor = "Neighbor1" in node.neighbors
            
            self.print_test("Add neighbor", has_neighbor,
                          f"Neighbors: {node.neighbors}")
            
            return has_id and has_pos and has_neighbor
            
        except Exception as e:
            self.print_test("Node Test", False, f"Exception: {e}")
            return False
    
    def test_4_simulator_creation(self) -> bool:
        """Test 4: MANET Simulator creation and basic operations"""
        self.print_section("TEST 4: MANET SIMULATOR CREATION")
        
        try:
            from simulator import MANETSimulator
            
            # Create simulator
            sim = MANETSimulator(width=800, height=600)
            self.print_test("Create MANETSimulator", True,
                          f"Size: {sim.width}x{sim.height}")
            
            # Add nodes
            for i in range(5):
                sim.add_node(f"N{i}", x=i*150, y=100)
            
            node_count = len(sim.nodes)
            self.print_test("Add 5 nodes", node_count == 5,
                          f"Node count: {node_count}")
            
            # Check protocols
            protocols = ["AODV", "OLSR", "DSR", "FLOODING", "ZRP"]
            has_protocols = all(protocol in sim.metrics for protocol in protocols)
            
            self.print_test("Has 5 protocols", has_protocols,
                          f"Protocols: {', '.join(protocols)}")
            
            # Check topology update
            sim.update_network_topology()
            
            # Count connections
            connections = sum(len(node.neighbors) for node in sim.nodes.values()) // 2
            self.print_test("Network topology updated", connections > 0,
                          f"Connections: {connections}")
            
            return node_count == 5 and has_protocols and connections > 0
            
        except Exception as e:
            self.print_test("Simulator Test", False, f"Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_5_aodv_routing(self) -> bool:
        """Test 5: AODV routing protocol"""
        self.print_section("TEST 5: AODV ROUTING PROTOCOL")
        
        try:
            from simulator import MANETSimulator
            
            # Create network
            sim = MANETSimulator()
            sim.routing_protocol = "AODV"
            
            # Line topology: N0 -- N1 -- N2 -- N3 -- N4
            for i in range(5):
                sim.add_node(f"N{i}", x=i*100, y=100)
            
            sim.update_network_topology()
            
            # Check connectivity
            n0_neighbors = len(sim.nodes["N0"].neighbors)
            self.print_test("N0 has neighbors", n0_neighbors > 0,
                          f"N0 neighbors: {sim.nodes['N0'].neighbors}")
            
            # Send packet
            sim.send_packet("N0", "N4", "AODV Test Data")
            
            # Run simulation
            events = sim.discrete_simulator.run_until(10.0)
            self.print_test("Events processed", events > 0,
                          f"Events: {events}")
            
            # Check metrics
            metrics = sim.metrics["AODV"]
            packets_sent = metrics.total_packets_sent
            packets_delivered = metrics.total_packets_delivered
            
            self.print_test("Packets sent", packets_sent > 0,
                          f"Sent: {packets_sent}")
            
            self.print_test("Packets delivered", packets_delivered > 0,
                          f"Delivered: {packets_delivered}")
            
            delivery_ratio = metrics.calculate_delivery_ratio()
            self.print_test("Delivery ratio", delivery_ratio > 0,
                          f"Ratio: {delivery_ratio:.1f}%")
            
            return packets_sent > 0 and events > 0
            
        except Exception as e:
            self.print_test("AODV Test", False, f"Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_6_olsr_routing(self) -> bool:
        """Test 6: OLSR routing protocol"""
        self.print_section("TEST 6: OLSR ROUTING PROTOCOL")
        
        try:
            from simulator import MANETSimulator
            
            # Create network
            sim = MANETSimulator()
            sim.routing_protocol = "OLSR"
            
            # Grid topology
            for i in range(6):
                x = (i % 3) * 100
                y = (i // 3) * 100
                sim.add_node(f"N{i}", x=x, y=y)
            
            sim.update_network_topology()
            
            # Schedule initial events
            sim.schedule_initial_events()
            
            # Run for convergence
            events = sim.discrete_simulator.run_until(5.0)
            self.print_test("OLSR convergence", events > 0,
                          f"Events: {events}")
            
            # Check routing tables
            n0_routes = len(sim.nodes["N0"].routing_table)
            self.print_test("Routing table populated", n0_routes > 0,
                          f"N0 routes: {n0_routes}")
            
            # Send packet
            sim.send_packet("N0", "N5", "OLSR Test Data")
            sim.discrete_simulator.run_until(10.0)
            
            metrics = sim.metrics["OLSR"]
            self.print_test("OLSR packet sent", metrics.total_packets_sent > 0,
                          f"Sent: {metrics.total_packets_sent}")
            
            return events > 0 and metrics.total_packets_sent > 0
            
        except Exception as e:
            self.print_test("OLSR Test", False, f"Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_7_mobility_system(self) -> bool:
        """Test 7: Mobility system"""
        self.print_section("TEST 7: MOBILITY SYSTEM")
        
        try:
            from simulator import MANETSimulator
            from models import MobilityParameters
            
            # Create simulator
            sim = MANETSimulator()
            
            # Add nodes
            for i in range(5):
                sim.add_node(f"M{i}", x=i*100, y=100)
            
            # Enable mobility
            params = MobilityParameters(
                model_type="random_waypoint",
                min_speed=1.0,
                max_speed=5.0,
                pause_time=0.5
            )
            
            sim.enable_mobility(True, params)
            self.print_test("Enable mobility", sim.mobility_enabled,
                          f"Model: {params.model_type}")
            
            # Check nodes have mobility
            has_mobility = all(hasattr(node, 'mobility') for node in sim.nodes.values())
            self.print_test("Nodes have mobility", has_mobility)
            
            # Record initial positions
            initial_positions = {nid: (n.x, n.y) for nid, n in sim.nodes.items()}
            
            # Run simulation with mobility
            sim.schedule_initial_events()
            sim.discrete_simulator.run_until(5.0)
            
            # Check if nodes moved
            moved_count = 0
            for nid, node in sim.nodes.items():
                if (node.x, node.y) != initial_positions[nid]:
                    moved_count += 1
            
            self.print_test("Nodes moved", moved_count > 0,
                          f"Moved nodes: {moved_count}/5")
            
            # Get mobility stats
            mob_stats = sim.get_mobility_stats()
            self.print_test("Mobility stats", mob_stats['total_distance_traveled'] > 0,
                          f"Distance: {mob_stats['total_distance_traveled']:.1f}m")
            
            return has_mobility and moved_count > 0
            
        except Exception as e:
            self.print_test("Mobility Test", False, f"Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_8_scalability(self) -> bool:
        """Test 8: Scalability with 50 nodes"""
        self.print_section("TEST 8: SCALABILITY (50 NODES)")
        
        try:
            from simulator import MANETSimulator
            
            # Create large network
            sim = MANETSimulator(width=1500, height=1500)
            
            # Add 50 nodes in grid
            for i in range(50):
                x = (i % 10) * 150
                y = (i // 10) * 150
                sim.add_node(f"N{i}", x=x, y=y)
            
            node_count = len(sim.nodes)
            self.print_test("50 nodes created", node_count == 50,
                          f"Nodes: {node_count}")
            
            # Update topology
            start_time = time.time()
            sim.update_network_topology()
            topology_time = time.time() - start_time
            
            self.print_test("Topology update time", topology_time < 1.0,
                          f"Time: {topology_time:.3f}s")
            
            # Count connections
            connections = sum(len(n.neighbors) for n in sim.nodes.values()) // 2
            self.print_test("Network connected", connections > 0,
                          f"Connections: {connections}")
            
            # Test routing
            sim.routing_protocol = "AODV"
            sim.send_packet("N0", "N49", "Scalability Test")
            
            events = sim.discrete_simulator.run_until(15.0)
            self.print_test("Routing works at scale", events > 0,
                          f"Events: {events}")
            
            return node_count == 50 and connections > 0 and events > 0
            
        except Exception as e:
            self.print_test("Scalability Test", False, f"Exception: {e}")
            return False
    
    def test_9_protocol_comparison(self) -> bool:
        """Test 9: Compare all protocols"""
        self.print_section("TEST 9: PROTOCOL COMPARISON")
        
        try:
            from simulator import MANETSimulator
            
            sim = MANETSimulator()
            
            # Create network
            for i in range(10):
                x = (i % 4) * 100
                y = (i // 4) * 100
                sim.add_node(f"P{i}", x=x, y=y)
            
            sim.update_network_topology()
            
            # Test each protocol
            protocols = ["AODV", "OLSR", "DSR", "FLOODING", "ZRP"]
            results = {}
            
            for protocol in protocols:
                sim.routing_protocol = protocol
                sim.reset_metrics()
                
                # Schedule events
                sim.schedule_initial_events()
                sim.discrete_simulator.run_until(5.0)
                
                # Send test packets
                sim.send_packet("P0", "P9", f"{protocol} Test")
                sim.discrete_simulator.run_until(15.0)
                
                # Get metrics
                metrics = sim.metrics[protocol]
                results[protocol] = {
                    'sent': metrics.total_packets_sent,
                    'delivered': metrics.total_packets_delivered,
                    'ratio': metrics.calculate_delivery_ratio()
                }
                
                self.print_test(f"{protocol} protocol", 
                              metrics.total_packets_sent > 0,
                              f"Sent: {metrics.total_packets_sent}, "
                              f"Delivered: {metrics.total_packets_delivered}")
            
            # Check if at least 3 protocols work
            working = sum(1 for r in results.values() if r['sent'] > 0)
            self.print_test("Multiple protocols work", working >= 3,
                          f"Working protocols: {working}/5")
            
            return working >= 3
            
        except Exception as e:
            self.print_test("Protocol Comparison", False, f"Exception: {e}")
            return False
    
    def test_10_energy_system(self) -> bool:
        """Test 10: Energy consumption system"""
        self.print_section("TEST 10: ENERGY SYSTEM")
        
        try:
            from simulator import MANETSimulator
            
            sim = MANETSimulator()
            
            # Create simple network
            for i in range(3):
                sim.add_node(f"E{i}", x=i*100, y=100)
            
            # Set low energy for middle node
            sim.nodes["E1"].energy.current_energy = 0.5
            initial_energy = sim.nodes["E1"].energy.current_energy
            
            self.print_test("Set low energy", initial_energy == 0.5,
                          f"E1 energy: {initial_energy}")
            
            # Send packets through middle node
            sim.routing_protocol = "AODV"
            sim.update_network_topology()
            sim.schedule_initial_events()
            sim.discrete_simulator.run_until(5.0)
            
            sim.send_packet("E0", "E2", "Energy Test")
            sim.discrete_simulator.run_until(10.0)
            
            # Check energy decreased
            final_energy = sim.nodes["E1"].energy.current_energy
            energy_consumed = initial_energy - final_energy
            
            self.print_test("Energy consumed", energy_consumed > 0,
                          f"Consumed: {energy_consumed:.3f}")
            
            self.print_test("Energy tracking", final_energy >= 0,
                          f"Final energy: {final_energy:.3f}")
            
            return energy_consumed > 0
            
        except Exception as e:
            self.print_test("Energy Test", False, f"Exception: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        self.print_section("TEST SUMMARY")
        
        total = self.passed_tests + self.failed_tests
        pass_rate = (self.passed_tests / total * 100) if total > 0 else 0
        
        print(f"\nTotal Tests:     {total}")
        print(f"Passed:          {self.passed_tests} ({pass_rate:.1f}%)")
        print(f"Failed:          {self.failed_tests}")
        
        if self.warnings:
            print(f"\nWarnings:        {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  ⚠  {warning}")
        
        # Overall status
        print("\n" + "="*80)
        if self.failed_tests == 0:
            print("✓✓✓ ALL TESTS PASSED! SIMULATOR IS FULLY FUNCTIONAL ✓✓✓")
        elif self.passed_tests >= total * 0.7:
            print("⚠ MOSTLY WORKING - Some minor issues need attention")
        else:
            print("✗ MAJOR ISSUES DETECTED - Simulator needs fixes")
        print("="*80)
        
        return self.failed_tests == 0
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("  COMPREHENSIVE MANET SIMULATOR TEST SUITE")
        print("  Testing all components and functionality")
        print("="*80)
        
        # Run all tests
        test_methods = [
            self.test_1_imports,
            self.test_2_discrete_simulator,
            self.test_3_node_creation,
            self.test_4_simulator_creation,
            self.test_5_aodv_routing,
            self.test_6_olsr_routing,
            self.test_7_mobility_system,
            self.test_8_scalability,
            self.test_9_protocol_comparison,
            self.test_10_energy_system
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"\n✗ Test crashed: {test_method.__name__}")
                print(f"   Exception: {e}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        return self.print_summary()


def main():
    """Main test runner"""
    tester = ComprehensiveTestSuite()
    success = tester.run_all_tests()
    
    # Return exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

