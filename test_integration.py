"""Quick integration test for MANET simulator"""

def test_basic_integration():
    """Test that all components integrate correctly"""
    print("Running integration test...")
    
    try:
        # Test imports
        from simulator import MANETSimulator
        from discrete_simulator import DiscreteEventSimulator
        from models import Packet, Event, MobilityParameters
        from node import Node
        from mobility import MobilityModel
        from mac_layer import IEEE80211MAC
        print("✓ All imports successful")
        
        # Test simulator creation
        sim = MANETSimulator()
        print("✓ Simulator created")
        
        # Test node creation
        sim.add_node("N0", 100, 100)
        sim.add_node("N1", 200, 100)
        print(f"✓ Created {len(sim.nodes)} nodes")
        
        # Test packet sending with AODV
        sim.routing_protocol = "AODV"
        success = sim.send_packet("N0", "N1", "Test")
        print(f"✓ Packet sent: {success}")
        
        # Test event processing
        events = sim.discrete_simulator.run_until(5.0)
        print(f"✓ Processed {events} events")
        
        # Test metrics
        stats = sim.get_network_stats()
        print(f"✓ Stats: {stats['packets_sent']} sent, {stats['packets_delivered']} delivered")
        
        # Test mobility
        sim.enable_mobility(True)
        print(f"✓ Mobility enabled: {sim.mobility_enabled}")
        
        print("\n✅ Integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_integration()
    exit(0 if success else 1)


