# test_functional_scenarios.py

class FunctionalTests:
    """Fonksiyonel test senaryoları"""
    
    def scenario_1_basic_routing(self):
        """Senaryo 1: Temel routing - 5 node, basit paket gönderimi"""
        from simulator import MANETSimulator
        
        print("\n" + "="*60)
        print("SENARYO 1: Temel Routing (5 Node)")
        print("="*60)
        
        sim = MANETSimulator()
        
        # 5 node ekle (line topology)
        nodes = []
        for i in range(5):
            node_id = f"Node_{i}"
            sim.add_node(node_id, x=i*100, y=100)
            nodes.append(node_id)
        
        print(f"✓ {len(nodes)} node eklendi")
        print(f"✓ Topology: Line (Node_0 -> Node_1 -> ... -> Node_4)")
        
        # Her protokol için test
        results = {}
        for protocol in ["AODV", "OLSR", "DSR", "FLOODING", "ZRP"]:
            sim.routing_protocol = protocol
            sim.reset_metrics()
            
            # Paket gönder
            sim.send_packet("Node_0", "Node_4", f"Test-{protocol}")
            
            # Simülasyon çalıştır
            sim.discrete_simulator.run_until(10.0)
            
            # Metrikler
            stats = sim.get_network_stats()
            results[protocol] = {
                'delivered': stats['packets_delivered'],
                'delivery_ratio': stats['delivery_ratio'],
                'avg_delay': stats['average_delay'] * 1000,  # ms
                'overhead': stats['routing_overhead']
            }
            
            print(f"\n{protocol}:")
            print(f"  Teslim edilen: {results[protocol]['delivered']}")
            print(f"  Teslim oranı: {results[protocol]['delivery_ratio']:.1f}%")
            print(f"  Ortalama gecikme: {results[protocol]['avg_delay']:.2f} ms")
            print(f"  Routing overhead: {results[protocol]['overhead']:.2f}")
        
        # Başarı kriteri: En az 3 protokol %80+ teslim oranı
        success_count = sum(1 for r in results.values() if r['delivery_ratio'] >= 80)
        
        print(f"\n{'✓ BAŞARILI' if success_count >= 3 else '✗ BAŞARISIZ'}: {success_count}/5 protokol başarılı")
        return success_count >= 3
    
    def scenario_2_scalability(self):
        """Senaryo 2: Ölçeklenebilirlik - 50 node"""
        from simulator import MANETSimulator
        
        print("\n" + "="*60)
        print("SENARYO 2: Ölçeklenebilirlik (50 Node)")
        print("="*60)
        
        sim = MANETSimulator(width=1500, height=1500)
        
        # 50 node ekle (grid topology)
        import math
        cols = 10
        rows = 5
        
        for i in range(50):
            row = i // cols
            col = i % cols
            x = 50 + col * 140
            y = 50 + row * 140
            sim.add_node(f"N{i}", x=x, y=y)
        
        print(f"✓ 50 node eklendi (10x5 grid)")
        
        # Network connectivity
        stats = sim.get_network_stats()
        connectivity = stats['network_connectivity']
        
        print(f"✓ Network bağlantısallığı: {connectivity:.1f}%")
        
        # Real packet transmission test with 50 nodes
        print(f"\n50 Node Paket Gönderim Testi:")
        sim.routing_protocol = "AODV"
        sim.reset_metrics()
        
        # Send packets between different distance ranges
        test_cases = [
            ("N0", "N9", "Yakın (1 satır)"),
            ("N0", "N24", "Orta (2.5 satır)"),
            ("N0", "N49", "Uzak (5 satır)"),
            ("N5", "N44", "Çapraz"),
            ("N10", "N39", "Diagonal")
        ]
        
        for source, dest, description in test_cases:
            sim.send_packet(source, dest, f"50NodeTest-{description}")
        
        # Run longer simulation for 50 nodes
        print("  Simülasyon çalışıyor (30 saniye)...")
        sim.discrete_simulator.run_until(30.0)
        
        # Get comprehensive metrics
        metrics = sim.get_network_stats()
        
        print(f"\n50 Node Performans Sonuçları:")
        print(f"  Gönderilen paket: {metrics['packets_sent']}")
        print(f"  Teslim edilen: {metrics['packets_delivered']}")
        print(f"  Düşürülen: {metrics['packets_dropped']}")
        print(f"  Teslim oranı: {metrics['delivery_ratio']:.1f}%")
        print(f"  Ortalama gecikme: {metrics['average_delay']*1000:.2f} ms")
        print(f"  Ortalama hop: {metrics['average_hop_count']:.1f}")
        print(f"  Routing overhead: {metrics['routing_overhead']:.2f}")
        
        # Success criteria: At least 60% delivery ratio with 50 nodes
        packets_delivered = metrics['packets_delivered']
        success = connectivity > 50 and packets_delivered >= 3
        
        print(f"\n{'✓ BAŞARILI' if success else '✗ BAŞARISIZ'}: "
              f"Connectivity: {connectivity:.1f}%, Delivered: {packets_delivered}/{len(test_cases)}")
        return success
    
    def scenario_3_mobility(self):
        """Senaryo 3: Mobility - Hareket eden nodelar"""
        from simulator import MANETSimulator
        from models import MobilityParameters
        
        print("\n" + "="*60)
        print("SENARYO 3: Mobility (Hareket Eden Nodelar)")
        print("="*60)
        
        sim = MANETSimulator()
        
        # 10 node ekle
        for i in range(10):
            sim.add_node(f"M{i}", x=i*80, y=100)
        
        # Mobility aktif et
        mobility_params = MobilityParameters(
            model_type="random_waypoint",
            min_speed=1.0,
            max_speed=5.0,
            pause_time=1.0
        )
        sim.enable_mobility(True, mobility_params)
        
        print("✓ 10 node eklendi")
        print("✓ Mobility model: Random Waypoint")
        print("✓ Hız aralığı: 1-5 m/s")
        
        # AODV ile test
        sim.routing_protocol = "AODV"
        
        # Paket gönder ve mobility ile test
        sim.send_packet("M0", "M9", "Mobility Test")
        
        # 20 saniye simülasyon (mobility etkisi görmek için)
        sim.discrete_simulator.run_until(20.0)
        
        # Mobility istatistikleri
        mobility_stats = sim.get_mobility_stats()
        
        print(f"\nMobility İstatistikleri:")
        print(f"  Hareketli node sayısı: {mobility_stats['nodes_moving']}")
        print(f"  Ortalama hız: {mobility_stats['average_speed']:.2f} m/s")
        print(f"  Toplam mesafe: {mobility_stats['total_distance_traveled']:.1f} m")
        
        # Network metrikleri
        metrics = sim.get_network_stats()
        print(f"\nPerformans:")
        print(f"  Teslim oranı: {metrics['delivery_ratio']:.1f}%")
        print(f"  Route yeniden keşif: {sum(n.stats.get('route_rediscoveries', 0) for n in sim.nodes.values())}")
        
        success = mobility_stats['nodes_moving'] > 0 and metrics['delivery_ratio'] > 0
        print(f"\n{'✓ BAŞARILI' if success else '✗ BAŞARISIZ'}: Mobility çalışıyor ve paket teslim ediliyor")
        return success
    
    def scenario_4_protocol_comparison(self):
        """Senaryo 4: Protokol karşılaştırma - Aynı ağ, farklı protokoller"""
        from simulator import MANETSimulator
        
        print("\n" + "="*60)
        print("SENARYO 4: Protokol Karşılaştırma")
        print("="*60)
        
        # Sabit network oluştur
        sim = MANETSimulator()
        for i in range(15):
            sim.add_node(f"P{i}", x=(i % 5) * 100, y=(i // 5) * 100)
        
        print("✓ 15 node (5x3 grid) oluşturuldu")
        
        # Her protokol için aynı testleri çalıştır
        protocols = ["AODV", "OLSR", "DSR", "FLOODING", "ZRP"]
        comparison = {}
        
        test_pairs = [
            ("P0", "P14"),
            ("P2", "P12"),
            ("P5", "P9")
        ]
        
        for protocol in protocols:
            sim.routing_protocol = protocol
            sim.reset_metrics()
            
            # Her test çifti için paket gönder
            for source, dest in test_pairs:
                sim.send_packet(source, dest, f"{protocol}-test")
            
            # Simülasyon
            sim.discrete_simulator.run_until(15.0)
            
            # Metrikler
            stats = sim.get_network_stats()
            comparison[protocol] = stats
        
        # Sonuçları göster
        print(f"\n{'Protokol':<12} {'Teslim %':<12} {'Gecikme(ms)':<15} {'Overhead':<12}")
        print("-" * 60)
        
        for protocol, stats in comparison.items():
            print(f"{protocol:<12} "
                  f"{stats['delivery_ratio']:<12.1f} "
                  f"{stats['average_delay']*1000:<15.2f} "
                  f"{stats['routing_overhead']:<12.2f}")
        
        # En az 3 protokol çalışmalı
        working_protocols = sum(1 for s in comparison.values() if s['packets_delivered'] > 0)
        
        print(f"\n{'✓ BAŞARILI' if working_protocols >= 3 else '✗ BAŞARISIZ'}: {working_protocols}/5 protokol çalışıyor")
        return working_protocols >= 3
    
    def run_all_scenarios(self):
        """Tüm senaryoları çalıştır"""
        print("\n" + "="*80)
        print("FONKSİYONEL TEST SENARYOLARI")
        print("="*80)
        
        results = []
        
        results.append(("Temel Routing", self.scenario_1_basic_routing()))
        results.append(("Ölçeklenebilirlik", self.scenario_2_scalability()))
        results.append(("Mobility", self.scenario_3_mobility()))
        results.append(("Protokol Karşılaştırma", self.scenario_4_protocol_comparison()))
        
        print("\n" + "="*80)
        print("SENARYO ÖZET")
        print("="*80)
        
        for name, passed in results:
            print(f"{'✓' if passed else '✗'} {name}: {'BAŞARILI' if passed else 'BAŞARISIZ'}")
        
        total_passed = sum(1 for _, p in results if p)
        print(f"\nToplam: {total_passed}/{len(results)} senaryo başarılı")
        
        return total_passed == len(results)


if __name__ == "__main__":
    tester = FunctionalTests()
    success = tester.run_all_scenarios()
    exit(0 if success else 1)

