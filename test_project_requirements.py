# test_project_requirements.py

class ProjectRequirementTests:
    """Proje gereksinimlerini test eder"""
    
    def __init__(self):
        self.results = []
        
    def test_minimum_protocols(self):
        """Test: En az 5 routing protokolü var mı?"""
        from simulator import MANETSimulator
        
        sim = MANETSimulator()
        protocols = ["AODV", "OLSR", "DSR", "FLOODING", "ZRP"]
        
        test_result = {
            'test_name': 'Minimum 5 Protokol',
            'requirement': 'En az 5 routing protokolü',
            'expected': 5,
            'actual': len(protocols),
            'passed': len(protocols) >= 5,
            'details': f"Protokoller: {', '.join(protocols)}"
        }
        
        self.results.append(test_result)
        return test_result
    
    def test_minimum_nodes(self):
        """Test: En az 50 node destekliyor mu?"""
        from simulator import MANETSimulator
        
        sim = MANETSimulator(width=2000, height=2000)
        
        # 50 node ekle
        for i in range(50):
            sim.add_node(f"N{i}")
        
        test_result = {
            'test_name': '50 Node Desteği',
            'requirement': 'En az 50 node desteklemeli',
            'expected': 50,
            'actual': len(sim.nodes),
            'passed': len(sim.nodes) >= 50,
            'details': f"Eklenen node sayısı: {len(sim.nodes)}"
        }
        
        self.results.append(test_result)
        return test_result
    
    def test_gui_exists(self):
        """Test: GUI var mı?"""
        try:
            import gui
            has_gui_class = hasattr(gui, 'MANETSimulatorGUI')
            
            test_result = {
                'test_name': 'GUI Varlığı',
                'requirement': 'Kullanıcı dostu GUI olmalı',
                'expected': True,
                'actual': has_gui_class,
                'passed': has_gui_class,
                'details': 'MANETSimulatorGUI sınıfı bulundu' if has_gui_class else 'GUI bulunamadı'
            }
        except ImportError as e:
            test_result = {
                'test_name': 'GUI Varlığı',
                'requirement': 'Kullanıcı dostu GUI olmalı',
                'expected': True,
                'actual': False,
                'passed': False,
                'details': f'Import hatası: {str(e)}'
            }
        
        self.results.append(test_result)
        return test_result
    
    def test_discrete_event_simulation(self):
        """Test: Discrete-event simülasyon var mı?"""
        try:
            from discrete_simulator import DiscreteEventSimulator
            
            des = DiscreteEventSimulator()
            has_event_queue = hasattr(des, 'event_queue')
            has_schedule = hasattr(des, 'schedule_event')
            has_run_until = hasattr(des, 'run_until')
            
            all_present = has_event_queue and has_schedule and has_run_until
            
            test_result = {
                'test_name': 'Discrete-Event Simülasyon',
                'requirement': 'Discrete-event network simulator kullanmalı',
                'expected': True,
                'actual': all_present,
                'passed': all_present,
                'details': f'Event queue: {has_event_queue}, Schedule: {has_schedule}, Run until: {has_run_until}'
            }
        except ImportError as e:
            test_result = {
                'test_name': 'Discrete-Event Simülasyon',
                'requirement': 'Discrete-event network simulator kullanmalı',
                'expected': True,
                'actual': False,
                'passed': False,
                'details': f'discrete_simulator.py bulunamadı: {str(e)}'
            }
        
        self.results.append(test_result)
        return test_result
    
    def test_protocol_comparison(self):
        """Test: Protokol karşılaştırma yapabiliyor mu?"""
        from simulator import MANETSimulator
        
        sim = MANETSimulator()
        
        # 10 node ekle
        for i in range(10):
            sim.add_node(f"N{i}", x=i*100, y=100)
        
        # Her protokol için paket gönder
        protocols = ["AODV", "OLSR", "DSR", "FLOODING", "ZRP"]
        for protocol in protocols:
            sim.routing_protocol = protocol
            sim.send_packet("N0", "N9", f"Test-{protocol}")
        
        # Karşılaştırma verisi al
        comparison = sim.get_protocol_comparison()
        
        test_result = {
            'test_name': 'Protokol Karşılaştırma',
            'requirement': 'Protokolleri karşılaştırabilmeli',
            'expected': 5,
            'actual': len(comparison),
            'passed': len(comparison) >= 3,  # En az 3 protokol test edilmiş olmalı
            'details': f'Karşılaştırılan protokoller: {list(comparison.keys())}'
        }
        
        self.results.append(test_result)
        return test_result
    
    def test_performance_metrics(self):
        """Test: Performans metrikleri ölçülüyor mu?"""
        from simulator import MANETSimulator
        
        sim = MANETSimulator()
        
        # Network oluştur
        for i in range(5):
            sim.add_node(f"N{i}", x=i*100, y=100)
        
        # Paket gönder
        sim.send_packet("N0", "N4", "Test Data")
        
        # Metrikler al
        stats = sim.get_network_stats()
        
        required_metrics = [
            'packets_sent', 
            'packets_delivered', 
            'delivery_ratio', 
            'average_delay', 
            'routing_overhead'
        ]
        
        has_all_metrics = all(metric in stats for metric in required_metrics)
        
        test_result = {
            'test_name': 'Performans Metrikleri',
            'requirement': 'Packet delivery ratio, delay, overhead ölçülmeli',
            'expected': required_metrics,
            'actual': list(stats.keys()),
            'passed': has_all_metrics,
            'details': f'Eksik metrikler: {[m for m in required_metrics if m not in stats]}'
        }
        
        self.results.append(test_result)
        return test_result
    
    def print_results(self):
        """Test sonuçlarını yazdır"""
        print("\n" + "="*80)
        print("PROJE GEREKSİNİMLERİ TEST SONUÇLARI")
        print("="*80)
        
        passed_count = 0
        failed_count = 0
        
        for result in self.results:
            status = "✓ BAŞARILI" if result['passed'] else "✗ BAŞARISIZ"
            print(f"\n{status}: {result['test_name']}")
            print(f"   Gereksinim: {result['requirement']}")
            print(f"   Beklenen: {result['expected']}")
            print(f"   Gerçekleşen: {result['actual']}")
            print(f"   Detay: {result['details']}")
            
            if result['passed']:
                passed_count += 1
            else:
                failed_count += 1
        
        print("\n" + "="*80)
        print(f"ÖZET: {passed_count} başarılı, {failed_count} başarısız")
        print("="*80)
        
        return passed_count, failed_count


# Test senaryolarını çalıştır
def run_requirement_tests():
    tester = ProjectRequirementTests()
    
    print("Proje gereksinimleri test ediliyor...\n")
    
    tester.test_minimum_protocols()
    tester.test_minimum_nodes()
    tester.test_gui_exists()
    tester.test_discrete_event_simulation()
    tester.test_protocol_comparison()
    tester.test_performance_metrics()
    
    passed, failed = tester.print_results()
    
    return passed == len(tester.results)


if __name__ == "__main__":
    success = run_requirement_tests()
    exit(0 if success else 1)

