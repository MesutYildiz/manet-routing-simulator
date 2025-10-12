"""
Simplified IEEE 802.11 MAC layer for MANET simulator
"""
import random

class IEEE80211MAC:
    """Simplified IEEE 802.11 DCF (Distributed Coordination Function)"""
    
    def __init__(self):
        # IEEE 802.11 timing parameters (in seconds)
        self.SIFS = 0.000010   # Short Inter-Frame Space: 10 microseconds
        self.DIFS = 0.000050   # DCF Inter-Frame Space: 50 microseconds
        self.slot_time = 0.000020  # Slot time: 20 microseconds
        
        # Contention window parameters
        self.CW_min = 31       # Minimum contention window
        self.CW_max = 1023     # Maximum contention window
        
        # Transmission parameters
        self.retry_limit = 7   # Maximum retry attempts
        self.data_rate = 250000  # 250 kbps (bytes per second)
    
    def calculate_backoff(self, retry_count):
        """Calculate exponential backoff time"""
        # Calculate contention window
        cw = min(self.CW_min * (2 ** retry_count), self.CW_max)
        
        # Random backoff slots
        backoff_slots = random.randint(0, cw)
        
        # Total backoff time
        backoff_time = self.DIFS + (backoff_slots * self.slot_time)
        
        return backoff_time
    
    def calculate_transmission_time(self, packet_size_bytes):
        """Calculate transmission time for a packet"""
        # Transmission time = packet size / data rate
        tx_time = (packet_size_bytes * 8) / self.data_rate
        
        # Add SIFS for ACK waiting
        total_time = tx_time + self.SIFS
        
        return total_time
    
    def should_retry(self, retry_count):
        """Check if packet should be retried after failure"""
        return retry_count < self.retry_limit
    
    def get_ack_timeout(self):
        """Get ACK timeout duration"""
        return self.SIFS * 2  # Wait 2 SIFS for ACK


class CSMACAHandler:
    """CSMA/CA (Carrier Sense Multiple Access with Collision Avoidance)"""
    
    def __init__(self, mac: IEEE80211MAC):
        self.mac = mac
        self.transmission_attempts = {}  # node_id -> retry_count
    
    def request_channel(self, node_id, is_busy):
        """Request channel access using CSMA/CA"""
        # Get retry count for this node
        retry_count = self.transmission_attempts.get(node_id, 0)
        
        if is_busy:
            # Channel busy - increment retry and calculate backoff
            self.transmission_attempts[node_id] = retry_count + 1
            backoff_time = self.mac.calculate_backoff(retry_count)
            
            return {
                'granted': False,
                'backoff_time': backoff_time,
                'retry_count': retry_count + 1
            }
        else:
            # Channel free - grant access after DIFS
            return {
                'granted': True,
                'backoff_time': self.mac.DIFS,
                'retry_count': retry_count
            }
    
    def transmission_success(self, node_id):
        """Reset retry count after successful transmission"""
        if node_id in self.transmission_attempts:
            del self.transmission_attempts[node_id]
    
    def transmission_failed(self, node_id):
        """Handle transmission failure"""
        retry_count = self.transmission_attempts.get(node_id, 0)
        
        if self.mac.should_retry(retry_count):
            return {
                'should_retry': True,
                'backoff_time': self.mac.calculate_backoff(retry_count)
            }
        else:
            # Max retries reached - drop packet
            if node_id in self.transmission_attempts:
                del self.transmission_attempts[node_id]
            return {
                'should_retry': False,
                'backoff_time': 0
            }


# Integration helper for simulator
def integrate_mac_layer(simulator):
    """Integrate IEEE 802.11 MAC layer into simulator"""
    simulator.mac_layer = IEEE80211MAC()
    simulator.csma_handler = CSMACAHandler(simulator.mac_layer)
    
    print("✓ IEEE 802.11 MAC layer integrated")
    return simulator.mac_layer

