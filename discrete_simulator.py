"""
Discrete event simulator for MANET
"""
import heapq
from models import Event

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
            try:
                self.event_handlers[event.event_type](event)
            except Exception as e:
                print(f"ERROR in event handler {event.event_type}: {e}")
                import traceback
                traceback.print_exc()
        
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
