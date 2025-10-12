"""
Mobility models for MANET simulator
"""
import random
import math
from models import MobilityParameters

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
