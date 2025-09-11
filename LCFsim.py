import pygame
import random
import time

# --- Simulation Parameters ---
WIDTH, HEIGHT = 800, 800
FPS = 30
SPAWN_RATE = 0.3
MAX_WAIT = 30  # max wait time in seconds for starvation prevention

# --- Colors ---
BLACK = (0, 0, 0)
GREY = (80, 80, 80)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 150, 255)
WHITE = (255, 255, 255)

# --- Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

# --- Vehicle Class ---
class Vehicle:
    def __init__(self, x, y, dx, dy, lane):
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.lane = lane
        self.spawn_time = time.time()
        self.wait_time = 0
        self.passed_intersection = False

    def must_stop_before_box(self, cx, cy):
        box_size = 40
        buffer = 1
        box_left = cx - box_size // 2
        box_right = cx + box_size // 2
        box_top = cy - box_size // 2
        box_bottom = cy + box_size // 2
        next_x = self.x + self.dx
        next_y = self.y + self.dy
        if self.lane == "N" and next_y + buffer >= box_top:
            return True
        elif self.lane == "S" and next_y - buffer <= box_bottom:
            return True
        elif self.lane == "E" and next_x + buffer >= box_left:
            return True
        elif self.lane == "W" and next_x - buffer <= box_right:
            return True
        return False

    def is_inside_box(self, cx, cy):
        box_size = 40
        box_left = cx - box_size // 2
        box_right = cx + box_size // 2
        box_top = cy - box_size // 2
        box_bottom = cy + box_size // 2
        return box_left < self.x < box_right and box_top < self.y < box_bottom

    def move(self, green_lane, cx, cy):
        # Only move into intersection box if green lane
        if self.must_stop_before_box(cx, cy) and self.lane != green_lane:
            self.wait_time += 1 / FPS
            return
        else:
            self.x += self.dx
            self.y += self.dy
            # Mark passed intersection when crossing past center
            if not self.passed_intersection:
                if self.lane == 'N' and self.y >= cy:
                    self.passed_intersection = True
                elif self.lane == 'S' and self.y <= cy:
                    self.passed_intersection = True
                elif self.lane == 'E' and self.x >= cx:
                    self.passed_intersection = True
                elif self.lane == 'W' and self.x <= cx:
                    self.passed_intersection = True

    def draw(self):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), 6)

# --- Intersection Class with Preemptive LJF Scheduling ---
class Intersection:
    def __init__(self, cx, cy):
        self.cx, self.cy = cx, cy
        self.green_lane = None
        self.last_switch_time = time.time()

    def get_clumps(self, vehicles):
        lane_vehicles = {"N": [], "S": [], "E": [], "W": []}
        for v in vehicles:
            dist = abs(v.x - self.cx) + abs(v.y - self.cy)
            if dist < 100:
                lane_vehicles[v.lane].append(v)
        for lane in lane_vehicles:
            lane_vehicles[lane].sort(key=lambda v: v.wait_time, reverse=True)
        return lane_vehicles

    def pick_lane(self, vehicles):
        clumps = self.get_clumps(vehicles)
        current_clump_size = len(clumps[self.green_lane]) if self.green_lane else 0
        # Starvation prevention
        starving_lanes = []
        for lane, vlist in clumps.items():
            for v in vlist:
                if v.wait_time > MAX_WAIT:
                    starving_lanes.append(lane)
                    break
        if starving_lanes:
            starving_lanes.sort(key=lambda ln: len(clumps[ln]), reverse=True)
            new_green = starving_lanes[0]
            if new_green != self.green_lane:
                self.green_lane = new_green
                self.last_switch_time = time.time()
            return
        lane_sizes = {lane: len(vlist) for lane, vlist in clumps.items()}
        max_lane = max(lane_sizes, key=lane_sizes.get)
        max_size = lane_sizes[max_lane]
        if (self.green_lane is None) or (max_size > current_clump_size and max_size > 0):
            self.green_lane = max_lane
            self.last_switch_time = time.time()
            return
        if current_clump_size == 0:
            alternatives = {lane: size for lane, size in lane_sizes.items() if size > 0}
            if alternatives:
                next_lane = max(alternatives, key=alternatives.get)
                if next_lane != self.green_lane:
                    self.green_lane = next_lane
                    self.last_switch_time = time.time()
            else:
                self.green_lane = None

    def draw(self):
        pygame.draw.rect(screen, GREY, (self.cx - 20, self.cy - 20, 40, 40))
        positions = {
            "N": (self.cx, self.cy - 40),
            "S": (self.cx, self.cy + 40),
            "E": (self.cx + 40, self.cy),
            "W": (self.cx - 40, self.cy),
        }
        for lane, pos in positions.items():
            color = GREEN if lane == self.green_lane else RED
            pygame.draw.circle(screen, color, pos, 8)

# --- Main Setup ---
cx, cy = WIDTH // 2, HEIGHT // 2
intersection = Intersection(cx, cy)
vehicles = []

# --- Main Loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Spawn vehicles randomly at edges
    if random.random() < SPAWN_RATE:
        direction = random.choice(['N', 'S', 'E', 'W'])
        if direction == 'N':
            vehicles.append(Vehicle(cx, 0, 0, 3, 'N'))
        elif direction == 'S':
            vehicles.append(Vehicle(cx, HEIGHT, 0, -3, 'S'))
        elif direction == 'E':
            vehicles.append(Vehicle(0, cy, 3, 0, 'E'))
        elif direction == 'W':
            vehicles.append(Vehicle(WIDTH, cy, -3, 0, 'W'))
    intersection.pick_lane(vehicles)
    for v in vehicles:
        v.move(intersection.green_lane, cx, cy)
    # Removal code: instantly delete any stopped vehicle in intersection box if lane not green
    remaining_vehicles = []
    for v in vehicles:
        out_of_bounds = not (0 <= v.x <= WIDTH and 0 <= v.y <= HEIGHT)
        in_box = v.is_inside_box(cx, cy)
        stopped = (v.lane != intersection.green_lane) and in_box
        if out_of_bounds or stopped:
            print(f"Vehicle from {v.lane} removed at intersection after waiting {v.wait_time:.2f} seconds")
        else:
            remaining_vehicles.append(v)
    vehicles = remaining_vehicles
    screen.fill(BLACK)
    pygame.draw.line(screen, GREY, (cx, 0), (cx, HEIGHT), 40)
    pygame.draw.line(screen, GREY, (0, cy), (WIDTH, cy), 40)
    screen.blit(font.render("NORTH", True, WHITE), (cx - 30, 20))
    screen.blit(font.render("SOUTH", True, WHITE), (cx - 30, HEIGHT - 40))
    screen.blit(font.render("WEST", True, WHITE), (20, cy - 20))
    screen.blit(font.render("EAST", True, WHITE), (WIDTH - 80, cy - 20))
    intersection.draw()
    for v in vehicles:
        v.draw()
    pygame.display.flip()
    clock.tick(FPS)
pygame.quit()

