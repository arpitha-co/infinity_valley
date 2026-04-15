"""
Infinite Valley Navigation Environment
========================================
A custom reinforcement learning environment with continuous state and action spaces.

Concept Overview:
-----------------
The agent navigates an INFINITE 2D terrain with periodic hills and valleys.
The terrain creates non-linear dynamics through gravity effects (rolling downhill),
state-dependent friction, and velocity drag. There are NO boundaries - the agent
can explore infinitely in any direction. The camera follows the agent.

State Space:
- x, y: Position (UNBOUNDED - can be any real number)
- vx, vy: Velocity in [-5, 5] x [-5, 5]

Action Space:
- ax, ay: Thrust forces in [-1, 1] x [-1, 1]

Author: Custom RL Environment
Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm


class InfiniteValleyEnv:
    """
    Infinite Valley Navigation Environment
    
    A continuous control environment where an agent navigates an infinite 2D terrain
    with elevation-based non-linear dynamics. No boundaries!
    """
    
    def __init__(self):
        # INFINITE WORLD - no bounds, just view size for rendering
        self.view_size = 25.0  # Size of visible area around agent
        
        # Velocity limits
        self.v_max = 5.0
        
        # Action limits - full power
        self.a_max = 1.0
        
        # Physics parameters - CHALLENGING but solvable
        self.dt = 0.1                    # Time step
        self.gravity_strength = 0.7      # Moderate terrain gradient effect
        self.base_friction = 0.25        # Good friction
        self.drag = 0.05                 # Normal drag
        
        # Wind disturbance - mild randomness
        self.wind_strength = 0.15        # Mild wind force
        self.wind_frequency = 0.05       # How often wind changes
        self.current_wind = np.array([0.0, 0.0])
        
        # Terrain parameters - moderate challenge
        self.terrain_scale = 14.0        # Scale of terrain features
        self.terrain_amplitude = 1.8     # Reasonable hills
        
        # Goal configuration
        self.goal = np.array([60.0, 60.0])
        self.goal_radius = 2.0
        
        # Start configuration
        self.start = np.array([0.0, 0.0])
        
        # Episode configuration - more steps for harder challenge
        self.max_steps = 3000
        
        # State variables
        self.state = None
        self.steps = 0
        self.trajectory = []
        
        # For visualization
        self.fig = None
        self.ax = None
        
    def terrain_height(self, x, y):
        """
        Compute terrain height at position (x, y).
        
        INFINITE periodic terrain with STEEPER hills and valleys.
        """
        s = self.terrain_scale
        amp = self.terrain_amplitude
        
        # Multiple frequency components - more dramatic terrain
        h1 = np.sin(2 * np.pi * x / s) * np.cos(2 * np.pi * y / s)
        h2 = 0.6 * np.sin(4 * np.pi * x / s) * np.sin(4 * np.pi * y / s)
        h3 = 0.4 * np.cos(2 * np.pi * (x + y) / (s * 0.7))
        h4 = 0.3 * np.sin(2 * np.pi * (x - y) / (s * 1.3))
        # Extra high-frequency roughness
        h5 = 0.2 * np.sin(6 * np.pi * x / s) * np.cos(6 * np.pi * y / s)
        
        h = amp * (h1 + h2 + h3 + h4 + h5)
        return h
    
    def terrain_gradient(self, x, y):
        """
        Compute terrain gradient using numerical differentiation.
        
        Returns:
            (dh/dx, dh/dy): Gradient components
        """
        eps = 0.01
        h = self.terrain_height(x, y)
        dh_dx = (self.terrain_height(x + eps, y) - h) / eps
        dh_dy = (self.terrain_height(x, y + eps) - h) / eps
        
        return dh_dx, dh_dy
    
    def reset(self, start_pos=None):
        """
        Reset the environment to initial state.
        
        Args:
            start_pos: Optional custom starting position [x, y]
            
        Returns:
            Initial state vector [x, y, vx, vy]
        """
        if start_pos is not None:
            pos = np.array(start_pos, dtype=np.float64)
        else:
            pos = self.start.copy()
        
        # Initialize with zero velocity
        self.state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
        self.steps = 0
        self.trajectory = [pos.copy()]
        
        return self.state.copy()
    
    def step(self, action):
        """
        Take one step in the environment.
        
        Args:
            action: [ax, ay] thrust forces in [-1, 1]
            
        Returns:
            next_state: New state vector [x, y, vx, vy]
            reward: Scalar reward
            done: Boolean terminal flag
            info: Dictionary with additional information
        """
        # Clip action to valid range
        action = np.clip(action, -self.a_max, self.a_max)
        ax, ay = action
        
        # Current state
        x, y, vx, vy = self.state
        
        # Compute terrain gradient (gravity effect)
        dh_dx, dh_dy = self.terrain_gradient(x, y)
        gx = -dh_dx * self.gravity_strength
        gy = -dh_dy * self.gravity_strength
        
        # WIND DISTURBANCE - changes periodically
        if self.steps % int(1.0 / self.wind_frequency) == 0:
            self.current_wind = self.wind_strength * (np.random.randn(2))
        wind_x, wind_y = self.current_wind
        
        # Compute state-dependent friction (slippery on hills)
        height = self.terrain_height(x, y)
        friction = self.base_friction * (1 + 0.3 * height)
        friction = max(0.05, friction)  # Can be very slippery!
        
        # Update velocity with forces INCLUDING WIND
        vx_new = vx + (ax + gx + wind_x - friction * vx - self.drag * vx * abs(vx)) * self.dt
        vy_new = vy + (ay + gy + wind_y - friction * vy - self.drag * vy * abs(vy)) * self.dt
        
        # Clip velocity
        vx_new = np.clip(vx_new, -self.v_max, self.v_max)
        vy_new = np.clip(vy_new, -self.v_max, self.v_max)
        
        # Update position - NO BOUNDS, infinite world!
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        
        # Update state
        self.state = np.array([x_new, y_new, vx_new, vy_new])
        self.steps += 1
        self.trajectory.append(np.array([x_new, y_new]))
        
        # Compute reward
        dist_to_goal = np.linalg.norm(self.state[:2] - self.goal)
        reward = -0.01 * dist_to_goal  # Distance penalty
        reward -= 0.001 * (ax**2 + ay**2)  # Action penalty
        
        # Check terminal conditions (NO BOUNDS CHECK!)
        done = False
        info = {"success": False, "timeout": False, "distance": dist_to_goal}
        
        # Goal reached
        if dist_to_goal < self.goal_radius:
            reward += 100.0
            done = True
            info["success"] = True
        
        # Timeout only (no out-of-bounds!)
        if self.steps >= self.max_steps:
            done = True
            info["timeout"] = True
        
        return self.state.copy(), reward, done, info
    
    def render(self, show_trajectory=True, show_agent=True, figsize=(12, 10)):
        """
        Render the environment with terrain visualization.
        CAMERA FOLLOWS THE AGENT in the infinite world.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        else:
            self.ax.clear()
        
        ax = self.ax
        
        # Create terrain mesh - CAMERA FOLLOWS AGENT
        # STABILIZED SAMPLING: Align grid to global coordinates to prevent "shimmering"
        if self.state is not None:
            cx, cy = self.state[0], self.state[1]
        else:
            cx, cy = self.start[0], self.start[1]
        
        # Define view bounds
        half_view = self.view_size / 2
        view_xmin, view_xmax = cx - half_view, cx + half_view
        view_ymin, view_ymax = cy - half_view, cy + half_view
        
        # Grid resolution (fixed global step size)
        res = 0.2
        
        # Align grid points to global resolution
        # This prevents the terrain from "wobbling" or "shimmering" as the camera moves
        x_start = np.floor(view_xmin / res) * res
        x_end = np.ceil(view_xmax / res) * res
        y_start = np.floor(view_ymin / res) * res
        y_end = np.ceil(view_ymax / res) * res
        
        # Create grid with fixed global resolution
        x_grid = np.arange(x_start, x_end + res, res)
        y_grid = np.arange(y_start, y_end + res, res)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = self.terrain_height(X, Y)
        
        # Plot terrain as filled contour with nicer colormap
        # 'viridis' or 'plasma' looks more modern than 'terrain'
        contour = ax.contourf(X, Y, Z, levels=40, cmap='viridis', alpha=0.9)
        
        # Add subtle contour lines for texture
        ax.contour(X, Y, Z, levels=20, colors='white', linewidths=0.3, alpha=0.3)
        
        # Constant grid lines (optional, can remove if still jittery)
        # ax.contour(X, Y, Z, levels=12, colors='darkgray', linewidths=0.5, alpha=0.5)
        
        # Turn off axis ticks to prevent layout jitter as numbers change
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Colorbar (only create once)
        if not hasattr(self, '_colorbar') or self._colorbar is None:
            self._colorbar = self.fig.colorbar(contour, ax=ax, label='Elevation')
        
        # Plot start position (if in view)
        if (view_xmin <= self.start[0] <= view_xmax and 
            view_ymin <= self.start[1] <= view_ymax):
            # Start marker with glow effect
            ax.scatter(*self.start, s=300, c='lime', marker='o', 
                       edgecolors='white', linewidths=3, zorder=10, label='Start')
            ax.scatter(*self.start, s=500, c='lime', marker='o', 
                       alpha=0.3, zorder=9)
        
        # Plot goal position (if in view)
        if (view_xmin <= self.goal[0] <= view_xmax and 
            view_ymin <= self.goal[1] <= view_ymax):
            # Goal marker with glow effect
            goal_circle = Circle(self.goal, self.goal_radius, 
                                fill=True, facecolor='red', edgecolor='white',
                                alpha=0.3, linewidth=2, zorder=10)
            ax.add_patch(goal_circle)
            ax.scatter(*self.goal, s=400, c='red', marker='*', 
                       edgecolors='white', linewidths=2, zorder=11, label='Goal')
        
        # Show direction to goal if not in view
        if not (view_xmin <= self.goal[0] <= view_xmax and 
                view_ymin <= self.goal[1] <= view_ymax) and self.state is not None:
            # Draw arrow pointing to goal
            dx = self.goal[0] - self.state[0]
            dy = self.goal[1] - self.state[1]
            dist = np.sqrt(dx**2 + dy**2)
            arrow_len = 3.0
            ax.annotate(f'Goal: {dist:.1f}m', 
                       xy=(cx + dx/dist * arrow_len, cy + dy/dist * arrow_len),
                       xytext=(cx, cy), fontsize=10, color='red',
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Plot trajectory with gradient fading
        if show_trajectory and len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            # Plot main line
            ax.plot(traj[:, 0], traj[:, 1], 'w-', linewidth=2, alpha=0.8, label='Trajectory')
            # Add glow
            ax.plot(traj[:, 0], traj[:, 1], 'c-', linewidth=4, alpha=0.3)
            
            # Plot dots for recent part of trajectory
            recent = min(100, len(traj) - 1)
            ax.scatter(traj[-recent:, 0], traj[-recent:, 1], s=15, c='white', alpha=0.6)
        
        # Plot current agent position - Futuristic drone look
        if show_agent and self.state is not None:
            # Main body
            ax.scatter(self.state[0], self.state[1], s=400, c='cyan', 
                      marker='H', edgecolors='white', linewidths=2, zorder=15, label='Agent')
            # Glow
            ax.scatter(self.state[0], self.state[1], s=700, c='cyan', 
                      marker='H', alpha=0.3, zorder=14)
            
            # Show velocity vector
            vx, vy = self.state[2], self.state[3]
            scale = 1.0
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                ax.arrow(self.state[0], self.state[1], vx * scale, vy * scale,
                        head_width=0.4, head_length=0.4, fc='white', ec='white', zorder=16)
        
        ax.set_xlim(view_xmin, view_xmax)
        ax.set_ylim(view_ymin, view_ymax)
        
        # Modern styling
        ax.set_facecolor('#1a1a2e')  # Dark background
        
        # Show position in title with cleaner font
        if self.state is not None:
            dist = np.linalg.norm(self.state[:2] - self.goal)
            ax.set_title(f'Infinite Valley | Pos: ({self.state[0]:.1f}, {self.state[1]:.1f}) | Dist: {dist:.1f}', 
                        fontsize=14, fontweight='bold', color='black')
        else:
            ax.set_title('Infinite Valley Navigation', fontsize=16, fontweight='bold')
        
        # Remove axis ticks for clean look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Legend styling
        legend = ax.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', framealpha=0.8)
        
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    def close(self):
        """Close the rendering window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self._colorbar = None


def heuristic_controller(state, goal):
    """
    Improved heuristic controller that can overcome terrain forces.
    """
    x, y, vx, vy = state
    gx, gy = goal
    
    # Direction to goal
    dx = gx - x
    dy = gy - y
    dist = np.sqrt(dx**2 + dy**2) + 1e-8
    
    # Normalize direction
    dx_norm = dx / dist
    dy_norm = dy / dist
    
    # STRONGER proportional control to overcome terrain
    kp = 1.0   # Full proportional gain
    kd = 0.3   # Less damping - be more aggressive
    
    ax = kp * dx_norm - kd * vx
    ay = kp * dy_norm - kd * vy
    
    # Reduce thrust when close to goal
    if dist < 3.0:
        ax *= dist / 3.0
        ay *= dist / 3.0
    
    # Clip to valid range
    ax = np.clip(ax, -1.0, 1.0)
    ay = np.clip(ay, -1.0, 1.0)
    
    return np.array([ax, ay])


class ManualController:
    """
    Manual keyboard controller for interactive exploration.
    """
    
    def __init__(self, env):
        self.env = env
        self.action = np.array([0.0, 0.0])
        self.running = True
        self.thrust_magnitude = 0.8
        
    def on_key_press(self, event):
        if event.key == 'up':
            self.action[1] = self.thrust_magnitude
        elif event.key == 'down':
            self.action[1] = -self.thrust_magnitude
        elif event.key == 'left':
            self.action[0] = -self.thrust_magnitude
        elif event.key == 'right':
            self.action[0] = self.thrust_magnitude
        elif event.key == ' ':
            self.env.reset()
            print("Environment reset!")
        elif event.key == 'q':
            self.running = False
            print("Quitting...")
    
    def on_key_release(self, event):
        if event.key in ['up', 'down']:
            self.action[1] = 0.0
        elif event.key in ['left', 'right']:
            self.action[0] = 0.0
    
    def run(self, max_episodes=5):
        print("\n" + "="*60)
        print("MANUAL CONTROL MODE - INFINITE WORLD")
        print("="*60)
        print("Controls:")
        print("  Arrow Keys: Apply thrust in that direction")
        print("  SPACE: Reset environment")
        print("  Q: Quit")
        print("\nExplore freely - there are NO BOUNDARIES!")
        print("="*60 + "\n")
        
        state = self.env.reset()
        fig, ax = self.env.render()
        
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        
        plt.ion()
        plt.show()
        
        episode = 0
        total_reward = 0
        
        while self.running and episode < max_episodes:
            state, reward, done, info = self.env.step(self.action)
            total_reward += reward
            
            self.env.render()
            plt.pause(0.05)
            
            if done:
                if info["success"]:
                    print(f"Episode {episode + 1}: SUCCESS! Reward: {total_reward:.2f}")
                else:
                    print(f"Episode {episode + 1}: Timeout. Dist to goal: {info['distance']:.2f}")
                
                episode += 1
                total_reward = 0
                state = self.env.reset()
        
        plt.ioff()
        self.env.close()


def demo_heuristic_controller(live=True):
    """
    Demonstrate the environment is solvable using the heuristic controller.
    
    Args:
        live: If True, show live animation. If False, just show final result.
    """
    print("\n" + "="*60)
    print("HEURISTIC CONTROLLER DEMO - INFINITE WORLD")
    print("="*60)
    if live:
        print("LIVE MODE: Watch the agent solve in real-time!")
    print("="*60 + "\n")
    
    env = InfiniteValleyEnv()
    state = env.reset()
    
    print(f"Start: ({env.start[0]:.1f}, {env.start[1]:.1f})")
    print(f"Goal: ({env.goal[0]:.1f}, {env.goal[1]:.1f})")
    print(f"Distance: {np.linalg.norm(env.goal - env.start):.1f}")
    print()
    
    total_reward = 0
    
    if live:
        # Setup live visualization
        fig, ax = env.render()
        plt.ion()
        plt.show()
    
    for step in range(env.max_steps):
        action = heuristic_controller(state, env.goal)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Live rendering every few steps
        if live and step % 3 == 0:
            env.render()
            plt.pause(0.02)
        
        if step % 100 == 0:
            print(f"Step {step}: Pos=({state[0]:.1f}, {state[1]:.1f}), Dist={info['distance']:.1f}")
        
        if done:
            break
    
    if live:
        plt.ioff()
    
    # Final render
    fig, ax = env.render()
    
    print(f"\nEpisode completed in {step + 1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Success: {info['success']}")
    print(f"Final position: ({state[0]:.2f}, {state[1]:.2f})")
    
    if info['success']:
        print("\n✓ Environment is SOLVABLE!")
    
    plt.show()
    return info['success']


def visualize_terrain():
    """
    Visualize a region of the infinite terrain.
    """
    env = InfiniteValleyEnv()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Show a larger region of the infinite terrain
    x_grid = np.linspace(-30, 80, 200)
    y_grid = np.linspace(-30, 80, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = env.terrain_height(X, Y)
    
    ax1 = axes[0]
    c1 = ax1.contourf(X, Y, Z, levels=40, cmap='terrain')
    ax1.contour(X, Y, Z, levels=20, colors='black', linewidths=0.3, alpha=0.5)
    fig.colorbar(c1, ax=ax1, label='Elevation')
    ax1.scatter(*env.start, s=300, c='lime', marker='o', edgecolors='black', 
                linewidths=2, zorder=10, label='Start (0,0)')
    ax1.scatter(*env.goal, s=300, c='red', marker='*', edgecolors='black', 
                linewidths=2, zorder=10, label='Goal (50,50)')
    ax1.set_title('Infinite Periodic Terrain', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (extends infinitely)')
    ax1.set_ylabel('Y (extends infinitely)')
    ax1.legend()
    ax1.set_aspect('equal')
    
    ax2 = axes[1]
    dZ_dx, dZ_dy = env.terrain_gradient(X, Y)
    gradient_mag = np.sqrt(dZ_dx**2 + dZ_dy**2)
    c2 = ax2.contourf(X, Y, gradient_mag, levels=40, cmap='hot')
    fig.colorbar(c2, ax=ax2, label='Slope Steepness')
    ax2.scatter(*env.start, s=300, c='cyan', marker='o', edgecolors='black', 
                linewidths=2, zorder=10, label='Start')
    ax2.scatter(*env.goal, s=300, c='lime', marker='*', edgecolors='black', 
                linewidths=2, zorder=10, label='Goal')
    ax2.set_title('Terrain Difficulty (Gradient)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('infinite_terrain.png', dpi=150, bbox_inches='tight')
    print("Saved: infinite_terrain.png")
    plt.show()


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("INFINITE VALLEY NAVIGATION ENVIRONMENT")
    print("="*60)
    print("\nUsage:")
    print("  python infinite_valley_env.py demo     - Run heuristic controller demo")
    print("  python infinite_valley_env.py manual   - Run manual keyboard control")
    print("  python infinite_valley_env.py terrain  - Visualize terrain")
    print("  python infinite_valley_env.py          - Default: run demo")
    print("="*60)
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    if mode == "demo":
        demo_heuristic_controller()
    elif mode == "manual":
        env = InfiniteValleyEnv()
        controller = ManualController(env)
        controller.run()
    elif mode == "terrain":
        visualize_terrain()
    else:
        print(f"Unknown mode: {mode}")
        print("Valid modes: demo, manual, terrain")
