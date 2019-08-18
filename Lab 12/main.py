import numpy as np
import pygame
from simulation import Simulation
from line_follower import LineFollower
from track import Track
from utils import Vector2, Pose, Params, clamp
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, PIX2M, SIMULATION_SAMPLE_TIME, \
    DRAW_FREQUENCY, DEFAULT_ACCELERATED_FACTOR, MAX_ACCELERATED_FACTOR, MAX_EPISODE_TIME
from math import pi, inf
from reinforcement_learning import Sarsa, QLearning, compute_greedy_policy_as_table
import matplotlib.pyplot as plt


def configure_rl_algorithm():
    num_states = 10
    num_actions = 9
    epsilon = 0.05
    alpha = 0.1
    gamma = 0.99
    # Todo: comment/uncomment the lines below to select the desired algorithm
    # rl_algorithm = Sarsa(num_states, num_actions, epsilon, alpha, gamma)
    rl_algorithm = QLearning(num_states, num_actions, epsilon, alpha, gamma)
    return rl_algorithm


def capture_screen():
    """
    Captures the screen.
    """
    capture_window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    capture_window.fill((224, 255, 255))
    simulation.draw(capture_window)
    pygame.image.save(window, 'line_follower_solution.jpeg')


def plot_results():
    """
    Plots the results of the optimization.
    """
    fig_format = 'png'
    plt.figure()
    plt.plot(return_history)
    plt.xlabel('Iteration')
    plt.ylabel('Return')
    plt.title('Return (Discounted Cumulative Reward) Convergence')
    plt.grid()
    plt.savefig('return_convergence.%s' % fig_format, format=fig_format)
    plt.figure()
    plt.imshow(rl_algorithm.q)
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.title('Q (Action-Value) Table')
    plt.grid()
    plt.colorbar()
    plt.savefig('action_value_table.%s' % fig_format, format=fig_format)
    policy = compute_greedy_policy_as_table(rl_algorithm.q)
    plt.figure()
    plt.imshow(policy)
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.title('Greedy Policy Table')
    plt.grid()
    plt.colorbar()
    plt.savefig('greedy_policy_table.%s' % fig_format, format=fig_format)
    plt.show()


def process_input():
    """
    Processes input from keyboard.
    """
    global accelerated_mode, training, episode_time, quality, position, controller_params, accelerated_factor
    if keys[pygame.K_a] and not previous_keys[pygame.K_a]:
        accelerated_mode = not accelerated_mode
    if keys[pygame.K_t] and not previous_keys[pygame.K_t]:
        training = not training
        episode_time = 0
        simulation.reset(training)
    if keys[pygame.K_p] and not previous_keys[pygame.K_p]:
        plot_results()
    if keys[pygame.K_UP] and not previous_keys[pygame.K_UP]:
        accelerated_factor += 1
    if keys[pygame.K_DOWN] and not previous_keys[pygame.K_DOWN]:
        accelerated_factor -= 1
    if keys[pygame.K_LEFT] and not previous_keys[pygame.K_LEFT]:
        accelerated_factor -= 10
    if keys[pygame.K_RIGHT] and not previous_keys[pygame.K_RIGHT]:
        accelerated_factor += 10
    accelerated_factor = clamp(accelerated_factor, 1, MAX_ACCELERATED_FACTOR)

def print_text():
    """
    Prints help text on screen.
    """
    text = font.render('Episode time: %.1f/%.1f' % (episode_time, MAX_EPISODE_TIME), True, (0, 0, 0))
    window.blit(text, (round(0.1 * SCREEN_WIDTH), round(0.05 * SCREEN_HEIGHT)))
    text = font.render('Training iteration: ' + str(training_iteration), True, (0, 0, 0))
    window.blit(text, (round(0.6 * SCREEN_WIDTH), round(0.05 * SCREEN_HEIGHT)))
    text = font.render('Accelerated factor: ' + str(accelerated_factor) + 'x', True, (0, 0, 0))
    window.blit(text, (round(0.6 * SCREEN_WIDTH), round(0.1 * SCREEN_HEIGHT)))
    text = font.render('Training? ' + str(training), True, (0, 0, 0))
    window.blit(text, (round(0.6 * SCREEN_WIDTH), round(0.15* SCREEN_HEIGHT)))
    text = font.render('Accelerated mode? ' + str(accelerated_mode), True, (0, 0, 0))
    window.blit(text, (round(0.6 * SCREEN_WIDTH), round(0.2 * SCREEN_HEIGHT)))
    text = font.render('A: activate/deactivate accelerated mode', True, (0, 0, 0))
    window.blit(text, (round(0.1 * SCREEN_WIDTH), round(0.8 * SCREEN_HEIGHT)))
    text = font.render('T: activate/deactivate training', True, (0, 0, 0))
    window.blit(text, (round(0.1 * SCREEN_WIDTH), round(0.85 * SCREEN_HEIGHT)))
    text = font.render('P: plot optimization results', True, (0, 0, 0))
    window.blit(text, (round(0.1 * SCREEN_WIDTH), round(0.9 * SCREEN_HEIGHT)))


def format_position(position):
    return '[%.6f, %.6f, %.6f, %.6f]' % (position[0], position[1], position[2], position[3])


def convert_particle_position_to_params(position):
    """
    Converts a particle position into controller params.
    :param position: particle position.
    :type position: numpy array.
    :return: controller params.
    """
    params = Params()
    params.max_linear_speed_command = position[0]
    params.kp = position[1]
    params.ki = position[2]
    params.kd = position[3]
    return params


def create_simple_track():
    """
    Creates a simple track for a line follower robot.
    :return: the simple track.
    :rtype: Track.
    """
    track_width = 2.0
    track_height = 1.0
    screen_width_m = SCREEN_WIDTH * PIX2M
    screen_height_m = SCREEN_HEIGHT * PIX2M
    padding_y = screen_height_m - track_height
    padding_x = screen_width_m - track_width
    track = Track()
    track.add_line_piece(Vector2(padding_x / 2.0 + track_height / 2.0, padding_y / 2.0),
                         Vector2(screen_width_m - padding_x / 2.0 - track_height / 2.0, padding_y / 2.0))
    track.add_arc_piece(Vector2(screen_width_m - padding_x / 2.0 - track_height / 2.0, padding_y / 2.0 + track_height / 2.0),
                        track_height / 2.0, -pi / 2.0, pi / 2.0)
    track.add_line_piece(Vector2(screen_width_m - padding_x / 2.0 - track_height / 2.0, screen_height_m - padding_y / 2.0),
                         Vector2(padding_x / 2.0 + track_height / 2.0, screen_height_m - padding_y / 2.0))
    track.add_arc_piece(Vector2(padding_x / 2.0 + track_height / 2.0, padding_y / 2.0 + track_height / 2.0),
                        track_height / 2.0, pi / 2.0, 3.0 * pi / 2.0)
    return track


def create_complex_track():
    """
    Creates a more complex track for a line follower robot.
    :return: the complex track.
    :rtype: Track.
    """
    track_width = 2.0
    track_height = 1.0
    screen_width_m = SCREEN_WIDTH * PIX2M
    screen_height_m = SCREEN_HEIGHT * PIX2M
    padding_y = screen_height_m - track_height
    padding_x = screen_width_m - track_width
    track = Track()
    track.add_line_piece(Vector2(padding_x / 2.0 + track_height / 2.0, padding_y / 2.0),
                         Vector2(screen_width_m - padding_x / 2.0 - track_height / 2.0, padding_y / 2.0))
    track.add_arc_piece(Vector2(screen_width_m - padding_x / 2.0 - track_height / 2.0, padding_y / 2.0 + track_height / 2.0),
                        track_height / 2.0, -pi / 2.0, pi / 2.0)
    track.add_line_piece(Vector2(screen_width_m - padding_x / 2.0 - track_height / 2.0, screen_height_m - padding_y / 2.0),
                         Vector2(padding_x / 2.0 + track_height / 4.0, screen_height_m - padding_y / 2.0))
    track.add_arc_piece(Vector2(padding_x / 2.0 + track_height / 4.0, padding_y / 2.0 + 3.0 * track_height / 4.0),
                        track_height / 4.0, pi / 2.0, 3.0 * pi / 2.0)
    track.add_line_piece(Vector2(padding_x / 2.0 + track_height / 4.0, padding_y / 2.0 + track_height / 2.0),
                         Vector2(padding_x / 2.0 + track_height / 2.0, padding_y / 2.0 + track_height / 2.0))
    track.add_arc_piece(Vector2(padding_x / 2.0 + track_height / 2.0, padding_y / 2.0 + 3.0 * track_height / 8.0),
                        track_height / 8.0, -pi / 2.0, pi / 2.0)
    track.add_line_piece(Vector2(padding_x / 2.0 + track_height / 2.0, padding_y / 2.0 + track_height / 4.0),
                         Vector2(padding_x / 2.0 + track_height / 8.0, padding_y / 2.0 + track_height / 4.0))
    track.add_arc_piece(Vector2(padding_x / 2.0 + track_height / 8.0, padding_y / 2.0 + track_height / 8.0),
                        track_height / 8.0, pi / 2.0, 3.0 * pi / 2.0)
    track.add_line_piece(Vector2(padding_x / 2.0 + track_height / 8.0, padding_y / 2.0),
                         Vector2(padding_x / 2.0 + track_height / 2.0, padding_y / 2.0))
    return track


# Defining robot parameters
robot_params = Params()
robot_params.sensor_offset = 0.05
robot_params.max_wheel_speed = 45.0
robot_params.wheel_radius = 0.02
robot_params.wheels_distance = 0.05
robot_params.wheel_bandwidth = 10.0 * 2.0 * pi
# Defining line sensor parameters
sensor_params = Params()
sensor_params.sensor_range = 0.015
sensor_params.num_sensors = 7
sensor_params.array_width = 0.06
# Defining controller params (including learning algorithm)
linear_speed = 0.7
rl_algorithm = configure_rl_algorithm()
line_follower = LineFollower(Pose(0.0, 0.0, 0.0), rl_algorithm, linear_speed, robot_params, sensor_params)

# Creating track
# Switch to simple track if you are having trouble to make the robot learn in the complex track
track = create_complex_track()  # create_simple_track()

# Creating the simulation
simulation = Simulation(line_follower, track)

# Initializing pygame
pygame.init()
window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Lab 12 - Model-Free RL Algorithms")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 20, True)

# Initializing auxiliary variables
run = True  # if the program is running
accelerated_mode = False  # if the execution is in accelerated mode (faster than realtime)
training = True  # if the robot is training (the optimization is executing)
draw_path = True
# Obs.: if the robot is not training, the best solution found so far will be shown
# how much faster than realtime the simulation is executed in accelerated mode
accelerated_factor = DEFAULT_ACCELERATED_FACTOR
previous_keys = pygame.key.get_pressed()
episode_time = 0.0  # the elapsed time of the current episode
training_iteration = 1  # the number of the training iteration

# Initializing history
return_history = []

simulation.reset()

# Configure numpy print options
np.set_printoptions(suppress=True)

# Main loop
while run:
    clock.tick(DRAW_FREQUENCY)

    # Close the program if the quit button was pressed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Processing input
    keys = pygame.key.get_pressed()
    process_input()

    # Executing the simulation
    # To allow faster than realtime execution, the simulation executes num_steps
    # steps for each draw
    num_steps = 1
    if accelerated_mode:
        num_steps = accelerated_factor
    for i in range(num_steps):
        simulation.update()
        episode_time += SIMULATION_SAMPLE_TIME
        # If the episode has reached its end
        if episode_time >= MAX_EPISODE_TIME:
            # Prints the results of the current training iteration
            print('iter: ' + str(training_iteration) +
                  ', return: ' + str(line_follower.discounted_cumulative_reward))
            if training:
                # If the robot is training, update the optimization algorithm
                training_iteration += 1
                return_history.append(line_follower.discounted_cumulative_reward)
            else:
                capture_screen()  # Captures the screen at the end of the episode
            # Resetting the simulation to evaluate the new position
            episode_time = 0.0
            simulation.reset(training)
            break

    # Update the screen
    window.fill((224, 255, 255))
    simulation.draw(window)
    print_text()
    pygame.display.update()

    # Save the keyboard input for the next iteration
    previous_keys = keys

# Quitting pygame
pygame.quit()