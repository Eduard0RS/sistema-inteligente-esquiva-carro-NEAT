import math
import random
import sys
import os
import neat
import pygame
import pickle

WIDTH = 800
HEIGHT = 600

CAR_SIZE_X = 50
CAR_SIZE_Y = 50

current_generation = 0

class Car:
    def __init__(self):
        # Create car
        self.sprite = pygame.image.load("car.png")
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.position = [WIDTH/2, 550]
        self.speed = 2
        self.center = [self.position[0] + CAR_SIZE_X/2, self.position[1] + CAR_SIZE_Y/2]
        self.alive = True

    def draw(self, screen):
        # Draw car
        screen.blit(self.sprite, self.position)

    def move(self):
        # Move car
        self.center = [self.position[0] + CAR_SIZE_X/2, self.position[1] + CAR_SIZE_Y/2]

    def move_right(self):
        # Move car
        if self.position[0] + self.speed <= 750:
            self.position[0] += self.speed
            self.center = [self.position[0] + CAR_SIZE_X/2, self.position[1] + CAR_SIZE_Y/2]

    def move_left(self):
        # Move car
        if self.position[0] - self.speed >= 0:
            self.position[0] -= self.speed
            self.center = [self.position[0] + CAR_SIZE_X/2, self.position[1] + CAR_SIZE_Y/2]

    def update(self):
        # Update car
        self.move()
        self.check_boundaries()

    def check_boundaries(self):
        # Check if car is out of screen
        if self.position[0] > WIDTH:
            self.position[0] = 0
        if self.position[0] < 0:
            self.position[0] = WIDTH
        if self.position[1] > HEIGHT:
            self.position[1] = 0
        if self.position[1] < 0:
            self.position[1] = HEIGHT

    def is_alive(self):
        # Basic alive function
        return self.alive

    def get_data(self,obstacle_x,obstacle_y):
        # Get data from car
        return [self.position[0], self.position[1], self.speed, obstacle_x, obstacle_y]

    def get_reward(self):
        # Calculate reward (maybe change?)
        return self.distance / (CAR_SIZE_X / 2)

def create_obstacle():
    obstacle_width = 50
    obstacle_height = 50
    obstacle_x = random.randint(0, WIDTH - obstacle_width)
    obstacle_y = 0
    return pygame.Rect(obstacle_x, obstacle_y, obstacle_width, obstacle_height)

def draw_obstacle(screen, obstacle_rect):
    # Desenhar obstáculo
    pygame.draw.rect(screen, (255, 0, 0), obstacle_rect)

def check_collision(car, obstacle):
    # Verificar colisão
    if car.position[0] < obstacle.x + obstacle.width and \
       car.position[0] + CAR_SIZE_X > obstacle.x and \
       car.position[1] < obstacle.y + obstacle.height and \
       car.position[1] + CAR_SIZE_Y > obstacle.y:
        car.alive = False
        return True
    else:
        return False


def run_simulator(genomes, config):
    nets = []
    cars = []
   
    # Carregar a fonte
    font = pygame.font.Font(None, 36)


    counter = 0
    clock = pygame.time.Clock()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Car AI")
    obstacle_rect = create_obstacle()
    obstacle_speed = 2

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    run = True
    global current_generation
    current_generation += 1
    while run:
        obstacle_rect.y += obstacle_speed
        draw_obstacle(screen, obstacle_rect)

         # Calcular o número de carros vivos
        still_alive = sum(car.is_alive() for car in cars)


        # Renderizar o texto com o número de carros vivos
        text = font.render("Vivos: " + str(still_alive), True, (0, 0, 0))


        # Exibir o texto na tela
        screen.blit(text, (50, 10))

        if obstacle_rect.y >= HEIGHT:
            obstacle_rect.x = random.randint(0, WIDTH - obstacle_rect.width)
            obstacle_rect.y = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()

        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data(obstacle_rect.x, obstacle_rect.y))
            choice = output.index(max(output))
            if choice == 0:
                car.move_left()
            elif choice == 1:
                car.move_right()
            check_collision(car, obstacle_rect)

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.speed += 0.01
                car.update()
                genomes[i][1].fitness += 0.1
        counter += 1
        if counter == 5000:
            run = False
            break

        if still_alive == 0:
            run = False
            break

        

        screen.fill((0, 0, 0))  # Preencher a tela com cor preta

        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Desenhar o obstáculo na tela
        pygame.draw.rect(screen, (255, 0, 0), obstacle_rect)
        obstacle_speed += 0.01

        pygame.display.update()
        clock.tick(60)
        pygame.display.flip()


if __name__ == "__main__":
    pygame.init()
    # Load config file
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                               neat.DefaultReproduction,
                               neat.DefaultSpeciesSet,
                               neat.DefaultStagnation,
                               config_path)

    # Create population
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run simulation
    winner = p.run(run_simulator, 10)

    # Save winner
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    # Show output
    print(winner)

    # Show stats
    print(stats)
