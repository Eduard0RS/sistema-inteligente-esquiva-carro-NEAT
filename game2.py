import pygame
import pickle
import sys
import neat
import random

WIDTH = 800
HEIGHT = 600

CAR_SIZE_X = 50
CAR_SIZE_Y = 50

class Car:
    def __init__(self):
        self.sprite = pygame.image.load("car.png")
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.position = [WIDTH/2, 550]
        self.speed = 2
        self.center = [self.position[0] + CAR_SIZE_X/2, self.position[1] + CAR_SIZE_Y/2]
        self.alive = True

    def draw(self, screen):
        screen.blit(self.sprite, self.position)

    def move(self):
        self.center = [self.position[0] + CAR_SIZE_X/2, self.position[1] + CAR_SIZE_Y/2]

    def move_right(self):
        if self.position[0] + self.speed <= 750:
            self.position[0] += self.speed
            self.center = [self.position[0] + CAR_SIZE_X/2, self.position[1] + CAR_SIZE_Y/2]

    def move_left(self):
        if self.position[0] - self.speed >= 0:
            self.position[0] -= self.speed
            self.center = [self.position[0] + CAR_SIZE_X/2, self.position[1] + CAR_SIZE_Y/2]

    def update(self):
        self.move()
        self.check_boundaries()

    def check_boundaries(self):
        if self.position[0] > WIDTH:
            self.position[0] = 0
        if self.position[0] < 0:
            self.position[0] = WIDTH
        if self.position[1] > HEIGHT:
            self.position[1] = 0
        if self.position[1] < 0:
            self.position[1] = HEIGHT

    def is_alive(self):
        return self.alive

    def get_data(self, obstacle_x, obstacle_y):
        return [self.position[0], self.position[1], self.speed, obstacle_x, obstacle_y]

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)


def create_obstacle():
    obstacle_width = 50
    obstacle_height = 50
    obstacle_x = random.randint(0, WIDTH - obstacle_width)
    obstacle_y = 0
    return pygame.Rect(obstacle_x, obstacle_y, obstacle_width, obstacle_height)


def draw_obstacle(screen, obstacle_rect):
    pygame.draw.rect(screen, (255, 0, 0), obstacle_rect)


def check_collision(car, obstacle):
    if car.position[0] < obstacle.x + obstacle.width and \
       car.position[0] + CAR_SIZE_X > obstacle.x and \
       car.position[1] < obstacle.y + obstacle.height and \
       car.position[1] + CAR_SIZE_Y > obstacle.y:
        car.alive = False
        print("Collision")
        return True
    else:
        print("No collision")
        return False


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Car AI - Game 2")

    # Carregar o vencedor do arquivo pickle
    with open("winner.pkl", "rb") as f:
        winner = pickle.load(f)
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                               neat.DefaultReproduction,
                               neat.DefaultSpeciesSet,
                               neat.DefaultStagnation,
                               config_path)
    # Criar instância do carro com o vencedor carregado
    car = Car()
    car.brain = neat.nn.FeedForwardNetwork.create(winner, config)

    obstacle_rect = create_obstacle()
    obstacle_speed = 2
    clock = pygame.time.Clock()

    run = True
    while run:
        obstacle_rect.y += obstacle_speed
        draw_obstacle(screen, obstacle_rect)
        if obstacle_rect.y >= HEIGHT:
            obstacle_rect.x = random.randint(0, WIDTH - obstacle_rect.width)
            obstacle_rect.y = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()

        # Obter os dados do carro
        data = car.get_data(obstacle_rect.x, obstacle_rect.y)

        # Alimentar os dados na rede neural
        output = car.brain.activate(data)

        # Tomar uma decisão com base na saída da rede neural
        choice = output.index(max(output))
        if choice == 0:
            car.move_left()
        elif choice == 1:
            car.move_right()
        car.update()

        if check_collision(car, obstacle_rect):
            run = False
            break

        

        screen.fill((0, 0, 0))
        car.draw(screen)
       
        pygame.draw.rect(screen, (255, 0, 0), obstacle_rect)
        obstacle_speed += 0.01
        car.speed += 0.01
        pygame.display.update()
        clock.tick(60)
        pygame.display.flip()


if __name__ == "__main__":
    main()
