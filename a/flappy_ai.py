import pygame
import numpy as np
import random
import time

# Constantes do jogo
WIDTH, HEIGHT = 400, 600
FPS = 30
PIPE_WIDTH = 80
GAP_HEIGHT = 150
BIRD_SIZE = 20
GRAVITY = 1
JUMP_STRENGTH = -10

# Parâmetros Q-Learning
ALTURAS = 30
DISTANCIAS = 20
ACTIONS = 2
q_table = np.zeros((ALTURAS, DISTANCIAS, ACTIONS))
alpha = 0.1
gamma = 0.95
epsilon = 0.1
episodes = 500

# Iniciar pygame
pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

def draw(bird_y, pipe_x, gap_y, score):
    win.fill((0, 0, 0))

    # Pássaro
    pygame.draw.rect(win, (255, 255, 0), (50, bird_y, BIRD_SIZE, BIRD_SIZE))

    # Cano
    pygame.draw.rect(win, (0, 255, 0), (pipe_x, 0, PIPE_WIDTH, gap_y))
    pygame.draw.rect(win, (0, 255, 0), (pipe_x, gap_y + GAP_HEIGHT, PIPE_WIDTH, HEIGHT))

    # Score
    text = font.render(f"Score: {score}", True, (255, 255, 255))
    win.blit(text, (10, 10))

    pygame.display.flip()

def get_state(bird_y, pipe_x, gap_y):
    y_idx = int(bird_y // (HEIGHT / ALTURAS))
    dist_idx = int((pipe_x - 50) // (WIDTH / DISTANCIAS))
    return max(0, min(ALTURAS - 1, y_idx)), max(0, min(DISTANCIAS - 1, dist_idx))

def get_reward(alive, passed):
    if not alive:
        return -100
    return 1 + (10 if passed else 0)

# Treinamento
NUM_BIRDS = 10
q_tables = [np.zeros((ALTURAS, DISTANCIAS, ACTIONS)) for _ in range(NUM_BIRDS)]
scores = [0] * NUM_BIRDS

for ep in range(episodes):
    print(f"\n=== Episódio {ep + 1} ===")
    max_score = 0
    best_q = None

    for i in range(NUM_BIRDS):
        bird_y = HEIGHT // 2
        bird_vel = 0
        pipe_x = WIDTH
        gap_y = random.randint(100, HEIGHT - 200)
        score = 0
        done = False
        frames = 0

        while not done:
            frames += 1
            state = get_state(bird_y, pipe_x, gap_y)
            action = np.argmax(q_tables[i][state]) if random.random() > epsilon else random.randint(0, 1)

            if action == 1:
                bird_vel = JUMP_STRENGTH
            bird_vel += GRAVITY
            bird_y += bird_vel
            pipe_x -= 5

            if pipe_x + PIPE_WIDTH < 0:
                pipe_x = WIDTH
                gap_y = random.randint(100, HEIGHT - 200)
                score += 1
                passed = True
            else:
                passed = False

            alive = True
            if bird_y < 0 or bird_y + BIRD_SIZE > HEIGHT:
                alive = False
            elif pipe_x < 70 < pipe_x + PIPE_WIDTH:
                if bird_y < gap_y or bird_y + BIRD_SIZE > gap_y + GAP_HEIGHT:
                    alive = False

            reward = get_reward(alive, passed)
            next_state = get_state(bird_y, pipe_x, gap_y)
            old_q = q_tables[i][state][action]
            future_q = np.max(q_tables[i][next_state])
            q_tables[i][state][action] = old_q + alpha * (reward + gamma * future_q - old_q)

            if not alive:
                done = True

        scores[i] = score
        print(f" Pássaro {i+1} - Score: {score} - Frames vivos: {frames}")

        if score > max_score:
            max_score = score
            best_q = np.copy(q_tables[i])

    # Evolução: herdar o melhor Q-table com mutações
    if best_q is not None:
        for i in range(NUM_BIRDS):
            if not np.array_equal(q_tables[i], best_q):
                q_tables[i] = best_q + np.random.normal(0, 0.01, best_q.shape)
    else:
        print("⚠️ Nenhum pássaro sobreviveu nesse episódio... tentando de novo.")


print("FIM do treinamento coletivo! 🧠💪")

pygame.quit()
print("Treinamento finalizado. O pássaro tá ninja agora! 😎")
