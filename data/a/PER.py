import pygame
import random
import numpy as np
import pickle
import os
import heapq
# deep reinforcement learning
# === CONFIGURAÇÕES ===
WIDTH = 800
HEIGHT = 600
BIRD_SIZE = 30
PIPE_WIDTH = 50
GAP_HEIGHT = 150
FPS = 60
NUM_BIRDS = 10

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 1000
VISUAL_MODE = False  # <- Toggle para ver ou não o jogo

JUMP_STRENGTH = -10
GRAVITY = 0.5

# === INICIALIZAÇÃO ===
if VISUAL_MODE:
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 24)
else:
    clock = pygame.time.Clock()

# === RECOMPENSA ===
def get_reward(alive, passed, just_died=False):
    if just_died:
        return -1000
    if passed:
        return 50
    return 1

# === DRAW ===
def draw_all(birds, pipe_x, gap_y, score):
    if not VISUAL_MODE:
        return
    win.fill((0, 0, 0))
    for i, (y, color) in enumerate(birds):
        pygame.draw.rect(win, color, (50, y, BIRD_SIZE, BIRD_SIZE))
    pygame.draw.rect(win, (0, 255, 0), (pipe_x, 0, PIPE_WIDTH, gap_y))
    pygame.draw.rect(win, (0, 255, 0), (pipe_x, gap_y + GAP_HEIGHT, PIPE_WIDTH, HEIGHT - gap_y - GAP_HEIGHT))
    text = font.render(f"Score: {score}", True, (255, 255, 255))
    win.blit(text, (10, 10))
    pygame.display.flip()

# === ESTADO ===
def get_state(bird_y, bird_vel, pipe_x, gap_y):
    y_idx = int(np.clip(bird_y, 0, HEIGHT - 1) // (HEIGHT / 10))
    vel_idx = int(np.clip(bird_vel, -15, 15) + 15) // 3  # 10 buckets
    dist_idx = int(np.clip(pipe_x - 50, 0, WIDTH) // (WIDTH / 10))
    gap_rel = gap_y - bird_y
    gap_idx = int(np.clip(gap_rel, -200, 200) // (400 / 10))

    # Garante que todos os índices fiquem entre 0 e 9
    y_idx = int(np.clip(y_idx, 0, 9))
    vel_idx = int(np.clip(vel_idx, 0, 9))
    dist_idx = int(np.clip(dist_idx, 0, 9))
    gap_idx = int(np.clip(gap_idx, 0, 9))

    return y_idx, vel_idx, dist_idx, gap_idx

# === PRIORITIZED EXPERIENCE REPLAY ===
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, experience, priority):
        if len(self.buffer) < self.capacity:
            heapq.heappush(self.buffer, experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        # Amostragem com probabilidade proporcional à prioridade
        priorities = np.array(self.priorities)
        prob = priorities**self.alpha
        prob /= prob.sum()
        
        indices = np.random.choice(len(self.buffer), size=batch_size, p=prob)
        
        batch = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * prob[indices]) ** (-beta)
        weights /= weights.max()
        
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        # Atualize as prioridades com base no erro de TD
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5  # Um pequeno valor para evitar prioridades 0

# === Q-TABLE SETUP ===
q_fallback = np.zeros((10, 10, 10, 10, 2))  # y, vel, dist, gap, action
q_tables = [np.copy(q_fallback) for _ in range(NUM_BIRDS)]
best_q_tables = []

save_path = "qtable_best_avg.pkl"
if os.path.exists(save_path):
    with open(save_path, "rb") as f:
        best_q_tables = pickle.load(f)
        q_fallback = np.mean(best_q_tables, axis=0)
    print("✅ Q-tables carregadas com sucesso!")
else:
    best_q_tables = []
    
# === INICIALIZAÇÃO DO REPLAY BUFFER ===
replay_buffer = PrioritizedReplayBuffer(capacity=10000)

# === TREINAMENTO ===
try:
    for ep in range(EPISODES):
        print(f"\n=== Episódio {ep + 1} ===")
        max_score = 0
        best_q_for_episode = []

        bird_y = [HEIGHT // 2 for _ in range(NUM_BIRDS)]
        bird_vel = [0 for _ in range(NUM_BIRDS)]
        pipe_x = WIDTH
        gap_y = random.randint(100, HEIGHT - 200)
        score = 0
        alive = [True] * NUM_BIRDS
        colors = [tuple(np.random.randint(50, 255, size=3)) for _ in range(NUM_BIRDS)]

        best_q_avg = np.mean(best_q_tables, axis=0) if best_q_tables else q_fallback

        while any(alive):
            clock.tick(60000 if not VISUAL_MODE else FPS)

            if VISUAL_MODE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        with open("checkpoint.pkl", "wb") as f:
                            pickle.dump((q_tables, best_q_tables), f)
                        pygame.quit()
                        exit()

            pipe_x -= 10

            passou = False
            if pipe_x + PIPE_WIDTH < 0:
                pipe_x = WIDTH
                gap_y = random.randint(100, HEIGHT - 200)
                score += 1
                passou = True
                print(f"Pássaros vivos passaram por um cano! Score: {score}")
                if score > max_score:
                    max_score = score
                    best_q_for_episode = [np.copy(q_tables[i]) for i in range(NUM_BIRDS) if alive[i]]

            for i in range(NUM_BIRDS):
                if not alive[i]: continue

                state = get_state(bird_y[i], bird_vel[i], pipe_x, gap_y)
                action = np.argmax(q_tables[i][*state]) if random.random() > EPSILON else random.randint(0, 1)

                if action == 1:
                    bird_vel[i] = JUMP_STRENGTH
                bird_vel[i] += GRAVITY
                bird_y[i] += bird_vel[i]

                just_died = False

                if bird_y[i] < 0 or bird_y[i] + BIRD_SIZE > HEIGHT:
                    just_died = True
                    alive[i] = False
                elif pipe_x < 70 < pipe_x + PIPE_WIDTH:
                    if bird_y[i] < gap_y or bird_y[i] + BIRD_SIZE > gap_y + GAP_HEIGHT:
                        just_died = True
                        alive[i] = False

                reward = get_reward(alive[i], passou, just_died)
                next_state = get_state(bird_y[i], bird_vel[i], pipe_x, gap_y)

                old_q = q_tables[i][*state, action]
                future_q = np.max(q_tables[i][*next_state])
                td_error = reward + GAMMA * future_q - old_q

                # Armazenando transições no replay buffer com prioridade
                replay_buffer.add((state, action, reward, next_state, alive[i]), abs(td_error))

                # Atualiza a Q-table
                q_tables[i][*state, action] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)

            # Amostragem de minibatch do replay buffer
            batch, indices, weights = replay_buffer.sample(batch_size=32)

            # Atualiza as prioridades no replay buffer
            td_errors = []
            for b in batch:
                state, action, reward, next_state, done = b
                future_q = np.max(q_tables[i][*next_state])
                td_error = reward + GAMMA * future_q - np.max(q_tables[i][*state, action])
                td_errors.append(td_error)
            replay_buffer.update_priorities(indices, td_errors)

            draw_all([(bird_y[i], colors[i]) for i in range(NUM_BIRDS) if alive[i]], pipe_x, gap_y, score)

        if best_q_for_episode:
            best_score_new = max_score
            best_score_existing = max([np.max(q) for q in best_q_tables], default=0)
            if best_score_new > best_score_existing:
                best_q_tables.extend(best_q_for_episode)
                best_q_tables = sorted(best_q_tables, key=lambda x: np.max(x), reverse=True)[:3]

        if not any(alive):
            print("⚠️ Nenhum pássaro sobreviveu... usando fallback.")
            best_q_avg = np.mean(best_q_tables, axis=0) if best_q_tables else q_fallback
            for i in range(NUM_BIRDS):
                q_tables[i] = np.copy(best_q_avg)

        for i in range(NUM_BIRDS):
            if not np.array_equal(q_tables[i], best_q_avg):
                q_tables[i] = best_q_avg + np.random.normal(0, 0.01, best_q_avg.shape).astype(np.float32)

        print(f"🏁 Score do episódio: {max_score}")
        print(f"🎯 Média das melhores Q-tables: {np.mean([np.max(q) for q in best_q_tables]):.2f}")

        with open("checkpoint.pkl", "wb") as f:
            pickle.dump((q_tables, best_q_tables), f)

except KeyboardInterrupt:
    print("⛔ Interrompido manualmente. Salvando progresso...")
    with open(save_path, "wb") as f:
        pickle.dump(best_q_tables, f)
    pygame.quit()
    exit()

with open(save_path, "wb") as f:
    pickle.dump(best_q_tables, f)

print("✅ Q-tables salvas com sucesso!")

if VISUAL_MODE:
    pygame.quit()