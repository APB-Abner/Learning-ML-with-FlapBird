import pygame
import random
import numpy as np
import pickle

# Parâmetros do jogo
WIDTH = 800
HEIGHT = 600
BIRD_SIZE = 30
PIPE_WIDTH = 50
GAP_HEIGHT = 150
FPS = 60
NUM_BIRDS = 10

# Parâmetros de aprendizado
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 1000

# Inicializando o Pygame
pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 24)

# Parâmetros do pássaro e gravidade
JUMP_STRENGTH = -10
GRAVITY = 0.5

# Função de recompensa
def get_reward(alive, passed):
    if not alive:
        return -10  # Morreu
    if passed:
        return 1  # Passou por um cano
    return 0  # Continua vivo

# Função de desenhar todos os pássaros e canos na tela
def draw_all(birds, pipe_x, gap_y, score):
    # Limpar o buffer da tela
    win.fill((0, 0, 0))  # Limpar a tela antes de desenhar

    # Desenhar todos os pássaros
    for i, (y, color) in enumerate(birds):
        pygame.draw.rect(win, color, (50, y, BIRD_SIZE, BIRD_SIZE))

    # Desenhar o cano superior
    pygame.draw.rect(win, (0, 255, 0), (pipe_x, 0, PIPE_WIDTH, gap_y))  # Canos de cima
    # Desenhar o cano inferior
    pygame.draw.rect(win, (0, 255, 0), (pipe_x, gap_y + GAP_HEIGHT, PIPE_WIDTH, HEIGHT - gap_y - GAP_HEIGHT))  # Canos de baixo

    # Mostrar a pontuação
    text = font.render(f"Score: {score}", True, (255, 255, 255))
    win.blit(text, (10, 10))

    pygame.display.flip()  # Atualizar a tela

scores = [0] * NUM_BIRDS
# Criar uma Q-table fallback, caso nenhum pássaro sobreviva
q_fallback = np.zeros((10, 10, 10, 2))
# Inicializar uma lista para armazenar as 3 melhores Q-tables
best_q_tables = []

# Inicialização das Q-tables
q_tables = [np.zeros((10, 10, 10, 2)) for _ in range(NUM_BIRDS)]

# Função de estado com verificação de limites
def get_state(bird_y, pipe_x, gap_y):
    y_idx = int(bird_y // (HEIGHT / 10))
    dist_idx = int((pipe_x - 50) // (WIDTH / 10))
    gap_pos = gap_y - bird_y
    gap_idx = int(np.clip(gap_pos, -200, 200) // (400 / 10))  # Limitando de -200 a 200 para ficar dentro do range
    y_idx = max(0, min(9, y_idx))
    dist_idx = max(0, min(9, dist_idx))
    gap_idx = max(0, min(9, gap_idx))
    return y_idx, dist_idx, gap_idx

# Loop principal do treinamento
for ep in range(EPISODES):
    print(f"\n=== Episódio {ep + 1} ===")
    max_score = 0
    best_q_for_episode = []

    bird_y = [HEIGHT // 2 for _ in range(NUM_BIRDS)]
    bird_vel = [0 for _ in range(NUM_BIRDS)]
    pipe_x = [WIDTH for _ in range(NUM_BIRDS)]
    gap_y = [random.randint(100, HEIGHT - 200) for _ in range(NUM_BIRDS)]
    score = 0
    alive = [True for _ in range(NUM_BIRDS)]
    frames = [0 for _ in range(NUM_BIRDS)]
    colors = [tuple(np.random.randint(50, 255, size=3)) for _ in range(NUM_BIRDS)]

    while any(alive):
        clock.tick(FPS)

        # Processar eventos do Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()  # Encerra o programa corretamente

        # Atualizar a posição dos canos
        for i in range(NUM_BIRDS):
            pipe_x[i] -= 5  # Velocidade do movimento dos canos para a esquerda

            # Incrementar o score se o pássaro passar pelo cano
            if pipe_x[i] + PIPE_WIDTH < 100 and alive[i]:  # Verifica se o pássaro passou pelo cano
                score += 1
                print(f"Pássaro {i} passou pelo cano! Score: {score}")

            # Reiniciar o cano quando ele sair da tela
            if pipe_x[i] + PIPE_WIDTH < 0:
                pipe_x[i] = WIDTH
                gap_y[i] = random.randint(100, HEIGHT - 200)

        for i in range(NUM_BIRDS):
            if not alive[i]: continue

            state = get_state(bird_y[i], pipe_x[i], gap_y[i])

            # Verificação da tabela Q para o pássaro
            if q_tables[i] is None:
                print(f"⚠️ Q-table para o pássaro {i} não foi inicializada!")
                continue

            # Seleção da ação utilizando a tabela Q
            action = np.argmax(q_tables[i][state[0], state[1], state[2], :]) if random.random() > EPSILON else random.randint(0, 1)

            # Se o pássaro pular
            if action == 1:
                bird_vel[i] = JUMP_STRENGTH
            bird_vel[i] += GRAVITY
            bird_y[i] += bird_vel[i]

            if bird_y[i] < 0 or bird_y[i] + BIRD_SIZE > HEIGHT:
                alive[i] = False
            elif pipe_x[i] < 70 < pipe_x[i] + PIPE_WIDTH:
                if bird_y[i] < gap_y[i] or bird_y[i] + BIRD_SIZE > gap_y[i] + GAP_HEIGHT:
                    alive[i] = False

            reward = get_reward(alive[i], pipe_x[i] + PIPE_WIDTH < 50)  # Passou pelo cano?
            next_state = get_state(bird_y[i], pipe_x[i], gap_y[i])

            # Acesso correto às Q-tables
            old_q = q_tables[i][state[0], state[1], state[2], action]
            future_q = np.max(q_tables[i][next_state[0], next_state[1], next_state[2], :])

            # Atualizando Q-table
            q_tables[i][state[0], state[1], state[2], action] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)

        # Desenho do estado e pássaros
        draw_all([(bird_y[i], colors[i]) for i in range(NUM_BIRDS) if alive[i]], pipe_x[0], gap_y[0], score)

        if score > max_score:
            max_score = score
            best_q_for_episode = [np.copy(q_tables[i]) for i in range(NUM_BIRDS)]  # Guardar as Q-tables melhores

    # Seleção das 3 melhores Q-tables
    best_q_tables.extend(best_q_for_episode)
    best_q_tables = sorted(best_q_tables, key=lambda x: np.max(x), reverse=True)[:3]  # Manter as 3 melhores Q-tables

    # Se nenhum pássaro sobreviveu, usar a média das 3 melhores Q-tables
    if not any(alive):
        print("⚠️ Nenhum pássaro sobreviveu nesse episódio... usando fallback.")
        if best_q_tables:  # Verificar se há Q-tables disponíveis
            best_q_avg = np.mean(best_q_tables, axis=0)  # Média das 3 melhores Q-tables
        else:
            print("⚠️ Lista de melhores Q-tables está vazia. Usando Q-table de fallback.")
            best_q_avg = q_fallback  # Usar Q-table de fallback

        # Atualizar todas as Q-tables dos pássaros com a média ou fallback
        for i in range(NUM_BIRDS):
            q_tables[i] = np.copy(best_q_avg)

    # Mutação das Q-tables para a próxima geração
    for i in range(NUM_BIRDS):
        if not np.array_equal(q_tables[i], best_q_avg):
            q_tables[i] = best_q_avg + np.random.normal(0, 0.01, best_q_avg.shape).astype(np.float32)

    print(f"Melhor Score desse episódio: {max_score}")

# Salvar as Q-tables médias das melhores
with open("qtable_best_avg.pkl", "wb") as f:
    pickle.dump(best_q_tables, f)
print("✅ Q-tables médias salvas como qtable_best_avg.pkl")
