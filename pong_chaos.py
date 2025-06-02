import pygame
import sys
import random
import math
import numpy as np

# Initialisation de Pygame
pygame.init()

# Dimensions de la fenêtre
WIDTH, HEIGHT = 800, 600

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Création de la fenêtre
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

# Raquettes
paddle_width, paddle_height = 15, 100
player_paddle = pygame.Rect(50, HEIGHT // 2 - paddle_height // 2, paddle_width, paddle_height)
opponent_paddle = pygame.Rect(WIDTH - 50 - paddle_width, HEIGHT // 2 - paddle_height // 2, paddle_width, paddle_height)

# Balle
ball_radius = 10
ball_diameter = ball_radius * 2 # Déplacé ici pour être utilisé globalement

def reset_ball():
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    while True:
        angle = random.uniform(-math.pi / 6, math.pi / 6) # Angle plus restreint pour éviter les balles trop verticales
        speed = 4 # Légèrement augmenté
        speed_x = speed * math.cos(angle)
        speed_y = speed * math.sin(angle)
        if random.choice([-1, 1]) < 0:
            speed_x *= -1
        # S'assurer que la balle ne part pas presque verticalement (ou trop lentement horizontalement)
        if abs(speed_x) > 1.5 and abs(speed_y) < abs(speed_x) * 0.8: # Condition pour une trajectoire plus "pong-like"
            break
    return np.array([
        WIDTH // 2 - ball_radius,
        HEIGHT // 2 - ball_radius,
        speed_x,
        speed_y,
        color[0], color[1], color[2]
    ], dtype=np.float32)

balls_data = np.array([reset_ball()])

player_speed = 0
paddle_speed_val = 7 # Renommé pour clarté
opponent_paddle_speed_val = paddle_speed_val * 0.8 # IA un peu moins parfaite

player_score = 0
opponent_score = 0
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

TRAJECTORY_VARIATION = 0.2

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                player_speed = paddle_speed_val
            if event.key == pygame.K_UP:
                player_speed = -paddle_speed_val
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                player_speed = 0

    player_paddle.y += player_speed
    player_paddle.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT)) # Assurer que le paddle reste dans l'écran

    if len(balls_data) > 0:
        # IA simple : suit la balle la plus proche (ou la première)
        # Pour une IA plus avancée, on pourrait prédire la trajectoire
        closest_ball_y = balls_data[0, 1] + ball_radius # Centre Y de la première balle
        if opponent_paddle.centery < closest_ball_y :
            opponent_paddle.y += opponent_paddle_speed_val
        elif opponent_paddle.centery > closest_ball_y:
            opponent_paddle.y -= opponent_paddle_speed_val
    opponent_paddle.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT)) # Assurer que le paddle reste dans l'écran


    if len(balls_data) == 0: # Devrait être rare avec la logique de respawn, mais sécurité
        balls_data = np.array([reset_ball()])
        continue

    balls_data[:, 0] += balls_data[:, 2]
    balls_data[:, 1] += balls_data[:, 3]

    # Collisions avec les murs haut/bas
    top_collision = balls_data[:, 1] <= 0
    bottom_collision = balls_data[:, 1] + ball_diameter >= HEIGHT # Correction: ball_diameter
    balls_data[top_collision, 1] = 0 # Empêche de sortir par le haut
    balls_data[bottom_collision, 1] = HEIGHT - ball_diameter # Empêche de sortir par le bas
    wall_collision = top_collision | bottom_collision
    balls_data[wall_collision, 3] *= -1 # Inversion simple de la vitesse Y

    # Définition des limites des balles pour les collisions avec les raquettes
    # Ces valeurs seront utilisées pour les collisions avec les raquettes ET pour la sortie d'écran plus tard
    # Il est important que ces valeurs soient à jour AVANT les vérifications de sortie d'écran si des balles sont ajoutées
    ball_rects_left = balls_data[:, 0]
    ball_rects_right = balls_data[:, 0] + ball_diameter
    ball_rects_top = balls_data[:, 1]
    ball_rects_bottom = balls_data[:, 1] + ball_diameter # Correction: ball_diameter

    # Collision avec raquette joueur
    player_collision_mask = (
        (ball_rects_left <= player_paddle.right) &
        (ball_rects_right >= player_paddle.left) &
        (ball_rects_top <= player_paddle.bottom) &
        (ball_rects_bottom >= player_paddle.top) &
        (balls_data[:, 2] < 0) # Balle allant vers le joueur
    )

    # Collision avec raquette adversaire
    opponent_collision_mask = (
        (ball_rects_left <= opponent_paddle.right) &
        (ball_rects_right >= opponent_paddle.left) &
        (ball_rects_top <= opponent_paddle.bottom) &
        (ball_rects_bottom >= opponent_paddle.top) &
        (balls_data[:, 2] > 0) # Balle allant vers l'adversaire
    )
    
    combined_paddle_collision_mask = player_collision_mask | opponent_collision_mask

    if np.any(combined_paddle_collision_mask):
        collided_indices = np.where(combined_paddle_collision_mask)[0]

        # Inverser et accélérer la vitesse X pour les balles qui ont touché
        balls_data[collided_indices, 2] *= -1.1
        # Optionnel: légère accélération de Y ou juste la variation
        # balls_data[collided_indices, 3] *= 1.05 # Légère accélération Y

        # Ajouter une variation aléatoire
        num_collided = len(collided_indices)
        variation_x = np.random.uniform(-TRAJECTORY_VARIATION, TRAJECTORY_VARIATION, num_collided)
        variation_y = np.random.uniform(-TRAJECTORY_VARIATION, TRAJECTORY_VARIATION, num_collided)
        
        balls_data[collided_indices, 2] += variation_x
        balls_data[collided_indices, 3] += variation_y

        # Assurer que la balle ne reste pas "collée" à la raquette
        # Pour les collisions avec le joueur
        player_collided_now = collided_indices[player_collision_mask[collided_indices]]
        balls_data[player_collided_now, 0] = player_paddle.right 

        # Pour les collisions avec l'adversaire
        opponent_collided_now = collided_indices[opponent_collision_mask[collided_indices]]
        balls_data[opponent_collided_now, 0] = opponent_paddle.left - ball_diameter

        # Mettre à jour les scores (si vous voulez marquer sur les touches de raquette)
        # player_score += np.sum(player_collision_mask) # Exemple
        # opponent_score += np.sum(opponent_collision_mask) # Exemple

        # Ajouter de nouvelles balles pour chaque collision
        num_new_balls_to_add = num_collided # Ou un nombre fixe si vous préférez
        if num_new_balls_to_add > 0:
            new_balls_list = [reset_ball() for _ in range(num_new_balls_to_add)]
            if new_balls_list: # Vérifier si la liste n'est pas vide
                balls_data = np.vstack([balls_data, np.array(new_balls_list)])
                # APRÈS AVOIR AJOUTÉ DES BALLES, LES LIMITES DOIVENT ÊTRE MISES À JOUR
                # POUR LA VÉRIFICATION DE SORTIE D'ÉCRAN CI-DESSOUS
                ball_rects_left = balls_data[:, 0]
                ball_rects_right = balls_data[:, 0] + ball_diameter
                # ball_rects_top et ball_rects_bottom ne sont pas nécessaires pour out_left/right
                # mais si elles l'étaient, il faudrait les recalculer ici aussi.

    # Retirer les balles qui sortent de l'écran ET GÉRER LES SCORES
    # Utilise les `ball_rects_left` et `ball_rects_right` qui ont été mis à jour si des balles ont été ajoutées
    out_left_mask = ball_rects_right < 0  # Balle complètement sortie à gauche
    out_right_mask = ball_rects_left > WIDTH # Balle complètement sortie à droite
    
    # Logique de score standard pour Pong
    if np.any(out_left_mask):
        opponent_score += np.sum(out_left_mask) # L'adversaire marque
    if np.any(out_right_mask):
        player_score += np.sum(out_right_mask) # Le joueur marque

    # Garder seulement les balles qui ne sont pas sorties
    keep_balls_mask = ~(out_left_mask | out_right_mask)
    balls_data = balls_data[keep_balls_mask]

    # S'assurer qu'il y a au moins une balle
    if len(balls_data) == 0:
        balls_data = np.array([reset_ball()])
        # Optionnel : réinitialiser les scores ou attribuer un point
        # print("All balls out, resetting.")

    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player_paddle)
    pygame.draw.rect(screen, WHITE, opponent_paddle)

    for i in range(len(balls_data)):
        ball = balls_data[i]
        pygame.draw.circle(screen, (int(ball[4]), int(ball[5]), int(ball[6])),
                           (int(ball[0] + ball_radius), int(ball[1] + ball_radius)), ball_radius)

    player_text = font.render(f"Player: {player_score}", True, WHITE)
    opponent_text = font.render(f"Opponent: {opponent_score}", True, WHITE)
    balls_text = font.render(f"Balles: {len(balls_data)}", True, WHITE)
    
    screen.blit(player_text, (50, 20))
    opponent_text_rect = opponent_text.get_rect()
    screen.blit(opponent_text, (WIDTH - opponent_text_rect.width - 50, 20))
    screen.blit(balls_text, (WIDTH // 2 - balls_text.get_rect().width // 2, 20)) # Centré

    pygame.display.flip()
    clock.tick(60)