import pygame
import sys
import random
import math
import numpy as np
import os # Ajouté pour gérer les chemins de fichiers son

# Initialisation de Pygame
pygame.init()

# --- INITIALISATION DU SON ---
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512) # Initialiser avec des paramètres courants
sound_folder = "sounds" # Optionnel: Mettez vos sons dans un sous-dossier "sounds"

try:
    bounce_sound = pygame.mixer.Sound(os.path.join(sound_folder, "bounce.wav")) # Remplacez par vos noms de fichiers
    yellow_star_sound = pygame.mixer.Sound(os.path.join(sound_folder, "yellow_star_bounce.wav"))
    red_star_sound = pygame.mixer.Sound(os.path.join(sound_folder, "red_star_effect.wav"))
    # Ajuster le volume si nécessaire
    bounce_sound.set_volume(0.6)
    yellow_star_sound.set_volume(0.8)
    red_star_sound.set_volume(0.9)
    sound_enabled = True
except pygame.error as e:
    print(f"Erreur lors du chargement des sons : {e}")
    print("Les sons seront désactivés.")
    # Créer des objets Sound "factices" pour éviter les erreurs si les fichiers sont manquants
    class DummySound:
        def play(self): pass
        def set_volume(self, vol): pass
    bounce_sound = DummySound()
    yellow_star_sound = DummySound()
    red_star_sound = DummySound()
    sound_enabled = False
# --- FIN INITIALISATION DU SON ---


# --- GESTION DU PLEIN ÉCRAN ---
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
pygame.display.set_caption("Pong - Audio Chaos!")
# --- FIN GESTION PLEIN ÉCRAN ---


# Couleurs et autres constantes...
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW_STAR_COLOR = (255, 255, 0)
RED_NUKE_STAR_COLOR = (255, 0, 0)

paddle_width, paddle_height = int(WIDTH * 0.01875), int(HEIGHT * 0.1667)
player_paddle = pygame.Rect(int(WIDTH * 0.0625), HEIGHT // 2 - paddle_height // 2, paddle_width, paddle_height)
opponent_paddle = pygame.Rect(WIDTH - int(WIDTH * 0.0625) - paddle_width, HEIGHT // 2 - paddle_height // 2, paddle_width, paddle_height)

DEFAULT_BALL_RADIUS = int(HEIGHT * 0.0167)
STAR_BALL_RADIUS_MULTIPLIER = 2
MIN_BALL_RADIUS = max(1, int(DEFAULT_BALL_RADIUS / 2.5))

BALL_TYPE_NORMAL = 0.0
BALL_TYPE_YELLOW_STAR = 1.0
BALL_TYPE_RED_NUKE_STAR = 2.0

YELLOW_STAR_SPAWN_CHANCE = 0.08
RED_NUKE_STAR_SPAWN_CHANCE = 0.03
YELLOW_STAR_EXPLOSION_COUNT = 30

BALL_X_IDX, BALL_Y_IDX, BALL_SX_IDX, BALL_SY_IDX = 0, 1, 2, 3
BALL_R_COL_IDX, BALL_G_COL_IDX, BALL_B_COL_IDX = 4, 5, 6
BALL_TYPE_IDX, BALL_RADIUS_IDX = 7, 8

def reset_ball(force_regular=False, specific_type=None, initial_radius=None):
    ball_type = BALL_TYPE_NORMAL
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    current_radius_val = initial_radius if initial_radius is not None else DEFAULT_BALL_RADIUS 

    if specific_type is not None:
        ball_type = specific_type
    elif not force_regular:
        rand_val = random.random()
        if rand_val < RED_NUKE_STAR_SPAWN_CHANCE:
            ball_type = BALL_TYPE_RED_NUKE_STAR
        elif rand_val < RED_NUKE_STAR_SPAWN_CHANCE + YELLOW_STAR_SPAWN_CHANCE:
            ball_type = BALL_TYPE_YELLOW_STAR

    if ball_type == BALL_TYPE_YELLOW_STAR:
        color = YELLOW_STAR_COLOR
        current_radius_val = DEFAULT_BALL_RADIUS * STAR_BALL_RADIUS_MULTIPLIER
    elif ball_type == BALL_TYPE_RED_NUKE_STAR:
        color = RED_NUKE_STAR_COLOR
        current_radius_val = DEFAULT_BALL_RADIUS * STAR_BALL_RADIUS_MULTIPLIER
    
    while True:
        angle_limit_rad = math.pi / 2.8 
        angle = random.uniform(-angle_limit_rad, angle_limit_rad)
        speed_magnitude = random.uniform(HEIGHT * 0.0075, HEIGHT * 0.011) 
        speed_x = speed_magnitude * math.cos(angle)
        speed_y = speed_magnitude * math.sin(angle)
        if abs(speed_x) < speed_magnitude * 0.35: continue 
        if random.choice([-1, 1]) < 0: speed_x *= -1
        break 
            
    return np.array([
        WIDTH // 2 - current_radius_val, HEIGHT // 2 - current_radius_val,
        speed_x, speed_y,
        color[0], color[1], color[2],
        ball_type, current_radius_val
    ], dtype=np.float32)

balls_data = np.array([reset_ball()])
opponent_paddle_max_speed = DEFAULT_BALL_RADIUS * 1.1 # IA un peu plus rapide

player_score = 0
opponent_score = 0
font_size = int(HEIGHT * 0.06)
font = pygame.font.Font(None, font_size) 
clock = pygame.time.Clock()
TRAJECTORY_VARIATION = 0.20

def draw_star_shape(surface, color, center_x, center_y, outer_radius, inner_radius_factor=0.5, points=5):
    if outer_radius < 2: return
    inner_radius = outer_radius * inner_radius_factor
    angle_step = math.pi / points
    pts = []
    for i in range(points * 2):
        radius = outer_radius if i % 2 == 0 else inner_radius
        angle = i * angle_step - (math.pi / 2) 
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        pts.append((x, y))
    pygame.draw.polygon(surface, color, pts)
    border_color = tuple(max(0, c - 60) for c in color)
    if color != BLACK and outer_radius > 4 :
        pygame.draw.polygon(surface, border_color, pts, max(1,int(outer_radius/10)))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
        if event.type == pygame.MOUSEMOTION:
            player_paddle.centery = event.pos[1] 
            if player_paddle.top < 0: player_paddle.top = 0
            if player_paddle.bottom > HEIGHT: player_paddle.bottom = HEIGHT

    # --- IA DE L'ORDINATEUR ---
    if len(balls_data) > 0:
        target_y_for_ai = opponent_paddle.centery # ... (logique IA inchangée) ...
        best_ball_to_predict = None; min_time_to_reach_paddle_line = float('inf'); priority_ball = None
        for i in range(len(balls_data)):
            ball = balls_data[i]
            if ball[BALL_SX_IDX] > 0:
                if ball[BALL_TYPE_IDX] == BALL_TYPE_RED_NUKE_STAR: priority_ball = ball; break 
                elif ball[BALL_TYPE_IDX] == BALL_TYPE_YELLOW_STAR and priority_ball is None: priority_ball = ball
        ball_for_prediction_loop = [priority_ball] if priority_ball is not None else balls_data
        for ball_idx_loop in range(len(ball_for_prediction_loop)):
            ball = ball_for_prediction_loop[ball_idx_loop]
            if ball is None or ball[BALL_SX_IDX] <= 0: continue
            current_ball_radius = ball[BALL_RADIUS_IDX]; current_ball_diameter = current_ball_radius * 2
            time_to_reach = (opponent_paddle.left - (ball[BALL_X_IDX] + current_ball_diameter)) / ball[BALL_SX_IDX] if ball[BALL_SX_IDX] !=0 else float('inf')
            if 0 <= time_to_reach < min_time_to_reach_paddle_line: min_time_to_reach_paddle_line = time_to_reach; best_ball_to_predict = ball
        if best_ball_to_predict is not None:
            pred_x, pred_y, pred_sx, pred_sy = best_ball_to_predict[BALL_X_IDX:BALL_SY_IDX+1]
            pred_radius = best_ball_to_predict[BALL_RADIUS_IDX]; pred_diameter = pred_radius * 2
            time_remaining = min_time_to_reach_paddle_line; final_predicted_y = pred_y
            current_y_at_impact = pred_y + pred_sy * time_remaining
            if current_y_at_impact < 0:
                time_to_top_wall = -pred_y / pred_sy if pred_sy !=0 else float('inf')
                if 0 <= time_to_top_wall < time_remaining : y_at_wall = 0; time_after_wall_hit = time_remaining - time_to_top_wall; final_predicted_y = y_at_wall - pred_sy * time_after_wall_hit 
            elif current_y_at_impact + pred_diameter > HEIGHT:
                time_to_bottom_wall = (HEIGHT - (pred_y + pred_diameter)) / pred_sy if pred_sy !=0 else float('inf')
                if 0 <= time_to_bottom_wall < time_remaining: y_at_wall = HEIGHT - pred_diameter; time_after_wall_hit = time_remaining - time_to_bottom_wall; final_predicted_y = y_at_wall - pred_sy * time_after_wall_hit
            else: final_predicted_y = current_y_at_impact
            target_y_for_ai = final_predicted_y + pred_radius
        elif len(balls_data) > 0: 
            closest_ball_idx = np.argmax(balls_data[:, BALL_X_IDX]) 
            target_y_for_ai = balls_data[closest_ball_idx, BALL_Y_IDX] + balls_data[closest_ball_idx, BALL_RADIUS_IDX]
        else: target_y_for_ai = HEIGHT // 2
        dist_to_target = target_y_for_ai - opponent_paddle.centery
        move_amount = min(opponent_paddle_max_speed, abs(dist_to_target))
        if dist_to_target > 0: opponent_paddle.y += move_amount
        elif dist_to_target < 0: opponent_paddle.y -= move_amount
    else: 
        target_y_for_ai = HEIGHT // 2; dist_to_target = target_y_for_ai - opponent_paddle.centery
        move_amount = min(opponent_paddle_max_speed / 1.5, abs(dist_to_target)) 
        if dist_to_target > 0 : opponent_paddle.y += move_amount
        elif dist_to_target < 0 : opponent_paddle.y -= move_amount
    opponent_paddle.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))

    if not running: break 

    if len(balls_data) == 0: balls_data = np.array([reset_ball()])

    balls_data[:, BALL_X_IDX] += balls_data[:, BALL_SX_IDX]
    balls_data[:, BALL_Y_IDX] += balls_data[:, BALL_SY_IDX]

    # Collisions murs haut/bas
    for i in range(len(balls_data)):
        ball_radius_current = balls_data[i, BALL_RADIUS_IDX]
        ball_diameter_current = ball_radius_current * 2
        played_wall_sound_this_ball = False # Pour éviter double son si touche haut et bas en même temps (peu probable)
        if balls_data[i, BALL_Y_IDX] <= 0:
            balls_data[i, BALL_Y_IDX] = 0
            balls_data[i, BALL_SY_IDX] *= -1.02
            bounce_sound.play()
            played_wall_sound_this_ball = True
        elif balls_data[i, BALL_Y_IDX] + ball_diameter_current >= HEIGHT:
            balls_data[i, BALL_Y_IDX] = HEIGHT - ball_diameter_current
            balls_data[i, BALL_SY_IDX] *= -1.02
            if not played_wall_sound_this_ball: # Ne jouer que si pas déjà joué pour ce rebond
                bounce_sound.play()

    # Préparation pour collisions raquettes
    ball_radii = balls_data[:, BALL_RADIUS_IDX]
    ball_diameters = ball_radii * 2
    ball_rects_left = balls_data[:, BALL_X_IDX]
    ball_rects_right = balls_data[:, BALL_X_IDX] + ball_diameters
    ball_rects_top = balls_data[:, BALL_Y_IDX]
    ball_rects_bottom = balls_data[:, BALL_Y_IDX] + ball_diameters

    player_collision_mask = (
        (ball_rects_left <= player_paddle.right) & (ball_rects_right >= player_paddle.left) &
        (ball_rects_top <= player_paddle.bottom) & (ball_rects_bottom >= player_paddle.top) &
        (balls_data[:, BALL_SX_IDX] < 0)
    )
    opponent_collision_mask = (
        (ball_rects_left <= opponent_paddle.right) & (ball_rects_right >= opponent_paddle.left) &
        (ball_rects_top <= opponent_paddle.bottom) & (ball_rects_bottom >= opponent_paddle.top) &
        (balls_data[:, BALL_SX_IDX] > 0)
    )
    combined_paddle_collision_mask = player_collision_mask | opponent_collision_mask
    
    balls_to_add_later = []
    nuke_effect_activated_this_frame = False
    indices_of_special_balls_consumed_this_frame = [] 
    
    if np.any(combined_paddle_collision_mask):
        collided_indices_current_frame = np.where(combined_paddle_collision_mask)[0]
        
        for idx in collided_indices_current_frame:
            if nuke_effect_activated_this_frame and balls_data[idx, BALL_TYPE_IDX] != BALL_TYPE_RED_NUKE_STAR : continue 

            ball_type_collided = balls_data[idx, BALL_TYPE_IDX]
            current_radius_collided = balls_data[idx, BALL_RADIUS_IDX]
            current_diameter_collided = current_radius_collided * 2

            balls_data[idx, BALL_SX_IDX] *= -1.15 
            variation_x = np.random.uniform(-TRAJECTORY_VARIATION, TRAJECTORY_VARIATION) * abs(balls_data[idx, BALL_SX_IDX])
            variation_y = np.random.uniform(-TRAJECTORY_VARIATION, TRAJECTORY_VARIATION) * 3.0 
            balls_data[idx, BALL_SX_IDX] += variation_x
            balls_data[idx, BALL_SY_IDX] += variation_y
            
            if player_collision_mask[idx]:
                balls_data[idx, BALL_X_IDX] = player_paddle.right
            elif opponent_collision_mask[idx]:
                balls_data[idx, BALL_X_IDX] = opponent_paddle.left - current_diameter_collided

            # --- JOUER LES SONS DE REBOND SUR RAQUETTE ---
            if ball_type_collided == BALL_TYPE_RED_NUKE_STAR and not nuke_effect_activated_this_frame:
                red_star_sound.play() # Jouer le son de la Nuke avant son effet
                nuke_effect_activated_this_frame = True
                indices_of_special_balls_consumed_this_frame.append(idx)
                
                num_balls_total = len(balls_data)
                if num_balls_total > 0: 
                    all_ball_indices = list(range(num_balls_total))
                    other_ball_indices = [i for i in all_ball_indices if i != idx]
                    if other_ball_indices: 
                        random.shuffle(other_ball_indices); half_point = len(other_ball_indices) // 2
                        for i_ball_mod in other_ball_indices[:half_point]:
                            balls_data[i_ball_mod, BALL_RADIUS_IDX] = max(MIN_BALL_RADIUS, balls_data[i_ball_mod, BALL_RADIUS_IDX] / 2)
                        for i_ball_mod in other_ball_indices[half_point:]:
                            new_angle = random.uniform(0, 2 * math.pi)
                            current_speed_magnitude = np.sqrt(balls_data[i_ball_mod, BALL_SX_IDX]**2 + balls_data[i_ball_mod, BALL_SY_IDX]**2)
                            current_speed_magnitude = max(current_speed_magnitude, HEIGHT * 0.005)
                            min_horizontal_speed_factor = 0.35; max_tries_dir = 5
                            for _ in range(max_tries_dir):
                                balls_data[i_ball_mod, BALL_SX_IDX] = current_speed_magnitude * math.cos(new_angle)
                                balls_data[i_ball_mod, BALL_SY_IDX] = current_speed_magnitude * math.sin(new_angle)
                                if abs(balls_data[i_ball_mod, BALL_SX_IDX]) > current_speed_magnitude * min_horizontal_speed_factor: break
                                new_angle = random.uniform(0, 2 * math.pi)
            elif ball_type_collided == BALL_TYPE_YELLOW_STAR:
                yellow_star_sound.play()
                for _ in range(YELLOW_STAR_EXPLOSION_COUNT):
                    balls_to_add_later.append(reset_ball(force_regular=True)) 
                indices_of_special_balls_consumed_this_frame.append(idx)
            elif ball_type_collided == BALL_TYPE_NORMAL: 
                bounce_sound.play()
                balls_to_add_later.append(reset_ball()) 
            else: # Si c'est une nuke mais que l'effet a déjà eu lieu, juste un rebond normal
                bounce_sound.play()


    # --- Gestion des états après collisions (scores, suppression, ajout) ---
    # ... (logique inchangée) ...
    ball_radii_updated = balls_data[:, BALL_RADIUS_IDX] if len(balls_data) > 0 else np.array([])
    ball_diameters_updated = ball_radii_updated * 2
    if len(balls_data) > 0:
        out_left_mask = (balls_data[:, BALL_X_IDX] + ball_diameters_updated) < 0
        out_right_mask = balls_data[:, BALL_X_IDX] > WIDTH
        if np.any(out_left_mask): opponent_score += np.sum(out_left_mask)
        if np.any(out_right_mask): player_score += np.sum(out_right_mask)
        keep_mask = np.ones(len(balls_data), dtype=bool)
        keep_mask[out_left_mask | out_right_mask] = False
    else: keep_mask = np.array([], dtype=bool)
    unique_consumed_indices = sorted(list(set(indices_of_special_balls_consumed_this_frame)), reverse=True)
    for idx_consumed in unique_consumed_indices:
        if idx_consumed < len(keep_mask): keep_mask[idx_consumed] = False
    if len(balls_data) > 0 : balls_data = balls_data[keep_mask]
    if balls_to_add_later:
        new_balls_array = np.array(balls_to_add_later)
        if len(balls_data) > 0: balls_data = np.vstack([balls_data, new_balls_array])
        else: balls_data = new_balls_array
    if len(balls_data) == 0: balls_data = np.array([reset_ball()])


    # --- DESSIN ---
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player_paddle)
    pygame.draw.rect(screen, WHITE, opponent_paddle)

    for i in range(len(balls_data)):
        ball = balls_data[i]
        center_x = int(ball[BALL_X_IDX] + ball[BALL_RADIUS_IDX])
        center_y = int(ball[BALL_Y_IDX] + ball[BALL_RADIUS_IDX])
        current_radius_draw = int(ball[BALL_RADIUS_IDX])
        current_ball_type = ball[BALL_TYPE_IDX]

        if current_ball_type == BALL_TYPE_YELLOW_STAR:
            draw_star_shape(screen, YELLOW_STAR_COLOR, center_x, center_y, current_radius_draw, 0.4, 5)
        elif current_ball_type == BALL_TYPE_RED_NUKE_STAR:
            draw_star_shape(screen, RED_NUKE_STAR_COLOR, center_x, center_y, current_radius_draw, 0.5, 6)
        else: 
            normal_ball_color = (int(ball[BALL_R_COL_IDX]), int(ball[BALL_G_COL_IDX]), int(ball[BALL_B_COL_IDX]))
            if current_radius_draw >=1:
                 pygame.draw.circle(screen, normal_ball_color, (center_x, center_y), current_radius_draw)

    player_text_surface = font.render(f"Player: {player_score}", True, WHITE)
    opponent_text_surface = font.render(f"Opponent: {opponent_score}", True, WHITE)
    balls_text_surface = font.render(f"Balles: {len(balls_data)}", True, WHITE)
    
    screen.blit(player_text_surface, (int(WIDTH * 0.05), int(HEIGHT * 0.02)))
    opponent_text_rect = opponent_text_surface.get_rect()
    screen.blit(opponent_text_surface, (WIDTH - opponent_text_rect.width - int(WIDTH * 0.05), int(HEIGHT * 0.02)))
    balls_text_rect = balls_text_surface.get_rect()
    screen.blit(balls_text_surface, (WIDTH // 2 - balls_text_rect.width // 2, int(HEIGHT * 0.02)))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()