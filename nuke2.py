import pygame
import sys
import random
import math
import numpy as np
import os

# Initialisation de Pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
sound_folder = "sounds"
try:
    bounce_sound = pygame.mixer.Sound(os.path.join(sound_folder, "bounce.wav"))
    yellow_star_sound = pygame.mixer.Sound(os.path.join(sound_folder, "yellow_star_bounce.wav")) # Rebond étoile jaune
    red_star_impact_sound = pygame.mixer.Sound(os.path.join(sound_folder, "red_star_effect.wav")) # Impact étoile rouge
    paddle_shrink_sound = pygame.mixer.Sound(os.path.join(sound_folder, "paddle_shrink.wav"))
    paddle_grow_sound = pygame.mixer.Sound(os.path.join(sound_folder, "paddle_grow.wav")) # NOUVEAU SON
    explosion_visual_sound = pygame.mixer.Sound(os.path.join(sound_folder, "explosion.wav")) # Son pour l'explosion visuelle

    bounce_sound.set_volume(0.5); yellow_star_sound.set_volume(0.7); red_star_impact_sound.set_volume(0.8)
    paddle_shrink_sound.set_volume(0.7); paddle_grow_sound.set_volume(0.7); explosion_visual_sound.set_volume(0.9)
    sound_enabled = True
except pygame.error as e:
    print(f"Erreur sons: {e}. Sons désactivés.")
    class DummySound:
        def play(self): pass
        def set_volume(self, vol): pass
    bounce_sound, yellow_star_sound, red_star_impact_sound, paddle_shrink_sound, paddle_grow_sound, explosion_visual_sound = (DummySound(),)*6
    sound_enabled = False

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
pygame.display.set_caption("Pong - Explosions Visuelles & Raquettes Dynamiques!")

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
YELLOW_STAR_COLOR, RED_NUKE_STAR_COLOR = (255, 255, 0), (255, 0, 0)
EXPLOSION_COLOR = (255, 100, 0, 150) # Orange semi-transparent pour l'explosion

# --- Raquettes ---
INITIAL_PADDLE_HEIGHT = int(HEIGHT * 0.18) # Un peu plus grandes initialement
PADDLE_WIDTH = int(WIDTH * 0.01875)
MIN_PADDLE_HEIGHT = int(INITIAL_PADDLE_HEIGHT * 0.25) 
PADDLE_SIZE_CHANGE_AMOUNT = int(INITIAL_PADDLE_HEIGHT * 0.15) # Montant fixe pour grandir/rétrécir
NUKE_SCORE_PENALTY = 200

player_paddle_rect = pygame.Rect(int(WIDTH * 0.0625), HEIGHT // 2 - INITIAL_PADDLE_HEIGHT // 2, PADDLE_WIDTH, INITIAL_PADDLE_HEIGHT)
opponent_paddle_rect = pygame.Rect(WIDTH - int(WIDTH * 0.0625) - PADDLE_WIDTH, HEIGHT // 2 - INITIAL_PADDLE_HEIGHT // 2, PADDLE_WIDTH, INITIAL_PADDLE_HEIGHT)

# --- Balles ---
DEFAULT_BALL_RADIUS = int(HEIGHT * 0.0167)
STAR_BALL_RADIUS_MULTIPLIER = 2
MIN_BALL_RADIUS = max(1, int(DEFAULT_BALL_RADIUS / 2.5))
NUKE_EXPLOSION_TARGET_RADIUS = WIDTH / 7 
NUKE_EXPLOSION_DURATION = 30 # en frames (0.5 seconde à 60fps)

BALL_TYPE_NORMAL, BALL_TYPE_YELLOW_STAR, BALL_TYPE_RED_NUKE_STAR = 0.0, 1.0, 2.0
YELLOW_STAR_SPAWN_CHANCE, RED_NUKE_STAR_SPAWN_CHANCE = 0.07, 0.04 # Ajustement des chances
YELLOW_STAR_NEW_BALLS_COUNT = 25 # Un peu moins de balles pour la lisibilité

BALL_X_IDX, BALL_Y_IDX, BALL_SX_IDX, BALL_SY_IDX = 0, 1, 2, 3
BALL_R_COL_IDX, BALL_G_COL_IDX, BALL_B_COL_IDX = 4, 5, 6
BALL_TYPE_IDX, BALL_RADIUS_IDX = 7, 8

# --- Explosions Visuelles ---
active_explosions = [] # Liste de dictionnaires: {'pos': (x,y), 'radius': current_radius, 'max_radius': R, 'timer': DURATION}

def reset_ball(force_regular=False, specific_type=None, initial_radius=None): # Inchangé
    ball_type = BALL_TYPE_NORMAL; color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    current_radius_val = initial_radius if initial_radius is not None else DEFAULT_BALL_RADIUS 
    if specific_type is not None: ball_type = specific_type
    elif not force_regular:
        rand_val = random.random()
        if rand_val < RED_NUKE_STAR_SPAWN_CHANCE: ball_type = BALL_TYPE_RED_NUKE_STAR
        elif rand_val < RED_NUKE_STAR_SPAWN_CHANCE + YELLOW_STAR_SPAWN_CHANCE: ball_type = BALL_TYPE_YELLOW_STAR
    if ball_type == BALL_TYPE_YELLOW_STAR: color, current_radius_val = YELLOW_STAR_COLOR, DEFAULT_BALL_RADIUS * STAR_BALL_RADIUS_MULTIPLIER
    elif ball_type == BALL_TYPE_RED_NUKE_STAR: color, current_radius_val = RED_NUKE_STAR_COLOR, DEFAULT_BALL_RADIUS * STAR_BALL_RADIUS_MULTIPLIER
    while True:
        angle = random.uniform(-math.pi / 2.8, math.pi / 2.8)
        speed_magnitude = random.uniform(HEIGHT * 0.0075, HEIGHT * 0.011) 
        speed_x, speed_y = speed_magnitude * math.cos(angle), speed_magnitude * math.sin(angle)
        if abs(speed_x) < speed_magnitude * 0.35: continue 
        if random.choice([-1, 1]) < 0: speed_x *= -1
        break 
    return np.array([WIDTH // 2 - current_radius_val, HEIGHT // 2 - current_radius_val, speed_x, speed_y, color[0], color[1], color[2], ball_type, current_radius_val], dtype=np.float32)

balls_data = np.array([reset_ball()])
opponent_paddle_max_speed = DEFAULT_BALL_RADIUS * 1.25 # IA un peu plus réactive

player_score, opponent_score = 0, 0
font = pygame.font.Font(None, int(HEIGHT * 0.06)) 
clock = pygame.time.Clock()
TRAJECTORY_VARIATION = 0.20

def draw_star_shape(surface, color, center_x, center_y, outer_radius, inner_radius_factor=0.5, points=5): # Inchangé
    if outer_radius < 2: return
    inner_radius = outer_radius * inner_radius_factor; angle_step = math.pi / points; pts = []
    for i in range(points * 2):
        radius = outer_radius if i % 2 == 0 else inner_radius
        angle = i * angle_step - (math.pi / 2); x, y = center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)
        pts.append((x,y))
    pygame.draw.polygon(surface, color, pts)
    border_color = tuple(max(0, c - 60) for c in color)
    if color != BLACK and outer_radius > 4 : pygame.draw.polygon(surface, border_color, pts, max(1,int(outer_radius/10)))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
        if event.type == pygame.MOUSEMOTION:
            player_paddle_rect.centery = event.pos[1] 
            if player_paddle_rect.top < 0: player_paddle_rect.top = 0
            if player_paddle_rect.bottom > HEIGHT: player_paddle_rect.bottom = HEIGHT
    
    # --- IA DE L'ORDINATEUR (logique inchangée, utilise opponent_paddle_rect) ---
    if len(balls_data) > 0: # ... (Copier/coller la logique IA de la version précédente) ...
        target_y_for_ai = opponent_paddle_rect.centery; best_ball_to_predict = None; min_time_to_reach_paddle_line = float('inf'); priority_ball = None
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
            time_to_reach = (opponent_paddle_rect.left - (ball[BALL_X_IDX] + current_ball_diameter)) / ball[BALL_SX_IDX] if ball[BALL_SX_IDX] !=0 else float('inf')
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
        dist_to_target = target_y_for_ai - opponent_paddle_rect.centery
        move_amount = min(opponent_paddle_max_speed, abs(dist_to_target))
        if dist_to_target > 0: opponent_paddle_rect.y += move_amount
        elif dist_to_target < 0: opponent_paddle_rect.y -= move_amount
    else: 
        target_y_for_ai = HEIGHT // 2; dist_to_target = target_y_for_ai - opponent_paddle_rect.centery
        move_amount = min(opponent_paddle_max_speed / 1.5, abs(dist_to_target)) 
        if dist_to_target > 0 : opponent_paddle_rect.y += move_amount
        elif dist_to_target < 0 : opponent_paddle_rect.y -= move_amount
    opponent_paddle_rect.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))


    if not running: break 
    if len(balls_data) == 0: balls_data = np.array([reset_ball()])

    balls_data[:, BALL_X_IDX] += balls_data[:, BALL_SX_IDX]
    balls_data[:, BALL_Y_IDX] += balls_data[:, BALL_SY_IDX]

    for i in range(len(balls_data)): # Collisions murs
        ball_radius_current = balls_data[i, BALL_RADIUS_IDX]; ball_diameter_current = ball_radius_current * 2
        played_wall_sound_this_ball = False
        if balls_data[i, BALL_Y_IDX] <= 0:
            balls_data[i, BALL_Y_IDX] = 0; balls_data[i, BALL_SY_IDX] *= -1.02; bounce_sound.play(); played_wall_sound_this_ball = True
        elif balls_data[i, BALL_Y_IDX] + ball_diameter_current >= HEIGHT:
            balls_data[i, BALL_Y_IDX] = HEIGHT - ball_diameter_current; balls_data[i, BALL_SY_IDX] *= -1.02
            if not played_wall_sound_this_ball: bounce_sound.play()

    ball_radii = balls_data[:, BALL_RADIUS_IDX]; ball_diameters = ball_radii * 2
    ball_rects_left = balls_data[:, BALL_X_IDX]; ball_rects_right = balls_data[:, BALL_X_IDX] + ball_diameters
    ball_rects_top = balls_data[:, BALL_Y_IDX]; ball_rects_bottom = balls_data[:, BALL_Y_IDX] + ball_diameters

    player_collision_mask = ((ball_rects_left <= player_paddle_rect.right) & (ball_rects_right >= player_paddle_rect.left) & (ball_rects_top <= player_paddle_rect.bottom) & (ball_rects_bottom >= player_paddle_rect.top) & (balls_data[:, BALL_SX_IDX] < 0))
    opponent_collision_mask = ((ball_rects_left <= opponent_paddle_rect.right) & (ball_rects_right >= opponent_paddle_rect.left) & (ball_rects_top <= opponent_paddle_rect.bottom) & (ball_rects_bottom >= opponent_paddle_rect.top) & (balls_data[:, BALL_SX_IDX] > 0))
    combined_paddle_collision_mask = player_collision_mask | opponent_collision_mask
    
    balls_to_add_later = []; nuke_effect_done_this_frame = False; indices_of_special_balls_consumed_this_frame = [] 
    balls_destroyed_by_nuke_indices = []

    if np.any(combined_paddle_collision_mask):
        collided_indices_current_frame = np.where(combined_paddle_collision_mask)[0]
        
        for idx in collided_indices_current_frame:
            if nuke_effect_done_this_frame and balls_data[idx, BALL_TYPE_IDX] != BALL_TYPE_RED_NUKE_STAR : continue 

            ball_type_collided = balls_data[idx, BALL_TYPE_IDX]
            current_radius_collided = balls_data[idx, BALL_RADIUS_IDX]
            current_diameter_collided = current_radius_collided * 2
            # Point d'impact central de la balle spéciale
            impact_x, impact_y = balls_data[idx, BALL_X_IDX] + current_radius_collided, balls_data[idx, BALL_Y_IDX] + current_radius_collided

            balls_data[idx, BALL_SX_IDX] *= -1.15 
            variation_x = np.random.uniform(-TRAJECTORY_VARIATION, TRAJECTORY_VARIATION) * abs(balls_data[idx, BALL_SX_IDX])
            variation_y = np.random.uniform(-TRAJECTORY_VARIATION, TRAJECTORY_VARIATION) * 3.0 
            balls_data[idx, BALL_SX_IDX] += variation_x; balls_data[idx, BALL_SY_IDX] += variation_y
            
            collided_with_player = player_collision_mask[idx]
            collided_with_opponent = opponent_collision_mask[idx]
            target_paddle_rect = None
            if collided_with_player: target_paddle_rect = player_paddle_rect
            elif collided_with_opponent: target_paddle_rect = opponent_paddle_rect

            if collided_with_player: balls_data[idx, BALL_X_IDX] = player_paddle_rect.right
            elif collided_with_opponent: balls_data[idx, BALL_X_IDX] = opponent_paddle_rect.left - current_diameter_collided

            if ball_type_collided == BALL_TYPE_RED_NUKE_STAR and not nuke_effect_done_this_frame:
                nuke_effect_done_this_frame = True
                indices_of_special_balls_consumed_this_frame.append(idx)
                red_star_impact_sound.play()
                
                if target_paddle_rect:
                    paddle_shrink_sound.play()
                    if target_paddle_rect.height > MIN_PADDLE_HEIGHT:
                        new_height = max(MIN_PADDLE_HEIGHT, target_paddle_rect.height - PADDLE_SIZE_CHANGE_AMOUNT)
                        current_centery = target_paddle_rect.centery
                        target_paddle_rect.height = new_height
                        target_paddle_rect.centery = current_centery
                    else:
                        if collided_with_player: player_score -= NUKE_SCORE_PENALTY
                        else: opponent_score -= NUKE_SCORE_PENALTY
                
                explosion_visual_sound.play()
                active_explosions.append({'pos': (impact_x, impact_y), 'radius': 0, 'max_radius': NUKE_EXPLOSION_TARGET_RADIUS, 'timer': NUKE_EXPLOSION_DURATION, 'alpha_surface': pygame.Surface((NUKE_EXPLOSION_TARGET_RADIUS*2, NUKE_EXPLOSION_TARGET_RADIUS*2), pygame.SRCALPHA)})
                for i in range(len(balls_data)):
                    if i == idx: continue
                    ball_center_x = balls_data[i, BALL_X_IDX] + balls_data[i, BALL_RADIUS_IDX]
                    ball_center_y = balls_data[i, BALL_Y_IDX] + balls_data[i, BALL_RADIUS_IDX]
                    distance_sq = (ball_center_x - impact_x)**2 + (ball_center_y - impact_y)**2
                    if distance_sq < NUKE_EXPLOSION_TARGET_RADIUS**2:
                        balls_destroyed_by_nuke_indices.append(i)

            elif ball_type_collided == BALL_TYPE_YELLOW_STAR:
                yellow_star_sound.play()
                if target_paddle_rect: # Si une raquette est touchée
                    paddle_grow_sound.play()
                    new_height = min(INITIAL_PADDLE_HEIGHT, target_paddle_rect.height + PADDLE_SIZE_CHANGE_AMOUNT)
                    current_centery = target_paddle_rect.centery
                    target_paddle_rect.height = new_height
                    target_paddle_rect.centery = current_centery

                for _ in range(YELLOW_STAR_NEW_BALLS_COUNT): balls_to_add_later.append(reset_ball(force_regular=True)) 
                indices_of_special_balls_consumed_this_frame.append(idx)
            elif ball_type_collided == BALL_TYPE_NORMAL: 
                bounce_sound.play(); balls_to_add_later.append(reset_ball()) 
            else: bounce_sound.play()

    # --- Gestion des états après collisions (scores, suppression, ajout) ---
    # (logique inchangée) ...
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
    all_indices_to_remove = list(set(indices_of_special_balls_consumed_this_frame + balls_destroyed_by_nuke_indices))
    all_indices_to_remove.sort(reverse=True)
    for idx_to_remove in all_indices_to_remove:
        if idx_to_remove < len(keep_mask): keep_mask[idx_to_remove] = False
    if len(balls_data) > 0 : balls_data = balls_data[keep_mask]
    if balls_to_add_later:
        new_balls_array = np.array(balls_to_add_later)
        if len(balls_data) > 0: balls_data = np.vstack([balls_data, new_balls_array])
        else: balls_data = new_balls_array
    if len(balls_data) == 0: balls_data = np.array([reset_ball()])


    # --- DESSIN ---
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player_paddle_rect)
    pygame.draw.rect(screen, WHITE, opponent_paddle_rect)

    # Dessiner les explosions actives
    explosions_to_keep = []
    for exp in active_explosions:
        exp['timer'] -= 1
        if exp['timer'] > 0:
            progress = (NUKE_EXPLOSION_DURATION - exp['timer']) / NUKE_EXPLOSION_DURATION
            current_exp_radius_float = exp['max_radius'] * math.sin(progress * math.pi) 
        
            alpha = int(200 * math.sin(progress * math.pi)) 
        
            # S'assurer que le rayon est au moins 1 pour dessiner quelque chose et éviter des erreurs
            current_exp_radius_int = max(1, int(current_exp_radius_float)) 
        
            if alpha > 0 : # Seulement dessiner si visible
                temp_surface = pygame.Surface((current_exp_radius_int * 2, current_exp_radius_int * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surface, (*EXPLOSION_COLOR[:3], alpha), 
                                   (current_exp_radius_int, current_exp_radius_int), current_exp_radius_int) # Utiliser le rayon int ici aussi
            
                # --- CORRECTION ICI ---
                blit_pos_x = int(exp['pos'][0] - current_exp_radius_int)
                blit_pos_y = int(exp['pos'][1] - current_exp_radius_int)
                screen.blit(temp_surface, (blit_pos_x, blit_pos_y))
                # --- FIN CORRECTION ---

                explosions_to_keep.append(exp)
        active_explosions = explosions_to_keep


    for i in range(len(balls_data)): # Dessiner les balles
        ball = balls_data[i]
        center_x, center_y = int(ball[BALL_X_IDX] + ball[BALL_RADIUS_IDX]), int(ball[BALL_Y_IDX] + ball[BALL_RADIUS_IDX])
        current_radius_draw, current_ball_type = int(ball[BALL_RADIUS_IDX]), ball[BALL_TYPE_IDX]
        if current_ball_type == BALL_TYPE_YELLOW_STAR: draw_star_shape(screen, YELLOW_STAR_COLOR, center_x, center_y, current_radius_draw, 0.4, 5)
        elif current_ball_type == BALL_TYPE_RED_NUKE_STAR: draw_star_shape(screen, RED_NUKE_STAR_COLOR, center_x, center_y, current_radius_draw, 0.5, 6)
        else: 
            normal_ball_color = (int(ball[BALL_R_COL_IDX]), int(ball[BALL_G_COL_IDX]), int(ball[BALL_B_COL_IDX]))
            if current_radius_draw >=1: pygame.draw.circle(screen, normal_ball_color, (center_x, center_y), current_radius_draw)

    player_text_surface = font.render(f"P: {player_score}", True, WHITE)
    opponent_text_surface = font.render(f"IA: {opponent_score}", True, WHITE)
    balls_text_surface = font.render(f"B: {len(balls_data)}", True, WHITE)
    screen.blit(player_text_surface, (int(WIDTH * 0.05), int(HEIGHT * 0.02)))
    opp_rect = opponent_text_surface.get_rect(topright=(WIDTH - int(WIDTH*0.05), int(HEIGHT*0.02)))
    screen.blit(opponent_text_surface, opp_rect)
    balls_rect = balls_text_surface.get_rect(midtop=(WIDTH // 2, int(HEIGHT*0.02)))
    screen.blit(balls_text_surface, balls_rect)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()