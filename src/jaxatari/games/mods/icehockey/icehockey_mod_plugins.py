import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import jaxatari.games
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.games.jax_icehockey import IceHockeyConstants
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


def _make_narrowed_goal_background(new_x0: int, new_x1: int) -> np.ndarray:
    """Load the icehockey background and close the goal mouths down to [new_x0, new_x1).

    The rink is baked into the background sprite: each goal is a black gap in
    the grey board band (rows PLAYER_GOAL_Y..+GOAL_HEIGHT_TOP at the top,
    ENEMY_GOAL_Y-GOAL_HEIGHT_BOTTOM+1..ENEMY_GOAL_Y at the bottom, columns
    GOAL_X0..GOAL_X1). The now-covered columns are filled with board pixels.
    """
    c = IceHockeyConstants()
    sprite_path = os.path.join(
        os.path.dirname(jaxatari.games.__file__),
        "sprites",
        "icehockey",
        "background.npy",
    )
    bg = np.load(sprite_path).copy()
    board = np.array([192, 192, 192, 255], dtype=np.uint8)
    top_rows = slice(c.PLAYER_GOAL_Y, c.PLAYER_GOAL_Y + c.GOAL_HEIGHT_TOP)
    bottom_rows = slice(c.ENEMY_GOAL_Y - c.GOAL_HEIGHT_BOTTOM + 1, c.ENEMY_GOAL_Y + 1)
    for rows in (top_rows, bottom_rows):
        bg[rows, c.GOAL_X0 : new_x0] = board
        bg[rows, new_x1 : c.GOAL_X1] = board
    return bg


# --- 1. Individual Mod Plugins ---
class NoAttackingZonesMod(JaxAtariInternalModPlugin):
    """Removes the attacking-zone restrictions from the rink.

    In the base game each character is confined to a horizontal band in front of
    one goal: _character_bounds derives an upper band and a lower band from
    CHARACTER_GRID_Y_ORIGIN together with the UPPER/LOWER_CHARACTER_GRID_Y_MIN
    and _MAX pairs, so a skater is kept out of its own defensive zone and a
    goalie out of the opponent's far zone. Widening both bands to the full
    skateable range collapses those zones and lets all four characters skate the
    whole rink.

    The band is y = ORIGIN - GRID_Y_MAX .. ORIGIN - GRID_Y_MIN, and the full
    skateable range is RINK_TOP - PLAYER_H + 9 .. RINK_BOTTOM - PLAYER_H
    (25..161 with the stock rink), so MAX = 166 - 25 = 141 and MIN = 166 - 161 = 5.
    """

    constants_overrides = {
        "UPPER_CHARACTER_GRID_Y_MIN": 5,
        "UPPER_CHARACTER_GRID_Y_MAX": 141,
        "LOWER_CHARACTER_GRID_Y_MIN": 5,
        "LOWER_CHARACTER_GRID_Y_MAX": 141,
    }


class DisableTacklingMod(JaxAtariInternalModPlugin):
    """Disables body-checks: no character can ever be knocked down.

    The base game has no tackle-probability constant to zero out - a check lands
    on a hardcoded random-phase match inside _contact_pair, which then calls
    _knock_down on the victim. Making _knock_down a no-op is the smallest change
    that guarantees nobody is ever downed, while leaving the ordinary body
    contact/push between characters intact.

    Note: contact still strips the puck from a carrier, since that is decided in
    _contact_pair rather than in _knock_down; only the knockdown itself is gone.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _knock_down(self, char, random_byte):
        return char


class DecreasedGoalSizeMod(JaxAtariInternalModPlugin):
    """Halves the width of both goals (mouth 64..96 -> 72..88, centred).

    GOAL_X0/GOAL_X1 drive goal detection in _goal_and_reset_step and the
    goalie-protection band in _goalie_protected, so shots outside the narrowed
    mouth now bounce off the boards instead of scoring. The background asset
    is rebuilt with the covered goal columns filled in as boards so the visuals
    match the new geometry.
    """

    _NEW_GOAL_X0 = 72
    _NEW_GOAL_X1 = 88

    constants_overrides = {
        "GOAL_X0": _NEW_GOAL_X0,
        "GOAL_X1": _NEW_GOAL_X1,
    }
    asset_overrides = {
        "background": {
            "name": "background",
            "type": "background",
            "data": _make_narrowed_goal_background(_NEW_GOAL_X0, _NEW_GOAL_X1),
        }
    }


class TackleSlowdownMod(JaxAtariInternalModPlugin):
    """Characters get permanently slower each time they are tackled.

    times_tackled is maintained by the base game (and carried across face-offs),
    so it is a per-match knockdown count. Each knockdown multiplies that
    character's movement by SLOWDOWN_PER_TACKLE, down to MIN_SPEED_FACTOR.

    The base _apply_action no longer accepts an injected velocity - it moves the
    character at the fixed CHARACTER_SPEED_X/Y - so rather than reimplementing
    movement this lets the base primitive move the character normally and then
    shrinks the displacement it produced. Freezing while tackled, orientation and
    the walk cycle therefore stay exactly as the base game computes them.
    """

    SLOWDOWN_PER_TACKLE = 0.8
    MIN_SPEED_FACTOR = 0.3

    @partial(jax.jit, static_argnums=(0,))
    def _apply_team_inputs(self, char1, char2, active, action):
        # Same routing as the base implementation: the active character gets the
        # real action, the teammate a NOOP.
        action1 = jnp.where(active == 0, action, Action.NOOP)
        action2 = jnp.where(active == 1, action, Action.NOOP)

        def slowed(prev_char, act):
            moved = self._env._apply_action(prev_char, act)
            factor = jnp.float32(self.SLOWDOWN_PER_TACKLE) ** prev_char.times_tackled.astype(
                jnp.float32
            )
            factor = jnp.maximum(factor, jnp.float32(self.MIN_SPEED_FACTOR))
            delta = moved.position - prev_char.position
            return moved.replace(position=prev_char.position + delta * factor)

        return slowed(char1, action1), slowed(char2, action2)


def _goal_offset_x(consts, remaining_time, amplitude, speed=0.15):
    """Signed triangle-wave offset from the rink's horizontal center, starting at 0.

    Shape over one period (4*amplitude/speed frames): 0 -> +amplitude -> 0 ->
    -amplitude -> 0. Driven by the game clock (remaining_time counts down only
    while play is active) so it freezes in sync with everything else during
    face-offs/goal pauses, and both the physics and render-side callers agree
    on the exact same value every frame.
    """
    t = (consts.TIME_LIMIT - remaining_time).astype(jnp.float32)
    x = jnp.mod(t * speed, 4.0 * amplitude)
    return jnp.where(
        x < amplitude,
        x,
        jnp.where(x < 3.0 * amplitude, 2.0 * amplitude - x, x - 4.0 * amplitude),
    )


class ChangeBorderShapeMod(JaxAtariInternalModPlugin):
    CORNER_CUT = 16.0
    asset_overrides = {
        "background": {
            "name": "background",
            "type": "background",
            "file": "background_octagon.npy",
        }
    }

    @partial(jax.jit, static_argnums=(0,))
    def _puck_step(self, puck):
        c = self._env.consts
        cut = self.CORNER_CUT

        tentative = puck.position + puck.velocity
        vel = puck.velocity

        # Straight walls (unchanged from the base game).
        hit_left = tentative[0] < c.RINK_LEFT
        hit_right = tentative[0] > c.RINK_RIGHT
        hit_top = tentative[1] < c.RINK_TOP
        hit_bot = tentative[1] > c.RINK_BOTTOM

        vx = jnp.where(hit_left | hit_right, -vel[0], vel[0])
        vy = jnp.where(hit_top | hit_bot, -vel[1], vel[1])
        vel = jnp.array([vx, vy], dtype=jnp.float32)

        pos = jnp.clip(
            tentative,
            jnp.array([float(c.RINK_LEFT), float(c.RINK_TOP)]),
            jnp.array([float(c.RINK_RIGHT), float(c.RINK_BOTTOM)]),
        )

        # Diagonal walls at the four corners. Each entry is (sx, sy, ox, oy) such
        # that edge_sum = sx*(x-ox) + sy*(y-oy) is the summed distance from the two
        # straight edges meeting at that corner; n_in = (sx, sy)/sqrt2 points from
        # the corner back into the rink.
        inv_sqrt2 = 0.70710678
        corners = (
            (1.0, 1.0, c.RINK_LEFT, c.RINK_TOP),  # top-left
            (-1.0, 1.0, c.RINK_RIGHT, c.RINK_TOP),  # top-right
            (1.0, -1.0, c.RINK_LEFT, c.RINK_BOTTOM),  # bottom-left
            (-1.0, -1.0, c.RINK_RIGHT, c.RINK_BOTTOM),  # bottom-right
        )
        for sx, sy, ox, oy in corners:
            n_in = jnp.array([sx, sy], dtype=jnp.float32) * inv_sqrt2
            edge_sum = sx * (pos[0] - ox) + sy * (pos[1] - oy)
            penetration = cut - edge_sum
            penetrating = penetration > 0.0
            # Only reflect velocity if the puck is actually heading further into the
            # wedge; if it's already moving back out (e.g. repositioned there by a
            # face-off/pickup), leave velocity alone so it isn't sent back in.
            approaching = jnp.dot(vel, n_in) < 0.0
            hit = penetrating & approaching

            pos = jnp.where(penetrating, pos + penetration * n_in, pos)
            vel = jnp.where(hit, vel - 2.0 * jnp.dot(vel, n_in) * n_in, vel)

        # Friction (unchanged from the base game). The base models this as a
        # fixed-point 1/256 decay in _decay_puck_velocity rather than a
        # coefficient, so delegate instead of reimplementing it.
        vel = self._env._decay_puck_velocity(vel)

        return puck.replace(position=pos, velocity=vel)


class MovingGoalsMod(JaxAtariInternalModPlugin):
    """Both goal mouths are smaller, start centered, and slide back and forth
    along the boards in opposite directions (when one moves left, the other
    moves right).

    Patches _goal_and_reset_step (env) so scoring is checked against each
    goal's own current dynamic x0/x1 instead of the fixed GOAL_X0/GOAL_X1, and
    _render_hook_post_background (renderer) so the drawn notches follow along.
    Both read the same _goal_offset_x(consts, remaining_time, ...) so they can
    never desync.
    """

    GOAL_WIDTH = 22.0  # smaller than the original GOAL_X1 - GOAL_X0 (32px)
    AMPLITUDE = 32.0  # max distance (px) each goal travels from rink center

    @partial(jax.jit, static_argnums=(0,))
    def _goal_and_reset_step(
        self, game_state, player_state, enemy_state, puck_state, frozen, random_byte
    ):
        c = self._env.consts

        mid_x = (c.RINK_LEFT + c.RINK_RIGHT) / 2.0
        half_width = self.GOAL_WIDTH / 2.0
        offset = _goal_offset_x(c, game_state.remaining_time, self.AMPLITUDE)

        # Top goal (defended by player, scored into by enemy) and bottom goal
        # (defended by enemy, scored into by player) move in opposite directions.
        top_center = mid_x + offset
        bottom_center = mid_x - offset
        top_x0, top_x1 = top_center - half_width, top_center + half_width
        bottom_x0, bottom_x1 = bottom_center - half_width, bottom_center + half_width

        puck_pos = puck_state.position
        carried = (
            player_state.skater.has_puck
            | player_state.goalie.has_puck
            | enemy_state.skater.has_puck
            | enemy_state.goalie.has_puck
        )
        in_bottom_mouth = (puck_pos[0] >= bottom_x0) & (puck_pos[0] <= bottom_x1)
        in_top_mouth = (puck_pos[0] >= top_x0) & (puck_pos[0] <= top_x1)
        player_scored = (
            ~frozen & ~carried & in_bottom_mouth & (puck_pos[1] >= c.ENEMY_GOAL_Y)
        )
        enemy_scored = (
            ~frozen & ~carried & in_top_mouth & (puck_pos[1] <= c.PLAYER_GOAL_Y)
        )
        goal = player_scored | enemy_scored

        remaining_time = game_state.remaining_time
        remaining_time = jnp.where(
            goal, ((remaining_time + 59) // 60) * 60, remaining_time
        )
        clock_runs = ~frozen & ~goal
        remaining_time = jnp.where(
            clock_runs, jnp.maximum(remaining_time - 1, 0), remaining_time
        )
        time_up = clock_runs & (remaining_time == 0)

        pause_counter = jnp.where(
            frozen,
            jnp.maximum(game_state.pause_counter - 1, 0),
            game_state.pause_counter,
        )
        goal_phase_over = game_state.goal_scored & (pause_counter == 0)
        faceoff_over = game_state.is_faceoff & (pause_counter == 0)

        new_goal_scored = (game_state.goal_scored & ~goal_phase_over) | goal
        new_is_faceoff = (game_state.is_faceoff & ~faceoff_over) | goal_phase_over
        pause_counter = jnp.where(
            goal,
            jnp.int32(c.GOAL_PAUSE_FRAMES),
            jnp.where(
                goal_phase_over,
                jnp.int32(c.FACE_OFF_FRAMES),
                pause_counter,
            ),
        )

        # After the goal pause everyone snaps back to the face-off spots. The
        # base game draws a fresh random byte for the face-off and carries
        # times_tackled across (it is a per-match statistic), so mirror both.
        faceoff_random_byte = self._env._next_random_byte(random_byte)
        fo_player, fo_enemy, fo_puck = self._env._faceoff_positions(faceoff_random_byte)
        fo_player = fo_player.replace(
            skater=fo_player.skater.replace(
                times_tackled=player_state.skater.times_tackled
            ),
            goalie=fo_player.goalie.replace(
                times_tackled=player_state.goalie.times_tackled
            ),
        )
        fo_enemy = fo_enemy.replace(
            skater=fo_enemy.skater.replace(
                times_tackled=enemy_state.skater.times_tackled
            ),
            goalie=fo_enemy.goalie.replace(
                times_tackled=enemy_state.goalie.times_tackled
            ),
        )
        player_state, enemy_state, puck_state = jax.lax.cond(
            goal_phase_over,
            lambda: (fo_player, fo_enemy, fo_puck),
            lambda: (player_state, enemy_state, puck_state),
        )

        # The face-off launch velocity now comes from _faceoff_positions itself,
        # so re-derive it from the same random byte when the face-off ends.
        faceoff_launch_velocity = self._env._faceoff_launch_velocity(
            faceoff_random_byte
        )
        puck_state = puck_state.replace(
            velocity=jnp.where(
                faceoff_over, faceoff_launch_velocity, puck_state.velocity
            )
        )

        game_state = game_state.replace(
            pause_counter=pause_counter,
            player_score=game_state.player_score + player_scored.astype(jnp.int32),
            enemy_score=game_state.enemy_score + enemy_scored.astype(jnp.int32),
            remaining_time=remaining_time,
            is_faceoff=new_is_faceoff,
            goal_scored=new_goal_scored,
            is_finished=game_state.is_finished | time_up,
        )
        return player_state, enemy_state, puck_state, game_state

    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_background(self, raster, state):
        env = self._env
        c = env.consts
        jr = env.renderer.jr
        ice_id = env.renderer.COLOR_TO_ID[(192, 192, 192)]
        board_id = env.renderer.COLOR_TO_ID[(0, 0, 0)]
        goal_width = c.GOAL_X1 - c.GOAL_X0  # width of the static art being erased

        mid_x = (c.RINK_LEFT + c.RINK_RIGHT) / 2.0
        offset = _goal_offset_x(c, state.game_state.remaining_time, self.AMPLITUDE)
        top_x0 = mid_x + offset - self.GOAL_WIDTH / 2.0
        bottom_x0 = mid_x - offset - self.GOAL_WIDTH / 2.0

        # Close the two static goal notches baked into the background, refilling
        # with ice. Measured directly from background.npy: row RINK_TOP-1 and row
        # RINK_BOTTOM are each fully boards-colored across the *entire* rink
        # width (not ice at all), so the true ice band is [RINK_TOP,
        # RINK_BOTTOM - 1]. The two notches are NOT the same height: the top is
        # GOAL_HEIGHT_TOP rows, [RINK_TOP, RINK_TOP+GOAL_HEIGHT_TOP), and the
        # bottom GOAL_HEIGHT_BOTTOM rows,
        # [RINK_BOTTOM-GOAL_HEIGHT_BOTTOM, RINK_BOTTOM). Painting ice into row
        # RINK_BOTTOM itself (as an earlier version of this code did, based on a
        # single-column measurement that couldn't tell "row is black because of
        # the notch" apart from "row is black everywhere regardless of the
        # notch") leaves a visible ice patch inside the boards whenever the
        # dynamic notch no longer overlaps that spot.
        close_positions = jnp.array(
            [
                [c.GOAL_X0 - 1.0, c.RINK_TOP],
                [c.GOAL_X0 - 1.0, c.RINK_BOTTOM - c.GOAL_HEIGHT_BOTTOM],
            ],
            dtype=jnp.float32,
        )
        close_sizes = jnp.array(
            [
                [goal_width + 2.0, c.GOAL_HEIGHT_TOP],
                [goal_width + 2.0, c.GOAL_HEIGHT_BOTTOM],
            ],
            dtype=jnp.float32,
        )
        raster = jr.draw_rects(raster, close_positions, close_sizes, ice_id)

        # Cut new, smaller notches at each goal's own current dynamic position.
        xs = jnp.stack([top_x0, bottom_x0])
        ys = jnp.array(
            [float(c.RINK_TOP), float(c.RINK_BOTTOM - c.GOAL_HEIGHT_BOTTOM)],
            dtype=jnp.float32,
        )
        open_positions = jnp.stack([xs, ys], axis=1)
        open_sizes = jnp.full((2, 2), 0.0, dtype=jnp.float32)
        open_sizes = open_sizes.at[:, 0].set(float(self.GOAL_WIDTH))
        # The two notches differ in height, so set them per row.
        open_sizes = open_sizes.at[:, 1].set(
            jnp.array(
                [float(c.GOAL_HEIGHT_TOP), float(c.GOAL_HEIGHT_BOTTOM)],
                dtype=jnp.float32,
            )
        )
        raster = jr.draw_rects(raster, open_positions, open_sizes, board_id)

        return raster


class PlayerSlidingMod(JaxAtariInternalModPlugin):

    # Closer to 1.0 = slides longer and resists direction changes more.
    FRICTION_COEFF = 0.92
    MIN_SLIDE_SPEED = 0.15

    @partial(jax.jit, static_argnums=(0,))
    def _apply_action(self, character, action):
        up = jnp.any(
            jnp.array(
                [
                    action == Action.UP,
                    action == Action.UPRIGHT,
                    action == Action.UPLEFT,
                    action == Action.UPFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                ]
            )
        )
        down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNRIGHT,
                    action == Action.DOWNLEFT,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                    action == Action.LEFTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                    action == Action.RIGHTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                ]
            )
        )

        # A tackled character is frozen: ignore input and kill any glide outright.
        movable = jnp.logical_not(character.is_tackled)

        # Target velocity from raw input: full speed on a pressed axis, else 0.
        # Note this is the same target regardless of the character's *current*
        # velocity, so pressing the opposite direction targets -velocity even while
        # still sliding the old way — the blend below is what makes that a slide
        # instead of an instant reversal.
        # The base primitive moves at fixed CHARACTER_SPEED_X/Y per update; use
        # those as the steady-state glide speed the blend converges to.
        speed_x = jnp.float32(self._env.consts.CHARACTER_SPEED_X)
        speed_y = jnp.float32(self._env.consts.CHARACTER_SPEED_Y)
        target_vx = jnp.where(right, speed_x, jnp.where(left, -speed_x, 0.0))
        target_vy = jnp.where(down, speed_y, jnp.where(up, -speed_y, 0.0))
        target = jnp.array([target_vx, target_vy], dtype=jnp.float32)

        # Blend current velocity toward the target (exponential decay of the
        # difference, same shape as the puck's friction). Once within
        # MIN_SLIDE_SPEED of the target, snap to it exactly so the character
        # doesn't asymptotically creep forever.
        blended = target + (character.velocity - target) * self.FRICTION_COEFF
        residual = jnp.linalg.norm(blended - target)
        settled_velocity = jnp.where(residual > self.MIN_SLIDE_SPEED, blended, target)

        new_velocity = jnp.where(
            movable, settled_velocity, jnp.zeros(2, dtype=jnp.float32)
        )

        # No clamping here: like the base _apply_action this only produces the
        # intended movement, and the base game clamps to the character's bounds
        # afterwards in _finalize_character_positions.
        new_position = character.position + new_velocity

        # Orientation: 0 = facing left, 1 = facing right. Input keeps the current
        # facing; a tackled character keeps it too (frozen).
        new_orientation = jnp.where(
            movable & right, 1, jnp.where(movable & left, 0, character.orientation)
        )

        # Leg walk-cycle advances whenever the character is actually moving
        # (either from input or still gliding) and freezes once it comes to rest.
        has_motion = movable & (jnp.linalg.norm(new_velocity) > self.MIN_SLIDE_SPEED)
        new_walk_counter = jnp.where(has_motion, character.walk_counter + 1, 0)


        return character.replace(
            position=new_position,
            velocity=new_velocity,
            orientation=new_orientation,
            walk_counter=new_walk_counter,
        )


class EnemySpeedUpMod(JaxAtariPostStepModPlugin):
    """Enemy skaters get progressively faster every time the player scores.

    Runs after the base step, which has already moved every character at the
    fixed base speed and resolved this frame's goal/reset logic. Rather than
    reaching into how speed is computed internally, this takes the enemy
    characters' already-computed per-frame displacement (new position minus
    previous position -- naturally zero while frozen/tackled/stationary) and
    stretches it by an extra factor derived from the player's cumulative
    goal count, then re-clips to the same zone bounds ``_characters_step``
    uses. This deliberately avoids ``CharacterState.velocity``, which
    ``PlayerSlidingMod`` repurposes for actual sliding physics, so the two
    mods don't fight over the same storage.
    """

    SPEED_INCREASE_PER_GOAL = 0.15  # +15% enemy speed per player goal
    MAX_SPEED_MULTIPLIER = 2.5  # cap so it stays playable at high scores

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        c = self._env.consts

        multiplier = jnp.minimum(
            1.0
            + self.SPEED_INCREASE_PER_GOAL
            * new_state.game_state.player_score.astype(jnp.float32),
            jnp.float32(self.MAX_SPEED_MULTIPLIER),
        )
        extra_scale = multiplier - 1.0

        # Zone limits are no longer one offset constant; take the authoritative
        # per-character bounds (x_min, x_max, y_min, y_max) from the base game.
        _, _, b_skater, b_goalie = self._env._character_bounds()
        x_min = c.RINK_LEFT
        x_max = c.RINK_RIGHT - c.PLAYER_W

        def boosted(prev_char, new_char, y_lo, y_hi):
            delta = new_char.position - prev_char.position
            pos = new_char.position + delta * extra_scale
            x = jnp.clip(pos[0], x_min, x_max)
            y = jnp.clip(pos[1], y_lo, y_hi)
            return new_char.replace(position=jnp.array([x, y], dtype=jnp.float32))

        new_skater = boosted(
            prev_state.enemy_state.skater,
            new_state.enemy_state.skater,
            b_skater[2],
            b_skater[3],
        )
        new_goalie = boosted(
            prev_state.enemy_state.goalie,
            new_state.enemy_state.goalie,
            b_goalie[2],
            b_goalie[3],
        )

        new_enemy_state = new_state.enemy_state.replace(
            skater=new_skater, goalie=new_goalie
        )
        return new_state.replace(enemy_state=new_enemy_state)