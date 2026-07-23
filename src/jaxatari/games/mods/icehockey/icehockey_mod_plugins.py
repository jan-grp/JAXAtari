from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.modification import JaxAtariInternalModPlugin


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

        # Friction (unchanged from the base game).
        current_speed = jnp.linalg.norm(vel)
        fric_coeff = jnp.where(
            current_speed > c.PUCK_MIN_SPEED, c.PUCK_FRICTION_COEFF, 1.0
        )
        vel = vel * fric_coeff

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
        self, game_state, player_state, enemy_state, puck_state, frozen
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

        fo_player, fo_enemy, fo_puck = self._env._faceoff_positions()
        player_state, enemy_state, puck_state = jax.lax.cond(
            goal_phase_over,
            lambda: (fo_player, fo_enemy, fo_puck),
            lambda: (player_state, enemy_state, puck_state),
        )

        faceoff_launch_velocity = jnp.asarray(
            c.FACE_OFF_PUCK_DIRECTION, dtype=jnp.float32
        ) * jnp.float32(c.FACE_OFF_PUCK_MAX_SPEED)
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
        # RINK_BOTTOM - 1] and both notches are exactly GOAL_HEIGHT rows,
        # symmetric: [RINK_TOP, RINK_TOP+GOAL_HEIGHT) and
        # [RINK_BOTTOM-GOAL_HEIGHT, RINK_BOTTOM). Painting ice into row
        # RINK_BOTTOM itself (as an earlier version of this code did, based on a
        # single-column measurement that couldn't tell "row is black because of
        # the notch" apart from "row is black everywhere regardless of the
        # notch") leaves a visible ice patch inside the boards whenever the
        # dynamic notch no longer overlaps that spot.
        close_positions = jnp.array(
            [
                [c.GOAL_X0 - 1.0, c.RINK_TOP],
                [c.GOAL_X0 - 1.0, c.RINK_BOTTOM - c.GOAL_HEIGHT],
            ],
            dtype=jnp.float32,
        )
        close_sizes = jnp.array(
            [
                [goal_width + 2.0, c.GOAL_HEIGHT],
                [goal_width + 2.0, c.GOAL_HEIGHT],
            ],
            dtype=jnp.float32,
        )
        raster = jr.draw_rects(raster, close_positions, close_sizes, ice_id)

        # Cut new, smaller notches at each goal's own current dynamic position.
        xs = jnp.stack([top_x0, bottom_x0])
        ys = jnp.array(
            [float(c.RINK_TOP), float(c.RINK_BOTTOM - c.GOAL_HEIGHT)], dtype=jnp.float32
        )
        open_positions = jnp.stack([xs, ys], axis=1)
        open_sizes = jnp.full((2, 2), 0.0, dtype=jnp.float32)
        open_sizes = open_sizes.at[:, 0].set(float(self.GOAL_WIDTH))
        open_sizes = open_sizes.at[:, 1].set(float(c.GOAL_HEIGHT))
        raster = jr.draw_rects(raster, open_positions, open_sizes, board_id)

        return raster
