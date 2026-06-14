import os
from functools import partial
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import chex
from flax import struct

import jaxatari.rendering.jax_rendering_utils as render_utils
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer


def _get_default_asset_config() -> tuple:
    return (
        {"name": "background", "type": "background", "file": "background.npy"},
        {"name": "player", "type": "single", "file": "player.npy"},
        {"name": "enemy", "type": "single", "file": "enemy.npy"},
        {"name": "puck", "type": "single", "file": "puck.npy"},
        {"name": "digits", "type": "digits", "pattern": "digit_{}.npy"},
    )


class IceHockeyConstants(struct.PyTreeNode):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    RINK_LEFT: int = struct.field(pytree_node=False, default=4)
    RINK_RIGHT: int = struct.field(pytree_node=False, default=155)
    RINK_TOP: int = struct.field(pytree_node=False, default=20)
    RINK_BOTTOM: int = struct.field(pytree_node=False, default=190)
    GOAL_X0: int = struct.field(pytree_node=False, default=60)
    GOAL_X1: int = struct.field(pytree_node=False, default=100)
    ENEMY_GOAL_Y: int = struct.field(pytree_node=False, default=20)
    PLAYER_GOAL_Y: int = struct.field(pytree_node=False, default=187)
    GOAL_HEIGHT: int = struct.field(pytree_node=False, default=7)
    PLAYER_W: int = struct.field(pytree_node=False, default=8)
    PLAYER_H: int = struct.field(pytree_node=False, default=12)
    PUCK_W: int = struct.field(pytree_node=False, default=4)
    PUCK_H: int = struct.field(pytree_node=False, default=3)
    PLAYER_SPEED: float = struct.field(pytree_node=False, default=1.5)
    PUCK_SPEED: float = struct.field(pytree_node=False, default=3.0)
    PUCK_SPEED_DECAY: float = struct.field(pytree_node=False, default=0.985)
    MIN_SEPARATION: float = struct.field(pytree_node=False, default=8.0)
    MIN_VERTICAL_DISTANCE: float = struct.field(pytree_node=False, default=14.0)
    PICKUP_RADIUS: float = struct.field(pytree_node=False, default=6.0)
    FRAMES_TACKLED: int = struct.field(pytree_node=False, default=60)
    MIN_SHOOTING_INTERVAL: int = struct.field(pytree_node=False, default=20)
    TIME_LIMIT: int = struct.field(pytree_node=False, default=10800)
    FACE_OFF_FRAMES: int = struct.field(pytree_node=False, default=40)
    FACEOFF_X: float = struct.field(pytree_node=False, default=78.0)
    FACEOFF_Y: float = struct.field(pytree_node=False, default=103.0)
    P1_X: float = struct.field(pytree_node=False, default=60.0)
    P1_Y: float = struct.field(pytree_node=False, default=115.0)
    P2_X: float = struct.field(pytree_node=False, default=85.0)
    P2_Y: float = struct.field(pytree_node=False, default=150.0)
    E1_X: float = struct.field(pytree_node=False, default=85.0)
    E1_Y: float = struct.field(pytree_node=False, default=89.0)
    E2_X: float = struct.field(pytree_node=False, default=60.0)
    E2_Y: float = struct.field(pytree_node=False, default=54.0)
    ASSET_CONFIG: tuple = struct.field(
        pytree_node=False, default_factory=_get_default_asset_config
    )


@struct.dataclass
class GameState:
    pause_counter: chex.Array
    player_score: chex.Array
    enemy_score: chex.Array
    remaining_time: chex.Array
    is_faceoff: chex.Array
    goal_scored: chex.Array
    is_finished: chex.Array


@struct.dataclass
class CharacterState:
    is_tackled: chex.Array       # int32 countdown; 0 = free
    position: chex.Array         # float32 [x, y]
    orientation: chex.Array      # 0 = left, 1 = right
    has_puck: chex.Array
    shooting_cooldown: chex.Array  # int32 countdown


@struct.dataclass
class PuckState:
    position: chex.Array   # float32 [x, y]
    velocity: chex.Array   # float32 [vx, vy]
    direction: chex.Array
    position_stick: chex.Array


@struct.dataclass
class PlayerState:
    player1: CharacterState
    player2: CharacterState
    active_character: chex.Array


@struct.dataclass
class EnemyState:
    enemy1: CharacterState
    enemy2: CharacterState
    active_character: chex.Array
    enemy_target: chex.Array   # float32 [x, y] — pursuit target, updated every 4 frames


@struct.dataclass
class IceHockeyState:
    player_state: PlayerState
    enemy_state: EnemyState
    puck_state: PuckState
    counter: chex.Array
    game_state: GameState
    lfsr: chex.Array   # int32, 16-bit Galois LFSR for zigzag noise


@struct.dataclass
class IceHockeyInfo:
    player_score: chex.Array
    enemy_score: chex.Array
    remaining_time: chex.Array


@struct.dataclass
class IceHockeyObservation:
    player1: ObjectObservation
    player2: ObjectObservation
    enemy1: ObjectObservation
    enemy2: ObjectObservation
    puck: ObjectObservation
    player_score: chex.Array
    enemy_score: chex.Array
    remaining_time: chex.Array
    active_player: chex.Array


def _tw(cond, a, b):
    """jnp.where applied element-wise over two same-structure pytrees."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


class JaxIceHockey(JaxEnvironment):

    ACTION_SET = jnp.array([
        Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
        Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
        Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
        Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE,
    ], dtype=jnp.int32)

    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        consts = consts or IceHockeyConstants()
        super().__init__(consts)
        self.renderer = IceHockeyRenderer(self.consts)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        obj = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))
        return spaces.Dict({
            "player1": obj, "player2": obj, "enemy1": obj, "enemy2": obj, "puck": obj,
            "player_score": spaces.Box(0, 99, shape=(), dtype=jnp.int32),
            "enemy_score": spaces.Box(0, 99, shape=(), dtype=jnp.int32),
            "remaining_time": spaces.Box(0, self.consts.TIME_LIMIT, shape=(), dtype=jnp.int32),
            "active_player": spaces.Box(0, 1, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = None) -> Tuple:
        c = self.consts

        def char(x, y):
            return CharacterState(
                is_tackled=jnp.array(0, dtype=jnp.int32),
                position=jnp.array([x, y], dtype=jnp.float32),
                orientation=jnp.array(0, dtype=jnp.int32),
                has_puck=jnp.array(False),
                shooting_cooldown=jnp.array(0, dtype=jnp.int32),
            )

        state = IceHockeyState(
            player_state=PlayerState(
                player1=char(c.P1_X, c.P1_Y),
                player2=char(c.P2_X, c.P2_Y),
                active_character=jnp.array(0, dtype=jnp.int32),
            ),
            enemy_state=EnemyState(
                enemy1=char(c.E1_X, c.E1_Y),
                enemy2=char(c.E2_X, c.E2_Y),
                active_character=jnp.array(0, dtype=jnp.int32),
                enemy_target=jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32),
            ),
            puck_state=PuckState(
                position=jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32),
                velocity=jnp.zeros(2, dtype=jnp.float32),
                direction=jnp.array(0, dtype=jnp.int32),
                position_stick=jnp.array(0, dtype=jnp.int32),
            ),
            counter=jnp.array(0, dtype=jnp.int32),
            lfsr=jnp.array(0xACE1, dtype=jnp.int32),
            game_state=GameState(
                pause_counter=jnp.array(c.FACE_OFF_FRAMES, dtype=jnp.int32),
                player_score=jnp.array(0, dtype=jnp.int32),
                enemy_score=jnp.array(0, dtype=jnp.int32),
                remaining_time=jnp.array(c.TIME_LIMIT, dtype=jnp.int32),
                is_faceoff=jnp.array(True),
                goal_scored=jnp.array(False),
                is_finished=jnp.array(False),
            ),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: IceHockeyState, action: chex.Array) -> Tuple:
        c = self.consts
        prev = state
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        frozen = state.game_state.pause_counter > 0

        enemy_action = self._enemy_policy(state)

        rink = jnp.array([float(c.RINK_LEFT), float(c.RINK_RIGHT),
                          float(c.RINK_TOP),  float(c.RINK_BOTTOM)], dtype=jnp.float32)

        new_ps, new_es = self._characters_step(
            state.player_state, state.enemy_state,
            state.puck_state.position,
            atari_action, enemy_action,
            jnp.float32(c.PLAYER_SPEED),
            jnp.float32(c.MIN_SEPARATION),
            jnp.float32(c.MIN_VERTICAL_DISTANCE),
            rink, rink, rink, rink,
        )
        new_puck, new_ps, new_es = self._puck_step(state, new_ps, new_es, atari_action, enemy_action)
        new_puck, new_ps, new_es = self._pickup_check(new_puck, new_ps, new_es)
        new_ps, new_es, new_puck = self._tackle_step(new_ps, new_es, new_puck, atari_action, enemy_action)
        new_ps, new_es = self._tick_cooldowns(new_ps, new_es)

        # Apply face-off freeze: discard all updates while paused.
        new_ps   = _tw(frozen, state.player_state, new_ps)
        new_es   = _tw(frozen, state.enemy_state,  new_es)
        new_puck = _tw(frozen, state.puck_state,   new_puck)

        player_scored, enemy_scored = self._goal_check(new_puck)
        new_gs = self._update_game_state(state.game_state, player_scored, enemy_scored)

        # Reset positions on goal.
        goal = player_scored | enemy_scored
        r_ps, r_es, r_puck = self._faceoff_positions(c)
        new_ps   = _tw(goal, r_ps,   new_ps)
        new_es   = _tw(goal, r_es,   new_es)
        new_puck = _tw(goal, r_puck, new_puck)

        # Advance LFSR every frame; recalculate enemy pursuit target every 4 frames.
        new_lfsr = self._lfsr_step(state.lfsr)
        enemy_has_puck  = new_es.enemy1.has_puck | new_es.enemy2.has_puck
        player_has_puck = new_ps.player1.has_puck | new_ps.player2.has_puck
        # 4 bits each → 0–15, then subtract 7 → signed range −7 to +8
        noise = jnp.array([
            (new_lfsr & jnp.int32(0xF)).astype(jnp.float32) - jnp.float32(7.0),
            ((new_lfsr >> 4) & jnp.int32(0xF)).astype(jnp.float32) - jnp.float32(7.0),
        ])
        candidate = jnp.where(player_has_puck, new_puck.position + noise, new_puck.position)
        new_target = jnp.where(
            ((state.counter % 4) == 0) & ~enemy_has_puck,
            candidate,
            new_es.enemy_target,
        )
        new_es = new_es.replace(enemy_target=new_target)

        new_state = IceHockeyState(
            player_state=new_ps, enemy_state=new_es, puck_state=new_puck,
            counter=state.counter + 1, game_state=new_gs, lfsr=new_lfsr,
        )
        obs    = self._get_observation(new_state)
        reward = self._get_reward(prev, new_state)
        done   = self._get_done(new_state)
        info   = self._get_info(new_state)
        return obs, new_state, reward, done, info

    def render(self, state: IceHockeyState) -> jnp.ndarray:
        return self.renderer.render(state)

    # ------------------------------------------------------------------
    # Character movement
    # ------------------------------------------------------------------

    def _resolve_active(self, c1, c2, puck_pos, cur_active):
        d1 = jnp.sum((c1.position - puck_pos) ** 2)
        d2 = jnp.sum((c2.position - puck_pos) ** 2)
        return jnp.where(d1 < d2, jnp.int32(0),
               jnp.where(d2 < d1, jnp.int32(1), cur_active)).astype(jnp.int32)

    def _apply_action(self, ch, action, bounds, speed):
        up    = jnp.any(jnp.array([action == Action.UP,    action == Action.UPRIGHT,  action == Action.UPLEFT,
                                    action == Action.UPFIRE, action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE]))
        down  = jnp.any(jnp.array([action == Action.DOWN,   action == Action.DOWNRIGHT, action == Action.DOWNLEFT,
                                    action == Action.DOWNFIRE, action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE]))
        left  = jnp.any(jnp.array([action == Action.LEFT,   action == Action.UPLEFT,   action == Action.DOWNLEFT,
                                    action == Action.LEFTFIRE, action == Action.UPLEFTFIRE, action == Action.DOWNLEFTFIRE]))
        right = jnp.any(jnp.array([action == Action.RIGHT,  action == Action.UPRIGHT,  action == Action.DOWNRIGHT,
                                    action == Action.RIGHTFIRE, action == Action.UPRIGHTFIRE, action == Action.DOWNRIGHTFIRE]))

        movable = ch.is_tackled == 0
        dx = jnp.where(movable & right, speed, jnp.where(movable & left, -speed, jnp.float32(0)))
        dy = jnp.where(movable & down,  speed, jnp.where(movable & up,   -speed, jnp.float32(0)))
        nx = jnp.clip(ch.position[0] + dx, bounds[0], bounds[1])
        ny = jnp.clip(ch.position[1] + dy, bounds[2], bounds[3])
        ori = jnp.where(movable & right, jnp.int32(1),
              jnp.where(movable & left,  jnp.int32(0), ch.orientation))
        return ch.replace(position=jnp.array([nx, ny]), orientation=ori)

    def _separate(self, a, b, min_sep):
        delta  = a - b
        dist2  = jnp.sum(delta ** 2)
        over   = dist2 < min_sep ** 2
        coinc  = dist2 <= 0.0
        inv_d  = jnp.where(coinc, jnp.float32(0), jax.lax.rsqrt(dist2))
        offset = 0.5 * delta * (min_sep * inv_d - 1.0)
        offset = jnp.where(coinc, jnp.array([min_sep * 0.5, 0.0]), offset)
        offset = jnp.where(over, offset, jnp.zeros(2))
        return a + offset, b - offset

    def _push_vertical(self, active_pos, passive_pos, min_v):
        dy      = passive_pos[1] - active_pos[1]
        close   = jnp.abs(dy) < min_v
        side    = jnp.where(dy != 0.0, jnp.sign(dy), jnp.float32(1))
        new_y   = jnp.where(close, active_pos[1] + side * min_v, passive_pos[1])
        return jnp.array([passive_pos[0], new_y])

    def _clamp(self, pos, bounds):
        return jnp.array([jnp.clip(pos[0], bounds[0], bounds[1]),
                          jnp.clip(pos[1], bounds[2], bounds[3])])

    def _characters_step(self, ps, es, puck_pos, p_action, e_action,
                         speed, min_sep, min_v, bp1, bp2, be1, be2):
        pa = self._resolve_active(ps.player1, ps.player2, puck_pos, ps.active_character)
        ea = self._resolve_active(es.enemy1,  es.enemy2,  puck_pos, es.active_character)

        a1 = jnp.where(pa == 0, p_action, Action.NOOP)
        a2 = jnp.where(pa == 1, p_action, Action.NOOP)
        p1 = self._apply_action(ps.player1, a1, bp1, speed)
        p2 = self._apply_action(ps.player2, a2, bp2, speed)

        a1e = jnp.where(ea == 0, e_action, Action.NOOP)
        a2e = jnp.where(ea == 1, e_action, Action.NOOP)
        e1 = self._apply_action(es.enemy1, a1e, be1, speed)
        e2 = self._apply_action(es.enemy2, a2e, be2, speed)

        pp1, pe1 = self._separate(p1.position, e1.position, min_sep)
        pp1, pe2 = self._separate(pp1,         e2.position, min_sep)
        pp2, pe1 = self._separate(p2.position, pe1,         min_sep)
        pp2, pe2 = self._separate(pp2,         pe2,         min_sep)

        pp2 = jnp.where(pa == 0, self._push_vertical(pp1, pp2, min_v), pp2)
        pp1 = jnp.where(pa == 1, self._push_vertical(pp2, pp1, min_v), pp1)
        pe2 = jnp.where(ea == 0, self._push_vertical(pe1, pe2, min_v), pe2)
        pe1 = jnp.where(ea == 1, self._push_vertical(pe2, pe1, min_v), pe1)

        pp1 = self._clamp(pp1, bp1); pp2 = self._clamp(pp2, bp2)
        pe1 = self._clamp(pe1, be1); pe2 = self._clamp(pe2, be2)

        new_ps = ps.replace(player1=p1.replace(position=pp1), player2=p2.replace(position=pp2), active_character=pa)
        new_es = es.replace(enemy1=e1.replace(position=pe1),  enemy2=e2.replace(position=pe2),  active_character=ea)
        return new_ps, new_es

    # ------------------------------------------------------------------
    # Puck physics
    # ------------------------------------------------------------------

    def _has_fire(self, action):
        return jnp.any(jnp.array([
            action == Action.FIRE,
            action == Action.UPFIRE,        action == Action.DOWNFIRE,
            action == Action.LEFTFIRE,      action == Action.RIGHTFIRE,
            action == Action.UPLEFTFIRE,    action == Action.UPRIGHTFIRE,
            action == Action.DOWNLEFTFIRE,  action == Action.DOWNRIGHTFIRE,
        ]))

    def _shoot_vel(self, action, is_player):
        c  = self.consts
        r  = jnp.any(jnp.array([action == Action.RIGHT, action == Action.UPRIGHT,  action == Action.DOWNRIGHT,
                                  action == Action.RIGHTFIRE, action == Action.UPRIGHTFIRE, action == Action.DOWNRIGHTFIRE]))
        l  = jnp.any(jnp.array([action == Action.LEFT,  action == Action.UPLEFT,   action == Action.DOWNLEFT,
                                  action == Action.LEFTFIRE,  action == Action.UPLEFTFIRE,  action == Action.DOWNLEFTFIRE]))
        u  = jnp.any(jnp.array([action == Action.UP,    action == Action.UPRIGHT,  action == Action.UPLEFT,
                                  action == Action.UPFIRE, action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE]))
        d  = jnp.any(jnp.array([action == Action.DOWN,  action == Action.DOWNRIGHT, action == Action.DOWNLEFT,
                                  action == Action.DOWNFIRE, action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE]))
        vx = jnp.where(r, jnp.float32(c.PUCK_SPEED * 0.5),
             jnp.where(l, jnp.float32(-c.PUCK_SPEED * 0.5), jnp.float32(0)))
        dvy = jnp.float32(-c.PUCK_SPEED if is_player else c.PUCK_SPEED)
        vy  = jnp.where(u, jnp.float32(-c.PUCK_SPEED),
              jnp.where(d, jnp.float32(c.PUCK_SPEED), dvy))
        return jnp.array([vx, vy], dtype=jnp.float32)

    def _puck_step(self, state, new_ps, new_es, p_action, e_action):
        c    = self.consts
        puck = state.puck_state
        p1h  = state.player_state.player1.has_puck
        p2h  = state.player_state.player2.has_puck
        e1h  = state.enemy_state.enemy1.has_puck
        e2h  = state.enemy_state.enemy2.has_puck
        has  = p1h | p2h | e1h | e2h

        carrier_pos = jnp.where(p1h, new_ps.player1.position,
                      jnp.where(p2h, new_ps.player2.position,
                      jnp.where(e1h, new_es.enemy1.position, new_es.enemy2.position)))
        carrier_ori = jnp.where(p1h, new_ps.player1.orientation,
                      jnp.where(p2h, new_ps.player2.orientation,
                      jnp.where(e1h, new_es.enemy1.orientation, new_es.enemy2.orientation)))
        stick_x     = jnp.where(carrier_ori == 1, jnp.float32(4), jnp.float32(-4))
        carried_pos = carrier_pos + jnp.array([stick_x, jnp.float32(2)])

        vel      = puck.velocity * jnp.float32(c.PUCK_SPEED_DECAY)
        tent_pos = puck.position + puck.velocity
        hit_l = tent_pos[0] < c.RINK_LEFT;  hit_r = tent_pos[0] > c.RINK_RIGHT
        hit_t = tent_pos[1] < c.RINK_TOP;   hit_b = tent_pos[1] > c.RINK_BOTTOM
        vx    = jnp.where(hit_l | hit_r, -vel[0], vel[0])
        vy    = jnp.where(hit_t | hit_b, -vel[1], vel[1])
        vel   = jnp.array([vx, vy], dtype=jnp.float32)
        free_pos = jnp.clip(tent_pos,
                            jnp.array([float(c.RINK_LEFT),  float(c.RINK_TOP)]),
                            jnp.array([float(c.RINK_RIGHT), float(c.RINK_BOTTOM)]))

        puck_pos = jnp.where(has, carried_pos, free_pos)
        puck_vel = jnp.where(has, jnp.zeros(2, dtype=jnp.float32), vel)

        p1_can = p1h & (state.player_state.player1.shooting_cooldown == 0)
        p2_can = p2h & (state.player_state.player2.shooting_cooldown == 0)
        e1_can = e1h & (state.enemy_state.enemy1.shooting_cooldown == 0)
        e2_can = e2h & (state.enemy_state.enemy2.shooting_cooldown == 0)

        p_fires = self._has_fire(p_action)
        e_fires = self._has_fire(e_action)
        p_shoots = p_fires & (p1_can | p2_can)
        e_shoots = e_fires & (e1_can | e2_can)
        shoots   = p_shoots | e_shoots

        sv       = jnp.where(p_shoots,
                              self._shoot_vel(p_action, is_player=True),
                              self._shoot_vel(e_action, is_player=False))
        puck_vel = jnp.where(shoots, sv, puck_vel)

        cd = jnp.array(c.MIN_SHOOTING_INTERVAL, dtype=jnp.int32)
        f  = jnp.array(False)
        new_ps = new_ps.replace(
            player1=new_ps.player1.replace(
                has_puck=jnp.where(p_shoots, f, new_ps.player1.has_puck),
                shooting_cooldown=jnp.where(p_shoots & p1h, cd, new_ps.player1.shooting_cooldown)),
            player2=new_ps.player2.replace(
                has_puck=jnp.where(p_shoots, f, new_ps.player2.has_puck),
                shooting_cooldown=jnp.where(p_shoots & p2h, cd, new_ps.player2.shooting_cooldown)),
        )
        new_es = new_es.replace(
            enemy1=new_es.enemy1.replace(
                has_puck=jnp.where(e_shoots, f, new_es.enemy1.has_puck),
                shooting_cooldown=jnp.where(e_shoots & e1h, cd, new_es.enemy1.shooting_cooldown)),
            enemy2=new_es.enemy2.replace(
                has_puck=jnp.where(e_shoots, f, new_es.enemy2.has_puck),
                shooting_cooldown=jnp.where(e_shoots & e2h, cd, new_es.enemy2.shooting_cooldown)),
        )
        return puck.replace(position=puck_pos, velocity=puck_vel), new_ps, new_es

    # ------------------------------------------------------------------
    # Pickup
    # ------------------------------------------------------------------

    def _pickup_check(self, puck, ps, es):
        c    = self.consts
        free = ~(ps.player1.has_puck | ps.player2.has_puck | es.enemy1.has_puck | es.enemy2.has_puck)
        r2   = jnp.float32(c.PICKUP_RADIUS ** 2)

        dp1 = jnp.sum((puck.position - ps.player1.position) ** 2)
        dp2 = jnp.sum((puck.position - ps.player2.position) ** 2)
        de1 = jnp.sum((puck.position - es.enemy1.position)  ** 2)
        de2 = jnp.sum((puck.position - es.enemy2.position)  ** 2)

        cp1 = free & (dp1 < r2) & (ps.player1.shooting_cooldown == 0)
        cp2 = free & (dp2 < r2) & (ps.player2.shooting_cooldown == 0)
        ce1 = free & (de1 < r2) & (es.enemy1.shooting_cooldown == 0)
        ce2 = free & (de2 < r2) & (es.enemy2.shooting_cooldown == 0)

        big  = jnp.float32(1e9)
        ep1  = jnp.where(cp1, dp1, big); ep2 = jnp.where(cp2, dp2, big)
        ee1  = jnp.where(ce1, de1, big); ee2 = jnp.where(ce2, de2, big)
        mind = jnp.minimum(jnp.minimum(ep1, ep2), jnp.minimum(ee1, ee2))
        any_ = mind < big

        g1 = any_ & (ep1 == mind)
        g2 = any_ & ~g1 & (ep2 == mind)
        g3 = any_ & ~g1 & ~g2 & (ee1 == mind)
        g4 = any_ & ~g1 & ~g2 & ~g3 & (ee2 == mind)

        ps = ps.replace(
            player1=ps.player1.replace(has_puck=ps.player1.has_puck | g1),
            player2=ps.player2.replace(has_puck=ps.player2.has_puck | g2),
        )
        es = es.replace(
            enemy1=es.enemy1.replace(has_puck=es.enemy1.has_puck | g3),
            enemy2=es.enemy2.replace(has_puck=es.enemy2.has_puck | g4),
        )
        return puck, ps, es

    # ------------------------------------------------------------------
    # Tackle
    # ------------------------------------------------------------------

    def _tackle_step(self, ps, es, puck, p_action, e_action):
        c  = self.consts
        p1 = ps.player1; p2 = ps.player2
        e1 = es.enemy1;  e2 = es.enemy2

        def close(pos_a, pos_b):
            # Characters are pushed to exactly MIN_SEPARATION apart by _separate().
            # Use a threshold just above that distance to detect contact.
            dist2 = jnp.sum((pos_a - pos_b) ** 2)
            return dist2 < jnp.float32(c.MIN_SEPARATION ** 2 + 10.0)

        p1e1 = close(p1.position, e1.position)
        p1e2 = close(p1.position, e2.position)
        p2e1 = close(p2.position, e1.position)
        p2e2 = close(p2.position, e2.position)

        p_fires = self._has_fire(p_action)
        e_fires = self._has_fire(e_action)

        # Only the active character acts
        p1_can_tackle = p_fires & (ps.active_character == 0) & ~p1.has_puck
        p2_can_tackle = p_fires & (ps.active_character == 1) & ~p2.has_puck
        e1_can_tackle = e_fires & (es.active_character == 0) & ~e1.has_puck
        e2_can_tackle = e_fires & (es.active_character == 1) & ~e2.has_puck

        # Who gets tackled: the other side's character when attacker is close + fires
        n1 = ((e1_can_tackle & p1e1) | (e2_can_tackle & p1e2)) & (p1.is_tackled == 0)
        n2 = ((e1_can_tackle & p2e1) | (e2_can_tackle & p2e2)) & (p2.is_tackled == 0)
        n3 = ((p1_can_tackle & p1e1) | (p2_can_tackle & p2e1)) & (e1.is_tackled == 0)
        n4 = ((p1_can_tackle & p1e2) | (p2_can_tackle & p2e2)) & (e2.is_tackled == 0)

        fr = jnp.array(c.FRAMES_TACKLED, dtype=jnp.int32)
        cd = jnp.array(c.MIN_SHOOTING_INTERVAL, dtype=jnp.int32)
        f  = jnp.array(False)

        d1 = n1 & p1.has_puck; d2 = n2 & p2.has_puck
        d3 = n3 & e1.has_puck; d4 = n4 & e2.has_puck
        drop = d1 | d2 | d3 | d4

        new_ps = ps.replace(
            player1=p1.replace(is_tackled=jnp.where(n1, fr, p1.is_tackled),
                               has_puck=jnp.where(d1, f, p1.has_puck),
                               shooting_cooldown=jnp.where(d1, cd, p1.shooting_cooldown)),
            player2=p2.replace(is_tackled=jnp.where(n2, fr, p2.is_tackled),
                               has_puck=jnp.where(d2, f, p2.has_puck),
                               shooting_cooldown=jnp.where(d2, cd, p2.shooting_cooldown)),
        )
        new_es = es.replace(
            enemy1=e1.replace(is_tackled=jnp.where(n3, fr, e1.is_tackled),
                              has_puck=jnp.where(d3, f, e1.has_puck),
                              shooting_cooldown=jnp.where(d3, cd, e1.shooting_cooldown)),
            enemy2=e2.replace(is_tackled=jnp.where(n4, fr, e2.is_tackled),
                              has_puck=jnp.where(d4, f, e2.has_puck),
                              shooting_cooldown=jnp.where(d4, cd, e2.shooting_cooldown)),
        )
        new_puck = puck.replace(velocity=jnp.where(drop, jnp.zeros(2, dtype=jnp.float32), puck.velocity))
        return new_ps, new_es, new_puck

    # ------------------------------------------------------------------
    # Cooldown tick
    # ------------------------------------------------------------------

    def _tick_cooldowns(self, ps, es):
        z = jnp.array(0, dtype=jnp.int32)

        def tick(ch):
            return ch.replace(
                is_tackled=jnp.maximum(z, ch.is_tackled - 1),
                shooting_cooldown=jnp.maximum(z, ch.shooting_cooldown - 1),
            )

        return (ps.replace(player1=tick(ps.player1), player2=tick(ps.player2)),
                es.replace(enemy1=tick(es.enemy1),   enemy2=tick(es.enemy2)))

    # ------------------------------------------------------------------
    # Goal / game state
    # ------------------------------------------------------------------

    def _goal_check(self, puck):
        c  = self.consts
        px = puck.position[0]; py = puck.position[1]
        in_x = (px >= c.GOAL_X0) & (px <= c.GOAL_X1)
        return in_x & (py <= c.ENEMY_GOAL_Y), in_x & (py >= c.PLAYER_GOAL_Y)

    def _update_game_state(self, gs, player_scored, enemy_scored):
        c    = self.consts
        goal = player_scored | enemy_scored
        new_t = jnp.maximum(jnp.array(0, dtype=jnp.int32), gs.remaining_time - 1)
        new_p = jnp.where(goal,
                          jnp.array(c.FACE_OFF_FRAMES, dtype=jnp.int32),
                          jnp.maximum(jnp.array(0, dtype=jnp.int32), gs.pause_counter - 1))
        return gs.replace(
            player_score=gs.player_score + player_scored.astype(jnp.int32),
            enemy_score=gs.enemy_score   + enemy_scored.astype(jnp.int32),
            remaining_time=new_t,
            is_finished=new_t <= 0,
            goal_scored=goal,
            is_faceoff=new_p > 0,
            pause_counter=new_p,
        )

    def _faceoff_positions(self, c):
        def char(x, y):
            return CharacterState(
                is_tackled=jnp.array(0, dtype=jnp.int32),
                position=jnp.array([x, y], dtype=jnp.float32),
                orientation=jnp.array(0, dtype=jnp.int32),
                has_puck=jnp.array(False),
                shooting_cooldown=jnp.array(0, dtype=jnp.int32),
            )
        return (
            PlayerState(player1=char(c.P1_X, c.P1_Y), player2=char(c.P2_X, c.P2_Y),
                        active_character=jnp.array(0, dtype=jnp.int32)),
            EnemyState(enemy1=char(c.E1_X, c.E1_Y),   enemy2=char(c.E2_X, c.E2_Y),
                       active_character=jnp.array(0, dtype=jnp.int32),
                       enemy_target=jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32)),
            PuckState(position=jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32),
                      velocity=jnp.zeros(2, dtype=jnp.float32),
                      direction=jnp.array(0, dtype=jnp.int32),
                      position_stick=jnp.array(0, dtype=jnp.int32)),
        )

    # ------------------------------------------------------------------
    # Enemy AI
    # ------------------------------------------------------------------

    @staticmethod
    def _lfsr_step(lfsr: chex.Array) -> chex.Array:
        """Galois 16-bit LFSR, polynomial 0xB400 (maximal length, period 65535)."""
        bit = lfsr & 1
        return jnp.where(bit, (lfsr >> 1) ^ jnp.int32(0xB400), lfsr >> 1).astype(jnp.int32)

    def _enemy_policy(self, state: IceHockeyState) -> chex.Array:
        c   = self.consts
        es  = state.enemy_state
        ai  = es.active_character
        pos = jnp.where(ai == 0, es.enemy1.position, es.enemy2.position)
        has = es.enemy1.has_puck | es.enemy2.has_puck
        tgt = jnp.where(has,
                        jnp.array([c.FACEOFF_X, float(c.PLAYER_GOAL_Y)], dtype=jnp.float32),
                        es.enemy_target)
        dx  = tgt[0] - pos[0]; dy = tgt[1] - pos[1]
        r   = dx >  2.0; l = dx < -2.0
        d   = dy >  2.0; u = dy < -2.0
        near_goal    = jnp.abs(pos[1] - float(c.PLAYER_GOAL_Y)) < 50.0
        should_shoot = has & near_goal

        ps = state.player_state
        thresh2 = jnp.float32(c.MIN_SEPARATION ** 2 + 10.0)
        p1_close = jnp.sum((pos - ps.player1.position) ** 2) < thresh2
        p2_close = jnp.sum((pos - ps.player2.position) ** 2) < thresh2
        should_tackle = ~has & (p1_close | p2_close)

        return jnp.where(should_shoot,  jnp.int32(Action.DOWNFIRE),
               jnp.where(should_tackle, jnp.int32(Action.FIRE),
               jnp.where(r & d,         jnp.int32(Action.DOWNRIGHT),
               jnp.where(l & d,         jnp.int32(Action.DOWNLEFT),
               jnp.where(r & u,         jnp.int32(Action.UPRIGHT),
               jnp.where(l & u,         jnp.int32(Action.UPLEFT),
               jnp.where(r,             jnp.int32(Action.RIGHT),
               jnp.where(l,             jnp.int32(Action.LEFT),
               jnp.where(d,             jnp.int32(Action.DOWN),
               jnp.where(u,             jnp.int32(Action.UP),
                                         jnp.int32(Action.NOOP)))))))))))

    # ------------------------------------------------------------------
    # RL helpers
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: IceHockeyState) -> IceHockeyObservation:
        c = self.consts

        def obj(pos, w, h):
            return ObjectObservation.create(
                x=pos[0].astype(jnp.int32), y=pos[1].astype(jnp.int32),
                width=jnp.array(w, dtype=jnp.int32), height=jnp.array(h, dtype=jnp.int32),
            )

        return IceHockeyObservation(
            player1=obj(state.player_state.player1.position, c.PLAYER_W, c.PLAYER_H),
            player2=obj(state.player_state.player2.position, c.PLAYER_W, c.PLAYER_H),
            enemy1=obj(state.enemy_state.enemy1.position,    c.PLAYER_W, c.PLAYER_H),
            enemy2=obj(state.enemy_state.enemy2.position,    c.PLAYER_W, c.PLAYER_H),
            puck=obj(state.puck_state.position,              c.PUCK_W,   c.PUCK_H),
            player_score=state.game_state.player_score,
            enemy_score=state.game_state.enemy_score,
            remaining_time=state.game_state.remaining_time,
            active_player=state.player_state.active_character,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: IceHockeyObservation) -> jnp.ndarray:
        def flat(o):
            return jnp.array([o.x, o.y, o.width, o.height, o.active], dtype=jnp.float32)
        return jnp.concatenate([
            flat(obs.player1), flat(obs.player2), flat(obs.enemy1), flat(obs.enemy2), flat(obs.puck),
            jnp.array([obs.player_score, obs.enemy_score, obs.remaining_time, obs.active_player],
                      dtype=jnp.float32),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: IceHockeyState) -> IceHockeyInfo:
        return IceHockeyInfo(player_score=state.game_state.player_score,
                             enemy_score=state.game_state.enemy_score,
                             remaining_time=state.game_state.remaining_time)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev: IceHockeyState, state: IceHockeyState) -> chex.Array:
        d0 = prev.game_state.player_score  - prev.game_state.enemy_score
        d1 = state.game_state.player_score - state.game_state.enemy_score
        return (d1 - d0).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: IceHockeyState) -> chex.Array:
        return state.game_state.is_finished


class IceHockeyRenderer(JAXGameRenderer):

    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        self.consts = consts or IceHockeyConstants()
        super().__init__(self.consts)
        self.config = render_utils.RendererConfig(game_dimensions=(210, 160), channels=3, downscale=None)
        self.jr     = render_utils.JaxRenderingUtils(self.config)
        self.sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "icehockey")
        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND,
         self.COLOR_TO_ID, self.FLIP_OFFSETS) = self.jr.load_and_setup_assets(
            list(self.consts.ASSET_CONFIG), self.sprite_path)
        self.PALETTE, self.RED_ID = self.jr.add_palette_color(self.PALETTE, [255, 0, 0])

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: IceHockeyState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)
        pm = self.SHAPE_MASKS["player"]; em = self.SHAPE_MASKS["enemy"]; pk = self.SHAPE_MASKS["puck"]

        def col(pos): return jnp.round(pos[0]).astype(jnp.int32)
        def row(pos): return jnp.round(pos[1]).astype(jnp.int32)

        p1 = state.player_state.player1.position; p2 = state.player_state.player2.position
        e1 = state.enemy_state.enemy1.position;   e2 = state.enemy_state.enemy2.position
        pp = state.puck_state.position

        raster = self.jr.render_at_clipped(raster, col(p2), row(p2), pm)
        raster = self.jr.render_at_clipped(raster, col(e2), row(e2), em)
        raster = self.jr.render_at_clipped(raster, col(p1), row(p1), pm)
        raster = self.jr.render_at_clipped(raster, col(e1), row(e1), em)
        raster = self.jr.render_at_clipped(raster, col(pp), row(pp), pk)

        def draw_tackle_outline(r, pos, is_tackled):
            x = jnp.round(pos[0]).astype(jnp.int32)
            y = jnp.round(pos[1]).astype(jnp.int32)
            xx, yy = self.jr._xx, self.jr._yy
            pw, ph = self.consts.PLAYER_W, self.consts.PLAYER_H
            outer = (xx >= x - 1) & (xx < x + pw + 1) & (yy >= y - 1) & (yy < y + ph + 1)
            inner = (xx >= x) & (xx < x + pw) & (yy >= y) & (yy < y + ph)
            border = outer & ~inner
            return jnp.where(is_tackled & border, jnp.asarray(self.RED_ID, r.dtype), r)

        raster = draw_tackle_outline(raster, p1, state.player_state.player1.is_tackled > 0)
        raster = draw_tackle_outline(raster, p2, state.player_state.player2.is_tackled > 0)
        raster = draw_tackle_outline(raster, e1, state.enemy_state.enemy1.is_tackled > 0)
        raster = draw_tackle_outline(raster, e2, state.enemy_state.enemy2.is_tackled > 0)

        dm = self.SHAPE_MASKS["digits"]

        def draw_score(r, value, xs, xd):
            digits = self.jr.int_to_digits(value, max_digits=2)
            single = value < 10
            start  = jax.lax.select(single, jnp.int32(1), jnp.int32(0))
            count  = jax.lax.select(single, jnp.int32(1), jnp.int32(2))
            x      = jax.lax.select(single, jnp.int32(xs), jnp.int32(xd))
            return self.jr.render_label_selective(r, x, 3, digits, dm, start, count,
                                                  spacing=7, max_digits_to_render=2)

        raster = draw_score(raster, state.game_state.enemy_score,  43,  33)
        raster = draw_score(raster, state.game_state.player_score, 113, 103)
        return self.jr.render_from_palette(raster, self.PALETTE)
