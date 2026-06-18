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
    """Manifest of the .npy sprites the renderer loads from sprites/icehockey/.

    Run scripts/make_icehockey_sprites.py once to create the placeholder files.
    """
    return (
        {"name": "background", "type": "background", "file": "background.npy"},
        {"name": "player", "type": "single", "file": "player.npy"},
        {"name": "enemy", "type": "single", "file": "enemy.npy"},
        {"name": "puck", "type": "single", "file": "puck.npy"},
        {"name": "digits", "type": "digits", "pattern": "digit_{}.npy"},
    )


class IceHockeyConstants(struct.PyTreeNode):
    # Static parameters. Marked pytree_node=False so JAX keeps them as static
    # metadata instead of tracing them.
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    # Rink interior in pixels (inside the boards).
    RINK_LEFT: int = struct.field(pytree_node=False, default=4)
    RINK_RIGHT: int = struct.field(pytree_node=False, default=155)
    RINK_TOP: int = struct.field(pytree_node=False, default=20)
    RINK_BOTTOM: int = struct.field(pytree_node=False, default=190)

    # Goals. Player defends the top, enemy the bottom.
    GOAL_X0: int = struct.field(pytree_node=False, default=60)
    GOAL_X1: int = struct.field(pytree_node=False, default=100)
    ENEMY_GOAL_Y: int = struct.field(pytree_node=False, default=187)
    PLAYER_GOAL_Y: int = struct.field(pytree_node=False, default=20)
    GOAL_HEIGHT: int = struct.field(pytree_node=False, default=7)

    # Sprite sizes, used for observation bounding boxes.
    PLAYER_W: int = struct.field(pytree_node=False, default=8)
    PLAYER_H: int = struct.field(pytree_node=False, default=12)
    PUCK_W: int = struct.field(pytree_node=False, default=4)
    PUCK_H: int = struct.field(pytree_node=False, default=3)

    PLAYER_SPEED: float = struct.field(pytree_node=False, default=1.5)
    # Speed multiplier while a character carries the puck (0.0 – 1.0).
    CARRY_SPEED_FACTOR: float = struct.field(pytree_node=False, default=0.6)

    # Offset from the goal lines defining zone where goalie/skater can't move
    ATTACKING_ZONE_OFFSET_Y: int = struct.field(pytree_node=False, default=50)

    # Phase-2 collision tunables for _characters_step
    MIN_SEPARATION: float = struct.field(pytree_node=False, default=8.0)
    MIN_VERTICAL_DISTANCE: float = struct.field(pytree_node=False, default=40.0)

    # 3 min * 60 s * 60 fps = 10800 raw frames.
    TIME_LIMIT: int = struct.field(pytree_node=False, default=10800)
    FACE_OFF_FRAMES: int = struct.field(pytree_node=False, default=40)

    # Face-off layout. [x, y] = [col, row]. Estimated from the ALE screen;
    # refine against captured frames once real sprites are in.
    FACEOFF_X: float = struct.field(pytree_node=False, default=78.0)
    FACEOFF_Y: float = struct.field(pytree_node=False, default=103.0)
    PLAYER_SKATER_X: float = struct.field(pytree_node=False, default=60.0)
    PLAYER_SKATER_Y: float = struct.field(pytree_node=False, default=80.0)
    PLAYER_GOALIE_X: float = struct.field(pytree_node=False, default=85.0)
    PLAYER_GOALIE_Y: float = struct.field(pytree_node=False, default=35.0)
    ENEMY_SKATER_X: float = struct.field(pytree_node=False, default=85.0)
    ENEMY_SKATER_Y: float = struct.field(pytree_node=False, default=110.0)
    ENEMY_GOALIE_X: float = struct.field(pytree_node=False, default=60.0)
    ENEMY_GOALIE_Y: float = struct.field(pytree_node=False, default=155.0)

    # Asset manifest lives in the constants so the modding framework can apply
    # asset_overrides before the renderer is constructed.
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
    is_tackled: chex.Array
    position: chex.Array        # float32 [x, y]
    orientation: chex.Array     # 0 = left, 1 = right
    has_puck: chex.Array
    shooting_cooldown: chex.Array


@struct.dataclass
class PuckState:
    position: chex.Array        # float32 [x, y]
    velocity: chex.Array        # float32 [vx, vy]
    direction: chex.Array       # shot angle slot, 0-31
    position_stick: chex.Array  # slot on the stick arc while carried, 0-31
    carry_offset: chex.Array    # float32 [dx, dy]: puck pos relative to carrier at pickup time


@struct.dataclass
class PlayerState:
    skater: CharacterState
    goalie: CharacterState
    active_character: chex.Array   # 0 = skater controlled, 1 = goalie controlled


@struct.dataclass
class EnemyState:
    skater: CharacterState
    goalie: CharacterState
    active_character: chex.Array


@struct.dataclass
class IceHockeyState:
    player_state: PlayerState
    enemy_state: EnemyState
    puck_state: PuckState
    counter: chex.Array
    game_state: GameState
    key: chex.Array  # PRNG key — advanced each time a random puck launch is needed


@struct.dataclass
class IceHockeyInfo:
    player_score: chex.Array
    enemy_score: chex.Array
    remaining_time: chex.Array


@struct.dataclass
class IceHockeyObservation:
    player_skater: ObjectObservation
    player_goalie: ObjectObservation
    enemy_skater: ObjectObservation
    enemy_goalie: ObjectObservation
    puck: ObjectObservation
    player_score: chex.Array
    enemy_score: chex.Array
    remaining_time: chex.Array
    active_player: chex.Array


class JaxIceHockey(JaxEnvironment):

    # IceHockey uses the full ALE action set, so the agent index maps straight
    # onto the ALE action integer.
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
            "player_skater": obj,
            "player_goalie": obj,
            "enemy_skater": obj,
            "enemy_goalie": obj,
            "puck": obj,
            "player_score": spaces.Box(0, 99, shape=(), dtype=jnp.int32),
            "enemy_score": spaces.Box(0, 99, shape=(), dtype=jnp.int32),
            "remaining_time": spaces.Box(0, self.consts.TIME_LIMIT, shape=(), dtype=jnp.int32),
            "active_player": spaces.Box(0, 1, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = None) -> Tuple:
        # Face-off: puck at centre, characters on start positions
        c = self.consts
        if key is None:
            key = jax.random.PRNGKey(0)

        def char(x, y):
            return CharacterState(
                is_tackled=jnp.array(False),
                position=jnp.array([x, y], dtype=jnp.float32),
                orientation=jnp.array(0, dtype=jnp.int32),
                has_puck=jnp.array(False),
                shooting_cooldown=jnp.array(0, dtype=jnp.int32),
            )

        state = IceHockeyState(
            player_state=PlayerState(
                skater=char(c.PLAYER_SKATER_X, c.PLAYER_SKATER_Y),
                goalie=char(c.PLAYER_GOALIE_X, c.PLAYER_GOALIE_Y),
                active_character=jnp.array(0, dtype=jnp.int32),
            ),
            enemy_state=EnemyState(
                skater=char(c.ENEMY_SKATER_X, c.ENEMY_SKATER_Y),
                goalie=char(c.ENEMY_GOALIE_X, c.ENEMY_GOALIE_Y),
                active_character=jnp.array(0, dtype=jnp.int32),
            ),
            puck_state=PuckState(
                position=jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32),
                velocity=jnp.array([0.0, 0.0], dtype=jnp.float32),
                direction=jnp.array(0, dtype=jnp.int32),
                position_stick=jnp.array(0, dtype=jnp.int32),
                carry_offset=jnp.zeros(2, dtype=jnp.float32),
            ),
            counter=jnp.array(0, dtype=jnp.int32),
            game_state=GameState(
                pause_counter=jnp.array(c.FACE_OFF_FRAMES, dtype=jnp.int32),
                player_score=jnp.array(0, dtype=jnp.int32),
                enemy_score=jnp.array(0, dtype=jnp.int32),
                remaining_time=jnp.array(c.TIME_LIMIT, dtype=jnp.int32),
                is_faceoff=jnp.array(True),
                goal_scored=jnp.array(False),
                is_finished=jnp.array(False),
            ),
            key=key,
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: IceHockeyState, action):
        previous_state = state

        new_player_state, new_enemy_state = self._characters_step(
            state.player_state,
            state.enemy_state,
            state.puck_state.position,
            player_action=action,
            enemy_action=jnp.array(Action.NOOP, dtype=jnp.int32),
        )
        state = state.replace(
            player_state=new_player_state,
            enemy_state=new_enemy_state,
            counter=state.counter + 1,
        )

        # Stick interactions — pickup / carry / shoot — symmetric for both teams.
        new_puck_state, new_player_state, new_enemy_state = self._stick_interaction_step(
            state.puck_state, state.player_state, state.enemy_state, state.game_state, action
        )
        state = state.replace(
            puck_state=new_puck_state,
            player_state=new_player_state,
            enemy_state=new_enemy_state,
        )

        # Tell _puck_step whether any character holds the puck so it never
        # overwrites a carry with a random face-off launch velocity.
        puck_is_carried = (
            state.player_state.skater.has_puck | state.player_state.goalie.has_puck |
            state.enemy_state.skater.has_puck  | state.enemy_state.goalie.has_puck
        )

        new_puck_state, new_game_state, new_key = self._puck_step(
            state.puck_state, state.game_state, state.key, puck_is_carried
        )

        # On a goal the puck teleports to face-off centre — release it from all characters.
        goal_this_frame = new_game_state.is_faceoff & ~state.game_state.is_faceoff

        def _clear_hp(char: CharacterState) -> CharacterState:
            return char.replace(
                has_puck=jnp.where(goal_this_frame, jnp.bool_(False), char.has_puck)
            )

        state = state.replace(
            puck_state=new_puck_state.replace(
                carry_offset=jnp.where(
                    goal_this_frame,
                    jnp.zeros(2, dtype=jnp.float32),
                    new_puck_state.carry_offset,
                )
            ),
            game_state=new_game_state,
            key=new_key,
            player_state=state.player_state.replace(
                skater=_clear_hp(state.player_state.skater),
                goalie=_clear_hp(state.player_state.goalie),
            ),
            enemy_state=state.enemy_state.replace(
                skater=_clear_hp(state.enemy_state.skater),
                goalie=_clear_hp(state.enemy_state.goalie),
            ),
        )

        obs = self._get_observation(state)
        reward = self._get_reward(previous_state, state)
        done = self._get_done(state)
        info = self._get_info(state)
        return obs, state, reward, done, info

    def _resolve_active_character(
        self,
        char1: CharacterState,
        char2: CharacterState,
        puck_position: chex.Array,
        current_active: chex.Array,
    ) -> chex.Array:
        """Resolve which of two characters is closest to the puck.

        Control in Ice Hockey goes to whichever of a team's two skaters is closest to
        the puck, so `⁠ _player_step ⁠` and `⁠ _enemy_step ⁠` both call this on their own
        pair.

        Args:
            char1: First character (corresponds to index 0).
            char2: Second character (corresponds to index 1).
            puck_position: `⁠ (x, y) ⁠` position of the puck.
            current_active: Index (0 or 1) returned on an exact distance tie, which
                avoids the result flickering between equidistant characters.

        Returns:
            An `⁠ int32 ⁠` scalar: 0 if `⁠ char1 ⁠` is closer, 1 if `⁠ char2 ⁠` is.
        """
        # Squared distance from each character to the puck; sqrt is unnecessary for ordering.
        dist1_sq = jnp.sum((char1.position - puck_position) ** 2)
        dist2_sq = jnp.sum((char2.position - puck_position) ** 2)

        # Closest character wins; on an exact tie keep whoever is currently active.
        closest = jnp.where(
            dist1_sq < dist2_sq,
            0,
            jnp.where(dist2_sq < dist1_sq, 1, current_active),
        )
        return closest.astype(jnp.int32)

    def _apply_action(
        self,
        character: CharacterState,
        action: chex.Array,
        bounds: chex.Array,
        velocity: chex.Array,
    ) -> CharacterState:
        """Apply one frame of joystick input movement to a single character.

        This is the per-character movement primitive shared by the human player and the
        computer opponent: each chooses an action through its own policy, but the action
        is applied identically here. Directions are absolute screen directions.

        Only the active skater of a team should receive a real action; the inactive
        teammate never moves from input (the caller passes NOOP or simply skips it).
        A tackled character ignores input entirely (it is frozen for the tackle period).

        Args:
            character: The character to move.
            action: The chosen Atari action integer.
            bounds: `⁠ (x_min, x_max, y_min, y_max) ⁠` provisional wall/zone clamp (see above).
            velocity: Per-axis movement distance for this frame (e.g. `⁠ PLAYER_SPEED ⁠`).

        Returns:
            The updated `⁠ CharacterState ⁠` (position + orientation; other fields kept).
        """
        up = jnp.any(jnp.array([
            action == Action.UP, action == Action.UPRIGHT, action == Action.UPLEFT,
            action == Action.UPFIRE, action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE,
        ]))
        down = jnp.any(jnp.array([
            action == Action.DOWN, action == Action.DOWNRIGHT, action == Action.DOWNLEFT,
            action == Action.DOWNFIRE, action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE,
        ]))
        left = jnp.any(jnp.array([
            action == Action.LEFT, action == Action.UPLEFT, action == Action.DOWNLEFT,
            action == Action.LEFTFIRE, action == Action.UPLEFTFIRE, action == Action.DOWNLEFTFIRE,
        ]))
        right = jnp.any(jnp.array([
            action == Action.RIGHT, action == Action.UPRIGHT, action == Action.DOWNRIGHT,
            action == Action.RIGHTFIRE, action == Action.UPRIGHTFIRE, action == Action.DOWNRIGHTFIRE,
        ]))

        # A tackled character is frozen: ignore all input movement this frame.
        movable = jnp.logical_not(character.is_tackled)
        dx = jnp.where(movable & right, velocity, jnp.where(movable & left, -velocity, 0.0))
        # Screen y grows downward, so DOWN increases y and UP decreases it.
        # NOTE: diagonals move by ⁠ velocity ⁠ on each axis, i.e. ~1.41x faster than a
        # straight move.    
        dy = jnp.where(movable & down, velocity, jnp.where(movable & up, -velocity, 0.0))

        new_x = jnp.clip(character.position[0] + dx, bounds[0], bounds[1])
        new_y = jnp.clip(character.position[1] + dy, bounds[2], bounds[3])
        new_position = jnp.array([new_x, new_y])

        # Orientation: 0 = facing left, 1 = facing right.
        # input keeps the current facing; a tackled character keeps it too (frozen).
        new_orientation = jnp.where(
            movable & right, 1, jnp.where(movable & left, 0, character.orientation)
        )

        return character.replace(position=new_position, orientation=new_orientation)

    # ------------------------------------------------------------------ #
    # Phase 1 — intended input movement (uniform over a team's two skaters)
    # ------------------------------------------------------------------ #
    def _apply_team_inputs(
        self,
        char1: CharacterState,
        char2: CharacterState,
        active: chex.Array,
        action: chex.Array,
        bounds1: chex.Array,
        bounds2: chex.Array,
        velocity: chex.Array,
    ) -> Tuple[CharacterState, CharacterState]:
        """Apply one team's chosen action as phase-1 intended movement.

        The reframed phase 1: instead of "only the active skater moves", every character
        is handled uniformly by the same `⁠ _apply_action ⁠` — the active skater receives
        the real action and the teammate receives `⁠ NOOP ⁠` (a zero intended delta). The
        active/passive split therefore collapses to "what action does this character get
        this frame", and the inactive teammate simply gets a no-op move.

        This is shared by both teams: the player's action comes from the agent, the
        computer's from its (future) policy, but routing + application are identical.

        Returns the two characters with their provisional (wall/zone-clamped) intended
        positions; the authoritative position is decided later by `⁠ _resolve_interactions ⁠`.
        """
        action1 = jnp.where(active == 0, action, Action.NOOP)
        action2 = jnp.where(active == 1, action, Action.NOOP)
        return (
            self._apply_action(char1, action1, bounds1, velocity),
            self._apply_action(char2, action2, bounds2, velocity),
        )

    # ------------------------------------------------------------------ #
    # Phase 2 — interaction resolution (pure geometry, single fixed-order pass)
    # ------------------------------------------------------------------ #
    def _separate_opponents(
        self,
        pos_a: chex.Array,
        pos_b: chex.Array,
        min_separation: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Resolve a cross-team (opponent) overlap: the confirmed "both shift".

        If the two characters are closer than `⁠ min_separation ⁠`, push BOTH apart along
        their centre-to-centre direction, each by half the penetration, so they end up
        exactly `⁠ min_separation ⁠` apart. The centre-to-centre normal is what makes the
        displacement diagonal. No-op when already separated.

        Pure geometry of the post-move positions only — it needs no pre-move state.

        Efficiency: the overlap test is done on squared distance (no sqrt). The push
        itself needs the true distance for the unit normal and the linear penetration,
        so a root is unavoidable here — but we fold sqrt + divide into a single
        reciprocal-sqrt: 1/dist == rsqrt(dist**2), giving
        `⁠ offset = 0.5 * delta * (min_separation / dist - 1) ⁠`.

        NOTE: `⁠ min_separation ⁠` (derived from body sizes) is not yet finalised; passed
        in. Whether a tackled/downed character is still pushable is also unverified.
        """
        delta = pos_a - pos_b
        dist_sq = jnp.sum(delta ** 2)
        overlapping = dist_sq < min_separation ** 2

        # True distance is only needed via its reciprocal: 1/dist == rsqrt(dist**2).
        coincident = dist_sq <= 0.0
        inv_dist = jnp.where(coincident, 0.0, jax.lax.rsqrt(dist_sq))
        # offset = 0.5 * (min_sep - dist) * (delta / dist) = 0.5 * delta * (min_sep/dist - 1)
        offset = 0.5 * delta * (min_separation * inv_dist - 1.0)
        # Coincident centres give no direction: default to separating along +x / -x.
        offset = jnp.where(coincident, jnp.array([min_separation * 0.5, 0.0]), offset)
        offset = jnp.where(overlapping, offset, jnp.array([0.0, 0.0]))
        return pos_a + offset, pos_b - offset

    def _enforce_min_vertical(
        self,
        active_pos: chex.Array,
        passive_pos: chex.Array,
        min_vertical_distance: chex.Array,
    ) -> chex.Array:
        """Resolve a same-team overlap: vertical-only push, mover holds / passive yields.

        Distinct from the opponent mechanic. The active (moving) skater keeps its
        position; the passive teammate is displaced along y only so the pair are at
        least `⁠ min_vertical_distance ⁠` apart (e.g. the goalie skating forward pushes
        his teammate straight up). The passive's x is untouched.

        Returns the passive teammate's new position. No-op when the gap is already large
        enough.
        """
        dy = passive_pos[1] - active_pos[1]
        too_close = jnp.abs(dy) < min_vertical_distance
        # Preserve which side the passive is on; if exactly level, default to below.
        side = jnp.where(dy != 0.0, jnp.sign(dy), 1.0)
        new_y = jnp.where(
            too_close, active_pos[1] + side * min_vertical_distance, passive_pos[1]
        )
        return jnp.array([passive_pos[0], new_y])

    def _clamp_to_bounds(self, pos: chex.Array, bounds: chex.Array) -> chex.Array:
        """Authoritative wall/zone clamp: `⁠ (x_min, x_max, y_min, y_max) ⁠`."""
        return jnp.array([
            jnp.clip(pos[0], bounds[0], bounds[1]),
            jnp.clip(pos[1], bounds[2], bounds[3]),
        ])

    def _resolve_interactions(
        self,
        player_skater: CharacterState,
        player_goalie: CharacterState,
        enemy_skater: CharacterState,
        enemy_goalie: CharacterState,
        player_active: chex.Array,
        enemy_active: chex.Array,
        min_separation: chex.Array,
        min_vertical_distance: chex.Array,
        bounds_p1: chex.Array,
        bounds_p2: chex.Array,
        bounds_e1: chex.Array,
        bounds_e2: chex.Array,
    ) -> Tuple[CharacterState, CharacterState, CharacterState, CharacterState]:
        """Phase 2: resolve all interactions on the four post-phase-1 characters.

        A single, fixed-order, fully branchless pass (deliberately NOT an iterative
        constraint solver — see the efficiency rationale: with 4 characters and shallow
        per-frame overlaps this is faithful and cheap):

          1. Opponent (cross-team) separations for the 4 player×enemy pairs.
          2. Same-team vertical pushes (active holds, passive yields).
          3. Authoritative wall/zone clamp on all four (covers pushes into walls and the
             passive teammate, which never passes through phase 1).

        The sub-step ORDER is a behavioural choice to verify against the game; in rare
        triple-contact frames a later step can nudge an earlier constraint sub-pixel,
        which the design guide tolerates.
        """
        p1, p2 = player_skater.position, player_goalie.position
        e1, e2 = enemy_skater.position, enemy_goalie.position

        # 1) Opponent collisions — both shift along centre-to-centre.
        p1, e1 = self._separate_opponents(p1, e1, min_separation)
        p1, e2 = self._separate_opponents(p1, e2, min_separation)
        p2, e1 = self._separate_opponents(p2, e1, min_separation)
        p2, e2 = self._separate_opponents(p2, e2, min_separation)

        # 2) Same-team vertical push — the active skater holds, the teammate yields.
        p2 = jnp.where(player_active == 0, self._enforce_min_vertical(p1, p2, min_vertical_distance), p2)
        p1 = jnp.where(player_active == 1, self._enforce_min_vertical(p2, p1, min_vertical_distance), p1)
        e2 = jnp.where(enemy_active == 0, self._enforce_min_vertical(e1, e2, min_vertical_distance), e2)
        e1 = jnp.where(enemy_active == 1, self._enforce_min_vertical(e2, e1, min_vertical_distance), e1)

        # 3) Authoritative clamp.
        p1 = self._clamp_to_bounds(p1, bounds_p1)
        p2 = self._clamp_to_bounds(p2, bounds_p2)
        e1 = self._clamp_to_bounds(e1, bounds_e1)
        e2 = self._clamp_to_bounds(e2, bounds_e2)

        return (
            player_skater.replace(position=p1),
            player_goalie.replace(position=p2),
            enemy_skater.replace(position=e1),
            enemy_goalie.replace(position=e2),
        )

    # ------------------------------------------------------------------ #
    # Orchestrator — runs phase 1 then phase 2 for all four characters
    # ------------------------------------------------------------------ #
    def _characters_step(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_position: chex.Array,
        player_action: chex.Array,
        enemy_action: chex.Array,
    ) -> Tuple[PlayerState, EnemyState]:
        """Advance all four characters one frame: active resolution -> phase 1 -> phase 2.

        This is the character-movement orchestrator shared by both teams. The movement
        speed, collision tunables, and zone bounds are read from `⁠ self.consts ⁠`;
        `⁠ enemy_action ⁠` will come from the (future) `⁠ _enemy_policy ⁠` and is `⁠ NOOP ⁠`
        until then. The lower-level geometry primitives still take these as parameters so
        they stay generic/unit-testable — only this orchestrator binds them to consts.

        Steps:
          1. Resolve each team's active (controlled) skater = closest to the puck.
          2. Phase 1: apply each team's action as intended input movement (uniformly via
             `⁠ _apply_team_inputs ⁠` — active skater gets the action, teammate gets NOOP).
          3. Phase 2: resolve interactions across all four post-move characters
             (opponent separation, teammate vertical push, authoritative clamp).

        Returns the updated `⁠ PlayerState ⁠` and `⁠ EnemyState ⁠` (positions/orientation
        from movement, plus the resolved `⁠ active_character ⁠` for each team).
        """
        c = self.consts
        base_velocity = jnp.float32(c.PLAYER_SPEED)
        carry_factor = jnp.float32(c.CARRY_SPEED_FACTOR)
        min_separation = jnp.float32(c.MIN_SEPARATION)
        min_vertical_distance = jnp.float32(c.MIN_VERTICAL_DISTANCE)
        # No per-team zones defined yet: all four skaters share the full-rink bounds.
        rink = jnp.array(
            [c.RINK_LEFT, c.RINK_RIGHT, c.RINK_TOP, c.RINK_BOTTOM], dtype=jnp.float32
        )
        # Player defends the top, enemy the bottom. Each goalie is barred from the
        # opponent's (attacking) zone; each skater is barred from its own crease.
        bounds_player_skater = bounds_enemy_goalie = jnp.array(
            [c.RINK_LEFT, c.RINK_RIGHT, c.RINK_TOP+c.ATTACKING_ZONE_OFFSET_Y, c.RINK_BOTTOM], dtype=jnp.float32
        )
        bounds_player_goalie = bounds_enemy_skater = jnp.array(
            [c.RINK_LEFT, c.RINK_RIGHT, c.RINK_TOP, c.RINK_BOTTOM-c.ATTACKING_ZONE_OFFSET_Y], dtype=jnp.float32
        )

        # 1) Active-skater resolution (per team, against the shared puck).
        player_active = self._resolve_active_character(
            player_state.skater, player_state.goalie,
            puck_position, player_state.active_character,
        )
        enemy_active = self._resolve_active_character(
            enemy_state.skater, enemy_state.goalie,
            puck_position, enemy_state.active_character,
        )
        
        # 2) Phase 1 — intended input movement.
        # Reduce speed when the active character is carrying the puck.
        player_active_hp = jnp.where(
            player_active == 0, player_state.skater.has_puck, player_state.goalie.has_puck
        )
        enemy_active_hp = jnp.where(
            enemy_active == 0, enemy_state.skater.has_puck, enemy_state.goalie.has_puck
        )
        player_velocity = jnp.where(player_active_hp, base_velocity * carry_factor, base_velocity)
        enemy_velocity  = jnp.where(enemy_active_hp,  base_velocity * carry_factor, base_velocity)

        p1, p2 = self._apply_team_inputs(
            player_state.skater, player_state.goalie,
            player_active, player_action, bounds_player_skater, bounds_player_goalie, player_velocity,
        )
        e1, e2 = self._apply_team_inputs(
            enemy_state.skater, enemy_state.goalie,
            enemy_active, enemy_action, bounds_enemy_skater, bounds_enemy_goalie, enemy_velocity,
        )

        # 3) Phase 2 — resolve interactions across all four post-move characters.
        p1, p2, e1, e2 = self._resolve_interactions(
            p1, p2, e1, e2,
            player_active, enemy_active,
            min_separation, min_vertical_distance,
            bounds_player_skater, bounds_player_goalie, bounds_enemy_skater, bounds_enemy_goalie,
        )

        new_player_state = player_state.replace(
            skater=p1, goalie=p2, active_character=player_active,
        )
        new_enemy_state = enemy_state.replace(
            skater=e1, goalie=e2, active_character=enemy_active,
        )
        return new_player_state, new_enemy_state

    # Fixed launch speed for the puck after every face-off (pixels per frame).
    _PUCK_LAUNCH_SPEED: float = 1.2
    # Speed of a player shot (pixels per frame).
    _PUCK_SHOT_SPEED: float = 2.0
    # Squared pixel radius within which the active player can pick up the puck.
    _STICK_REACH_SQ: float = 12.0 ** 2
    # Frames the active player must wait between shots.
    _SHOOT_COOLDOWN: int = 20

    def _team_puck_step(
        self,
        puck_state: PuckState,
        char1: CharacterState,
        char2: CharacterState,
        active: chex.Array,
        game_state: GameState,
        action: chex.Array,
        puck_globally_free: chex.Array,
    ) -> Tuple[PuckState, CharacterState, CharacterState]:
        """Generic puck interaction for one team: pickup, carry, shoot.

        Works identically for player and enemy — the caller supplies the action
        (human input for the player, a computed action for the enemy).

        puck_globally_free: True when neither team holds the puck.  The caller
        recomputes this between the player and enemy calls so the enemy cannot
        steal a puck that the player just picked up in the same frame.
        """
        c              = self.consts
        SHOT_SPEED     = jnp.float32(self._PUCK_SHOT_SPEED)
        STICK_REACH_SQ = jnp.float32(self._STICK_REACH_SQ)
        COOLDOWN       = jnp.int32(self._SHOOT_COOLDOWN)

        # Active character snapshot
        active_pos    = jnp.where(active == 0, char1.position,         char2.position)
        active_orient = jnp.where(active == 0, char1.orientation,       char2.orientation)
        active_hp     = jnp.where(active == 0, char1.has_puck,          char2.has_puck)
        active_cd     = jnp.where(active == 0, char1.shooting_cooldown, char2.shooting_cooldown)

        # FIRE action decoding
        has_fire  = jnp.any(jnp.array([
            action == Action.FIRE,
            action == Action.UPFIRE,        action == Action.DOWNFIRE,
            action == Action.RIGHTFIRE,     action == Action.LEFTFIRE,
            action == Action.UPRIGHTFIRE,   action == Action.UPLEFTFIRE,
            action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE,
        ]))
        fire_only = action == Action.FIRE
        has_right = jnp.any(jnp.array([action == Action.RIGHTFIRE, action == Action.UPRIGHTFIRE,  action == Action.DOWNRIGHTFIRE]))
        has_left  = jnp.any(jnp.array([action == Action.LEFTFIRE,  action == Action.UPLEFTFIRE,   action == Action.DOWNLEFTFIRE]))
        has_up    = jnp.any(jnp.array([action == Action.UPFIRE,    action == Action.UPRIGHTFIRE,  action == Action.UPLEFTFIRE]))
        has_down  = jnp.any(jnp.array([action == Action.DOWNFIRE,  action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE]))

        # Shot velocity — normalised to constant SHOT_SPEED (diagonal shots don't get √2× faster)
        raw_vx = jnp.where(
            fire_only,
            jnp.where(active_orient == 1, SHOT_SPEED, -SHOT_SPEED),
            jnp.where(has_right, SHOT_SPEED, jnp.where(has_left, -SHOT_SPEED, jnp.float32(0.0))),
        )
        raw_vy   = jnp.where(has_up, -SHOT_SPEED, jnp.where(has_down, SHOT_SPEED, jnp.float32(0.0)))
        speed_sq = raw_vx * 2 + raw_vy * 2
        norm     = SHOT_SPEED / jnp.where(speed_sq > 0.0, jnp.sqrt(speed_sq), jnp.float32(1.0))
        shot_vel = jnp.array([raw_vx * norm, raw_vy * norm], dtype=jnp.float32)

        # Team puck ownership
        char1_hp    = char1.has_puck
        char2_hp    = char2.has_puck
        any_team_hp = char1_hp | char2_hp

        # Proximity check (centre-to-centre distance)
        player_ctr = active_pos + jnp.array([c.PLAYER_W / 2.0, c.PLAYER_H / 2.0])
        puck_ctr   = puck_state.position + jnp.array([c.PUCK_W / 2.0, c.PUCK_H / 2.0])
        in_reach   = jnp.sum((player_ctr - puck_ctr) ** 2) <= STICK_REACH_SQ

        can_shoot  = active_hp & has_fire & (active_cd <= 0)
        # Pickup only when the puck is globally free (no team holds it)
        can_pickup = in_reach & ~has_fire & ~any_team_hp & puck_globally_free & ~game_state.is_faceoff

        # Carry: puck follows the carrier at the relative offset recorded at pickup.
        # This prevents snapping to a fixed side — the puck stays exactly where
        # it was touched and then moves with the carrier.
        carrier_pos = jnp.where(char1_hp, char1.position, char2.position)

        # Offset to record on this pickup frame (puck pos relative to active char pos)
        pickup_offset = puck_state.position - active_pos

        # carry_offset for next frame:
        #   pickup  → record offset now
        #   carry   → keep existing offset
        #   shoot   → clear (puck is free again)
        #   nothing → keep existing
        new_carry_offset = jnp.where(
            can_shoot,   jnp.zeros(2, dtype=jnp.float32),
            jnp.where(can_pickup, pickup_offset,
                                  puck_state.carry_offset),
        )

        # Current carry position: carrier's current pos + the offset stored last frame
        carry_pos = carrier_pos + puck_state.carry_offset

        # New puck position:
        #   shoot   → from current carry pos (where puck actually is)
        #   carry   → follow carrier at stored offset
        #   pickup  → stay exactly where puck is now (no snap)
        #   free    → unchanged
        new_puck_pos = jnp.where(
            can_shoot,   carry_pos,
            jnp.where(any_team_hp, carry_pos,
            jnp.where(can_pickup,  puck_state.position,
                                   puck_state.position)),
        )
        # New puck velocity
        new_puck_vel = jnp.where(
            can_shoot, shot_vel,
            jnp.where(any_team_hp | can_pickup, jnp.zeros(2, dtype=jnp.float32), puck_state.velocity),
        )
        new_puck_state = puck_state.replace(
            position=new_puck_pos,
            velocity=new_puck_vel,
            carry_offset=new_carry_offset,
        )

        # has_puck updates
        new_char1_hp = jnp.where(
            active == 0,
            jnp.where(can_shoot, jnp.bool_(False), jnp.where(can_pickup, jnp.bool_(True), char1.has_puck)),
            char1.has_puck,
        )
        new_char2_hp = jnp.where(
            active == 1,
            jnp.where(can_shoot, jnp.bool_(False), jnp.where(can_pickup, jnp.bool_(True), char2.has_puck)),
            char2.has_puck,
        )

        # Shooting cooldown (decrement every frame; reset on shot)
        new_cd1 = jnp.maximum(0, char1.shooting_cooldown - 1)
        new_cd2 = jnp.maximum(0, char2.shooting_cooldown - 1)
        new_cd1 = jnp.where(can_shoot & (active == 0), COOLDOWN, new_cd1)
        new_cd2 = jnp.where(can_shoot & (active == 1), COOLDOWN, new_cd2)

        new_char1 = char1.replace(has_puck=new_char1_hp, shooting_cooldown=new_cd1)
        new_char2 = char2.replace(has_puck=new_char2_hp, shooting_cooldown=new_cd2)
        # Return can_shoot so _stick_interaction_step can block the opposing team
        # from stealing the puck in the same frame it was released by a shot.
        return new_puck_state, new_char1, new_char2, can_shoot

    def _stick_interaction_step(
        self,
        puck_state: PuckState,
        player_state: PlayerState,
        enemy_state: EnemyState,
        game_state: GameState,
        action: chex.Array,
    ) -> Tuple[PuckState, PlayerState, EnemyState]:
        """Pickup / carry / shoot for both teams, in priority order (player first).

        Enemy shooting uses a placeholder AI: the active enemy character shoots
        straight toward the player goal (UPFIRE) whenever it holds the puck.
        This is replaced when the real enemy policy is implemented.
        """
        # Player can pickup whenever the PLAYER team doesn't already hold the puck —
        # this allows stealing the puck from the enemy.
        puck_free_for_player = ~(
            player_state.skater.has_puck | player_state.goalie.has_puck
        )

        # --- Player interaction (higher priority) ---
        player_had_puck = player_state.skater.has_puck | player_state.goalie.has_puck
        puck_state, new_p_sk, new_p_go, player_just_shot = self._team_puck_step(
            puck_state,
            player_state.skater, player_state.goalie, player_state.active_character,
            game_state, action, puck_free_for_player,
        )
        new_player_state = player_state.replace(skater=new_p_sk, goalie=new_p_go)

        # If the player just picked up the puck while the enemy was holding it,
        # immediately release it from the enemy so the enemy doesn't keep carrying.
        player_just_picked_up = (new_p_sk.has_puck | new_p_go.has_puck) & ~player_had_puck
        enemy_had_puck = enemy_state.skater.has_puck | enemy_state.goalie.has_puck
        steal = player_just_picked_up & enemy_had_puck

        cleared_e_sk = enemy_state.skater.replace(
            has_puck=jnp.where(steal, jnp.bool_(False), enemy_state.skater.has_puck)
        )
        cleared_e_go = enemy_state.goalie.replace(
            has_puck=jnp.where(steal, jnp.bool_(False), enemy_state.goalie.has_puck)
        )

        # Enemy can only pickup when nobody holds the puck (no stealing from player).
        puck_free_for_enemy = ~(
            new_p_sk.has_puck | new_p_go.has_puck |
            cleared_e_sk.has_puck | cleared_e_go.has_puck
        ) & ~player_just_shot

        # Enemy holds the puck but never shoots — always NOOP.
        enemy_action = jnp.int32(Action.NOOP)

        # --- Enemy interaction (uses cleared has_puck if player stole) ---
        puck_state, new_e_sk, new_e_go, _ = self._team_puck_step(
            puck_state,
            cleared_e_sk, cleared_e_go, enemy_state.active_character,
            game_state, enemy_action, puck_free_for_enemy,
        )
        new_enemy_state = enemy_state.replace(skater=new_e_sk, goalie=new_e_go)

        return puck_state, new_player_state, new_enemy_state

    def _puck_step(
        self, puck_state: PuckState, game_state: GameState, key: chex.Array,
        puck_is_carried: chex.Array,
    ) -> Tuple[PuckState, GameState, chex.Array]:
        """Advance the puck one frame: face-off countdown, random launch, movement, wall/goal."""
        c = self.consts

        # --- Random velocity for next face-off launch ---
        # Always generated (pure JAX), conditionally applied below.
        key, sk1, sk2 = jax.random.split(key, 3)
        # Angle in [π/4, 3π/4] (downward diagonal) or [5π/4, 7π/4] (upward diagonal).
        # This guarantees a meaningful vertical component and avoids near-horizontal shots.
        theta = jax.random.uniform(sk1, minval=0.0, maxval=jnp.pi / 2.0)
        base_angle = theta + jnp.pi / 4.0           # [π/4, 3π/4]
        going_up   = jax.random.uniform(sk2) > 0.5
        angle      = jnp.where(going_up, base_angle + jnp.pi, base_angle)
        random_vel = jnp.array(
            [self._PUCK_LAUNCH_SPEED * jnp.cos(angle),
             self._PUCK_LAUNCH_SPEED * jnp.sin(angle)],
            dtype=jnp.float32,
        )

        # --- Face-off countdown ---
        new_pause_counter = jnp.where(
            game_state.is_faceoff,
            game_state.pause_counter - 1,
            game_state.pause_counter,
        )
        counter_expired  = new_pause_counter <= 0
        face_off_ending  = game_state.is_faceoff & counter_expired

        is_faceoff  = jnp.where(face_off_ending, jnp.bool_(False), game_state.is_faceoff)
        goal_scored = jnp.where(face_off_ending, jnp.bool_(False), game_state.goal_scored)

        # Inject random velocity when face-off ends — but NOT if someone is carrying the puck
        # (a player who grabbed the puck during the countdown keeps possession).
        active_vel = jnp.where(
            face_off_ending & ~puck_is_carried, random_vel, puck_state.velocity
        )

        # --- Puck movement (frozen to zero while still in face-off) ---
        vel     = jnp.where(is_faceoff, jnp.zeros(2, dtype=jnp.float32), active_vel)
        new_pos = puck_state.position + vel

        # Puck centre-x for goal-opening detection.
        center_x  = new_pos[0] + c.PUCK_W / 2.0
        in_goal_x = (center_x >= c.GOAL_X0) & (center_x <= c.GOAL_X1)

        # Wall-hit detection (top-left corner + size).
        hit_left   = new_pos[0] < c.RINK_LEFT
        hit_right  = new_pos[0] + c.PUCK_W > c.RINK_RIGHT
        hit_top    = new_pos[1] < c.RINK_TOP
        hit_bottom = new_pos[1] + c.PUCK_H > c.RINK_BOTTOM

        # Distinguish goal mouth vs. plain board at top/bottom.
        top_goal    = hit_top    & in_goal_x   # puck into player goal → enemy scores
        bottom_goal = hit_bottom & in_goal_x   # puck into enemy goal  → player scores
        top_wall    = hit_top    & ~in_goal_x
        bottom_wall = hit_bottom & ~in_goal_x

        # Reflect velocity components on board contact.
        new_vx = jnp.where(hit_left | hit_right,   -vel[0], vel[0])
        new_vy = jnp.where(top_wall  | bottom_wall, -vel[1], vel[1])

        # Clamp position so puck never exits rink bounds.
        clamped_x = jnp.clip(new_pos[0], c.RINK_LEFT, c.RINK_RIGHT  - c.PUCK_W)
        clamped_y = jnp.clip(new_pos[1], c.RINK_TOP,  c.RINK_BOTTOM - c.PUCK_H)

        # --- Goal: reset puck to face-off spot, start new countdown ---
        any_goal    = top_goal | bottom_goal
        faceoff_pos = jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32)

        final_pos = jnp.where(any_goal, faceoff_pos,
                               jnp.array([clamped_x, clamped_y], dtype=jnp.float32))
        final_vel = jnp.where(any_goal, jnp.zeros(2, dtype=jnp.float32),
                               jnp.array([new_vx, new_vy], dtype=jnp.float32))

        new_player_score  = game_state.player_score + bottom_goal.astype(jnp.int32)
        new_enemy_score   = game_state.enemy_score  + top_goal.astype(jnp.int32)
        new_is_faceoff    = is_faceoff  | any_goal
        new_goal_scored   = goal_scored | any_goal
        new_pause_counter = jnp.where(any_goal, jnp.int32(c.FACE_OFF_FRAMES), new_pause_counter)

        new_game_state = game_state.replace(
            player_score=new_player_score,
            enemy_score=new_enemy_score,
            pause_counter=new_pause_counter,
            is_faceoff=new_is_faceoff,
            goal_scored=new_goal_scored,
        )
        new_puck_state = puck_state.replace(position=final_pos, velocity=final_vel)
        return new_puck_state, new_game_state, key

    def render(self, state: IceHockeyState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: IceHockeyState) -> IceHockeyObservation:
        c = self.consts

        def obj(pos, w, h):
            return ObjectObservation.create(
                x=pos[0].astype(jnp.int32),
                y=pos[1].astype(jnp.int32),
                width=jnp.array(w, dtype=jnp.int32),
                height=jnp.array(h, dtype=jnp.int32),
            )

        return IceHockeyObservation(
            player_skater=obj(state.player_state.skater.position, c.PLAYER_W, c.PLAYER_H),
            player_goalie=obj(state.player_state.goalie.position, c.PLAYER_W, c.PLAYER_H),
            enemy_skater=obj(state.enemy_state.skater.position, c.PLAYER_W, c.PLAYER_H),
            enemy_goalie=obj(state.enemy_state.goalie.position, c.PLAYER_W, c.PLAYER_H),
            puck=obj(state.puck_state.position, c.PUCK_W, c.PUCK_H),
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
            flat(obs.player_skater), flat(obs.player_goalie),
            flat(obs.enemy_skater), flat(obs.enemy_goalie),
            flat(obs.puck),
            jnp.array([obs.player_score, obs.enemy_score,
                       obs.remaining_time, obs.active_player], dtype=jnp.float32),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: IceHockeyState) -> IceHockeyInfo:
        return IceHockeyInfo(
            player_score=state.game_state.player_score,
            enemy_score=state.game_state.enemy_score,
            remaining_time=state.game_state.remaining_time,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: IceHockeyState, state: IceHockeyState) -> chex.Array:
        # Reward is the change in goal difference: +1 scored, -1 conceded.
        prev_diff = previous_state.game_state.player_score - previous_state.game_state.enemy_score
        diff = state.game_state.player_score - state.game_state.enemy_score
        return (diff - prev_diff).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: IceHockeyState) -> chex.Array:
        return state.game_state.is_finished


class IceHockeyRenderer(JAXGameRenderer):
    # Palette-based renderer. The rink (boards, lines, goals, score bars) is
    # baked into the background, so render() only stamps the moving objects.

    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        self.consts = consts or IceHockeyConstants()
        super().__init__(self.consts)

        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160), channels=3, downscale=None,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Branch-local sprite folder for now; move to the shared sprite dir later.
        self.sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "icehockey")

        final_asset_config = list(self.consts.ASSET_CONFIG)
        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND,
         self.COLOR_TO_ID, self.FLIP_OFFSETS) = self.jr.load_and_setup_assets(
            final_asset_config, self.sprite_path
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: IceHockeyState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        pm = self.SHAPE_MASKS["player"]
        em = self.SHAPE_MASKS["enemy"]
        puck_m = self.SHAPE_MASKS["puck"]

        def col(pos):
            return jnp.round(pos[0]).astype(jnp.int32)

        def row(pos):
            return jnp.round(pos[1]).astype(jnp.int32)

        p1 = state.player_state.skater.position
        p2 = state.player_state.goalie.position
        e1 = state.enemy_state.skater.position
        e2 = state.enemy_state.goalie.position
        pp = state.puck_state.position

        # render_at_clipped because skaters can reach the board pixels at the
        # edge; render_at would slice out of bounds there.
        raster = self.jr.render_at_clipped(raster, col(p2), row(p2), pm)
        raster = self.jr.render_at_clipped(raster, col(e2), row(e2), em)
        raster = self.jr.render_at_clipped(raster, col(p1), row(p1), pm)
        raster = self.jr.render_at_clipped(raster, col(e1), row(e1), em)
        raster = self.jr.render_at_clipped(raster, col(pp), row(pp), puck_m)

        dm = self.SHAPE_MASKS["digits"]

        def draw_score(r, value, x_single, x_double):
            digits = self.jr.int_to_digits(value, max_digits=2)
            is_single = value < 10
            start = jax.lax.select(is_single, jnp.int32(1), jnp.int32(0))
            count = jax.lax.select(is_single, jnp.int32(1), jnp.int32(2))
            x = jax.lax.select(is_single, jnp.int32(x_single), jnp.int32(x_double))
            return self.jr.render_label_selective(
                r, x, 3, digits, dm, start, count, spacing=7, max_digits_to_render=2
            )

        raster = draw_score(raster, state.game_state.enemy_score, 43, 33)
        raster = draw_score(raster, state.game_state.player_score, 113, 103)

        return self.jr.render_from_palette(raster, self.PALETTE)