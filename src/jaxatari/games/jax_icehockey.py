import os
from functools import partial
from typing import Tuple, Optional, Union
from jax import lax
import jax
import jax.numpy as jnp
import chex
from flax import struct
import jaxatari.rendering.jax_rendering_utils as render_utils
import jaxatari.spaces as spaces
from jaxatari.environment import (
    JaxEnvironment,
    JAXAtariAction as Action,
    ObjectObservation,
)
from jaxatari.renderers import JAXGameRenderer


def _get_default_asset_config() -> tuple:
    """Manifest of the .npy sprites the renderer loads from sprites/icehockey/.

    Run scripts/make_icehockey_sprites.py once to create the placeholder files.
    """
    return (
        {"name": "background", "type": "background", "file": "background.npy"},
        {
            "name": "player_walking_left",
            "type": "group",
            "files": [
                "player_walking_left_0.npy",
                "player_walking_left_1.npy",
                "player_walking_left_2.npy",
                "player_walking_left_3.npy",
            ],
        },
        {
            "name": "player_walking_right",
            "type": "group",
            "files": [
                "player_walking_right_0.npy",
                "player_walking_right_1.npy",
                "player_walking_right_2.npy",
                "player_walking_right_3.npy",
            ],
        },
        {
            "name": "enemy_walking_left",
            "type": "group",
            "files": [
                "enemy_walking_left_0.npy",
                "enemy_walking_left_1.npy",
                "enemy_walking_left_2.npy",
                "enemy_walking_left_3.npy",
            ],
        },
        {
            "name": "enemy_walking_right",
            "type": "group",
            "files": [
                "enemy_walking_right_0.npy",
                "enemy_walking_right_1.npy",
                "enemy_walking_right_2.npy",
                "enemy_walking_right_3.npy",
            ],
        },
        {"name": "player_idle_left", "type": "single", "file": "player_idle_left.npy"},
        {
            "name": "player_idle_right",
            "type": "single",
            "file": "player_idle_right.npy",
        },
        {"name": "enemy_idle_left", "type": "single", "file": "enemy_idle_left.npy"},
        {"name": "enemy_idle_right", "type": "single", "file": "enemy_idle_right.npy"},
        {
            "name": "player_active_standing_left",
            "type": "single",
            "file": "player_active_standing_left.npy",
        },
        {
            "name": "player_active_standing_right",
            "type": "single",
            "file": "player_active_standing_right.npy",
        },
        {
            "name": "enemy_active_standing_left",
            "type": "single",
            "file": "enemy_active_standing_left.npy",
        },
        {
            "name": "enemy_active_standing_right",
            "type": "single",
            "file": "enemy_active_standing_right.npy",
        },
        {"name": "player_faceoff", "type": "single", "file": "player_faceoff.npy"},
        {"name": "enemy_faceoff", "type": "single", "file": "enemy_faceoff.npy"},
        {"name": "player_tackled", "type": "single", "file": "player_tackled.npy"},
        {"name": "enemy_tackled", "type": "single", "file": "enemy_tackled.npy"},
        {
            "name": "player_shooting_right",
            "type": "group",
            "files": [
                "player_shooting_right_0.npy",
                "player_shooting_right_1.npy",
                "player_shooting_right_2.npy",
                "player_shooting_right_3.npy",
                "player_shooting_right_4.npy",
            ],
        },
        {
            "name": "player_shooting_left",
            "type": "group",
            "files": [
                "player_shooting_left_0.npy",
                "player_shooting_left_1.npy",
                "player_shooting_left_2.npy",
                "player_shooting_left_3.npy",
                "player_shooting_left_4.npy",
            ],
        },
        {
            "name": "enemy_shooting_right",
            "type": "group",
            "files": [
                "enemy_shooting_right_0.npy",
                "enemy_shooting_right_1.npy",
                "enemy_shooting_right_2.npy",
                "enemy_shooting_right_3.npy",
                "enemy_shooting_right_4.npy",
            ],
        },
        {
            "name": "enemy_shooting_left",
            "type": "group",
            "files": [
                "enemy_shooting_left_0.npy",
                "enemy_shooting_left_1.npy",
                "enemy_shooting_left_2.npy",
                "enemy_shooting_left_3.npy",
                "enemy_shooting_left_4.npy",
            ],
        },
        {"name": "puck", "type": "single", "file": "puck.npy"},
        {
            "name": "digits",
            "type": "digits",
            "pattern": "digit_{}.npy",
            "recolorings": {"gold": (236, 200, 96)},
        },
    )


class IceHockeyConstants(struct.PyTreeNode):
    DEBUG_RENDER: bool = struct.field(pytree_node=False, default=False)

    # Static parameters. Marked pytree_node=False so JAX keeps them as static
    # metadata instead of tracing them.
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    # Rink interior in pixels (inside the boards)
    RINK_LEFT: int = struct.field(pytree_node=False, default=32)
    RINK_RIGHT: int = struct.field(pytree_node=False, default=128)
    RINK_TOP: int = struct.field(pytree_node=False, default=42)
    RINK_BOTTOM: int = struct.field(pytree_node=False, default=187)

    # Goals. Player defends the top, enemy the bottom.
    GOAL_X0: int = struct.field(pytree_node=False, default=64)
    GOAL_X1: int = struct.field(pytree_node=False, default=96)
    ENEMY_GOAL_Y: int = struct.field(pytree_node=False, default=186)  # Bottom goal line
    PLAYER_GOAL_Y: int = struct.field(pytree_node=False, default=42)  # top goal line
    GOAL_HEIGHT_TOP: int = struct.field(pytree_node=False, default=4)
    GOAL_HEIGHT_BOTTOM: int = struct.field(pytree_node=False, default=3)

    # Sprite sizes, used for observation bounding boxes
    PLAYER_W: int = struct.field(pytree_node=False, default=26)
    PLAYER_H: int = struct.field(pytree_node=False, default=26)
    PUCK_W: int = struct.field(pytree_node=False, default=2)
    PUCK_H: int = struct.field(pytree_node=False, default=2)

    # Character movement per update tick. Horizontal coordinates advance by one
    # pixel, while vertical coordinates advance by two.
    PLAYER_SPEED: float = struct.field(pytree_node=False, default=1.0)
    PLAYER_VERTICAL_SPEED: float = struct.field(pytree_node=False, default=2.0)
    # Characters and a free puck advance on separate fixed cadences.
    # The game clock still advances every frame.
    CHARACTER_UPDATE_CADENCE: int = struct.field(pytree_node=False, default=4)
    FREE_PUCK_UPDATE_CADENCE: int = struct.field(pytree_node=False, default=2)

    # Skater leg walk-cycle: number of frames in the loop and how many game
    # frames each phase is shown for. The cycle advances only while a skater has
    # directional input.
    ANIM_CADENCE: int = struct.field(pytree_node=False, default=4)

    # A stick swing remains active for six character-update ticks.
    SHOOT_ANIM_FRAMES: int = struct.field(pytree_node=False, default=6)

    # Offset from the goal lines defining zone where goalie/skater can't move.
    # A skater is kept out of its own defensive zone (this deep); a goalie is kept
    # out of the opponent's far zone (this deep).
    # Skater's own-goal margin, measured at the PICKUP BOX: the forward's stick is
    # kept ~35% of the rink from its own goal (manual: "prevented from moving too
    # close to his own goal"). Box-referenced so the active-zone test below lines up
    # with where the stick can actually collect the puck.
    ATTACKING_ZONE_OFFSET_Y: int = struct.field(pytree_node=False, default=47)
    # Goalie's forward limit
    GOALIE_FORWARD_OFFSET: int = struct.field(pytree_node=False, default=50)
    # Character contact box and fixed separation impulse.
    CONTACT_X_MIN: float = struct.field(pytree_node=False, default=-8.0)
    CONTACT_X_MAX: float = struct.field(pytree_node=False, default=7.0)
    CONTACT_Y_MIN: float = struct.field(pytree_node=False, default=-3.0)
    CONTACT_Y_MAX: float = struct.field(pytree_node=False, default=5.0)
    CONTACT_PUSH_X: float = struct.field(pytree_node=False, default=1.0)
    CONTACT_PUSH_Y: float = struct.field(pytree_node=False, default=2.0)
    # Minimum vertical gap between two same-team players (and between the two goalies
    # of opposing teams), ~30% of the rink, measured top-left corner to top-left
    # corner. The passive player is pushed vertically to keep this gap.
    MIN_VERTICAL_DISTANCE: float = struct.field(pytree_node=False, default=44.0)

    STICK_VISIBLE_CADENCE: int = struct.field(pytree_node=False, default=4)

    # Body-check timing and goalie protection. Knockdown timers are counted in
    # eighth-frame ticks;
    TACKLE_TIMER_CADENCE: int = struct.field(pytree_node=False, default=8)
    PLAYER_GOALIE_PROTECTED_GRID_Y: int = struct.field(pytree_node=False, default=129)
    ENEMY_GOALIE_PROTECTED_GRID_Y: int = struct.field(pytree_node=False, default=19)

    # 3 min * 60 s * 60 fps = 10800 frames, in frames of active play
    TIME_LIMIT: int = struct.field(pytree_node=False, default=10800)
    # freeze duration of the face-off countdown (after reset and after goals)
    FACE_OFF_FRAMES: int = struct.field(pytree_node=False, default=63)
    # freeze after a goal before everyone snaps back to the face-off spots
    GOAL_PAUSE_FRAMES: int = struct.field(pytree_node=False, default=64)

    # Integer coordinate grid used by the controller and puck-contact rules.
    # These values are fixed game mechanics rather than tunable AI parameters.
    PUCK_GRID_X_OFFSET: int = struct.field(pytree_node=False, default=10)
    CHARACTER_GRID_X_OFFSET: int = struct.field(pytree_node=False, default=17)
    PUCK_GRID_Y_ORIGIN: int = struct.field(pytree_node=False, default=188)
    CHARACTER_GRID_Y_ORIGIN: int = struct.field(pytree_node=False, default=166)
    ACTIVE_TOP_THRESHOLD: int = struct.field(pytree_node=False, default=95)
    ACTIVE_BOTTOM_THRESHOLD: int = struct.field(pytree_node=False, default=55)
    GOAL_CENTER_GRID_X: int = struct.field(pytree_node=False, default=89)
    PLAYER_GOAL_GRID_Y: int = struct.field(pytree_node=False, default=146)
    REGRAB_BLOCK_FRAMES: int = struct.field(pytree_node=False, default=48)

    # Face-off layout. [x, y] = [col, row].
    FACEOFF_X: float = struct.field(pytree_node=False, default=79.0)
    FACEOFF_Y: float = struct.field(pytree_node=False, default=114.0)
    PLAYER_SKATER_X: float = struct.field(pytree_node=False, default=52.0)
    PLAYER_SKATER_Y: float = struct.field(pytree_node=False, default=81.0)
    PLAYER_GOALIE_X: float = struct.field(pytree_node=False, default=61.0)
    PLAYER_GOALIE_Y: float = struct.field(pytree_node=False, default=33.0)
    ENEMY_SKATER_X: float = struct.field(pytree_node=False, default=83.0)
    ENEMY_SKATER_Y: float = struct.field(pytree_node=False, default=103.0)
    ENEMY_GOALIE_X: float = struct.field(pytree_node=False, default=63.0)
    ENEMY_GOALIE_Y: float = struct.field(pytree_node=False, default=152.0)

    #  Puck wandert über 32 Slots am Stock hin und her.
    STICK_SLOTS: int = struct.field(pytree_node=False, default=32)
    STICK_CADENCE: int = struct.field(pytree_node=False, default=1)  # Frames pro Slot
    # Stock-Endpunkte relativ zum Box-MITTELPUNKT, autoriert für Blick nach rechts.
    # # Slot 0 = an den Händen (Basis), Slot 31 = Schlägerspitze.
    STICK_MIN_DX: float = struct.field(pytree_node=False, default=5.0)  # Slot 0
    STICK_MAX_DX: float = struct.field(pytree_node=False, default=12.0)  # Slot 31
    # feste Höhe — bewegt sich NICHT mit.
    STICK_DY: float = struct.field(pytree_node=False, default=8.0)

    ENEMY_STICK_DY: float = struct.field(pytree_node=False, default=8.0)

    PUCK_MAX_SPEED: float = struct.field(pytree_node=False, default=2.0)
    # Unit [x, y] direction used to release the puck when the face-off ends.
    # Keep this direction valid for a normal puck shot; it is scaled by
    # PUCK_MAX_SPEED when applied.
    FACE_OFF_PUCK_DIRECTION: Tuple[float, float] = struct.field(
        pytree_node=False, default=(-0.692, 0.692)
    )
    FACE_OFF_PUCK_MAX_SPEED: float = struct.field(pytree_node=False, default=1)

    # Pick up region around the end of the stick in which the puck can be picked up
    PICKUP_BOX_W: float = struct.field(pytree_node=False, default=16.0)
    PICKUP_BOX_H: float = struct.field(pytree_node=False, default=4.0)
    PLAYER_PICKUP_BOX_OFFSET_Y: float = struct.field(pytree_node=False, default=19.0)
    ENEMY_PICKUP_BOX_OFFSET_Y: float = struct.field(pytree_node=False, default=19.0)
    PICKUP_BOX_OFFSET_X_LEFT: float = struct.field(pytree_node=False, default=0.0)
    PICKUP_BOX_OFFSET_X_RIGHT: float = struct.field(pytree_node=False, default=9.0)

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
    position: chex.Array  # float32 [x, y]
    velocity: chex.Array  # float32 [vx, vy]; unused by the base game, available to mods
    orientation: chex.Array  # 0 = left, 1 = right
    has_puck: chex.Array
    shooting_cooldown: chex.Array
    walk_counter: chex.Array  # leg walk-cycle phase counter (advances while moving)
    tackle_timer: (
        chex.Array
    )  # eighth-frame ticks left while down; is_tackled == (tackle_timer > 0)
    times_tackled: (
        chex.Array
    )  # knockdowns suffered this match; unused by the base game, read by mods


@struct.dataclass
class PuckState:
    position: chex.Array  # float32 [x, y]
    velocity: chex.Array  # float32 [vx, vy]
    position_stick: chex.Array  # slot on the stick arc while carried, 0-31
    # -1 = no blocked shooter; otherwise 0..3 identifies the last shooter.
    pickup_blocker: chex.Array
    pickup_blocker_timer: chex.Array  # active-play ticks left in the re-grab lockout
    carry_timer: chex.Array
    holder: chex.Array


@struct.dataclass
class PlayerState:
    skater: CharacterState
    goalie: CharacterState
    active_character: chex.Array  # 0 = skater controlled, 1 = goalie controlled


@struct.dataclass
class EnemyState:
    skater: CharacterState
    goalie: CharacterState
    active_character: chex.Array


@struct.dataclass
class EnemyControllerState:
    """Persistent state for the computer-controlled team."""

    target_grid: chex.Array  # int32 [x, y] in the controller coordinate grid
    move_dx: chex.Array  # int32 in {-1, 0, 1}
    move_dy: chex.Array  # int32 in {-1, 0, 1}
    fire: chex.Array  # bool
    random_byte: chex.Array  # int32, low 8 bits used


@struct.dataclass
class IceHockeyState:
    player_state: PlayerState
    enemy_state: EnemyState
    enemy_controller: EnemyControllerState
    puck_state: PuckState
    counter: chex.Array
    game_state: GameState


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
    ACTION_SET = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        consts = consts or IceHockeyConstants()
        super().__init__(consts)
        self.renderer = IceHockeyRenderer(self.consts)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        obj = spaces.get_object_space(
            n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)
        )
        return spaces.Dict(
            {
                "player_skater": obj,
                "player_goalie": obj,
                "enemy_skater": obj,
                "enemy_goalie": obj,
                "puck": obj,
                "player_score": spaces.Box(0, 99, shape=(), dtype=jnp.int32),
                "enemy_score": spaces.Box(0, 99, shape=(), dtype=jnp.int32),
                "remaining_time": spaces.Box(
                    0, self.consts.TIME_LIMIT, shape=(), dtype=jnp.int32
                ),
                "active_player": spaces.Box(0, 1, shape=(), dtype=jnp.int32),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    def _faceoff_positions(self) -> Tuple[PlayerState, EnemyState, PuckState]:
        """Fresh character/puck states on the face-off spots (reset + after goals)."""
        c = self.consts

        def char(x, y, orientation):
            return CharacterState(
                is_tackled=jnp.array(False),
                position=jnp.array([x, y], dtype=jnp.float32),
                velocity=jnp.zeros(2, dtype=jnp.float32),
                orientation=jnp.array(orientation, dtype=jnp.int32),
                has_puck=jnp.array(False),
                shooting_cooldown=jnp.array(0, dtype=jnp.int32),
                walk_counter=jnp.array(0, dtype=jnp.int32),
                tackle_timer=jnp.array(0, dtype=jnp.int32),
                times_tackled=jnp.array(0, dtype=jnp.int32),
            )

        player_state = PlayerState(
            skater=char(c.PLAYER_SKATER_X, c.PLAYER_SKATER_Y, orientation=1),
            goalie=char(c.PLAYER_GOALIE_X, c.PLAYER_GOALIE_Y, orientation=1),
            active_character=jnp.array(0, dtype=jnp.int32),
        )
        enemy_state = EnemyState(
            skater=char(c.ENEMY_SKATER_X, c.ENEMY_SKATER_Y, orientation=0),
            goalie=char(c.ENEMY_GOALIE_X, c.ENEMY_GOALIE_Y, orientation=0),
            active_character=jnp.array(0, dtype=jnp.int32),
        )
        puck_state = PuckState(
            position=jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32),
            velocity=jnp.array([0, 0], dtype=jnp.float32),
            position_stick=jnp.array(0, dtype=jnp.int32),
            pickup_blocker=jnp.array(-1, dtype=jnp.int32),
            pickup_blocker_timer=jnp.array(0, dtype=jnp.int32),
            carry_timer=jnp.array(0, dtype=jnp.int32),
            holder=jnp.array(-1, dtype=jnp.int32),
        )
        return player_state, enemy_state, puck_state

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = None) -> Tuple:
        # Face-off: puck at centre, characters on start positions
        c = self.consts
        player_state, enemy_state, puck_state = self._faceoff_positions()

        enemy_controller = EnemyControllerState(
            target_grid=self._character_grid_position(enemy_state.skater.position),
            move_dx=jnp.array(0, dtype=jnp.int32),
            move_dy=jnp.array(0, dtype=jnp.int32),
            fire=jnp.array(False),
            random_byte=jnp.array(0, dtype=jnp.int32),
        )

        state = IceHockeyState(
            player_state=player_state,
            enemy_state=enemy_state,
            enemy_controller=enemy_controller,
            puck_state=puck_state,
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
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: IceHockeyState, action):
        previous_state = state
        gs = state.game_state

        # no movement and no clock during goal pause, face-off and after game end
        frozen = gs.is_finished | gs.goal_scored | gs.is_faceoff

        # Keep the controller's deterministic random sequence advancing through pauses.
        advanced_controller = state.enemy_controller.replace(
            random_byte=self._next_random_byte(state.enemy_controller.random_byte)
        )

        if isinstance(action, (tuple, list)):
            player_action, enemy_action = action
            new_enemy_controller = advanced_controller
        else:
            player_action = action
            controller_state = state.replace(enemy_controller=advanced_controller)
            policy_action, policy_controller = self._enemy_policy(controller_state)
            # Direction, fire, and target remain unchanged during pauses.
            new_enemy_controller = jax.lax.cond(
                frozen,
                lambda _: advanced_controller,
                lambda _: policy_controller,
                operand=None,
            )
            enemy_action = jnp.where(frozen, Action.NOOP, policy_action)
        character_update = (
            state.counter % self.consts.CHARACTER_UPDATE_CADENCE == 0
        )
        free_puck_update = (
            state.counter % self.consts.FREE_PUCK_UPDATE_CADENCE == 0
        )

        tackle_timer_tick = (
            state.counter % self.consts.TACKLE_TIMER_CADENCE == 0
        )

        def play_step(_):
            player_state = self._tick_tackle_timers(
                state.player_state, tackle_timer_tick
            )
            enemy_state = self._tick_tackle_timers(
                state.enemy_state, tackle_timer_tick
            )

            new_player_state, new_enemy_state, contacts, protections = jax.lax.cond(
                character_update,
                lambda _: self._characters_step(
                    player_state,
                    enemy_state,
                    state.puck_state.position,
                    player_action=player_action,
                    enemy_action=enemy_action,
                ),
                lambda _: (
                    player_state,
                    enemy_state,
                    jnp.zeros(3, dtype=jnp.bool_),
                    jnp.zeros(2, dtype=jnp.bool_),
                ),
                operand=None,
            )

            new_player_state, new_enemy_state, new_puck_state, contact_fire = (
                self._tackle_step(
                    new_player_state,
                    new_enemy_state,
                    state.puck_state,
                    contacts,
                    protections,
                    new_enemy_controller.random_byte,
                    state.counter,
                    gs.player_score,
                    gs.enemy_score,
                )
            )

            new_player_state, new_enemy_state, new_puck_state = self._puck_pickup(
                new_player_state,
                new_enemy_state,
                new_puck_state,
                random_byte=new_enemy_controller.random_byte,
            )

            new_puck_state = self._puck_carry(
                new_player_state,
                new_enemy_state,
                new_puck_state,
                advance_free_puck=free_puck_update,
            )
            new_player_state, new_enemy_state, new_puck_state = self._puck_shoot(
                new_player_state,
                new_enemy_state,
                new_puck_state,
                advance_free_puck=free_puck_update,
            )
            return new_player_state, new_enemy_state, new_puck_state, contact_fire

        def hold_step(_):
            return (
                state.player_state,
                state.enemy_state,
                state.puck_state,
                jnp.array(False),
            )

        new_player_state, new_enemy_state, new_puck_state, contact_fire = jax.lax.cond(
            frozen, hold_step, play_step, None
        )

        if not isinstance(action, (tuple, list)):
            new_enemy_controller = new_enemy_controller.replace(
                fire=new_enemy_controller.fire | contact_fire
            )

        (
            new_player_state,
            new_enemy_state,
            new_puck_state,
            new_game_state,
        ) = self._goal_and_reset_step(
            gs,
            new_player_state,
            new_enemy_state,
            new_puck_state,
            frozen,
        )

        state = state.replace(
            player_state=new_player_state,
            enemy_state=new_enemy_state,
            enemy_controller=new_enemy_controller,
            puck_state=new_puck_state,
            counter=state.counter + 1,
            game_state=new_game_state,
        )

        obs = self._get_observation(state)
        reward = self._get_reward(previous_state, state)
        done = self._get_done(state)
        info = self._get_info(state)
        return obs, state, reward, done, info

    def _next_random_byte(self, value: chex.Array) -> chex.Array:
        """Advance the controller's deterministic 8-bit random sequence."""
        old = value.astype(jnp.int32) & jnp.int32(255)
        feedback = jnp.where(
            old == 0,
            jnp.int32(1),
            ((old >> 7) ^ (old >> 4)) & jnp.int32(1),
        )
        return ((old << 1) & jnp.int32(255)) | feedback

    def _puck_grid_position(self, position: chex.Array) -> chex.Array:
        c = self.consts
        return jnp.array(
            [
                jnp.round(position[0] + c.PUCK_GRID_X_OFFSET),
                jnp.round(c.PUCK_GRID_Y_ORIGIN - position[1]),
            ],
            dtype=jnp.int32,
        ) & jnp.int32(255)

    def _character_grid_position(self, position: chex.Array) -> chex.Array:
        c = self.consts
        return jnp.array(
            [
                jnp.round(position[0] + c.CHARACTER_GRID_X_OFFSET),
                jnp.round(c.CHARACTER_GRID_Y_ORIGIN - position[1]),
            ],
            dtype=jnp.int32,
        ) & jnp.int32(255)

    def _enemy_policy(
        self, state: IceHockeyState
    ) -> Tuple[chex.Array, EnemyControllerState]:
        """Stateful controller for the enemy's active character.

        The controller follows a persistent integer-grid target and refreshes it only on
        selected frames. It chases a jittered point around the puck when defending,
        moves toward randomized goalward targets while carrying, and times shots from
        puck distance and stick phase. Direction and fire inputs persist between
        controller updates.
        """
        c = self.consts
        controller = state.enemy_controller
        enemy = state.enemy_state
        player = state.player_state
        puck = state.puck_state

        counter = state.counter.astype(jnp.int32) & jnp.int32(255)
        random_byte = controller.random_byte.astype(jnp.int32) & jnp.int32(255)
        puck_grid = self._puck_grid_position(puck.position)

        control_tick = (counter & jnp.int32(3)) == jnp.int32(2)

        def update_control(current: EnemyControllerState) -> EnemyControllerState:
            active_is_goalie = enemy.active_character == 1
            active = jax.lax.cond(
                active_is_goalie, lambda: enemy.goalie, lambda: enemy.skater
            )

            refresh_mask = jnp.where(
                state.game_state.player_score >= state.game_state.enemy_score,
                jnp.int32(7),
                jnp.int32(31),
            )
            refresh_target = (counter & refresh_mask) == jnp.int32(2)

            player_has_puck = player.skater.has_puck | player.goalie.has_puck
            enemy_has_puck = enemy.skater.has_puck | enemy.goalie.has_puck
            random_3 = random_byte & jnp.int32(7)
            random_5 = random_byte & jnp.int32(31)
            random_6 = random_byte & jnp.int32(63)
            counter_high = (counter & jnp.int32(128)) != 0

            def refresh(_: None):
                previous_target = current.target_grid

                def chase_puck(__: None):
                    extra_y = jnp.where(player_has_puck & ~counter_high, 8, 0)
                    target_grid_y_raw = (
                        puck_grid[1] - 4 + random_3 - extra_y
                    ) & jnp.int32(255)
                    y_clamped = target_grid_y_raw >= jnp.int32(c.PLAYER_GOAL_GRID_Y)
                    target_grid_y = jnp.where(
                        y_clamped, jnp.int32(0), target_grid_y_raw
                    )
                    target_grid_x = (
                        puck_grid[0]
                        + random_3
                        + jnp.where(active.orientation == 1, -11, -4)
                        + y_clamped.astype(jnp.int32)
                    ) & jnp.int32(255)
                    target_grid = jnp.array(
                        [target_grid_x, target_grid_y], dtype=jnp.int32
                    )
                    return target_grid, jnp.array(False)

                def carry_puck(__: None):
                    goalie_pass = active_is_goalie & counter_high
                    target_grid_x = (
                        jnp.int32(48)
                        + random_6
                        + active_is_goalie.astype(jnp.int32)
                    )

                    player_goalie_down = player.goalie.is_tackled
                    forward_past_goalie = (
                        enemy.skater.position[1] <= player.goalie.position[1]
                    )
                    force_goalward = player_goalie_down | forward_past_goalie
                    shot_distance_carry = (~player_goalie_down) & forward_past_goalie
                    target_grid_y = jnp.where(
                        force_goalward,
                        jnp.int32(c.PLAYER_GOAL_GRID_Y),
                        random_byte | jnp.int32(120),
                    )
                    target_grid = jnp.array(
                        [target_grid_x, target_grid_y], dtype=jnp.int32
                    )

                    near_goal = (
                        puck_grid[1]
                        + random_5
                        + shot_distance_carry.astype(jnp.int32)
                    ) >= jnp.int32(c.PLAYER_GOAL_GRID_Y)
                    slot_low_half = puck.position_stick < 16
                    facing_left = enemy.skater.orientation == 0
                    puck_left = puck_grid[0] < jnp.int32(c.GOAL_CENTER_GRID_X)
                    angle_ok = jnp.logical_xor(
                        slot_low_half, jnp.logical_xor(facing_left, puck_left)
                    )
                    normal_shot = near_goal & angle_ok

                    target_grid = jnp.where(
                        goalie_pass, previous_target, target_grid
                    )
                    fire = goalie_pass | normal_shot
                    return target_grid, fire

                return jax.lax.cond(
                    enemy_has_puck, carry_puck, chase_puck, operand=None
                )

            target_grid, fire = jax.lax.cond(
                refresh_target,
                refresh,
                lambda _: (current.target_grid, current.fire),
                operand=None,
            )

            # Exact equality deliberately preserves the previous direction.
            active_grid = self._character_grid_position(active.position)
            dx = jnp.where(
                active_grid[0] < target_grid[0],
                jnp.int32(1),
                jnp.where(
                    active_grid[0] > target_grid[0],
                    jnp.int32(-1),
                    current.move_dx,
                ),
            )
            dy = jnp.where(
                active_grid[1] < target_grid[1],
                jnp.int32(-1),
                jnp.where(
                    active_grid[1] > target_grid[1],
                    jnp.int32(1),
                    current.move_dy,
                ),
            )
            return current.replace(
                target_grid=target_grid,
                move_dx=dx,
                move_dy=dy,
                fire=fire,
            )

        controller = jax.lax.cond(
            control_tick, update_control, lambda current: current, controller
        )
        action = self._compose_action(
            controller.move_dx, controller.move_dy, controller.fire
        )
        return action, controller

    def _compose_action(
        self, dx: chex.Array, dy: chex.Array, fire: chex.Array
    ) -> chex.Array:
        """Map (dx, dy in {-1, 0, 1}, fire) onto the matching ALE action integer."""
        table = jnp.array(
            [
                [
                    [Action.UPLEFT, Action.UPLEFTFIRE],
                    [Action.UP, Action.UPFIRE],
                    [Action.UPRIGHT, Action.UPRIGHTFIRE],
                ],
                [
                    [Action.LEFT, Action.LEFTFIRE],
                    [Action.NOOP, Action.FIRE],
                    [Action.RIGHT, Action.RIGHTFIRE],
                ],
                [
                    [Action.DOWNLEFT, Action.DOWNLEFTFIRE],
                    [Action.DOWN, Action.DOWNFIRE],
                    [Action.DOWNRIGHT, Action.DOWNRIGHTFIRE],
                ],
            ],
            dtype=jnp.int32,
        )
        return table[dy + 1, dx + 1, fire.astype(jnp.int32)]

    def _goal_and_reset_step(
        self,
        game_state: GameState,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_state: PuckState,
        frozen: chex.Array,
    ) -> Tuple[PlayerState, EnemyState, PuckState, GameState]:
        """Detect goals, advance pause phases, and reset to face-off positions."""
        c = self.consts

        # Goal: a free (not carried) puck touching a goal line inside the mouth.
        puck_pos = puck_state.position
        carried = (
            player_state.skater.has_puck
            | player_state.goalie.has_puck
            | enemy_state.skater.has_puck
            | enemy_state.goalie.has_puck
        )
        # GOAL_X0 and GOAL_X1 are rigid post pixels.  Only the open interval
        # between them is a scoring mouth.
        in_mouth = (puck_pos[0] > c.GOAL_X0) & (puck_pos[0] < c.GOAL_X1)
        player_scored = ~frozen & ~carried & in_mouth & (puck_pos[1] >= c.ENEMY_GOAL_Y)
        enemy_scored = ~frozen & ~carried & in_mouth & (puck_pos[1] <= c.PLAYER_GOAL_Y)
        goal = player_scored | enemy_scored

        # Clock only runs during play; a goal rounds it up to the next full
        # second so the tick restarts fresh after the face-off.
        remaining_time = game_state.remaining_time
        remaining_time = jnp.where(
            goal, ((remaining_time + 59) // 60) * 60, remaining_time
        )
        clock_runs = ~frozen & ~goal
        remaining_time = jnp.where(
            clock_runs, jnp.maximum(remaining_time - 1, 0), remaining_time
        )
        time_up = clock_runs & (remaining_time == 0)

        # Pause phases: goal pause -> face-off countdown -> play.
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

        # After the goal pause everyone snaps back to the face-off spots.
        # times_tackled is a per-match statistic, so the fresh face-off
        # characters inherit it instead of restarting at 0.
        fo_player, fo_enemy, fo_puck = self._faceoff_positions()
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

        # The face-off countdown holds the puck still. Release it when that countdown ends; 
        # movement begins next frame.
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

    def _tick_tackle_timers(self, team, tick: chex.Array):
        """Advance downed-player timers on their fixed cadence."""

        def update(char: CharacterState) -> CharacterState:
            timer = jnp.where(tick, jnp.maximum(char.tackle_timer - 1, 0), char.tackle_timer)
            return char.replace(tackle_timer=timer, is_tackled=timer > 0)

        return team.replace(skater=update(team.skater), goalie=update(team.goalie))

    @staticmethod
    def _signed_byte(value: chex.Array) -> chex.Array:
        value = value.astype(jnp.int32) & jnp.int32(255)
        return jnp.where(value < 128, value, value - 256)

    def _dropped_puck_velocity(
        self, random_byte: chex.Array, second_player_was_hit: chex.Array
    ) -> chex.Array:
        """Small deterministic impulse applied when a puck carrier is knocked down."""
        r = random_byte.astype(jnp.int32) & jnp.int32(255)
        vx = self._signed_byte((r << 1) & jnp.int32(255)).astype(jnp.float32) / 256.0
        vy_first_hits_second = -self._signed_byte(r).astype(jnp.float32) / 256.0
        vy_second_hits_first = (jnp.float32(256) - r.astype(jnp.float32)) / 256.0
        vy = jnp.where(second_player_was_hit, vy_first_hits_second, vy_second_hits_first)
        return jnp.array([vx, vy], dtype=jnp.float32)

    def _tackle_step(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_state: PuckState,
        contacts: chex.Array,
        protections: chex.Array,
        random_byte: chex.Array,
        counter: chex.Array,
        player_score: chex.Array,
        enemy_score: chex.Array,
    ) -> Tuple[PlayerState, EnemyState, PuckState, chex.Array]:
        """Resolve body checks for the three possible opponent contacts."""
        r = random_byte.astype(jnp.int32) & jnp.int32(255)
        phase = r & jnp.int32(7)
        duration = (r & jnp.int32(31)) | jnp.int32(8)

        p_go = player_state.goalie
        e_sk = enemy_state.skater
        p_sk = player_state.skater
        e_go = enemy_state.goalie
        enemy_had_puck = e_sk.has_puck | e_go.has_puck

        def knock_down(char: CharacterState, hit: chex.Array) -> CharacterState:
            return char.replace(
                tackle_timer=jnp.where(hit, duration, char.tackle_timer),
                is_tackled=char.is_tackled | hit,
                has_puck=char.has_puck & ~hit,
                times_tackled=char.times_tackled + hit.astype(jnp.int32),
            )

        def resolve_pair(
            first: CharacterState,
            second: CharacterState,
            touching: chex.Array,
            first_protected: chex.Array = jnp.array(False),
            second_protected: chex.Array = jnp.array(False),
        ):
            pair_active = touching & ~first.is_tackled
            first_hits_second = (
                pair_active
                & (first.shooting_cooldown > 0)
                & (phase == 4)
                & ~second_protected
            )
            second_hits_first = (
                pair_active
                & (second.shooting_cooldown > 0)
                & (phase == 0)
                & ~first_protected
            )

            second_had_puck = second.has_puck
            first_had_puck = first.has_puck
            first = knock_down(first, second_hits_first)
            second = knock_down(second, first_hits_second)

            drop_second = first_hits_second & second_had_puck
            drop_first = second_hits_first & first_had_puck
            dropped = drop_second | drop_first
            drop_velocity = self._dropped_puck_velocity(r, drop_second)
            new_puck = puck_state.replace(
                velocity=jnp.where(dropped, drop_velocity, puck_state.velocity),
                holder=jnp.where(dropped, jnp.int32(-1), puck_state.holder),
                carry_timer=jnp.where(dropped, jnp.int32(0), puck_state.carry_timer),
            )
            return first, second, new_puck, dropped

        # Pair order follows physical contact resolution: skater/goalie, skater/skater,
        # goalie/skater. Later pairs see any knockdown produced by an earlier pair.
        p_sk, e_go, puck_state, _ = resolve_pair(
            p_sk, e_go, contacts[2], second_protected=protections[1]
        )
        e_sk, p_sk, puck_state, _ = resolve_pair(e_sk, p_sk, contacts[1])
        p_go, e_sk, puck_state, _ = resolve_pair(
            p_go, e_sk, contacts[0], first_protected=protections[0]
        )

        player_state = player_state.replace(goalie=p_go, skater=p_sk)
        enemy_state = enemy_state.replace(skater=e_sk, goalie=e_go)

        aggression_window = (counter.astype(jnp.int32) & jnp.int32(255)) < 64
        contact_fire = (
            jnp.any(contacts)
            & ~enemy_had_puck
            & ((player_score >= enemy_score) | aggression_window)
        )
        return player_state, enemy_state, puck_state, contact_fire

    def _resolve_active_characters(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_position: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Choose the controlled character for each team."""
        c = self.consts
        puck_grid = self._puck_grid_position(puck_position)

        # While carried, control selection uses the puck's logical game coordinate,
        # which stays vertically aligned with the holder rather than with the rendered
        # puck sprite. Player-team holders use the holder y directly; enemy-team
        # holders use the corresponding two-unit offset. This keeps a goalie controllable
        # all the way to its legal forward limit without changing the rendered puck.
        player_holds = player_state.skater.has_puck | player_state.goalie.has_puck
        enemy_holds = enemy_state.skater.has_puck | enemy_state.goalie.has_puck
        carried = player_holds | enemy_holds
        holder_grid_y = jnp.where(
            player_state.skater.has_puck,
            self._character_grid_position(player_state.skater.position)[1],
            jnp.where(
                player_state.goalie.has_puck,
                self._character_grid_position(player_state.goalie.position)[1],
                jnp.where(
                    enemy_state.skater.has_puck,
                    self._character_grid_position(enemy_state.skater.position)[1],
                    self._character_grid_position(enemy_state.goalie.position)[1],
                ),
            ),
        )
        carried_grid_y = (
            holder_grid_y + jnp.where(enemy_holds, jnp.int32(2), jnp.int32(0))
        ) & jnp.int32(255)
        puck_grid = puck_grid.at[1].set(jnp.where(carried, carried_grid_y, puck_grid[1]))
        puck_y_even = puck_grid[1] & jnp.int32(254)

        def wrapped_abs_delta(a, b):
            d = (a - b) & jnp.int32(255)
            return jnp.where(d < 128, d, d ^ jnp.int32(255))

        def distance(char: CharacterState):
            char_grid = self._character_grid_position(char.position)
            return wrapped_abs_delta(puck_y_even, char_grid[1]) + wrapped_abs_delta(
                puck_grid[0], char_grid[0]
            )

        player_goalie_distance = distance(player_state.goalie)
        player_skater_distance = distance(player_state.skater)
        enemy_skater_distance = distance(enemy_state.skater)
        enemy_goalie_distance = distance(enemy_state.goalie)

        # Ties favor the forward for the player and the goalie for the enemy.
        player_active = jnp.where(
            player_goalie_distance < player_skater_distance, 1, 0
        ).astype(jnp.int32)
        enemy_active = jnp.where(
            enemy_skater_distance < enemy_goalie_distance, 0, 1
        ).astype(jnp.int32)

        bottom_region = puck_grid[1] < jnp.int32(c.ACTIVE_BOTTOM_THRESHOLD)
        top_region = puck_grid[1] >= jnp.int32(c.ACTIVE_TOP_THRESHOLD)
        player_active = jnp.where(
            bottom_region,
            jnp.int32(0),
            jnp.where(top_region, jnp.int32(1), player_active),
        )
        enemy_active = jnp.where(
            bottom_region,
            jnp.int32(1),
            jnp.where(top_region, jnp.int32(0), enemy_active),
        )

        # A downed character always hands control to its teammate.
        player_active = jnp.where(
            player_state.goalie.is_tackled,
            jnp.int32(0),
            jnp.where(player_state.skater.is_tackled, jnp.int32(1), player_active),
        )
        enemy_active = jnp.where(
            enemy_state.skater.is_tackled,
            jnp.int32(1),
            jnp.where(enemy_state.goalie.is_tackled, jnp.int32(0), enemy_active),
        )
        return player_active, enemy_active

    def _apply_action(
        self,
        character: CharacterState,
        action: chex.Array,
        bounds: chex.Array,
        horizontal_speed: chex.Array,
        vertical_speed: chex.Array,
    ) -> CharacterState:
        """Apply one frame of joystick *input* movement to a single character.

        This is the per-character movement primitive shared by the human player and the
        computer opponent: each chooses an action through its own policy, but the action
        is applied identically here. Directions are absolute screen directions.

        Only the active skater of a team should receive a real action; the inactive
        teammate never moves from input (the caller passes NOOP or simply skips it).
        A tackled character ignores input entirely (it is frozen for the tackle period).

        Args:
            character: The character to move.
            action: The chosen Atari action integer.
            bounds: ``(x_min, x_max, y_min, y_max)`` provisional wall/zone clamp (see above).
            horizontal_speed: Horizontal movement distance for this update.
            vertical_speed: Vertical movement distance for this update.

        Returns:
            The updated ``CharacterState`` (position + orientation; other fields kept).
        """
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

        # A tackled character is frozen: ignore all input movement this frame.
        movable = jnp.logical_not(character.is_tackled)
        dx = jnp.where(
            movable & right,
            horizontal_speed,
            jnp.where(movable & left, -horizontal_speed, 0.0),
        )
        # Screen y grows downward, so DOWN increases y and UP decreases it.
        dy = jnp.where(
            movable & down,
            vertical_speed,
            jnp.where(movable & up, -vertical_speed, 0.0),
        )

        new_x = jnp.clip(character.position[0] + dx, bounds[0], bounds[1])
        new_y = jnp.clip(character.position[1] + dy, bounds[2], bounds[3])
        new_position = jnp.array([new_x, new_y])

        # Orientation: 0 = facing left, 1 = facing right.
        # input keeps the current facing; a tackled character keeps it too (frozen).
        new_orientation = jnp.where(
            movable & right, 1, jnp.where(movable & left, 0, character.orientation)
        )

        # Leg walk-cycle advances whenever the skater has any directional input
        # and freezes on frame 0 when idle (NOOP) or tackled.
        has_input = movable & (up | down | left | right)
        new_walk_counter = jnp.where(has_input, character.walk_counter + 1, 0)

        # Shooting/swing animation: a FIRE press starts the swing.
        # shooting_cooldown counts the swing pose down to 0;
        # a fresh press only (re)starts it when not already
        # swinging, so holding FIRE replays the full swing.
        # A tackled character cannot swing.
        fire = movable & jnp.any(
            jnp.array(
                [
                    action == Action.FIRE,
                    action == Action.UPFIRE,
                    action == Action.DOWNFIRE,
                    action == Action.LEFTFIRE,
                    action == Action.RIGHTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        decremented = jnp.maximum(character.shooting_cooldown - 1, 0)
        new_cooldown = jnp.where(
            fire & (character.shooting_cooldown == 0),
            self.consts.SHOOT_ANIM_FRAMES,
            decremented,
        )

        return character.replace(
            position=new_position,
            orientation=new_orientation,
            walk_counter=new_walk_counter,
            shooting_cooldown=new_cooldown,
        )

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
        horizontal_speed: chex.Array,
        vertical_speed: chex.Array,
    ) -> Tuple[CharacterState, CharacterState]:
        """Apply one team's chosen action as phase-1 intended movement.

        The reframed phase 1: instead of "only the active skater moves", every character
        is handled uniformly by the same ``_apply_action`` — the active skater receives
        the real action and the teammate receives ``NOOP`` (a zero intended delta). The
        active/passive split therefore collapses to "what action does this character get
        this frame", and the inactive teammate simply gets a no-op move.

        This is shared by both teams: the player's action comes from the agent, the
        computer's from ``_enemy_policy``, but routing + application are identical.

        Returns the two characters with their *provisional* (wall/zone-clamped) intended
        positions; the authoritative position is decided later by ``_resolve_interactions``.
        """
        action1 = jnp.where(active == 0, action, Action.NOOP)
        action2 = jnp.where(active == 1, action, Action.NOOP)
        return (
            self._apply_action(
                char1, action1, bounds1, horizontal_speed, vertical_speed
            ),
            self._apply_action(
                char2, action2, bounds2, horizontal_speed, vertical_speed
            ),
        )

    # ------------------------------------------------------------------ #
    # Phase 2 — interaction resolution (pure geometry, single fixed-order pass)
    # ------------------------------------------------------------------ #
    def _separate_opponents(
        self,
        pos_a: chex.Array,
        pos_b: chex.Array,
    ):
        """Resolve close opponent contact and report whether the pair touched."""
        c = self.consts
        delta = pos_a - pos_b
        touching = (
            (delta[0] >= c.CONTACT_X_MIN)
            & (delta[0] <= c.CONTACT_X_MAX)
            & (delta[1] >= c.CONTACT_Y_MIN)
            & (delta[1] <= c.CONTACT_Y_MAX)
        )

        # Ties deliberately choose a direction instead of preserving perfect alignment.
        shift_x = jnp.where(delta[0] >= 0.0, c.CONTACT_PUSH_X, -c.CONTACT_PUSH_X)
        shift_y = jnp.where(delta[1] <= 0.0, -c.CONTACT_PUSH_Y, c.CONTACT_PUSH_Y)
        shift = jnp.where(
            touching,
            jnp.array([shift_x, shift_y], dtype=jnp.float32),
            jnp.zeros(2, dtype=jnp.float32),
        )
        return pos_a + shift, pos_b - shift, touching

    def _enforce_min_vertical(
        self,
        active_pos: chex.Array,
        passive_pos: chex.Array,
        min_vertical_distance: chex.Array,
    ) -> chex.Array:
        """Resolve a same-team overlap: vertical-only push, mover holds / passive yields.

        Distinct from the opponent mechanic. The active (moving) skater keeps its
        position; the passive teammate is displaced along y only so the pair are at
        least ``min_vertical_distance`` apart (e.g. the goalie skating forward pushes
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
        """Authoritative wall/zone clamp: ``(x_min, x_max, y_min, y_max)``."""
        return jnp.array(
            [
                jnp.clip(pos[0], bounds[0], bounds[1]),
                jnp.clip(pos[1], bounds[2], bounds[3]),
            ]
        )

    def _resolve_interactions(
        self,
        player_skater: CharacterState,
        player_goalie: CharacterState,
        enemy_skater: CharacterState,
        enemy_goalie: CharacterState,
        player_active: chex.Array,
        enemy_active: chex.Array,
        min_vertical_distance: chex.Array,
        bounds_p1: chex.Array,
        bounds_p2: chex.Array,
        bounds_e1: chex.Array,
        bounds_e2: chex.Array,
    ):
        """Resolve contacts, teammate spacing, and final movement bounds.

        A single fixed-order pass resolves the three opponent pairs that can meet, then
        applies the teammate spacing constraints and the final rink/zone clamps.
        """
        c = self.consts
        p1, p2 = player_skater.position, player_goalie.position
        e1, e2 = enemy_skater.position, enemy_goalie.position

        player_goalie_protected = (
            self._character_grid_position(p2)[1] >= c.PLAYER_GOALIE_PROTECTED_GRID_Y
        )
        enemy_goalie_protected = (
            self._character_grid_position(e2)[1] < c.ENEMY_GOALIE_PROTECTED_GRID_Y
        )

        # Opponent contacts. The fixed order matters when one skater touches two players.
        p1, e2, contact_2 = self._separate_opponents(p1, e2)
        e1, p1, contact_1 = self._separate_opponents(e1, p1)
        p2, e1, contact_0 = self._separate_opponents(p2, e1)

        # 2) Same-team vertical push — the active skater holds, the teammate yields.
        p2 = jnp.where(
            player_active == 0,
            self._enforce_min_vertical(p1, p2, min_vertical_distance),
            p2,
        )
        p1 = jnp.where(
            player_active == 1,
            self._enforce_min_vertical(p2, p1, min_vertical_distance),
            p1,
        )
        e2 = jnp.where(
            enemy_active == 0,
            self._enforce_min_vertical(e1, e2, min_vertical_distance),
            e2,
        )
        e1 = jnp.where(
            enemy_active == 1,
            self._enforce_min_vertical(e2, e1, min_vertical_distance),
            e1,
        )

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
            jnp.array([contact_0, contact_1, contact_2], dtype=jnp.bool_),
            jnp.array([player_goalie_protected, enemy_goalie_protected], dtype=jnp.bool_),
        )

    # ------------------------------------------------------------------ #
    # Orchestrator — runs phase 1 then phase 2 for all four characters
    # ------------------------------------------------------------------ #
    def _advance_puck_with_walls(
        self, position: chex.Array, velocity: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Advance a puck and resolve the rink boards and rigid goal posts.

        The goal-post pixels use the same clamp-and-reflect rule as the outer
        rink walls.  Their vertical span is the rink-facing goal-height
        pixels at the top and bottom board, while ``GOAL_X0`` and ``GOAL_X1``
        are the left and right post pixels respectively.
        """
        c = self.consts
        tentative = position + velocity

        hit_left = tentative[0] < c.RINK_LEFT
        hit_right = tentative[0] > c.RINK_RIGHT
        hit_top = tentative[1] < c.RINK_TOP
        hit_bottom = tentative[1] > c.RINK_BOTTOM
        resolved_y = jnp.clip(tentative[1], c.RINK_TOP, c.RINK_BOTTOM)

        # A goal-height pixel segment includes its board pixel, so its inward
        # extent is height - 1.  A post only blocks a puck entering the
        # goal mouth from its corresponding outside side; this mirrors the
        # outer-board convention, which permits occupying a wall pixel and
        # reflects only on the attempted step beyond it.
        top_post_depth = c.GOAL_HEIGHT_TOP - 1
        bottom_post_depth = c.GOAL_HEIGHT_BOTTOM - 1
        in_top_post_band = (resolved_y >= c.RINK_TOP) & (
            resolved_y <= c.RINK_TOP + top_post_depth
        )
        in_bottom_post_band = (resolved_y >= c.RINK_BOTTOM - bottom_post_depth) & (
            resolved_y <= c.RINK_BOTTOM
        )
        in_post_band = in_top_post_band | in_bottom_post_band
        hit_goal_left = (
            in_post_band
            & (position[0] <= c.GOAL_X0)
            & (tentative[0] > c.GOAL_X0)
        )
        hit_goal_right = (
            in_post_band
            & (position[0] >= c.GOAL_X1)
            & (tentative[0] < c.GOAL_X1)
        )

        vx = jnp.where(
            hit_left | hit_right | hit_goal_left | hit_goal_right,
            -velocity[0],
            velocity[0],
        )
        vy = jnp.where(hit_top | hit_bottom, -velocity[1], velocity[1])

        board_x = jnp.clip(tentative[0], c.RINK_LEFT, c.RINK_RIGHT)
        resolved_x = jnp.where(
            hit_goal_left,
            jnp.float32(c.GOAL_X0),
            jnp.where(hit_goal_right, jnp.float32(c.GOAL_X1), board_x),
        )
        resolved_position = jnp.array([resolved_x, resolved_y], dtype=jnp.float32)
        resolved_velocity = jnp.array([vx, vy], dtype=jnp.float32)
        return resolved_position, resolved_velocity

    def _decay_puck_velocity(self, velocity: chex.Array) -> chex.Array:
        # Puck velocity uses 1/256-pixel increments. Each component loses roughly
        # 1/64 of its value per puck update, with the smallest quantized speeds
        # retained instead of decaying to zero.
        fixed = jnp.rint(velocity * 256.0).astype(jnp.int32)
        decay = fixed >> jnp.int32(6)
        decay = jnp.where((decay == 0) | (decay == -1), 0, decay)
        fixed = fixed - decay
        return fixed.astype(jnp.float32) / 256.0

    def _puck_step(self, puck: PuckState) -> PuckState:
        vel = self._decay_puck_velocity(puck.velocity)
        pos, vel = self._advance_puck_with_walls(puck.position, vel)
        return puck.replace(position=pos, velocity=vel)

    def _stick_slot(self, carry_timer: chex.Array) -> chex.Array:
        c = self.consts
        n = c.STICK_SLOTS  # 32
        phase = (carry_timer // c.STICK_CADENCE) % (2 * n)  # 0..63
        slot = jnp.where(phase < n, phase, 2 * n - 1 - phase)  # 0..31, 31..0
        return slot.astype(jnp.int32)

    def _visible_stick_pos(self, carry_timer: chex.Array) -> chex.Array:
        # 8 visible carry positions as their own (coarser, faster) triangle
        # wave, independent of the fine 32-slot shot-angle wave in
        # _stick_slot. Same reflecting-triangle shape, so each outer position
        # (0 and 7) is naturally held for 2 ticks at the turnaround while the
        # inner ones hold for 1 tick each: 0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0,...
        c = self.consts
        n = 8
        phase = (carry_timer // c.STICK_VISIBLE_CADENCE) % (2 * n)  # 0..15
        visible = jnp.where(phase < n, phase, 2 * n - 1 - phase)  # 0..7, 7..0
        return visible.astype(jnp.int32)

    def _carried_puck_pos(self, char, visible, stick_dy):
        c = self.consts
        base = jnp.round(char.position)
        t = visible.astype(jnp.float32) / 7.0
        dx = c.STICK_MIN_DX + t * (c.STICK_MAX_DX - c.STICK_MIN_DX)
        x = jnp.where(
            char.orientation == 1,
            base[0] + c.PLAYER_W / 2.0 + dx,
            base[0] + c.PLAYER_W / 2.0 - dx - c.PUCK_W,
        )
        y = base[1] + c.PLAYER_H / 2.0 + stick_dy
        return jnp.array([x, y], dtype=jnp.float32)

    def _puck_pickup(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_state: PuckState,
        random_byte: chex.Array,
    ) -> tuple[PlayerState, EnemyState, PuckState]:
        """Resolve free-puck pickups and steals."""
        chars = (
            player_state.skater,
            player_state.goalie,
            enemy_state.skater,
            enemy_state.goalie,
        )
        indices = jnp.arange(4, dtype=jnp.int32)

        # Character-specific pickup phase and simultaneous-contact priority.
        pickup_rank = jnp.array([2, 0, 1, 3], dtype=jnp.int32)
        puck_grid = self._puck_grid_position(puck_state.position)

        def in_pickup_region(char: CharacterState) -> chex.Array:
            char_grid = self._character_grid_position(char.position)
            dx = (puck_grid[0] - char_grid[0]) & jnp.int32(255)
            dy = (puck_grid[1] - char_grid[1]) & jnp.int32(255)

            left_x_ok = (dx < 8) | (dx >= 249)
            right_x_ok = dx < 18
            x_ok = jnp.where(char.orientation == 0, left_x_ok, right_x_ok)
            y_ok = (dy < 4) | (dy == 255)
            return x_ok & y_ok

        in_range = jnp.array([in_pickup_region(char) for char in chars])
        holds = jnp.array([char.has_puck for char in chars])
        tackled = jnp.array([char.is_tackled for char in chars])
        puck_is_held = jnp.any(holds)

        blocker_timer = jnp.maximum(puck_state.pickup_blocker_timer - 1, 0)
        blocked = (
            (blocker_timer > 0)
            & (indices == puck_state.pickup_blocker)
        )
        eligible = in_range & ~holds & ~tackled & ~blocked

        # A free puck is collected immediately. Stealing an attached puck succeeds
        # on one of four deterministic random phases for each character.
        random_phase = random_byte.astype(jnp.int32) & jnp.int32(255)
        steal_allowed = ((pickup_rank ^ random_phase) & jnp.int32(3)) == 0
        successful = eligible & (~puck_is_held | steal_allowed)

        priority = jnp.where(successful, pickup_rank, jnp.int32(-1))
        acquired = jnp.max(priority) >= 0
        winner = jnp.argmax(priority)
        new_has_puck = jnp.where(acquired, indices == winner, holds)

        blocker_timer = jnp.where(acquired, 0, blocker_timer).astype(jnp.int32)
        pickup_blocker = jnp.where(
            acquired | (blocker_timer == 0),
            jnp.int32(-1),
            puck_state.pickup_blocker,
        )

        player_state = player_state.replace(
            skater=player_state.skater.replace(has_puck=new_has_puck[0]),
            goalie=player_state.goalie.replace(has_puck=new_has_puck[1]),
        )
        enemy_state = enemy_state.replace(
            skater=enemy_state.skater.replace(has_puck=new_has_puck[2]),
            goalie=enemy_state.goalie.replace(has_puck=new_has_puck[3]),
        )
        puck_state = puck_state.replace(
            pickup_blocker=pickup_blocker,
            pickup_blocker_timer=blocker_timer,
        )
        return player_state, enemy_state, puck_state

    def _puck_carry(
        self,
        player_state,
        enemy_state,
        puck_state,
        advance_free_puck: chex.Array,
    ):
        c = self.consts
        p_sk, p_go = player_state.skater, player_state.goalie
        e_sk, e_go = enemy_state.skater, enemy_state.goalie
        anyone_has = p_sk.has_puck | p_go.has_puck | e_sk.has_puck | e_go.has_puck

        holder = jnp.where(
            p_sk.has_puck,
            0,
            jnp.where(
                p_go.has_puck,
                1,
                jnp.where(e_sk.has_puck, 2, jnp.where(e_go.has_puck, 3, -1)),
            ),
        ).astype(jnp.int32)

        same_holder = holder == puck_state.holder
        # A new holder starts at phase 1 of the stick cycle.
        carry_timer = jnp.where(
            anyone_has,
            jnp.where(same_holder, puck_state.carry_timer + 1, jnp.int32(1)),
            jnp.int32(0),
        ).astype(jnp.int32)

        slot = self._stick_slot(carry_timer)
        visible = self._visible_stick_pos(carry_timer)

        carry_pos = jnp.where(
            p_sk.has_puck,
            self._carried_puck_pos(p_sk, visible, stick_dy=c.STICK_DY + 2.0),
            jnp.where(
                p_go.has_puck,
                self._carried_puck_pos(p_go, visible, stick_dy=c.STICK_DY + 2.0),
                jnp.where(
                    e_sk.has_puck,
                    self._carried_puck_pos(e_sk, visible, stick_dy=c.ENEMY_STICK_DY),
                    self._carried_puck_pos(e_go, visible, stick_dy=c.ENEMY_STICK_DY),
                ),
            ),
        )

        free_puck = jax.lax.cond(
            advance_free_puck,
            self._puck_step,
            lambda puck: puck,
            puck_state,
        )
        new_puck_pos = jnp.where(anyone_has, carry_pos, free_puck.position)
        new_puck_vel = jnp.where(
            anyone_has, jnp.zeros(2, dtype=jnp.float32), free_puck.velocity
        )
        return puck_state.replace(
            position=new_puck_pos,
            velocity=new_puck_vel,
            position_stick=slot,
            carry_timer=carry_timer,
            holder=holder,
        )

    def _puck_shoot(
        self,
        player_state,
        enemy_state,
        puck_state,
        advance_free_puck: chex.Array,
    ):
        c = self.consts
        p_sk, p_go = player_state.skater, player_state.goalie
        e_sk, e_go = enemy_state.skater, enemy_state.goalie
        sk_shoots = p_sk.has_puck & (p_sk.shooting_cooldown == c.SHOOT_ANIM_FRAMES)
        go_shoots = p_go.has_puck & (p_go.shooting_cooldown == c.SHOOT_ANIM_FRAMES)
        e_sk_shoots = e_sk.has_puck & (e_sk.shooting_cooldown == c.SHOOT_ANIM_FRAMES)
        e_go_shoots = e_go.has_puck & (e_go.shooting_cooldown == c.SHOOT_ANIM_FRAMES)
        should_shoot = sk_shoots | go_shoots | e_sk_shoots | e_go_shoots

        slot = puck_state.position_stick  # 0..31

        def shot_for(char, team_sign):
            s_eff = jnp.where(char.orientation == 1, slot, 31 - slot)
            vx = (s_eff.astype(jnp.float32) - 16.0) / 8.0
            vy = team_sign * jnp.float32(c.PUCK_MAX_SPEED)
            return jnp.array([vx, vy], dtype=jnp.float32)

        shot_vel = jnp.where(
            sk_shoots,
            shot_for(p_sk, 1.0),
            jnp.where(
                go_shoots,
                shot_for(p_go, 1.0),
                jnp.where(e_sk_shoots, shot_for(e_sk, -1.0), shot_for(e_go, -1.0)),
            ),
        )
        released_puck = puck_state.replace(velocity=shot_vel)
        released_puck = jax.lax.cond(
            should_shoot & advance_free_puck,
            self._puck_step,
            lambda puck: puck,
            released_puck,
        )
        new_velocity = jnp.where(
            should_shoot, released_puck.velocity, puck_state.velocity
        )
        new_position = jnp.where(
            should_shoot, released_puck.position, puck_state.position
        )
        shooter_index = jnp.where(
            sk_shoots,
            jnp.array(0, dtype=jnp.int32),
            jnp.where(
                go_shoots,
                jnp.array(1, dtype=jnp.int32),
                jnp.where(
                    e_sk_shoots,
                    jnp.array(2, dtype=jnp.int32),
                    jnp.array(3, dtype=jnp.int32),
                ),
            ),
        )
        new_pickup_blocker = jnp.where(
            should_shoot, shooter_index, puck_state.pickup_blocker
        )
        new_pickup_blocker_timer = jnp.where(
            should_shoot,
            jnp.int32(c.REGRAB_BLOCK_FRAMES),
            puck_state.pickup_blocker_timer,
        )
        new_player_state = player_state.replace(
            skater=p_sk.replace(has_puck=p_sk.has_puck & ~sk_shoots),
            goalie=p_go.replace(has_puck=p_go.has_puck & ~go_shoots),
        )
        new_enemy_state = enemy_state.replace(
            skater=e_sk.replace(has_puck=e_sk.has_puck & ~e_sk_shoots),
            goalie=e_go.replace(has_puck=e_go.has_puck & ~e_go_shoots),
        )
        return (
            new_player_state,
            new_enemy_state,
            puck_state.replace(
                position=new_position,
                velocity=new_velocity,
                pickup_blocker=new_pickup_blocker,
                pickup_blocker_timer=new_pickup_blocker_timer,
            ),
        )

    def _characters_step(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_position: chex.Array,
        player_action: chex.Array,
        enemy_action: chex.Array,
    ):
        """Advance all four characters and report the contacts seen this update.

        This is the character-movement orchestrator shared by both teams. The movement
        speed, collision tunables, and zone bounds are read from ``self.consts``;
        ``enemy_action`` comes from ``_enemy_policy``. The lower-level geometry
        primitives still take these as parameters so
        they stay generic/unit-testable — only this orchestrator binds them to consts.

        Steps:
          1. Resolve each team's active (controlled) skater = closest to the puck.
          2. Phase 1: apply each team's action as intended input movement (uniformly via
             ``_apply_team_inputs`` — active skater gets the action, teammate gets NOOP).
          3. Phase 2: resolve close opponent contacts, teammate spacing, and bounds.

        Returns the updated ``PlayerState`` and ``EnemyState`` (positions/orientation
        from movement, plus the resolved ``active_character`` for each team).
        """
        c = self.consts
        horizontal_speed = jnp.float32(c.PLAYER_SPEED)
        vertical_speed = jnp.float32(c.PLAYER_VERTICAL_SPEED)
        min_vertical_distance = jnp.float32(c.MIN_VERTICAL_DISTANCE)
        x_min = c.RINK_LEFT
        x_max = c.RINK_RIGHT - c.PLAYER_W
        y_top = c.RINK_TOP - c.PLAYER_H + 9
        y_bot = c.RINK_BOTTOM - c.PLAYER_H
        off = c.ATTACKING_ZONE_OFFSET_Y  # skater: depth of its own defensive zone
        goff = c.GOALIE_FORWARD_OFFSET    # goalie: how far up it may leave its goal
        # Player defends the TOP goal, enemy the BOTTOM. A skater is kept out of its
        # own defensive zone (so it plays toward the goal it attacks). A goalie may
        # leave its goal only up to GOALIE_FORWARD_OFFSET (~2/3 of the ice).
        bounds_player_skater = jnp.array(
            [x_min, x_max, y_top + off, y_bot], dtype=jnp.float32
        )
        bounds_player_goalie = jnp.array(
            [x_min, x_max, y_top, y_bot - goff], dtype=jnp.float32
        )
        bounds_enemy_skater = jnp.array(
            [x_min, x_max, y_top, y_bot - off], dtype=jnp.float32
        )
        bounds_enemy_goalie = jnp.array(
            [x_min, x_max, y_top + goff, y_bot], dtype=jnp.float32
        )

        # 1) Active-skater resolution (per team, against the shared puck).
        active_player, active_enemy = self._resolve_active_characters(
            player_state, enemy_state, puck_position
        )

        # 2) Phase 1 — intended input movement, uniform over each team's two skaters.
        p1, p2 = self._apply_team_inputs(
            player_state.skater,
            player_state.goalie,
            active_player,
            player_action,
            bounds_player_skater,
            bounds_player_goalie,
            horizontal_speed,
            vertical_speed,
        )
        e1, e2 = self._apply_team_inputs(
            enemy_state.skater,
            enemy_state.goalie,
            active_enemy,
            enemy_action,
            bounds_enemy_skater,
            bounds_enemy_goalie,
            horizontal_speed,
            vertical_speed,
        )

        # 3) Phase 2 — resolve interactions across all four post-move characters.
        p1, p2, e1, e2, contacts, protections = self._resolve_interactions(
            p1,
            p2,
            e1,
            e2,
            active_player,
            active_enemy,
            min_vertical_distance,
            bounds_player_skater,
            bounds_player_goalie,
            bounds_enemy_skater,
            bounds_enemy_goalie,
        )

        new_player_state = player_state.replace(
            skater=p1,
            goalie=p2,
            active_character=active_player,
        )
        new_enemy_state = enemy_state.replace(
            skater=e1,
            goalie=e2,
            active_character=active_enemy,
        )
        return new_player_state, new_enemy_state, contacts, protections

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
            player_skater=obj(
                state.player_state.skater.position, c.PLAYER_W, c.PLAYER_H
            ),
            player_goalie=obj(
                state.player_state.goalie.position, c.PLAYER_W, c.PLAYER_H
            ),
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

        return jnp.concatenate(
            [
                flat(obs.player_skater),
                flat(obs.player_goalie),
                flat(obs.enemy_skater),
                flat(obs.enemy_goalie),
                flat(obs.puck),
                jnp.array(
                    [
                        obs.player_score,
                        obs.enemy_score,
                        obs.remaining_time,
                        obs.active_player,
                    ],
                    dtype=jnp.float32,
                ),
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: IceHockeyState) -> IceHockeyInfo:
        return IceHockeyInfo(
            player_score=state.game_state.player_score,
            enemy_score=state.game_state.enemy_score,
            remaining_time=state.game_state.remaining_time,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(
        self, previous_state: IceHockeyState, state: IceHockeyState
    ) -> chex.Array:
        # Reward is the change in goal difference: +1 scored, -1 conceded.
        prev_diff = (
            previous_state.game_state.player_score
            - previous_state.game_state.enemy_score
        )
        diff = state.game_state.player_score - state.game_state.enemy_score
        return (diff - prev_diff).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: IceHockeyState) -> chex.Array:
        return state.game_state.is_finished


class IceHockeyRenderer(JAXGameRenderer):
    # Palette-based renderer. The rink (boards, lines, goals, score bars) is
    # baked into the background, so render() only stamps the moving objects.

    def __init__(
        self,
        consts: Optional[IceHockeyConstants] = None,
        config: Optional[render_utils.RendererConfig] = None,
    ):
        self.consts = consts or IceHockeyConstants()
        super().__init__(self.consts)

        self.config = config or render_utils.RendererConfig(
            game_dimensions=(210, 160), channels=3, downscale=None
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Branch-local sprite folder for now; move to the shared sprite dir later.
        self.sprite_path = os.path.join(
            os.path.dirname(__file__), "sprites", "icehockey"
        )

        final_asset_config = list(self.consts.ASSET_CONFIG)
        debug_red = jnp.array([255, 0, 0, 255], dtype=jnp.uint8)
        debug_dot = debug_red.reshape(1, 1, 4)
        debug_box_h = int(round(self.consts.PICKUP_BOX_H)) + 1
        debug_box_w = int(round(self.consts.PICKUP_BOX_W)) + 1
        debug_pickup_box = jnp.zeros((debug_box_h, debug_box_w, 4), dtype=jnp.uint8)
        debug_pickup_box = debug_pickup_box.at[0, :, :].set(debug_red)
        debug_pickup_box = debug_pickup_box.at[-1, :, :].set(debug_red)
        debug_pickup_box = debug_pickup_box.at[:, 0, :].set(debug_red)
        debug_pickup_box = debug_pickup_box.at[:, -1, :].set(debug_red)
        # colon for the clock, two dots in the blue scoreboard colour
        clock_blue = jnp.array([84, 92, 214, 255], dtype=jnp.uint8)
        clock_colon = jnp.zeros((7, 2, 4), dtype=jnp.uint8)
        clock_colon = clock_colon.at[1, :, :].set(clock_blue)
        clock_colon = clock_colon.at[5, :, :].set(clock_blue)
        final_asset_config.extend(
            [
                {"name": "debug_position_dot", "type": "procedural", "data": debug_dot},
                {
                    "name": "debug_pickup_box",
                    "type": "procedural",
                    "data": debug_pickup_box,
                },
                {"name": "clock_colon", "type": "procedural", "data": clock_colon},
            ]
        )
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_background(
        self, raster: jnp.ndarray, state: IceHockeyState
    ) -> jnp.ndarray:
        """No-op hook for mods to redraw the ice/boards before objects are stamped."""
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: IceHockeyState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self._render_hook_post_background(raster, state)

        puck_m = self.SHAPE_MASKS["puck"]

        # Skater sprites. Player and enemy walking/standing/idle/shooting poses
        p_walk_l = self.SHAPE_MASKS["player_walking_left"]
        p_walk_r = self.SHAPE_MASKS["player_walking_right"]
        e_walk_l = self.SHAPE_MASKS["enemy_walking_left"]
        e_walk_r = self.SHAPE_MASKS["enemy_walking_right"]
        p_idle_l = self.SHAPE_MASKS["player_idle_left"]
        p_idle_r = self.SHAPE_MASKS["player_idle_right"]
        e_idle_l = self.SHAPE_MASKS["enemy_idle_left"]
        e_idle_r = self.SHAPE_MASKS["enemy_idle_right"]
        p_stand_l = self.SHAPE_MASKS["player_active_standing_left"]
        p_stand_r = self.SHAPE_MASKS["player_active_standing_right"]
        e_stand_l = self.SHAPE_MASKS["enemy_active_standing_left"]
        e_stand_r = self.SHAPE_MASKS["enemy_active_standing_right"]
        p_faceoff = self.SHAPE_MASKS["player_faceoff"]
        e_faceoff = self.SHAPE_MASKS["enemy_faceoff"]
        p_tackled = self.SHAPE_MASKS["player_tackled"]
        e_tackled = self.SHAPE_MASKS["enemy_tackled"]
        p_shoot_l = self.SHAPE_MASKS["player_shooting_left"]
        p_shoot_r = self.SHAPE_MASKS["player_shooting_right"]
        e_shoot_l = self.SHAPE_MASKS["enemy_shooting_left"]
        e_shoot_r = self.SHAPE_MASKS["enemy_shooting_right"]
        debug_dot = self.SHAPE_MASKS["debug_position_dot"]
        debug_pickup_box = self.SHAPE_MASKS["debug_pickup_box"]
        cadence = self.consts.ANIM_CADENCE

        def col(pos):
            return jnp.round(pos[0]).astype(jnp.int32)

        def row(pos):
            return jnp.round(pos[1]).astype(jnp.int32)

        def draw_position_dot(r, char):
            return self.jr.render_at_clipped(
                r,
                col(char.position),
                row(char.position),
                debug_dot,
            )

        def pickup_box_pos(char, offset_y):
            offset_x = jnp.where(
                char.orientation == 0,
                self.consts.PICKUP_BOX_OFFSET_X_LEFT,
                self.consts.PICKUP_BOX_OFFSET_X_RIGHT,
            )
            return char.position + jnp.array([offset_x, offset_y])

        def draw_pickup_box(r, char, offset_y):
            box_pos = pickup_box_pos(char, offset_y)
            return self.jr.render_at_clipped(
                r,
                col(box_pos),
                row(box_pos),
                debug_pickup_box,
            )

        def draw_oriented(r, char, left_mask, right_mask):
            return jax.lax.cond(
                char.orientation == 0,
                lambda rr: self.jr.render_at_clipped(
                    rr,
                    col(char.position),
                    row(char.position),
                    left_mask,
                ),
                lambda rr: self.jr.render_at_clipped(
                    rr,
                    col(char.position),
                    row(char.position),
                    right_mask,
                ),
                r,
            )

        def draw_oriented_frame(r, char, left_masks, right_masks, frame):
            return jax.lax.cond(
                char.orientation == 0,
                lambda rr: self.jr.render_at_clipped(
                    rr,
                    col(char.position),
                    row(char.position),
                    left_masks[frame],
                ),
                lambda rr: self.jr.render_at_clipped(
                    rr,
                    col(char.position),
                    row(char.position),
                    right_masks[frame],
                ),
                r,
            )

        def draw_player_shooting(r, char, shoot_frame):
            return jax.lax.cond(
                char.orientation == 0,
                lambda rr: self.jr.render_at_clipped(
                    rr,
                    col(char.position),
                    row(char.position),
                    p_shoot_l[shoot_frame],
                ),
                lambda rr: self.jr.render_at_clipped(
                    rr,
                    col(char.position),
                    row(char.position),
                    p_shoot_r[shoot_frame],
                ),
                r,
            )

        def draw_player(r, char, is_active):
            frame = (char.walk_counter // cadence) % p_walk_l.shape[0]
            moving = char.walk_counter > 0
            shooting = char.shooting_cooldown > 0
            elapsed = self.consts.SHOOT_ANIM_FRAMES - char.shooting_cooldown
            shoot_frame = (elapsed // cadence) % p_shoot_l.shape[0]

            def draw_active(rr):
                return jax.lax.cond(
                    shooting,
                    lambda rrr: draw_player_shooting(rrr, char, shoot_frame),
                    lambda rrr: jax.lax.cond(
                        moving,
                        lambda rrrr: draw_oriented_frame(
                            rrrr, char, p_walk_l, p_walk_r, frame
                        ),
                        lambda rrrr: draw_oriented(rrrr, char, p_stand_l, p_stand_r),
                        rrr,
                    ),
                    rr,
                )

            return jax.lax.cond(
                is_active,
                draw_active,
                lambda rr: draw_oriented(rr, char, p_idle_l, p_idle_r),
                r,
            )

        def draw_enemy(r, char, is_active):
            frame = (char.walk_counter // cadence) % e_walk_l.shape[0]
            moving = char.walk_counter > 0
            shooting = char.shooting_cooldown > 0
            elapsed = self.consts.SHOOT_ANIM_FRAMES - char.shooting_cooldown
            shoot_frame = (elapsed // cadence) % e_shoot_l.shape[0]

            def draw_active(rr):
                return jax.lax.cond(
                    shooting,
                    lambda rrr: draw_oriented_frame(
                        rrr,
                        char,
                        e_shoot_l,
                        e_shoot_r,
                        shoot_frame,
                    ),
                    lambda rrr: jax.lax.cond(
                        moving,
                        lambda rrrr: draw_oriented_frame(
                            rrrr, char, e_walk_l, e_walk_r, frame
                        ),
                        lambda rrrr: draw_oriented(rrrr, char, e_stand_l, e_stand_r),
                        rrr,
                    ),
                    rr,
                )

            return jax.lax.cond(
                is_active,
                draw_active,
                lambda rr: draw_oriented(rr, char, e_idle_l, e_idle_r),
                r,
            )

        def draw_faceoff(r, char, mask):
            return self.jr.render_at_clipped(
                r,
                col(char.position),
                row(char.position),
                mask,
            )

        def draw_tackled(r, char, mask):
            return self.jr.render_at_clipped(
                r,
                col(char.position),
                row(char.position),
                mask,
            )

        def draw_player_with_tackle(r, char, is_active):
            return jax.lax.cond(
                char.is_tackled,
                lambda rr: draw_tackled(rr, char, p_tackled),
                lambda rr: draw_player(rr, char, is_active),
                r,
            )

        def draw_enemy_with_tackle(r, char, is_active):
            return jax.lax.cond(
                char.is_tackled,
                lambda rr: draw_tackled(rr, char, e_tackled),
                lambda rr: draw_enemy(rr, char, is_active),
                r,
            )

        # Active character of each team (0 = skater controlled, 1 = goalie).
        p_act = state.player_state.active_character
        e_act = state.enemy_state.active_character

        # render_at_clipped because skaters can reach the board pixels at the
        # edge; render_at would slice out of bounds there.
        raster = draw_player_with_tackle(raster, state.player_state.goalie, p_act == 1)
        raster = draw_enemy_with_tackle(raster, state.enemy_state.goalie, e_act == 1)
        raster = jax.lax.cond(
            state.player_state.skater.is_tackled,
            lambda r: draw_tackled(r, state.player_state.skater, p_tackled),
            lambda r: jax.lax.cond(
                state.game_state.is_faceoff,
                lambda rr: draw_faceoff(rr, state.player_state.skater, p_faceoff),
                lambda rr: draw_player(rr, state.player_state.skater, p_act == 0),
                r,
            ),
            raster,
        )
        raster = jax.lax.cond(
            state.enemy_state.skater.is_tackled,
            lambda r: draw_tackled(r, state.enemy_state.skater, e_tackled),
            lambda r: jax.lax.cond(
                state.game_state.is_faceoff,
                lambda rr: draw_faceoff(rr, state.enemy_state.skater, e_faceoff),
                lambda rr: draw_enemy(rr, state.enemy_state.skater, e_act == 0),
                r,
            ),
            raster,
        )
        raster = self.jr.render_at_clipped(
            raster,
            col(state.puck_state.position),
            row(state.puck_state.position),
            puck_m,
        )
        if self.consts.DEBUG_RENDER:
            player_offset_y = self.consts.PLAYER_PICKUP_BOX_OFFSET_Y
            enemy_offset_y = self.consts.ENEMY_PICKUP_BOX_OFFSET_Y
            raster = draw_pickup_box(raster, state.player_state.goalie, player_offset_y)
            raster = draw_pickup_box(raster, state.enemy_state.goalie, enemy_offset_y)
            raster = draw_pickup_box(raster, state.player_state.skater, player_offset_y)
            raster = draw_pickup_box(raster, state.enemy_state.skater, enemy_offset_y)
            raster = draw_position_dot(raster, state.player_state.goalie)
            raster = draw_position_dot(raster, state.enemy_state.goalie)
            raster = draw_position_dot(raster, state.player_state.skater)
            raster = draw_position_dot(raster, state.enemy_state.skater)

        dm_blue = self.SHAPE_MASKS["digits"]  # player score (blue team)
        dm_gold = self.SHAPE_MASKS["digits_gold"]  # enemy score (gold team)

        def draw_score(r, value, x_single, x_double, dm):
            digits = self.jr.int_to_digits(value, max_digits=2)
            is_single = value < 10
            start = jax.lax.select(is_single, jnp.int32(1), jnp.int32(0))
            count = jax.lax.select(is_single, jnp.int32(1), jnp.int32(2))
            x = jax.lax.select(is_single, jnp.int32(x_single), jnp.int32(x_double))
            return self.jr.render_label_selective(
                r, x, 14, digits, dm, start, count, spacing=7, max_digits_to_render=2
            )

        # Blue (player) score on the left, gold (enemy) on the right.
        raster = draw_score(raster, state.game_state.player_score, 46, 33, dm_blue)
        raster = draw_score(raster, state.game_state.enemy_score, 110, 103, dm_gold)

        # Clock "M:SS" at the top. remaining_time is in frames; round up so
        # 3:00 stays visible until the first tick.
        clock_secs = (state.game_state.remaining_time + 59) // 60
        clock_min = clock_secs // 60
        clock_sec = clock_secs % 60
        min_digits = self.jr.int_to_digits(clock_min, max_digits=2)
        sec_digits = self.jr.int_to_digits(clock_sec, max_digits=2)
        raster = self.jr.render_label_selective(
            raster, 65, 5, min_digits, dm_blue, 1, 1, spacing=8, max_digits_to_render=2
        )
        raster = self.jr.render_at_clipped(
            raster, 75, 5, self.SHAPE_MASKS["clock_colon"]
        )
        raster = self.jr.render_label_selective(
            raster, 81, 5, sec_digits, dm_blue, 0, 2, spacing=8, max_digits_to_render=2
        )

        frame = self.jr.render_from_palette(raster, self.PALETTE)

        if self.consts.DEBUG_RENDER:
            c = self.consts
            blue = jnp.array([0, 0, 255], dtype=frame.dtype)
            x0 = c.RINK_LEFT
            x1 = min(c.RINK_RIGHT + 1, c.WIDTH)
            player_attack_line_y = c.PLAYER_GOAL_Y + c.ATTACKING_ZONE_OFFSET_Y
            enemy_attack_line_y = c.ENEMY_GOAL_Y - c.ATTACKING_ZONE_OFFSET_Y

            frame = frame.at[player_attack_line_y, x0:x1, :].set(blue)
            frame = frame.at[enemy_attack_line_y, x0:x1, :].set(blue)

        return frame
