from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.modification import JaxAtariInternalModPlugin


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
