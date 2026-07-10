"""Growatt MIN inverter controller using solax_modbus with VPP remote power control.

This controller replaces the single-segment TOU approach with Growatt's VPP
remote power control registers, giving per-period power control without any
TOU segment management.

VPP control entities (via solax_modbus HA integration):
    select.growatt_inverter_vpp_remote_control  — enable/disable per period
    select.growatt_inverter_vpp_status          — enable once at startup
    select.growatt_inverter_vpp_allow_ac_charging — enable/disable AC charging
    number.growatt_inverter_vpp_power           — power % (-100..100)
    number.growatt_inverter_vpp_time            — fallback duration in minutes

Enable sequence:
    VPP Status → wait 1s (once at startup or if disabled)
    VPP Remote Control → written every period together with power commands

VPP Time is written every period (20 min) to reset the fallback timer.
If BESS stops writing for any reason, the inverter returns to load_first
after 20 minutes automatically. Note: TOU control does NOT resume until
VPP Remote Control is explicitly Disabled (via deinitialize_hardware).

Intent → VPP mapping:
    BATTERY_EXPORT (discharge_rate >= 50%) → vpp_power=-100, vpp_control=1
    BATTERY_EXPORT (discharge_rate <  50%) → vpp_power=0,    vpp_control=0
    GRID_CHARGING                          → vpp_power=100,  vpp_control=1
    SOLAR_STORAGE                          → vpp_power=0,    vpp_control=0
    SOLAR_EXPORT                           → vpp_power=0,    vpp_control=1
    LOAD_SUPPORT                           → vpp_power=0,    vpp_control=0
    IDLE                                   → vpp_power=0,    vpp_control=0
"""

import logging
import time
from typing import ClassVar

from . import time_utils
from .dp_schedule import DPSchedule
from .growatt_min_controller import GrowattMinController
from .health_check import perform_health_check
from .settings import BatterySettings

logger = logging.getLogger(__name__)

# VPP entity IDs (via solax_modbus HA integration)
VPP_REMOTE_CONTROL_ENTITY = "select.growatt_inverter_vpp_remote_control"
VPP_STATUS_ENTITY = "select.growatt_inverter_vpp_status"
VPP_ALLOW_AC_CHARGING_ENTITY = "select.growatt_inverter_vpp_allow_ac_charging"
VPP_POWER_ENTITY = "number.growatt_inverter_vpp_power"
VPP_TIME_ENTITY = "number.growatt_inverter_vpp_time"

VPP_ENABLE = "Enabled"
VPP_DISABLE = "Disabled"

# Fallback duration in minutes — inverter returns to load_first if BESS
# stops writing. Must be > 15 (period length) to avoid spurious fallback
# during normal operation.
VPP_FALLBACK_MINUTES = 20

# Discharge rate threshold below which we use load_first instead of
# active VPP export — load_first handles low discharge reactively and better.
VPP_EXPORT_THRESHOLD_PCT = 50


class SolaxModbusGrowattController(GrowattMinController):
    """Growatt MIN controller using VPP remote power control.

    Manages per-period charge/discharge power via VPP registers instead of
    TOU segments. Schedule creation and comparison logic is inherited from
    GrowattMinController via strategic intents.

    VPP state is stored as class variables so it survives instance replacement
    by battery_system_manager each optimization cycle.
    """

    # Class-level VPP state — shared across all instances so that when
    # battery_system_manager replaces the controller with a new instance
    # each optimization cycle, the VPP enable state is preserved and
    # flash registers are not written unnecessarily.
    _class_vpp_status_enabled: bool = False
    _class_vpp_enabled: bool = False
    _class_ac_charging_enabled: bool = False
    _class_last_written_vpp_power: int | None = None

    # VPP controls charge/discharge power directly — disable the separate
    # EMS charging rate register to avoid conflicting with VPP commands.
    supports_charge_rate_control: ClassVar[bool] = False

    def __init__(self, battery_settings: BatterySettings) -> None:
        """Initialize the VPP controller."""
        super().__init__(battery_settings)
        self._active_tou_intervals: list[dict] = []

    # ── Class-level state properties ─────────────────────────────────────────

    @property
    def _vpp_status_enabled(self) -> bool:
        return SolaxModbusGrowattController._class_vpp_status_enabled

    @_vpp_status_enabled.setter
    def _vpp_status_enabled(self, value: bool) -> None:
        SolaxModbusGrowattController._class_vpp_status_enabled = value

    @property
    def _vpp_enabled(self) -> bool:
        return SolaxModbusGrowattController._class_vpp_enabled

    @_vpp_enabled.setter
    def _vpp_enabled(self, value: bool) -> None:
        SolaxModbusGrowattController._class_vpp_enabled = value

    @property
    def _ac_charging_enabled(self) -> bool:
        return SolaxModbusGrowattController._class_ac_charging_enabled

    @_ac_charging_enabled.setter
    def _ac_charging_enabled(self, value: bool) -> None:
        SolaxModbusGrowattController._class_ac_charging_enabled = value

    @property
    def _last_written_vpp_power(self) -> int | None:
        return SolaxModbusGrowattController._class_last_written_vpp_power

    @_last_written_vpp_power.setter
    def _last_written_vpp_power(self, value: int | None) -> None:
        SolaxModbusGrowattController._class_last_written_vpp_power = value

    # ── Abstract property (required by parent) ───────────────────────────────

    @property
    def active_tou_intervals(self) -> list[dict]:
        return self._active_tou_intervals

    @active_tou_intervals.setter
    def active_tou_intervals(self, value: list[dict]) -> None:
        self._active_tou_intervals = value

    # ── Schedule creation ────────────────────────────────────────────────────

    def create_schedule(
        self,
        schedule: DPSchedule,
        current_period: int = 0,
        previous_tou_intervals: list[dict] | None = None,
    ) -> None:
        """Store strategic intents — VPP power is applied per-period.

        Args:
            schedule: DPSchedule containing strategic_intent list.
            current_period: Current 15-minute period (0-95).
            previous_tou_intervals: Unused for VPP approach.
        """
        logger.info("Creating VPP schedule from strategic intents")

        self.strategic_intents = schedule.original_dp_results["strategic_intent"]
        self.current_schedule = schedule

        logger.info(
            "VPP: %d strategic intents loaded",
            len(self.strategic_intents),
        )

        for period in range(1, len(self.strategic_intents)):
            if self.strategic_intents[period] != self.strategic_intents[period - 1]:
                logger.info(
                    "Intent transition at period %d: %s -> %s",
                    period,
                    self.strategic_intents[period - 1],
                    self.strategic_intents[period],
                )

        self._update_tou_display_state()

    # ── VPP enable/disable ───────────────────────────────────────────────────

    def _enable_vpp(self, controller) -> None:
        """Enable VPP control.

        VPP Status is written once (or if disabled) followed by 1s pause.
        VPP Remote Control is written every period together with power commands.
        """
        if not self._vpp_status_enabled:
            logger.info("HARDWARE: VPP Status -> Enabled")
            controller._service_call_with_retry(
                "select",
                "select_option",
                operation="VPP enable status",
                entity_id=VPP_STATUS_ENTITY,
                option=VPP_ENABLE,
            )
            if not controller.test_mode:
                self._vpp_status_enabled = True
            logger.info("HARDWARE: Waiting 1s after enabling VPP Status")
            time.sleep(1)

        logger.info("HARDWARE: VPP Remote Control -> Enabled")
        controller._service_call_with_retry(
            "select",
            "select_option",
            operation="VPP enable remote control",
            entity_id=VPP_REMOTE_CONTROL_ENTITY,
            option=VPP_ENABLE,
        )
        if not controller.test_mode:
            self._vpp_enabled = True

    def _disable_vpp(self, controller) -> None:
        """Disable VPP control cleanly.

        Sequence: Status → wait 1s → Remote Control
        """
        logger.info("HARDWARE: VPP Status -> Disabled")
        controller._service_call_with_retry(
            "select",
            "select_option",
            operation="VPP disable status",
            entity_id=VPP_STATUS_ENTITY,
            option=VPP_DISABLE,
        )
        self._vpp_status_enabled = False
        logger.info("HARDWARE: Waiting 1s before disabling VPP Remote Control")
        time.sleep(1)
        logger.info("HARDWARE: VPP Remote Control -> Disabled")
        controller._service_call_with_retry(
            "select",
            "select_option",
            operation="VPP disable remote control",
            entity_id=VPP_REMOTE_CONTROL_ENTITY,
            option=VPP_DISABLE,
        )
        self._vpp_enabled = False

    def deinitialize_hardware(self, controller) -> None:
        """Disable VPP control cleanly on BESS shutdown.

        Mirrors initialize_hardware. Should be called from battery_system_manager
        shutdown hook, e.g. via SIGTERM handler in the addon entry point:

            signal.signal(signal.SIGTERM, lambda s, f: manager.deinitialize_hardware())

        Without this call, the inverter remains in VPP mode until the 20-minute
        fallback timer expires and returns to load_first. TOU control will NOT
        resume until VPP Remote Control is explicitly Disabled.
        """
        if self._vpp_enabled or self._vpp_status_enabled:
            try:
                self._disable_vpp(controller)
                logger.info("VPP shutdown complete")
            except Exception as e:
                logger.error("FAILED: VPP shutdown: %s", e)
        else:
            logger.info("VPP already disabled, no shutdown action needed")

    # ── Intent → VPP power ───────────────────────────────────────────────────

    def _intent_to_vpp(
        self, intent: str, discharge_rate: int, grid_charge: bool,
        current_soc: float | None = None
    ) -> tuple[int, int]:
        """Convert strategic intent to (vpp_power, vpp_control).

        vpp_control: 1 = VPP active (Remote Control Enabled)
                     0 = load first (Remote Control Disabled)
        vpp_power:   -100 = full discharge/export
                      0   = no active power command
                      100 = full charge

        Args:
            intent: Strategic intent string
            discharge_rate: Discharge rate 0-100% from schedule
            grid_charge: Whether grid charging is active
            current_soc: Current battery SOC (0-100%), used for low export decision

        Returns:
            Tuple of (vpp_power, vpp_control)
        """
        if intent == "BATTERY_EXPORT":
            if discharge_rate >= VPP_EXPORT_THRESHOLD_PCT:
                return -100, 1
            else:
                # Low discharge rate — use load first if battery full,
                # otherwise grid first with actual power for reactive control
                if current_soc is not None and current_soc >= 100:
                    return 0, 0
                return -discharge_rate, 1
        elif intent == "GRID_CHARGING":
            # TODO: AC charging power set to 40% — change value here if needed
            return 40, 1
        else:
            # SOLAR_STORAGE, LOAD_SUPPORT, IDLE
            return 0, 0

    # ── Hardware interface ────────────────────────────────────────────────────

    def apply_period(
        self, controller, grid_charge: bool, discharge_rate: int
    ) -> tuple[bool, str]:
        """Write VPP power setting for the current period.

        Called every 15 minutes by BESS. Always resets the fallback timer
        by writing VPP Time, so the inverter returns to load_first if BESS
        stops for any reason.

        Args:
            controller: HomeAssistantAPIController instance
            grid_charge: Whether grid charging is active this period
            discharge_rate: Discharge power rate (0-100%), post-inhibit

        Returns:
            Tuple of (success, error_message).
        """
        errors = []
        now = time_utils.now()
        current_period = now.hour * 4 + now.minute // 15

        intent = "IDLE"
        if current_period < len(self.strategic_intents):
            intent = self.strategic_intents[current_period]

        # Read current SOC for low BATTERY_EXPORT decision
        current_soc = controller.get_battery_soc()

        vpp_power, vpp_control = self._intent_to_vpp(
            intent, discharge_rate, grid_charge, current_soc
        )

        logger.info(
            "Period %d (%02d:%02d): intent=%s discharge_rate=%d%% "
            "vpp_power=%d%% vpp_control=%d",
            current_period,
            now.hour,
            now.minute,
            intent,
            discharge_rate,
            vpp_power,
            vpp_control,
        )

        # VPP Status written once; Remote Control set based on vpp_control
        try:
            if vpp_control == 1:
                self._enable_vpp(controller)
            else:
                # Load first — disable Remote Control so inverter manages naturally
                if not self._vpp_status_enabled:
                    # Still need Status enabled for VPP to work when needed
                    logger.info("HARDWARE: VPP Status -> Enabled (one-time)")
                    controller._service_call_with_retry(
                        "select",
                        "select_option",
                        operation="VPP enable status",
                        entity_id=VPP_STATUS_ENTITY,
                        option=VPP_ENABLE,
                    )
                    if not controller.test_mode:
                        self._vpp_status_enabled = True
                    time.sleep(1)
                if self._vpp_enabled:
                    logger.info("HARDWARE: VPP Remote Control -> Disabled (load first)")
                    controller._service_call_with_retry(
                        "select",
                        "select_option",
                        operation="VPP disable remote control (load first)",
                        entity_id=VPP_REMOTE_CONTROL_ENTITY,
                        option=VPP_DISABLE,
                    )
                    if not controller.test_mode:
                        self._vpp_enabled = False
        except Exception as e:
            logger.error("FAILED: Set VPP control: %s", e)
            errors.append(str(e))

        # Reset fallback timer only when VPP control is active
        if vpp_control == 1:
            try:
                logger.info(
                    "HARDWARE: VPP Time -> %d min (fallback timer reset)",
                    VPP_FALLBACK_MINUTES,
                )
                controller._service_call_with_retry(
                    "number",
                    "set_value",
                    operation="VPP reset fallback timer",
                    entity_id=VPP_TIME_ENTITY,
                    value=VPP_FALLBACK_MINUTES,
                )
            except Exception as e:
                logger.error("FAILED: Reset VPP timer: %s", e)
                errors.append(str(e))

        # AC charging is permanently enabled — no dynamic control needed

        # Signal reactive automation when VPP is active with low power —
        # automation varies VPP Power to keep grid exchange near zero.
        # Not needed for full export (>= 50%) or charging (fixed values).
        try:
            reactive_control = (
                vpp_control == 1
                and intent == "BATTERY_EXPORT"
                and discharge_rate < VPP_EXPORT_THRESHOLD_PCT
            )
            controller._service_call_with_retry(
                "input_boolean",
                "turn_on" if reactive_control else "turn_off",
                operation="VPP reactive control signal",
                entity_id="input_boolean.vpp_reactive_control",
            )
        except Exception as e:
            logger.error("FAILED: Set VPP reactive control signal: %s", e)

        # Write VPP power — only on change
        if vpp_power != self._last_written_vpp_power:
            try:
                logger.info(
                    "HARDWARE: VPP power %s%% -> %d%%",
                    self._last_written_vpp_power,
                    vpp_power,
                )
                controller._service_call_with_retry(
                    "number",
                    "set_value",
                    operation=f"VPP set power -> {vpp_power}%",
                    entity_id=VPP_POWER_ENTITY,
                    value=vpp_power,
                )
                if not controller.test_mode:
                    self._last_written_vpp_power = vpp_power
            except Exception as e:
                logger.error("FAILED: Set VPP power to %d%%: %s", vpp_power, e)
                errors.append(str(e))
        else:
            logger.debug("VPP power unchanged at %d%%, skipping write", vpp_power)

        if errors:
            return False, "; ".join(errors)
        return True, ""

    def write_schedule_to_hardware(
        self,
        controller,
        effective_period: int,
        current_tou: list,
    ) -> tuple[int, int]:
        """Enable VPP and write initial power for the current period.

        Returns:
            Tuple of (writes, disables) — disables always 0 for VPP.
        """
        now = time_utils.now()
        current_period = now.hour * 4 + now.minute // 15

        intent = "IDLE"
        if current_period < len(self.strategic_intents):
            intent = self.strategic_intents[current_period]

        # Get discharge rate from schedule actions if available
        discharge_rate = 0
        if self.current_schedule:
            actions = self.current_schedule.original_dp_results.get("action", [])
            if current_period < len(actions):
                action_kwh = actions[current_period]
                if action_kwh < 0 and self.battery_settings.max_discharge_power_kw > 0:
                    discharge_rate = int(
                        min(
                            abs(action_kwh * 4)
                            / self.battery_settings.max_discharge_power_kw
                            * 100,
                            100,
                        )
                    )

        grid_charge = intent == "GRID_CHARGING"
        current_soc = controller.get_battery_soc()
        vpp_power, vpp_control = self._intent_to_vpp(
            intent, discharge_rate, grid_charge, current_soc
        )

        try:
            if vpp_control == 1:
                self._enable_vpp(controller)
                logger.info(
                    "HARDWARE: VPP Time -> %d min (fallback timer)",
                    VPP_FALLBACK_MINUTES,
                )
                controller._service_call_with_retry(
                    "number",
                    "set_value",
                    operation="VPP set fallback timer (initial)",
                    entity_id=VPP_TIME_ENTITY,
                    value=VPP_FALLBACK_MINUTES,
                )
            else:
                if self._vpp_status_enabled and not self._vpp_enabled:
                    pass  # Status already enabled, Remote Control already disabled
                elif not self._vpp_status_enabled:
                    logger.info("HARDWARE: VPP Status -> Enabled (one-time)")
                    controller._service_call_with_retry(
                        "select",
                        "select_option",
                        operation="VPP enable status (initial)",
                        entity_id=VPP_STATUS_ENTITY,
                        option=VPP_ENA
