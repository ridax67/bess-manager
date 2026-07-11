"""Growatt MIN inverter controller using solax_modbus with VPP remote power control.

This controller replaces the single-segment TOU approach with Growatt's VPP
remote power control registers, giving per-period power control without any
TOU segment management.

VPP control entities (via solax_modbus HA integration):
    select.growatt_inverter_vpp_remote_control  — enable/disable per period
    select.growatt_inverter_vpp_status          — enable once at startup
    select.growatt_inverter_vpp_allow_ac_charging — permanently enabled
    number.growatt_inverter_vpp_power           — power % (-100..100)
    number.growatt_inverter_vpp_time            — fallback duration in minutes
    number.growatt_inverter_max_charge_power_from_grid — desired AC charge power (W)

Reactive control signal:
    input_select.vpp_reactive_control — tri-state for reactive automation:
        "off"          — no reactive control
        "grid_first"   — vary VPP Power between -discharge_rate and 0
        "battery_first"— vary VPP Power around charge start value

Enable sequence:
    VPP Status → wait 1s (once at startup or if disabled)
    VPP Remote Control → written every period together with power commands

VPP Time is written every period when VPP control is active (20 min) to
reset the fallback timer. If BESS stops writing, inverter returns to
load_first after 20 minutes automatically.

Intent → VPP mapping:
    BATTERY_EXPORT (SOC=100%, rate<50%)  → vpp_power=0,            vpp_control=0 (load first)
    BATTERY_EXPORT (SOC<100% or rate>=50%)→ vpp_power=-rate,        vpp_control=1
    GRID_CHARGING                        → vpp_power=<calculated>,  vpp_control=1
    SOLAR_STORAGE                        → vpp_power=0,             vpp_control=0
    LOAD_SUPPORT                         → vpp_power=0,             vpp_control=0
    IDLE                                 → vpp_power=0,             vpp_control=0

Reactive automation signal (input_select.vpp_reactive_control):
    BATTERY_EXPORT rate<50%, SOC<100%   → "grid_first"
    GRID_CHARGING (first period only)   → "battery_first"
    all others                          → "off"
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
VPP_MAX_CHARGE_POWER_ENTITY = "number.growatt_inverter_max_charge_power_from_grid"

# Reactive control signal entity (input_select with options: off, grid_first, battery_first)
VPP_REACTIVE_CONTROL_ENTITY = "input_select.vpp_reactive_control"
REACTIVE_OFF = "off"
REACTIVE_GRID_FIRST = "grid_first"
REACTIVE_BATTERY_FIRST = "battery_first"

VPP_ENABLE = "Enabled"
VPP_DISABLE = "Disabled"

# Fallback duration in minutes — inverter returns to load_first if BESS
# stops writing. Must be > 15 (period length) to avoid spurious fallback
# during normal operation.
VPP_FALLBACK_MINUTES = 20

# Discharge rate threshold — below this, reactive automation handles export.
# Above this, we use the actual discharge_rate directly.
VPP_EXPORT_THRESHOLD_PCT = 50


class SolaxModbusGrowattController(GrowattMinController):
    """Growatt MIN controller using VPP remote power control."""

    # Class-level VPP state — shared across all instances so that when
    # battery_system_manager replaces the controller with a new instance
    # each optimization cycle, the VPP enable state is preserved and
    # flash registers are not written unnecessarily.
    _class_vpp_status_enabled: bool = False
    _class_vpp_enabled: bool = False
    _class_last_written_vpp_power: int | None = None
    _class_last_intent: str | None = None

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
    def _last_written_vpp_power(self) -> int | None:
        return SolaxModbusGrowattController._class_last_written_vpp_power

    @_last_written_vpp_power.setter
    def _last_written_vpp_power(self, value: int | None) -> None:
        SolaxModbusGrowattController._class_last_written_vpp_power = value

    @property
    def _last_intent(self) -> str | None:
        return SolaxModbusGrowattController._class_last_intent

    @_last_intent.setter
    def _last_intent(self, value: str | None) -> None:
        SolaxModbusGrowattController._class_last_intent = value

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
        """Store strategic intents — VPP power is applied per-period."""
        logger.info("Creating VPP schedule from strategic intents")

        self.strategic_intents = schedule.original_dp_results["strategic_intent"]
        self.current_schedule = schedule

        logger.info("VPP: %d strategic intents loaded", len(self.strategic_intents))

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

    def _disable_vpp_remote_control(self, controller) -> None:
        """Disable VPP Remote Control (load first mode)."""
        if not self._vpp_status_enabled:
            logger.info("HARDWARE: VPP Status -> Enabled (one-time)")
            controller._service_call_with_retry(
                "select",
                "select_option",
                operation="VPP enable status (one-time)",
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

    def _disable_vpp(self, controller) -> None:
        """Disable VPP control fully on shutdown.

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
        shutdown hook, e.g. via SIGTERM handler in the addon entry point.
        Without this call, TOU control will NOT resume until VPP Remote Control
        is explicitly Disabled.
        """
        if self._vpp_enabled or self._vpp_status_enabled:
            try:
                self._disable_vpp(controller)
                logger.info("VPP shutdown complete")
            except Exception as e:
                logger.error("FAILED: VPP shutdown: %s", e)
        else:
            logger.info("VPP already disabled, no shutdown action needed")

    # ── Charge power calculation ─────────────────────────────────────────────

    def _calculate_charge_power_pct(self, controller) -> int:
        """Calculate VPP charge power percentage from HA entity and battery settings.

        Reads desired AC charge power from number.growatt_inverter_max_charge_power_from_grid
        and divides by inverter max charge power from battery_settings.

        Returns:
            Charge power as percentage (0-100). Falls back to 40% if unavailable.
        """
        try:
            response = controller.get_entity_state_raw(VPP_MAX_CHARGE_POWER_ENTITY)
            if response and "state" in response:
                desired_w = float(response["state"])
                max_w = self.battery_settings.max_charge_power_kw * 1000
                if max_w > 0:
                    pct = round(desired_w / max_w * 100)
                    pct = max(0, min(100, pct))
                    logger.info(
                        "VPP charge power: %.0fW / %.0fW = %d%%",
                        desired_w,
                        max_w,
                        pct,
                    )
                    return pct
        except Exception as e:
            logger.warning("Could not read max charge power entity: %s", e)

        # TODO: fallback charge power — change if needed
        logger.info("VPP charge power: using fallback 40%%")
        return 40

    # ── Intent → VPP power ───────────────────────────────────────────────────

    def _intent_to_vpp(
        self,
        intent: str,
        discharge_rate: int,
        charge_power_pct: int,
        current_soc: float | None = None,
    ) -> tuple[int, int]:
        """Convert strategic intent to (vpp_power, vpp_control).

        vpp_control: 1 = VPP active (Remote Control Enabled)
                     0 = load first (Remote Control Disabled)
        vpp_power:   negative = discharge/export to grid
                     0        = no active power command
                     positive = charge from grid

        Args:
            intent: Strategic intent string
            discharge_rate: Discharge rate 0-100% from schedule
            charge_power_pct: Calculated charge power percentage
            current_soc: Current battery SOC (0-100%)

        Returns:
            Tuple of (vpp_power, vpp_control)
        """
        if intent == "BATTERY_EXPORT":
            # Load first if battery full and low discharge — solar exports naturally
            if discharge_rate < VPP_EXPORT_THRESHOLD_PCT and (
                current_soc is not None and current_soc >= 100
            ):
                return 0, 0
            # Otherwise use actual discharge rate — reactive automation handles low values
            return -discharge_rate, 1
        elif intent == "GRID_CHARGING":
            return charge_power_pct, 1
        else:
            # SOLAR_STORAGE, LOAD_SUPPORT, IDLE
            return 0, 0

    def _get_reactive_signal(
        self, intent: str, discharge_rate: int, vpp_control: int, is_new_intent: bool
    ) -> str:
        """Determine reactive control signal for automation.

        Args:
            intent: Current strategic intent
            discharge_rate: Current discharge rate
            vpp_control: Computed vpp_control value
            is_new_intent: True if intent changed from previous period

        Returns:
            One of: REACTIVE_OFF, REACTIVE_GRID_FIRST, REACTIVE_BATTERY_FIRST
        """
        if intent == "BATTERY_EXPORT" and vpp_control == 1 and discharge_rate < VPP_EXPORT_THRESHOLD_PCT:
            return REACTIVE_GRID_FIRST
        elif intent == "GRID_CHARGING":
            return REACTIVE_BATTERY_FIRST
        else:
            return REACTIVE_OFF

    # ── Hardware interface ────────────────────────────────────────────────────

    def apply_period(
        self, controller, grid_charge: bool, discharge_rate: int
    ) -> tuple[bool, str]:
        """Write VPP power setting for the current period."""
        errors = []
        now = time_utils.now()
        current_period = now.hour * 4 + now.minute // 15

        intent = "IDLE"
        if current_period < len(self.strategic_intents):
            intent = self.strategic_intents[current_period]

        # Read current SOC for BATTERY_EXPORT load first decision
        current_soc = controller.get_battery_soc()

        # Calculate charge power from HA entity only when needed
        charge_power_pct = 0
        if intent == "GRID_CHARGING":
            charge_power_pct = self._calculate_charge_power_pct(controller)

        vpp_power, vpp_control = self._intent_to_vpp(
            intent, discharge_rate, charge_power_pct, current_soc
        )

        is_new_intent = intent != self._last_intent

        logger.info(
            "Period %d (%02d:%02d): intent=%s discharge_rate=%d%% soc=%s%% "
            "vpp_power=%d%% vpp_control=%d new_intent=%s",
            current_period,
            now.hour,
            now.minute,
            intent,
            discharge_rate,
            f"{current_soc:.0f}" if current_soc is not None else "?",
            vpp_power,
            vpp_control,
            is_new_intent,
        )

        # Set VPP Remote Control based on vpp_control
        try:
            if vpp_control == 1:
                self._enable_vpp(controller)
            else:
                self._disable_vpp_remote_control(controller)
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

        # Write VPP power only when intent changes — reactive automation
        # handles subsequent periods in the same intent series.
        if is_new_intent and vpp_power != self._last_written_vpp_power:
            try:
                logger.info(
                    "HARDWARE: VPP power %s%% -> %d%% (new intent: %s)",
                    self._last_written_vpp_power,
                    vpp_power,
                    intent,
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
            logger.debug(
                "VPP power write skipped — same intent series (intent=%s power=%d%%)",
                intent,
                vpp_power,
            )

        # Set reactive control signal
        reactive = self._get_reactive_signal(intent, discharge_rate, vpp_control, is_new_intent)
        try:
            controller._service_call_with_retry(
                "input_select",
                "select_option",
                operation=f"VPP reactive control -> {reactive}",
                entity_id=VPP_REACTIVE_CONTROL_ENTITY,
                option=reactive,
            )
        except Exception as e:
            logger.error("FAILED: Set VPP reactive control signal: %s", e)

        if not controller.test_mode:
            self._last_intent = intent

        if errors:
            return False, "; ".join(errors)
        return True, ""

    def write_schedule_to_hardware(
        self,
        controller,
        effective_period: int,
        current_tou: list,
    ) -> tuple[int, int]:
        """Enable VPP and write initial power for the current period."""
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

        current_soc = controller.get_battery_soc()

        charge_power_pct = 0
        if intent == "GRID_CHARGING":
            charge_power_pct = self._calculate_charge_power_pct(controller)

        grid_charge = intent == "GRID_CHARGING"
        vpp_power, vpp_control = self._intent_to_vpp(
            intent, discharge_rate, charge_power_pct, current_soc
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
                self._disable_vpp_remote_control(controller)

            logger.info("HARDWARE: VPP Power -> %d%%", vpp_power)
            controller._service_call_with_retry(
                "number",
                "set_value",
                operation=f"VPP set initial power -> {vpp_power}%",
                entity_id=VPP_POWER_ENTITY,
                value=vpp_power,
            )
            if not controller.test_mode:
                self._last_written_vpp_power = vpp_power

            logger.info(
                "VPP: Initial write — power=%d%% control=%d (period %d, intent %s)",
                vpp_power,
                vpp_control,
                current_period,
                intent,
            )
            return 1, 0
        except Exception as e:
            logger.error("FAILED: VPP initial write: %s", e)
            return 0, 0

    def read_and_initialize_from_hardware(self, controller, current_hour: int) -> None:
        """Read VPP state from hardware and seed internal state."""
        self.current_hour = current_hour
        try:
            rc = controller.get_entity_state_raw(VPP_REMOTE_CONTROL_ENTITY)
            self._vpp_enabled = rc["state"] == VPP_ENABLE if rc else False

            status = controller.get_entity_state_raw(VPP_STATUS_ENTITY)
            self._vpp_status_enabled = status["state"] == VPP_ENABLE if status else False

            power = controller.get_entity_state_raw(VPP_POWER_ENTITY)
            self._last_written_vpp_power = (
                int(float(power["state"])) if power else None
            )

            logger.info(
                "VPP: Initialised from hardware — remote_control=%s "
                "status=%s power=%s%%",
                self._vpp_enabled,
                self._vpp_status_enabled,
                self._last_written_vpp_power,
            )
        except Exception as e:
            logger.warning("VPP: Could not read hardware state: %s — resetting", e)
            self._vpp_enabled = False
            self._vpp_status_enabled = False
            self._last_written_vpp_power = None

        self._update_tou_display_state()

    def initialize_hardware(self, controller) -> None:
        """Sync SOC limits and enable AC charging permanently."""
        self.sync_soc_limits(controller)
        try:
            response = controller.get_entity_state_raw(VPP_ALLOW_AC_CHARGING_ENTITY)
            if response and response.get("state") == VPP_ENABLE:
                logger.info("VPP Allow AC charging already Enabled, skipping write")
            else:
                logger.info("HARDWARE: VPP Allow AC charging -> Enabled (permanent)")
                controller._service_call_with_retry(
                    "select",
                    "select_option",
                    operation="VPP enable AC charging (permanent)",
                    entity_id=VPP_ALLOW_AC_CHARGING_ENTITY,
                    option=VPP_ENABLE,
                )
        except Exception as e:
            logger.error("FAILED: Set VPP Allow AC charging: %s", e)

    # ── Schedule comparison ──────────────────────────────────────────────────

    def compare_schedules(
        self,
        other_schedule: "SolaxModbusGrowattController",
        from_period: int = 0,
    ) -> tuple[bool, str]:
        """Compare schedules by strategic intent list."""
        current = self.strategic_intents
        new = other_schedule.strategic_intents

        if not current and not new:
            return False, ""

        if len(current) != len(new):
            return True, f"Intent count differs: {len(current)} vs {len(new)}"

        for period in range(from_period, len(current)):
            if current[period] != new[period]:
                logger.info(
                    "DECISION: Intent differs at period %d — current=%s new=%s",
                    period,
                    current[period],
                    new[period],
                )
                return True, f"Strategic intents differ from period {period}"

        logger.info("DECISION: Schedules match")
        return False, ""

    # ── TOU display (kept for API/UI compatibility) ───────────────────────────

    def _update_tou_display_state(self) -> None:
        """Update TOU interval lists for API/display compatibility."""
        groups = self.get_detailed_period_groups()
        if not groups:
            self.tou_intervals = []
            self._active_tou_intervals = []
            return

        now = time_utils.now()
        current_p = now.hour * 4 + now.minute // 15
        segments = []
        for group in groups:
            mode = self.INTENT_TO_MODE.get(group["intent"], "load_first")
            is_current = group["start_period"] <= current_p <= group["end_period"]
            segments.append(
                {
                    "segment_id": len(segments) + 1,
                    "start_time": group["start_time"],
                    "end_time": group["end_time"],
                    "batt_mode": mode,
                    "enabled": mode != "load_first",
                    "is_default": mode == "load_first",
                    "is_current": is_current,
                    "strategic_intent": group["intent"],
                }
            )
        self.tou_intervals = segments
        self._active_tou_intervals = segments

    def get_daily_TOU_settings(self) -> list[dict]:
        return [seg.copy() for seg in self.tou_intervals]

    def get_all_tou_segments(self, current_period: int | None = None):
        self._update_tou_display_state()
        return self.tou_intervals

    def log_current_TOU_schedule(self, header=None) -> None:
        if header:
            logger.info(header)
        logger.info(
            "VPP: remote_control=%s status=%s power=%s%% last_intent=%s",
            VPP_ENABLE if self._vpp_enabled else VPP_DISABLE,
            VPP_ENABLE if self._vpp_status_enabled else VPP_DISABLE,
            self._last_written_vpp_power,
            self._last_intent,
        )

    # ── Health check ─────────────────────────────────────────────────────────

    def check_health(self, controller) -> list:
        """Check VPP control entity availability."""
        health_check = perform_health_check(
            component_name="Battery Control",
            description="Controls battery via VPP remote power control",
            is_required=True,
            controller=controller,
            all_methods=[
                "get_charging_power_rate",
                "get_discharging_power_rate",
                "get_charge_stop_soc",
                "get_discharge_stop_soc",
            ],
        )

        vpp_entities = [
            VPP_REMOTE_CONTROL_ENTITY,
            VPP_STATUS_ENTITY,
            VPP_ALLOW_AC_CHARGING_ENTITY,
            VPP_POWER_ENTITY,
            VPP_TIME_ENTITY,
            VPP_MAX_CHARGE_POWER_ENTITY,
        ]
        for entity_id in vpp_entities:
            try:
                response = controller.get_entity_state_raw(entity_id)
                status = "OK" if response is not None else "ERROR"
                error = None if response is not None else "Entity not found or unavailable"
            except Exception as e:
                status = "ERROR"
                error = str(e)

            health_check["checks"].append(
                {
                    "name": f"VPP Entity: {entity_id}",
                    "key": entity_id,
                    "method_name": None,
                    "entity_id": entity_id,
                    "status": status,
                    "rawValue": None,
                    "displayValue": entity_id,
                    "error": error,
                }
            )

        has_error = any(c["status"] == "ERROR" for c in health_check["checks"])
        if has_error:
            health_check["status"] = "ERROR"

        return [health_check]
